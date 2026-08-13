"""Generate exact-chance Q labels and distill them into a DQN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import secrets
from time import perf_counter
from typing import Any

import numpy as np

from engine import GameSession
from solver.RL import (
    ACTION_COUNT,
    DistillationConfig,
    ExactChanceDQNPolicy,
    PASS_ACTION_INDEX,
    TeacherDatabase,
    distill_teacher_database,
    encode_decision,
    pack_teacher_sample,
)

from .records import write_json


@dataclass(slots=True)
class _TeacherGame:
    ordinal: int
    seed: int
    validation: bool
    game: GameSession
    samples: list[tuple[bytes, bytes, bytes]] = field(default_factory=list)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _load_excluded_seeds(path: Path | None) -> set[int]:
    if path is None:
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(seed) for seed in raw.get("game_seeds", ())}


def _new_seed(excluded: set[int], database: TeacherDatabase) -> int:
    while True:
        seed = secrets.randbits(63)
        if seed in excluded or database.contains_seed(seed):
            continue
        excluded.add(seed)
        return seed


def _start_game(
    ordinal: int,
    seed: int,
    validation_every: int,
) -> _TeacherGame:
    game = GameSession(seed=seed)
    game.draw()
    return _TeacherGame(
        ordinal=ordinal,
        seed=seed,
        validation=ordinal % validation_every == 0,
        game=game,
    )


def _teacher_q_vector(
    decision: Any,
    reward_scale: float,
) -> np.ndarray:
    q_values = np.zeros(ACTION_COUNT, dtype=np.float32)
    for estimate in decision.estimates:
        index = (
            PASS_ACTION_INDEX
            if estimate.action is None
            else estimate.action.edge_id
        )
        q_values[index] = estimate.expected_gain / reward_scale
    return q_values


def generate_teacher_database(
    *,
    teacher_checkpoint: Path,
    database_path: Path,
    games: int,
    parallel_games: int,
    validation_every: int,
    excluded_seeds: set[int],
    device: str,
    leaf_batch_size: int,
) -> dict[str, float | int | str]:
    if validation_every < 2:
        raise ValueError("validation_every must be at least 2")
    checkpoint = teacher_checkpoint.resolve()
    policy = ExactChanceDQNPolicy.from_checkpoint(
        checkpoint,
        2,
        device=device,
        leaf_batch_size=leaf_batch_size,
    )
    reward_scale = policy.reward_scale
    started = perf_counter()

    with TeacherDatabase(
        database_path,
        teacher_checkpoint=checkpoint,
        depth=2,
        reward_scale=reward_scale,
    ) as database:
        completed_before = database.completed_games
        if completed_before > games:
            raise ValueError(
                f"database already contains {completed_before} games, "
                f"more than requested {games}"
            )
        pending_ordinals = [
            ordinal
            for ordinal in range(games)
            if ordinal not in database.completed_ordinals()
        ]
        pending_cursor = 0
        active: list[_TeacherGame] = []
        while pending_cursor < len(pending_ordinals) and len(active) < parallel_games:
            ordinal = pending_ordinals[pending_cursor]
            pending_cursor += 1
            seed = _new_seed(excluded_seeds, database)
            active.append(_start_game(ordinal, seed, validation_every))

        completed = completed_before
        last_report = completed
        while active:
            encodings = [encode_decision(slot.game) for slot in active]
            decisions = policy.choose_many([slot.game for slot in active])
            finished: list[_TeacherGame] = []
            for slot, encoded, decision in zip(active, encodings, decisions):
                q_values = _teacher_q_vector(decision, reward_scale)
                slot.samples.append(
                    pack_teacher_sample(
                        encoded.observation,
                        encoded.action_mask,
                        q_values,
                    )
                )
                slot.game.apply_legal_action(decision.action)
                if slot.game.status == "playing":
                    slot.game.draw()
                    continue

                final = slot.game.final_score()
                if final is None:
                    raise RuntimeError("teacher game ended without final score")
                database.append_game(
                    ordinal=slot.ordinal,
                    seed=slot.seed,
                    validation=slot.validation,
                    score=final.total,
                    samples=slot.samples,
                )
                completed += 1
                finished.append(slot)

            for slot in finished:
                active.remove(slot)
            while (
                pending_cursor < len(pending_ordinals)
                and len(active) < parallel_games
            ):
                ordinal = pending_ordinals[pending_cursor]
                pending_cursor += 1
                seed = _new_seed(excluded_seeds, database)
                active.append(_start_game(ordinal, seed, validation_every))

            if completed == games or completed - last_report >= 25:
                summary = database.game_summary()
                elapsed = perf_counter() - started
                rate = (completed - completed_before) / max(elapsed, 1e-9)
                print(
                    f"teacher games={completed}/{games}; "
                    f"positions={summary['positions']}; "
                    f"mean_score={summary['mean_score']:.2f}; "
                    f"rate={rate:.3f} games/s",
                    flush=True,
                )
                last_report = completed

        train_positions, validation_positions = database.split_counts()
        summary = database.game_summary()

    result: dict[str, float | int | str] = {
        **summary,
        "teacher_checkpoint": str(checkpoint),
        "teacher_depth": 2,
        "reward_scale": reward_scale,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "validation_every": validation_every,
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(database_path.with_suffix(".summary.json"), result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate teacher Q labels")
    generate.add_argument("--teacher-checkpoint", type=Path, required=True)
    generate.add_argument("--database", type=Path, required=True)
    generate.add_argument("--games", type=_positive_int, default=5_000)
    generate.add_argument("--parallel-games", type=_positive_int, default=16)
    generate.add_argument("--validation-every", type=_positive_int, default=10)
    generate.add_argument("--exclude-seeds-from", type=Path)
    generate.add_argument("--leaf-batch-size", type=_positive_int, default=8192)
    generate.add_argument("--device", default="auto")

    train = commands.add_parser("train", help="distill one teacher database")
    train.add_argument("--database", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--init-checkpoint", type=Path, required=True)
    train.add_argument("--epochs", type=_positive_int, default=30)
    train.add_argument("--batch-size", type=_positive_int, default=2048)
    train.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    train.add_argument("--reward-scale", type=_positive_float, default=10.0)
    train.add_argument(
        "--advantage-coefficient", type=float, default=0.0
    )
    train.add_argument("--policy-coefficient", type=float, default=0.0)
    train.add_argument(
        "--policy-temperature-points", type=_positive_float, default=2.0
    )
    train.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "generate":
        checkpoint = args.teacher_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        excluded = _load_excluded_seeds(
            None
            if args.exclude_seeds_from is None
            else args.exclude_seeds_from.resolve()
        )
        result = generate_teacher_database(
            teacher_checkpoint=checkpoint,
            database_path=args.database.resolve(),
            games=args.games,
            parallel_games=args.parallel_games,
            validation_every=args.validation_every,
            excluded_seeds=excluded,
            device=args.device,
            leaf_batch_size=args.leaf_batch_size,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if args.command == "train":
        checkpoint = args.init_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config = DistillationConfig(
            init_checkpoint=str(checkpoint),
            reward_scale=args.reward_scale,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            advantage_coefficient=args.advantage_coefficient,
            policy_coefficient=args.policy_coefficient,
            policy_temperature_points=args.policy_temperature_points,
            device=args.device,
        )
        best = distill_teacher_database(
            args.database.resolve(),
            args.run_dir.resolve(),
            config,
        )
        print(f"best_checkpoint={best}", flush=True)
        return

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
