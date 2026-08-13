"""Compare DQN with exact-chance DQN expectimax on shared game seeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from engine import GameSession
from solver.RL import (
    DQNPolicy,
    ExactChanceDQNPolicy,
    PASS_ACTION_INDEX,
    encode_decision,
)
from solver.RL.dqn import select_action_indices

from .records import (
    append_jsonl,
    build_game_record,
    describe,
    load_json,
    load_jsonl,
    write_json,
)


@dataclass(slots=True)
class _ActiveGame:
    index: int
    seed: int
    game: GameSession
    started: float
    sections: int = 0
    passes: int = 0
    strategic_passes: int = 0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _load_seeds(path: Path, games: int | None) -> list[int]:
    seeds = [int(seed) for seed in load_json(path).get("game_seeds", ())]
    if not seeds:
        raise ValueError(f"{path} has no game_seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} contains duplicate game seeds")
    count = len(seeds) if games is None else games
    if count > len(seeds):
        raise ValueError(f"requested {count} games from {len(seeds)} seeds")
    return seeds[:count]


def _start_game(index: int, seed: int) -> _ActiveGame:
    game = GameSession(seed=seed)
    game.draw()
    return _ActiveGame(index, seed, game, perf_counter())


def _validate_records(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
    policy: str,
    checkpoint: Path,
    *,
    depth: int | None = None,
    leaf_weights: str | None = None,
) -> None:
    if set(records) - set(seeds):
        raise ValueError(f"{policy} results contain unexpected seeds")
    for record in records.values():
        algorithm = record.get("algorithm", {})
        if record.get("policy") != policy:
            raise ValueError(f"result file contains a policy other than {policy}")
        if algorithm.get("checkpoint") != str(checkpoint):
            raise ValueError(f"{policy} results use a different checkpoint")
        if depth is not None and algorithm.get("depth") != depth:
            raise ValueError(f"{policy} results use a different depth")
        stored_weights = algorithm.get("leaf_weights", "online")
        if leaf_weights is not None and stored_weights != leaf_weights:
            raise ValueError(f"{policy} results use different leaf weights")


def _pending_games(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
) -> list[tuple[int, int]]:
    return [
        (index, seed)
        for index, seed in enumerate(seeds)
        if seed not in records
    ]


def _fill_active(
    active: list[_ActiveGame],
    pending: list[tuple[int, int]],
    cursor: int,
    parallel_games: int,
) -> int:
    while cursor < len(pending) and len(active) < parallel_games:
        active.append(_start_game(*pending[cursor]))
        cursor += 1
    return cursor


def _finish_record(
    slot: _ActiveGame,
    *,
    policy: str,
    checkpoint: Path,
    family: str,
    depth: int | None = None,
    leaf_weights: str | None = None,
) -> dict[str, Any]:
    algorithm: dict[str, Any] = {
        "family": family,
        "checkpoint": str(checkpoint),
    }
    if depth is not None:
        algorithm["depth"] = depth
    if leaf_weights is not None:
        algorithm["leaf_weights"] = leaf_weights
    return build_game_record(
        slot.game,
        policy=policy,
        seed_index=slot.index,
        game_seed=slot.seed,
        elapsed_seconds=perf_counter() - slot.started,
        actions={
            "sections": slot.sections,
            "passes": slot.passes,
            "strategic_passes": slot.strategic_passes,
        },
        algorithm=algorithm,
    )


def _print_progress(policy: str, completed: int, total: int) -> None:
    if completed == total or completed % 10 == 0:
        print(f"{policy}: {completed}/{total}", flush=True)


def _run_dqn(
    seeds: list[int],
    checkpoint: Path,
    output_dir: Path,
    parallel_games: int,
    device: str,
) -> dict[int, dict[str, Any]]:
    policy_name = "dqn"
    output_path = output_dir / "games" / "dqn.jsonl"
    records = load_jsonl(output_path)
    _validate_records(records, seeds, policy_name, checkpoint)
    pending = _pending_games(records, seeds)
    print(
        f"{policy_name}: completed={len(seeds) - len(pending)}/{len(seeds)}",
        flush=True,
    )
    if not pending:
        return records

    policy = DQNPolicy.from_checkpoint(checkpoint, device=device)
    rng = np.random.default_rng(0)
    active: list[_ActiveGame] = []
    cursor = _fill_active(active, pending, 0, parallel_games)
    while active:
        encoded = [encode_decision(slot.game) for slot in active]
        action_indices = select_action_indices(
            policy.network,
            np.stack([item.observation for item in encoded]),
            np.stack([item.action_mask for item in encoded]),
            device=policy.device,
            epsilon=0.0,
            rng=rng,
        )
        finished: list[_ActiveGame] = []
        for slot, decision, raw_index in zip(active, encoded, action_indices):
            action_index = int(raw_index)
            action = (
                None
                if action_index == PASS_ACTION_INDEX
                else decision.actions[action_index]
            )
            if action is None:
                slot.passes += 1
                slot.strategic_passes += int(bool(decision.actions))
            else:
                slot.sections += 1
            slot.game.apply_legal_action(action)
            if slot.game.status == "playing":
                slot.game.draw()
                continue

            finished.append(slot)
            record = _finish_record(
                slot,
                policy=policy_name,
                checkpoint=checkpoint,
                family="double-dqn",
            )
            records[slot.seed] = record
            append_jsonl(output_path, record)
            _print_progress(policy_name, len(records), len(seeds))

        for slot in finished:
            active.remove(slot)
        cursor = _fill_active(active, pending, cursor, parallel_games)
    return records


def _run_exact_chance(
    depth: int,
    seeds: list[int],
    checkpoint: Path,
    output_dir: Path,
    parallel_games: int,
    leaf_batch_size: int,
    device: str,
    leaf_weights: str = "online",
) -> dict[int, dict[str, Any]]:
    suffix = "" if leaf_weights == "online" else f"-{leaf_weights}"
    policy_name = f"exact-chance-depth-{depth}{suffix}"
    output_path = output_dir / "games" / f"{policy_name}.jsonl"
    records = load_jsonl(output_path)
    _validate_records(
        records,
        seeds,
        policy_name,
        checkpoint,
        depth=depth,
        leaf_weights=leaf_weights,
    )
    pending = _pending_games(records, seeds)
    print(
        f"{policy_name}: completed={len(seeds) - len(pending)}/{len(seeds)}",
        flush=True,
    )
    if not pending:
        return records

    policy = ExactChanceDQNPolicy.from_checkpoint(
        checkpoint,
        depth,
        device=device,
        leaf_batch_size=leaf_batch_size,
        leaf_weights=leaf_weights,
    )
    active: list[_ActiveGame] = []
    cursor = _fill_active(active, pending, 0, parallel_games)
    while active:
        decisions = policy.choose_many([slot.game for slot in active])
        finished: list[_ActiveGame] = []
        for slot, decision in zip(active, decisions):
            if decision.action is None:
                slot.passes += 1
                slot.strategic_passes += int(bool(slot.game.legal_actions()))
            else:
                slot.sections += 1
            slot.game.apply_legal_action(decision.action)
            if slot.game.status == "playing":
                slot.game.draw()
                continue

            finished.append(slot)
            record = _finish_record(
                slot,
                policy=policy_name,
                checkpoint=checkpoint,
                family="exact-chance-dqn",
                depth=depth,
                leaf_weights=leaf_weights,
            )
            records[slot.seed] = record
            append_jsonl(output_path, record)
            _print_progress(policy_name, len(records), len(seeds))

        for slot in finished:
            active.remove(slot)
        cursor = _fill_active(active, pending, cursor, parallel_games)
    return records


def _score_summary(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
) -> dict[str, float | int]:
    return describe([records[seed]["score"]["total"] for seed in seeds])


def _paired_summary(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    differences = [
        candidate[seed]["score"]["total"]
        - baseline[seed]["score"]["total"]
        for seed in seeds
    ]
    stats = describe(differences)
    mean = float(stats["mean"])
    error = float(stats["standard_error"])
    return {
        "mean_difference": mean,
        "standard_error": error,
        "ci95": [mean - 1.96 * error, mean + 1.96 * error],
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def _write_summary(
    output_dir: Path,
    checkpoint: Path,
    seeds: list[int],
    records: dict[str, dict[int, dict[str, Any]]],
) -> None:
    scores = {
        policy: _score_summary(policy_records, seeds)
        for policy, policy_records in records.items()
    }
    paired: dict[str, dict[str, Any]] = {}
    names = tuple(records)
    if "dqn" in records:
        for policy in names:
            if policy != "dqn":
                paired[f"{policy}-vs-dqn"] = _paired_summary(
                    records["dqn"], records[policy], seeds
                )
    depth1 = next((name for name in names if "depth-1" in name), None)
    depth2 = next((name for name in names if "depth-2" in name), None)
    if depth1 is not None and depth2 is not None:
        paired[f"{depth2}-vs-{depth1}"] = _paired_summary(
            records[depth1], records[depth2], seeds
        )
    write_json(
        output_dir / "summary.json",
        {
            "games": len(seeds),
            "checkpoint": str(checkpoint),
            "scores": scores,
            "paired": paired,
        },
    )

    labels = {
        "dqn": "DQN",
        **{
            policy: policy.replace("exact-chance-", "Exact chance ")
            for policy in records
            if policy != "dqn"
        },
    }
    dqn_mean = (
        float(scores["dqn"]["mean"])
        if "dqn" in scores
        else None
    )
    lines = [
        "# Exact-Chance DQN",
        "",
        f"Checkpoint: `{checkpoint}`  ",
        f"Games: `{len(seeds)}`",
        "",
        "| Policy | Mean | SE | Delta vs DQN | Min | Median | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in records:
        score = scores[policy]
        delta = (
            "-"
            if dqn_mean is None or policy == "dqn"
            else f"{float(score['mean']) - dqn_mean:+.2f}"
        )
        lines.append(
            f"| {labels[policy]} | {score['mean']:.2f} | "
            f"{score['standard_error']:.2f} | {delta} | {score['min']:.0f} | "
            f"{score['median']:.1f} | {score['max']:.0f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, comparison in paired.items():
        lower, upper = comparison["ci95"]
        lines.extend(
            [
                (
                    f"`{name}`: `{comparison['mean_difference']:+.2f} +/- "
                    f"{comparison['standard_error']:.2f}`, 95% CI "
                    f"`[{lower:+.2f}, {upper:+.2f}]`; wins/ties/losses "
                    f"`{comparison['wins']}/{comparison['ties']}/"
                    f"{comparison['losses']}`."
                ),
                "",
            ]
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument("--parallel-games", type=_positive_int, default=8)
    parser.add_argument("--leaf-batch-size", type=_positive_int, default=8192)
    parser.add_argument(
        "--leaf-weights",
        choices=("online", "target", "mean-online-target"),
        default="online",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("dqn", "depth-1", "depth-2"),
        default=("dqn", "depth-1", "depth-2"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    seeds = _load_seeds(args.seeds_from.resolve(), args.games)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[int, dict[str, Any]]] = {}
    if "dqn" in args.policies:
        records["dqn"] = _run_dqn(
            seeds,
            checkpoint,
            output_dir,
            args.parallel_games,
            args.device,
        )
    if "depth-1" in args.policies:
        name = "exact-chance-depth-1" + (
            "" if args.leaf_weights == "online" else f"-{args.leaf_weights}"
        )
        records[name] = _run_exact_chance(
            1,
            seeds,
            checkpoint,
            output_dir,
            args.parallel_games,
            args.leaf_batch_size,
            args.device,
            args.leaf_weights,
        )
    if "depth-2" in args.policies:
        name = "exact-chance-depth-2" + (
            "" if args.leaf_weights == "online" else f"-{args.leaf_weights}"
        )
        records[name] = _run_exact_chance(
            2,
            seeds,
            checkpoint,
            output_dir,
            args.parallel_games,
            args.leaf_batch_size,
            args.device,
            args.leaf_weights,
        )
    _write_summary(output_dir, checkpoint, seeds, records)


if __name__ == "__main__":
    main()
