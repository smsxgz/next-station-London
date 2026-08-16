"""Evaluate fixed-budget paired-scenario DQN rollouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from engine_cpp import GameSession
from solver.RL import PairedScenarioDQNPolicy

from .records import append_jsonl, build_game_record, describe, load_json, load_jsonl, write_json


@dataclass(slots=True)
class _ActiveGame:
    index: int
    seed: int
    game: GameSession
    started: float
    sections: int = 0
    passes: int = 0
    strategic_passes: int = 0
    allocated_search_seconds: float = 0.0
    trajectories: int = 0
    rollout_decisions: int = 0
    chance_samples: int = 0
    inference_batches: int = 0
    max_inference_batch_size: int = 0


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


def _policy_name(scenarios: int) -> str:
    return f"paired-scenario-dqn-{scenarios}"


def _validate_records(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
    checkpoint: Path,
    scenarios: int,
) -> None:
    expected_policy = _policy_name(scenarios)
    if set(records) - set(seeds):
        raise ValueError("result file contains unexpected game seeds")
    for record in records.values():
        algorithm = record.get("algorithm", {})
        if record.get("policy") != expected_policy:
            raise ValueError("result file contains another policy")
        if algorithm.get("checkpoint") != str(checkpoint):
            raise ValueError("result file uses another checkpoint")
        if algorithm.get("scenarios_per_action") != scenarios:
            raise ValueError("result file uses another scenario budget")


def _new_slot(index: int, seed: int) -> _ActiveGame:
    game = GameSession(seed=seed)
    game.draw()
    return _ActiveGame(index=index, seed=seed, game=game, started=perf_counter())


def _finish_record(
    slot: _ActiveGame,
    checkpoint: Path,
    scenarios: int,
) -> dict[str, Any]:
    record = build_game_record(
        slot.game,
        policy=_policy_name(scenarios),
        seed_index=slot.index,
        game_seed=slot.seed,
        elapsed_seconds=perf_counter() - slot.started,
        actions={
            "sections": slot.sections,
            "passes": slot.passes,
            "strategic_passes": slot.strategic_passes,
        },
        algorithm={
            "family": "paired-scenario-dqn-rollout",
            "checkpoint": str(checkpoint),
            "scenarios_per_action": scenarios,
            "rollout_policy": "dqn",
        },
    )
    record["search"] = {
        "allocated_seconds": slot.allocated_search_seconds,
        "trajectories": slot.trajectories,
        "rollout_decisions": slot.rollout_decisions,
        "chance_samples": slot.chance_samples,
        "inference_batches": slot.inference_batches,
        "max_inference_batch_size": slot.max_inference_batch_size,
    }
    return record


def _write_summary(
    output_dir: Path,
    checkpoint: Path,
    scenarios: int,
    seeds: list[int],
    records: dict[int, dict[str, Any]],
) -> None:
    ordered = [records[seed] for seed in seeds]
    score = describe([record["score"]["total"] for record in ordered])
    elapsed = describe([record["elapsed_seconds"] for record in ordered])
    search = describe([record["search"]["allocated_seconds"] for record in ordered])
    write_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint),
            "games": len(ordered),
            "policy": _policy_name(scenarios),
            "scenarios_per_action": scenarios,
            "score": score,
            "elapsed_seconds": elapsed,
            "allocated_search_seconds": search,
        },
    )
    lines = [
        "# Paired-Scenario DQN Rollout",
        "",
        f"Checkpoint: `{checkpoint}`  ",
        f"Games: `{len(ordered)}`  ",
        f"Scenarios per root action: `{scenarios}`",
        "",
        f"- Score: `{score['mean']:.2f} +/- {score['standard_error']:.2f}`",
        f"- Range: `{score['min']:.0f}..{score['max']:.0f}`",
        f"- Median: `{score['median']:.1f}`",
        f"- Mean wall time per game: `{elapsed['mean']:.2f}s`",
        f"- Mean allocated search time per game: `{search['mean']:.2f}s`",
        "",
    ]
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
    parser.add_argument("--scenarios", type=_positive_int, default=32)
    parser.add_argument("--parallel-games", type=_positive_int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    seeds = _load_seeds(args.seeds_from.resolve(), args.games)
    output_dir = args.output_dir.resolve()
    output_path = output_dir / "games" / f"{_policy_name(args.scenarios)}.jsonl"
    records = load_jsonl(output_path)
    _validate_records(records, seeds, checkpoint, args.scenarios)
    pending = [
        (index, seed)
        for index, seed in enumerate(seeds)
        if seed not in records
    ]
    print(
        f"{_policy_name(args.scenarios)}: "
        f"completed={len(records)}/{len(seeds)}",
        flush=True,
    )

    policy = PairedScenarioDQNPolicy.from_checkpoint(
        checkpoint,
        args.scenarios,
        device=args.device,
    )
    active: list[_ActiveGame] = []
    cursor = 0
    while cursor < len(pending) and len(active) < args.parallel_games:
        active.append(_new_slot(*pending[cursor]))
        cursor += 1

    while active:
        decisions = policy.choose_many([slot.game for slot in active])
        stats = policy.last_batch_stats
        allocation = stats.elapsed_seconds / len(active)
        rollout_allocation = stats.rollout_decisions // len(active)
        chance_allocation = stats.chance_samples // len(active)
        batch_allocation = stats.inference_batches / len(active)

        finished: list[_ActiveGame] = []
        for slot, decision in zip(active, decisions):
            legal_exists = bool(slot.game.legal_actions())
            if decision.action is None:
                slot.passes += 1
                slot.strategic_passes += int(legal_exists)
            else:
                slot.sections += 1
            slot.allocated_search_seconds += allocation
            slot.trajectories += len(decision.estimates) * args.scenarios
            slot.rollout_decisions += rollout_allocation
            slot.chance_samples += chance_allocation
            slot.inference_batches += batch_allocation
            slot.max_inference_batch_size = max(
                slot.max_inference_batch_size,
                stats.max_inference_batch_size,
            )
            slot.game.apply_legal_action(decision.action)
            if slot.game.status == "playing":
                slot.game.draw()
                continue

            finished.append(slot)
            record = _finish_record(slot, checkpoint, args.scenarios)
            records[slot.seed] = record
            append_jsonl(output_path, record)
            if len(records) == len(seeds) or len(records) % 10 == 0:
                values = [
                    item["score"]["total"]
                    for item in records.values()
                ]
                print(
                    f"{_policy_name(args.scenarios)}: "
                    f"{len(records)}/{len(seeds)}; "
                    f"mean={sum(values) / len(values):.2f}",
                    flush=True,
                )

        for slot in finished:
            active.remove(slot)
        while cursor < len(pending) and len(active) < args.parallel_games:
            active.append(_new_slot(*pending[cursor]))
            cursor += 1

    _write_summary(output_dir, checkpoint, args.scenarios, seeds, records)


if __name__ == "__main__":
    main()
