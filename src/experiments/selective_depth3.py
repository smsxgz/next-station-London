"""Evaluate bounded selective depth-3 and exact round-end search."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from engine_cpp import GameSession
from solver.RL import SelectiveDepth3Policy

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
    seed_index: int
    seed: int
    game: GameSession
    started: float
    decision_index: int = 0
    sections: int = 0
    passes: int = 0
    strategic_passes: int = 0
    triggered: int = 0
    depth3_completed: int = 0
    round_end_completed: int = 0
    budget_exceeded: int = 0
    action_changes: int = 0
    action_branches: int = 0
    unique_leaf_states: int = 0
    deep_search_seconds: float = 0.0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _load_seeds(path: Path, games: int | None) -> list[int]:
    available = [int(seed) for seed in load_json(path).get("game_seeds", ())]
    if not available:
        raise ValueError(f"{path} has no game_seeds")
    count = len(available) if games is None else games
    if count > len(available):
        raise ValueError(f"requested {count} games from {len(available)} seeds")
    return available[:count]


def _start_game(index: int, seed: int) -> _ActiveGame:
    game = GameSession(seed=seed)
    game.draw()
    return _ActiveGame(index, seed, game, perf_counter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument("--parallel-games", type=_positive_int, default=16)
    parser.add_argument("--leaf-batch-size", type=_positive_int, default=8192)
    parser.add_argument("--depth2-gap", type=_nonnegative_float, default=0.5)
    parser.add_argument("--max-action-branches", type=_positive_int, default=75000)
    parser.add_argument("--round-end-remaining", type=int, default=4)
    parser.add_argument(
        "--disable-depth-triggers",
        action="store_true",
        help="only trigger the configured round-end search",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.round_end_remaining < -1:
        parser.error("round-end-remaining must be -1 or non-negative")
    round_end_remaining = (
        None if args.round_end_remaining == -1 else args.round_end_remaining
    )
    seeds = _load_seeds(args.seeds_from.resolve(), args.games)
    output_dir = args.output_dir.resolve()
    games_path = output_dir / "games.jsonl"
    states_path = output_dir / "states.jsonl"
    if games_path.exists() or states_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    policy = SelectiveDepth3Policy.from_checkpoint(
        checkpoint,
        device=args.device,
        leaf_batch_size=args.leaf_batch_size,
        depth2_gap=args.depth2_gap,
        max_action_branches=args.max_action_branches,
        round_end_remaining=round_end_remaining,
        use_depth_triggers=not args.disable_depth_triggers,
    )
    pending = list(enumerate(seeds))
    cursor = 0
    active: list[_ActiveGame] = []
    while cursor < len(pending) and len(active) < args.parallel_games:
        active.append(_start_game(*pending[cursor]))
        cursor += 1

    records: list[dict[str, Any]] = []
    trigger_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    completed = 0
    started = perf_counter()
    while active:
        decisions = policy.choose_many([slot.game for slot in active])
        finished: list[_ActiveGame] = []
        for slot, decision in zip(active, decisions):
            for reason in decision.trigger_reasons:
                trigger_counts[reason] += 1
            selected_counts[decision.selected_search] += 1
            slot.triggered += int(bool(decision.trigger_reasons))
            slot.depth3_completed += int(decision.selected_search == "depth-3")
            slot.round_end_completed += int(
                decision.selected_search == "round-end"
            )
            slot.budget_exceeded += int(decision.budget_exceeded)
            slot.action_changes += int(decision.action_changed)
            slot.action_branches += decision.action_branches
            slot.unique_leaf_states += decision.unique_leaf_states
            slot.deep_search_seconds += decision.search_seconds
            append_jsonl(
                states_path,
                {
                    "seed_index": slot.seed_index,
                    "game_seed": slot.seed,
                    "decision_index": slot.decision_index,
                    "round": slot.game.round_index + 1,
                    "draw_count": slot.game.draw_count,
                    "remaining_cards": len(slot.game.remaining_card_ids()),
                    "selected_search": decision.selected_search,
                    "trigger_reasons": list(decision.trigger_reasons),
                    "depth1_action": decision.depth1_action_index,
                    "depth2_action": decision.depth2_action_index,
                    "selected_action": decision.action_index,
                    "depth2_gap": decision.depth2_gap,
                    "selected_gap": decision.selected_gap,
                    "action_changed": decision.action_changed,
                    "budget_exceeded": decision.budget_exceeded,
                    "action_branches": decision.action_branches,
                    "unique_leaf_states": decision.unique_leaf_states,
                    "search_seconds": decision.search_seconds,
                },
            )

            if decision.action is None:
                slot.passes += 1
                slot.strategic_passes += int(bool(slot.game.legal_actions()))
            else:
                slot.sections += 1
            slot.game.apply_legal_action(decision.action)
            slot.decision_index += 1
            if slot.game.status == "playing":
                slot.game.draw()
                continue

            record = build_game_record(
                slot.game,
                policy="selective-depth-3",
                seed_index=slot.seed_index,
                game_seed=slot.seed,
                elapsed_seconds=perf_counter() - slot.started,
                actions={
                    "sections": slot.sections,
                    "passes": slot.passes,
                    "strategic_passes": slot.strategic_passes,
                },
                algorithm={
                    "family": "selective-exact-chance-dqn",
                    "checkpoint": str(checkpoint),
                    "depth2_gap": args.depth2_gap,
                    "max_action_branches": args.max_action_branches,
                    "round_end_remaining": round_end_remaining,
                    "use_depth_triggers": not args.disable_depth_triggers,
                    "triggered": slot.triggered,
                    "depth3_completed": slot.depth3_completed,
                    "round_end_completed": slot.round_end_completed,
                    "budget_exceeded": slot.budget_exceeded,
                    "action_changes": slot.action_changes,
                    "action_branches": slot.action_branches,
                    "unique_leaf_states": slot.unique_leaf_states,
                    "deep_search_seconds": slot.deep_search_seconds,
                },
            )
            append_jsonl(games_path, record)
            records.append(record)
            finished.append(slot)
            completed += 1
            if completed == len(seeds) or completed % 5 == 0:
                score = describe(
                    [item["score"]["total"] for item in records]
                )
                print(
                    f"games={completed}/{len(seeds)}; "
                    f"mean={score['mean']:.2f}; "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )

        for slot in finished:
            active.remove(slot)
        while cursor < len(pending) and len(active) < args.parallel_games:
            active.append(_start_game(*pending[cursor]))
            cursor += 1

    baseline_path = Path(
        "artifacts/dqn/distill_depth2_5000/student/"
        "exact_chance_eval/games/exact-chance-depth-2.jsonl"
    )
    baseline = load_jsonl(baseline_path)
    differences = [
        int(record["score"]["total"])
        - int(baseline[int(record["game_seed"])]["score"]["total"])
        for record in records
    ]
    difference = describe(differences)
    mean = float(difference["mean"])
    error = float(difference["standard_error"])
    summary = {
        "checkpoint": str(checkpoint),
        "games": len(records),
        "configuration": {
            "depth2_gap": args.depth2_gap,
            "max_action_branches": args.max_action_branches,
            "round_end_remaining": round_end_remaining,
            "use_depth_triggers": not args.disable_depth_triggers,
        },
        "scores": describe([record["score"]["total"] for record in records]),
        "paired_vs_depth2": {
            **difference,
            "ci95": [mean - 1.96 * error, mean + 1.96 * error],
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        },
        "trigger_counts": dict(trigger_counts),
        "selected_search_counts": dict(selected_counts),
        "totals": {
            key: sum(int(record["algorithm"][key]) for record in records)
            for key in (
                "triggered",
                "depth3_completed",
                "round_end_completed",
                "budget_exceeded",
                "action_changes",
                "action_branches",
                "unique_leaf_states",
            )
        },
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(output_dir / "summary.json", summary)
    score = summary["scores"]
    paired = summary["paired_vs_depth2"]
    lines = [
        "# Selective Depth-3",
        "",
        f"Games: `{len(records)}`  ",
        f"Score: `{score['mean']:.2f} +/- {score['standard_error']:.2f}`  ",
        (
            f"Vs depth-2: `{paired['mean']:+.2f} +/- "
            f"{paired['standard_error']:.2f}`, 95% CI "
            f"`[{paired['ci95'][0]:+.2f}, {paired['ci95'][1]:+.2f}]`, "
            f"W/T/L `{paired['wins']}/{paired['ties']}/{paired['losses']}`."
        ),
        "",
        f"Selected searches: `{dict(selected_counts)}`  ",
        f"Budget exceeded: `{summary['totals']['budget_exceeded']}`  ",
        f"Actions changed from depth-2: `{summary['totals']['action_changes']}`  ",
        f"Elapsed seconds: `{summary['elapsed_seconds']:.1f}`",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
