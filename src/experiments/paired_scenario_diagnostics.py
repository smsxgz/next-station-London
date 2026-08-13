"""Measure paired-scenario action stability on fixed public states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Sequence

from engine import Action, GameSession
from solver.RL import (
    DQNPolicy,
    ExactChanceDQNPolicy,
    PASS_ACTION_INDEX,
    PairedScenarioDQNPolicy,
)

from .records import load_json, write_json


_BUDGET_REPEATS = ((32, 8), (128, 4), (256, 2))
_REFERENCE_BUDGET = 2048
_STATE_FRACTIONS = (0.2, 0.5, 0.8)
_STATE_BATCH_SIZE = 4


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _action_index(action: Action | None) -> int:
    return PASS_ACTION_INDEX if action is None else action.edge_id


def _load_seeds(path: Path, seed_count: int) -> list[tuple[int, int]]:
    all_seeds = [int(seed) for seed in load_json(path).get("game_seeds", ())]
    if seed_count > len(all_seeds):
        raise ValueError(f"requested {seed_count} seeds from {len(all_seeds)}")
    if seed_count == 1:
        indices = [0]
    else:
        indices = [
            round(position * (len(all_seeds) - 1) / (seed_count - 1))
            for position in range(seed_count)
        ]
    return [(index, all_seeds[index]) for index in indices]


def _fixed_states(
    seeds: list[tuple[int, int]],
    dqn: DQNPolicy,
) -> tuple[list[GameSession], list[dict[str, Any]]]:
    states: list[GameSession] = []
    metadata: list[dict[str, Any]] = []
    for seed_index, seed in seeds:
        game = GameSession(seed=seed)
        game.draw()
        trajectory: list[GameSession] = []
        while game.status == "playing":
            trajectory.append(game.copy_public_state())
            decision = dqn.choose(game)
            game.apply_legal_action(decision.action)
            if game.status == "playing":
                game.draw()

        selected: set[int] = set()
        for fraction in _STATE_FRACTIONS:
            state_index = round(fraction * (len(trajectory) - 1))
            if state_index in selected:
                continue
            selected.add(state_index)
            state = trajectory[state_index]
            states.append(state)
            metadata.append(
                {
                    "game_seed_index": seed_index,
                    "game_seed": seed,
                    "trajectory_decisions": len(trajectory),
                    "decision_index": state_index,
                    "fraction": fraction,
                    "round": state.round_index + 1,
                    "draw_count": state.draw_count,
                    "legal_actions": len(state.legal_actions()) + 1,
                }
            )
    return states, metadata


def _deterministic_choices(decisions: Sequence[Any]) -> list[int]:
    return [_action_index(decision.estimates[0].action) for decision in decisions]


def _reference_values(decisions: Sequence[Any]) -> list[dict[int, float]]:
    return [
        {
            _action_index(estimate.action): float(estimate.mean_gain)
            for estimate in decision.estimates
        }
        for decision in decisions
    ]


def _run_paired(
    checkpoint: Path,
    states: list[GameSession],
    budget: int,
    repeats: int,
    *,
    device: str,
) -> tuple[list[list[int]], list[float]]:
    policy = PairedScenarioDQNPolicy.from_checkpoint(
        checkpoint,
        budget,
        device=device,
    )
    choices: list[list[int]] = []
    elapsed: list[float] = []
    for repetition in range(repeats):
        policy.scenario_rng.seed(
            0x9E3779B97F4A7C15 ^ (budget << 16) ^ repetition
        )
        started = perf_counter()
        decisions: list[Any] = []
        max_batch = 0
        for start in range(0, len(states), _STATE_BATCH_SIZE):
            decisions.extend(
                policy.choose_many(states[start : start + _STATE_BATCH_SIZE])
            )
            max_batch = max(
                max_batch,
                policy.last_batch_stats.max_inference_batch_size,
            )
        elapsed.append(perf_counter() - started)
        choices.append(_deterministic_choices(decisions))
        print(
            f"paired-{budget}: repeat {repetition + 1}/{repeats}; "
            f"elapsed={elapsed[-1]:.2f}s; "
            f"max_batch={max_batch}",
            flush=True,
        )
    return choices, elapsed


def _summarize_budget(
    choices: list[list[int]],
    elapsed: list[float],
    reference_choices: list[int],
    reference_values: list[dict[int, float]],
    depth2_choices: list[int],
) -> dict[str, Any]:
    state_count = len(reference_choices)
    repeats = len(choices)
    flat = [
        (state, choice)
        for repetition in choices
        for state, choice in enumerate(repetition)
    ]
    regrets = [
        reference_values[state][reference_choices[state]]
        - reference_values[state][choice]
        for state, choice in flat
    ]
    per_state_unique = [
        len({repetition[state] for repetition in choices})
        for state in range(state_count)
    ]
    return {
        "repeats": repeats,
        "decisions": len(flat),
        "reference_agreement": sum(
            choice == reference_choices[state] for state, choice in flat
        )
        / len(flat),
        "depth2_agreement": sum(
            choice == depth2_choices[state] for state, choice in flat
        )
        / len(flat),
        "mean_reference_regret": fmean(regrets),
        "max_reference_regret": max(regrets),
        "states_always_stable": sum(value == 1 for value in per_state_unique),
        "mean_unique_choices_per_state": fmean(per_state_unique),
        "mean_elapsed_seconds": fmean(elapsed),
        "choices": choices,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    reference = result["reference"]
    lines = [
        "# Paired-Scenario Stability Diagnostic",
        "",
        f"Fixed public states: `{result['states']}` from `{result['seeds']}` games.  ",
        f"Reference: Paired-`{reference['budget']}`.  ",
        (
            "Reference agreement with Exact-chance Depth-2: "
            f"`{reference['depth2_agreement'] * 100:.1f}%`."
        ),
        "",
        "| Budget | Repeats | Ref. agreement | Depth-2 agreement | Ref. regret | Stable states | Time / repeat |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget, _ in _BUDGET_REPEATS:
        item = result["budgets"][str(budget)]
        lines.append(
            f"| {budget} | {item['repeats']} | "
            f"{item['reference_agreement'] * 100:.1f}% | "
            f"{item['depth2_agreement'] * 100:.1f}% | "
            f"{item['mean_reference_regret']:.2f} | "
            f"{item['states_always_stable']}/{result['states']} | "
            f"{item['mean_elapsed_seconds']:.2f}s |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-count", type=_positive_int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_seeds = _load_seeds(args.seeds_from.resolve(), args.seed_count)
    dqn = DQNPolicy.from_checkpoint(checkpoint, device=args.device)
    states, metadata = _fixed_states(selected_seeds, dqn)
    print(f"collected {len(states)} fixed public states", flush=True)

    depth2 = ExactChanceDQNPolicy.from_checkpoint(
        checkpoint,
        2,
        device=args.device,
    )
    started = perf_counter()
    depth2_decisions: list[Any] = []
    for start in range(0, len(states), _STATE_BATCH_SIZE):
        depth2_decisions.extend(
            depth2.choose_many(states[start : start + _STATE_BATCH_SIZE])
        )
    depth2_elapsed = perf_counter() - started
    depth2_choices = _deterministic_choices(depth2_decisions)
    print(f"depth-2 reference actions: elapsed={depth2_elapsed:.2f}s", flush=True)

    paired_reference = PairedScenarioDQNPolicy.from_checkpoint(
        checkpoint,
        _REFERENCE_BUDGET,
        device=args.device,
    )
    paired_reference.scenario_rng.seed(0xD1B54A32D192ED03)
    started = perf_counter()
    reference_decisions: list[Any] = []
    reference_max_batch = 0
    for index, state in enumerate(states):
        reference_decisions.extend(paired_reference.choose_many([state]))
        reference_max_batch = max(
            reference_max_batch,
            paired_reference.last_batch_stats.max_inference_batch_size,
        )
        print(
            f"paired-{_REFERENCE_BUDGET} reference: "
            f"state {index + 1}/{len(states)}",
            flush=True,
        )
    reference_elapsed = perf_counter() - started
    reference_choices = _deterministic_choices(reference_decisions)
    reference_values = _reference_values(reference_decisions)
    print(
        f"paired-{_REFERENCE_BUDGET} reference: "
        f"elapsed={reference_elapsed:.2f}s; "
        f"max_batch={reference_max_batch}",
        flush=True,
    )

    budgets: dict[str, Any] = {}
    for budget, repeats in _BUDGET_REPEATS:
        choices, elapsed = _run_paired(
            checkpoint,
            states,
            budget,
            repeats,
            device=args.device,
        )
        budgets[str(budget)] = _summarize_budget(
            choices,
            elapsed,
            reference_choices,
            reference_values,
            depth2_choices,
        )

    reference_depth2_agreement = sum(
        left == right
        for left, right in zip(reference_choices, depth2_choices)
    ) / len(states)
    for item, depth2_choice, reference_choice, values in zip(
        metadata,
        depth2_choices,
        reference_choices,
        reference_values,
    ):
        item["depth2_action"] = depth2_choice
        item["paired_reference_action"] = reference_choice
        item["paired_reference_regret_of_depth2"] = (
            values[reference_choice] - values[depth2_choice]
        )

    result = {
        "checkpoint": str(checkpoint),
        "seeds": len(selected_seeds),
        "states": len(states),
        "selected_seeds": [
            {"seed_index": index, "game_seed": seed}
            for index, seed in selected_seeds
        ],
        "state_metadata": metadata,
        "depth2_elapsed_seconds": depth2_elapsed,
        "reference": {
            "budget": _REFERENCE_BUDGET,
            "elapsed_seconds": reference_elapsed,
            "depth2_agreement": reference_depth2_agreement,
            "choices": reference_choices,
        },
        "budgets": budgets,
    }
    write_json(output_dir / "summary.json", result)
    _write_markdown(output_dir / "summary.md", result)
    print(json.dumps(result["budgets"], indent=2), flush=True)


if __name__ == "__main__":
    main()
