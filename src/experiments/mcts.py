"""Run resumable, fixed-configuration MCTS benchmarks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from engine_cpp import GameSession
from solver import (
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_MCTS_SIMULATIONS,
    MCTSPolicy,
)

from .records import (
    append_jsonl,
    build_game_record,
    load_json,
    load_jsonl,
    summarize,
)


@dataclass(frozen=True, slots=True)
class CompletedGame:
    index: int
    seed: int
    score: int
    elapsed_seconds: float
    record: dict[str, Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def policy_name(
    simulations: int,
    exploration: float,
    *,
    shared_objectives: bool = False,
    pencil_powers: bool = False,
) -> str:
    exploration_label = f"{exploration:g}".replace(".", "p")
    name = f"mcts-uct-{simulations}-c{exploration_label}"
    if shared_objectives and pencil_powers:
        return f"{name}-advanced"
    if shared_objectives:
        return f"{name}-shared-objectives"
    if pencil_powers:
        return f"{name}-pencil-powers"
    return name


def run_mcts_game(
    task: tuple[int, int, int, float, bool, bool],
) -> CompletedGame:
    index, seed, simulations, exploration, objectives, powers = task
    game = GameSession(
        seed=seed,
        shared_objectives_enabled=objectives,
        pencil_powers_enabled=powers,
    )
    policy = MCTSPolicy(simulations, exploration=exploration)
    sections = passes = strategic_passes = candidate_actions = 0
    second_section_stops = powered_sections = 0
    selected_errors: list[float] = []
    decision_nodes = tree_chance = rollout_chance = 0
    terminal_rollouts = tree_terminal_hits = 0
    tree_depths: list[float] = []
    max_tree_depth = 0
    search_seconds = 0.0
    started = perf_counter()

    while game.status == "playing":
        if game.pending is None:
            game.draw()
        legal = game.legal_actions()
        decision = policy.choose(game)
        candidate_actions += len(decision.estimates)
        selected_errors.append(decision.standard_error)
        stats = decision.stats
        decision_nodes += stats.decision_nodes
        tree_chance += stats.tree_chance_samples
        rollout_chance += stats.rollout_chance_samples
        terminal_rollouts += stats.terminal_rollouts
        tree_terminal_hits += stats.tree_terminal_hits
        tree_depths.append(stats.mean_tree_depth)
        max_tree_depth = max(max_tree_depth, stats.max_tree_depth)
        search_seconds += stats.elapsed_seconds

        action = decision.action
        second_phase = game.double_section_pending
        if action is None:
            if second_phase:
                second_section_stops += 1
            else:
                passes += 1
                strategic_passes += int(bool(legal))
        else:
            sections += 1
            powered_sections += int(action.power is not None)
        game.apply_legal_action(action)

    elapsed = perf_counter() - started
    final = game.final_score()
    if final is None:
        raise RuntimeError("MCTS game ended before final scoring")
    name = policy_name(
        simulations,
        exploration,
        shared_objectives=objectives,
        pencil_powers=powers,
    )
    record = build_game_record(
        game,
        policy=name,
        seed_index=index,
        game_seed=seed,
        elapsed_seconds=elapsed,
        actions={
            "sections": sections,
            "passes": passes,
            "strategic_passes": strategic_passes,
            "second_section_stops": second_section_stops,
            "powered_sections": powered_sections,
        },
        algorithm={
            "family": "chance-sampled-uct",
            "simulations_per_decision": simulations,
            "exploration": exploration,
            "rollout_policy": "greedy",
            "shared_objectives": objectives,
            "pencil_powers": powers,
            "total_simulations": simulations * len(selected_errors),
            "candidate_actions": candidate_actions,
            "selected_standard_error_mean": fmean(selected_errors),
            "decision_nodes_total": decision_nodes,
            "decision_nodes_mean": decision_nodes / len(selected_errors),
            "tree_chance_samples": tree_chance,
            "rollout_chance_samples": rollout_chance,
            "terminal_rollouts": terminal_rollouts,
            "tree_terminal_hits": tree_terminal_hits,
            "mean_tree_depth": fmean(tree_depths),
            "max_tree_depth": max_tree_depth,
            "search_seconds": search_seconds,
        },
    )
    return CompletedGame(index, seed, final.total, elapsed, record)


def _manifest_seeds(path: Path) -> list[int]:
    manifest = load_json(path)
    seeds = [int(seed) for seed in manifest.get("game_seeds", ())]
    if not seeds:
        raise ValueError(f"{path} has no game_seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} contains duplicate game seeds")
    return seeds


def _validate_existing(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
    simulations: int,
    exploration: float,
    objectives: bool,
    powers: bool,
) -> None:
    for seed in seeds:
        record = records.get(seed)
        if record is None:
            continue
        algorithm = record.get("algorithm", {})
        if (
            record.get("policy")
            != policy_name(
                simulations,
                exploration,
                shared_objectives=objectives,
                pencil_powers=powers,
            )
            or algorithm.get("simulations_per_decision") != simulations
            or algorithm.get("rollout_policy", "greedy") != "greedy"
            or algorithm.get("shared_objectives", False) is not objectives
            or algorithm.get("pencil_powers", False) is not powers
            or not math.isclose(
                float(algorithm.get("exploration", -1.0)),
                exploration,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"existing result for seed {seed} uses another configuration"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chance-sampled UCT with Greedy rollouts"
    )
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="first real-game seed when --seeds-from is not used",
    )
    parser.add_argument("--seeds-from", type=Path, help="manifest JSON")
    parser.add_argument(
        "--simulations", type=_positive_int, default=DEFAULT_MCTS_SIMULATIONS
    )
    parser.add_argument(
        "--exploration", type=_nonnegative_float, default=DEFAULT_MCTS_EXPLORATION
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="enable both Shared Objectives and Pencil Powers",
    )
    parser.add_argument("--shared-objectives", action="store_true")
    parser.add_argument("--pencil-powers", action="store_true")
    parser.add_argument("--output", type=Path, help="resumable JSONL output")
    args = parser.parse_args()
    objectives = args.advanced or args.shared_objectives
    powers = args.advanced or args.pencil_powers

    source_manifest: Path | None = None
    if args.seeds_from is not None:
        source_manifest = args.seeds_from.resolve()
        available = _manifest_seeds(source_manifest)
        game_count = args.games or len(available)
        if game_count > len(available):
            parser.error(
                f"--games {game_count} exceeds {len(available)} manifest seeds"
            )
        seeds = available[:game_count]
    else:
        game_count = args.games or 1
        seeds = [args.seed + index for index in range(game_count)]

    name = policy_name(
        args.simulations,
        args.exploration,
        shared_objectives=objectives,
        pencil_powers=powers,
    )
    if args.output is not None:
        output_path = args.output.resolve()
    elif source_manifest is not None:
        output_path = source_manifest.parent / "games" / f"{name}.jsonl"
    else:
        output_path = None

    records = load_jsonl(output_path) if output_path is not None else {}
    _validate_existing(
        records,
        seeds,
        args.simulations,
        args.exploration,
        objectives,
        powers,
    )
    tasks = [
        (
            index,
            seed,
            args.simulations,
            args.exploration,
            objectives,
            powers,
        )
        for index, seed in enumerate(seeds)
        if seed not in records
    ]
    print(
        f"policy={name}; games={game_count}; completed={game_count - len(tasks)}; "
        f"remaining={len(tasks)}; workers={args.workers}; "
        f"shared_objectives={objectives}; pencil_powers={powers}",
        flush=True,
    )

    def accept(result: CompletedGame) -> None:
        records[result.seed] = result.record
        if output_path is not None:
            append_jsonl(output_path, result.record)
        print(
            f"game {result.index + 1}/{game_count}: seed={result.seed}, "
            f"score={result.score}, elapsed={result.elapsed_seconds:.2f}s",
            flush=True,
        )

    if args.workers == 1:
        for task in tasks:
            accept(run_mcts_game(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_mcts_game, task) for task in tasks]
            for future in as_completed(futures):
                accept(future.result())

    selected = [records[seed] for seed in seeds]
    summary = summarize(selected)
    if output_path is not None:
        print(f"results={output_path}", flush=True)
    score = summary["score"]
    print(
        f"score mean={score['mean']:.2f}, SE={score['standard_error']:.2f}, "
        f"min={score['min']:.0f}, median={score['median']:.1f}, "
        f"max={score['max']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
