"""Run the maintained policies on one shared set of game seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
from time import perf_counter
from typing import Any

from engine_cpp import GameSession
from solver import (
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_MCTS_SIMULATIONS,
    Depth2Policy,
    DepthKPolicy,
    GreedyPolicy,
)

from .mcts import policy_name as mcts_policy_name
from .mcts import run_mcts_game
from .records import (
    append_jsonl,
    build_game_record,
    describe,
    load_json,
    load_jsonl,
    summarize,
    write_json,
)


POLICIES = (
    "simple-random",
    "greedy",
    "lookahead-2",
    "lookahead-3",
    "lookahead-4",
    "mcts",
)

POLICY_LABELS = {
    "simple-random": "Random",
    "greedy": "Greedy",
    "lookahead-2": "Lookahead-2",
    "lookahead-3": "Lookahead-3",
    "lookahead-4": "Lookahead-4",
}


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


def _module_suffix(objectives: bool, powers: bool) -> str:
    if objectives and powers:
        return "-advanced"
    if objectives:
        return "-shared-objectives"
    if powers:
        return "-pencil-powers"
    return ""


def _stored_name(
    policy: str,
    objectives: bool = False,
    powers: bool = False,
) -> str:
    if policy == "mcts":
        return mcts_policy_name(
            DEFAULT_MCTS_SIMULATIONS,
            DEFAULT_MCTS_EXPLORATION,
            shared_objectives=objectives,
            pencil_powers=powers,
        )
    return f"{policy}{_module_suffix(objectives, powers)}"


def _configuration(
    policy: str,
    objectives: bool = False,
    powers: bool = False,
) -> dict[str, Any]:
    if policy.startswith("lookahead-"):
        configuration: dict[str, Any] = {
            "family": "lookahead",
            "depth": int(policy.rsplit("-", 1)[1]),
        }
        if policy == "lookahead-2":
            configuration["implementation"] = "specialized"
    elif policy == "mcts":
        configuration = {
            "family": "chance-sampled-uct",
            "simulations_per_decision": DEFAULT_MCTS_SIMULATIONS,
            "exploration": DEFAULT_MCTS_EXPLORATION,
            "rollout_policy": "greedy",
        }
    else:
        configuration = {"family": policy}
    configuration["shared_objectives"] = objectives
    configuration["pencil_powers"] = powers
    return configuration


def _run_standard_game(
    task: tuple[str, int, int, bool, bool],
) -> CompletedGame:
    policy_name, index, seed, objectives, powers = task
    game = GameSession(
        seed=seed,
        shared_objectives_enabled=objectives,
        pencil_powers_enabled=powers,
    )
    system_rng = random.SystemRandom()
    if policy_name == "greedy":
        policy: GreedyPolicy | DepthKPolicy | None = GreedyPolicy()
    elif policy_name == "lookahead-2":
        policy = Depth2Policy()
    elif policy_name.startswith("lookahead-"):
        policy = DepthKPolicy(int(policy_name.rsplit("-", 1)[1]))
    elif policy_name == "simple-random":
        policy = None
    else:
        raise ValueError(f"unknown standard policy: {policy_name}")

    sections = passes = strategic_passes = candidate_actions = 0
    second_section_stops = powered_sections = 0
    decision_nodes = chance_nodes = chance_outcomes = cache_hits = 0
    started = perf_counter()
    while game.status == "playing":
        if game.pending is None:
            game.draw()
        legal = game.legal_actions()
        if policy is None:
            action = system_rng.choice(legal) if legal else None
        else:
            decision = policy.choose(game)
            action = decision.action
            stats = getattr(decision, "stats", None)
            if stats is not None:
                decision_nodes += stats.decision_nodes
                chance_nodes += stats.chance_nodes
                chance_outcomes += stats.chance_outcomes
                cache_hits += stats.cache_hits
        candidate_actions += len(legal) + int(policy_name.startswith("lookahead-"))
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
        raise RuntimeError("benchmark game ended before final scoring")
    algorithm = {
        **_configuration(policy_name, objectives, powers),
        "candidate_actions": candidate_actions,
    }
    if policy_name.startswith("lookahead-"):
        algorithm.update(
            {
                "decision_nodes": decision_nodes,
                "chance_nodes": chance_nodes,
                "chance_outcomes": chance_outcomes,
                "cache_hits": cache_hits,
            }
        )
    record = build_game_record(
        game,
        policy=_stored_name(policy_name, objectives, powers),
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
        algorithm=algorithm,
    )
    return CompletedGame(index, seed, final.total, elapsed, record)


def _manifest_seeds(path: Path, games: int | None) -> list[int]:
    source = load_json(path)
    available = [int(seed) for seed in source.get("game_seeds", ())]
    if not available:
        raise ValueError(f"{path} has no game_seeds")
    if len(set(available)) != len(available):
        raise ValueError(f"{path} contains duplicate game seeds")
    count = games or len(available)
    if count > len(available):
        raise ValueError(f"requested {count} games from {len(available)} seeds")
    return available[:count]


def _validate_records(
    records: dict[int, dict[str, Any]],
    seeds: list[int],
    policy: str,
    objectives: bool,
    powers: bool,
) -> None:
    name = _stored_name(policy, objectives, powers)
    expected = _configuration(policy, objectives, powers)
    for seed in seeds:
        record = records.get(seed)
        if record is None:
            continue
        if record.get("policy") != name:
            raise ValueError(
                f"seed {seed} has policy {record.get('policy')}, not {name}"
            )
        algorithm = record.get("algorithm", {})
        if any(algorithm.get(key) != value for key, value in expected.items()):
            raise ValueError(f"seed {seed} has a different {name} configuration")


def _policy_label(name: str) -> str:
    if name.startswith("mcts-uct-"):
        return "MCTS"
    return POLICY_LABELS.get(name, name)


def _write_suite_summary(
    output_dir: Path,
    seeds: list[int],
    policies: list[str] | tuple[str, ...],
    objectives: bool,
    powers: bool,
) -> None:
    names = [
        _stored_name(policy, objectives, powers) for policy in policies
    ]
    if len(set(names)) != len(names):
        raise ValueError("benchmark policies map to duplicate stored names")

    records_by_policy: dict[str, dict[int, dict[str, Any]]] = {}
    policy_summaries: dict[str, dict[str, Any]] = {}
    for policy, name in zip(policies, names):
        records = load_jsonl(output_dir / "games" / f"{name}.jsonl")
        missing = [seed for seed in seeds if seed not in records]
        if missing:
            raise ValueError(
                f"cannot summarize {name}: missing {len(missing)} game seeds"
            )
        _validate_records(records, seeds, policy, objectives, powers)
        records_by_policy[name] = records
        policy_summaries[name] = summarize([records[seed] for seed in seeds])

    winner_appearances = dict.fromkeys(names, 0)
    sole_wins = dict.fromkeys(names, 0)
    current_highest_scores: list[int] = []
    tied_seed_count = 0
    for seed in seeds:
        scores = {
            name: int(records_by_policy[name][seed]["score"]["total"])
            for name in names
        }
        current_highest = max(scores.values())
        winners = [name for name in names if scores[name] == current_highest]
        for winner in winners:
            winner_appearances[winner] += 1
        if len(winners) == 1:
            sole_wins[winners[0]] += 1
        else:
            tied_seed_count += 1
        current_highest_scores.append(current_highest)

    current_best_score = describe(current_highest_scores)
    strongest_fixed = max(
        names,
        key=lambda name: policy_summaries[name]["score"]["mean"],
    )
    strongest_fixed_mean = policy_summaries[strongest_fixed]["score"]["mean"]
    mean_gain = current_best_score["mean"] - strongest_fixed_mean
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    markdown = [
        f"# 当前 {len(seeds)}-seed 基准汇总",
        "",
        f"游戏数：`{len(seeds)}`  ",
        f"生成时间：`{generated_at}`",
        "",
        (
            "> **口径说明：**“逐 seed 当前最高分”是在本次纳入的 policy "
            "结果中事后取最大值。它不是一个可执行 agent 的成绩，也不是理论最优分。"
        ),
        "",
        "## 固定 policy 表现",
        "",
        "| Policy | Mean | SE | Min | Median | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        score = policy_summaries[name]["score"]
        markdown.append(
            f"| {_policy_label(name)} | {score['mean']:.2f} | "
            f"{score['standard_error']:.2f} | {score['min']:.0f} | "
            f"{score['median']:.1f} | {score['max']:.0f} |"
        )

    markdown.extend(
        [
            "",
            "## 逐 seed 当前最高分",
            "",
            (
                f"事后最佳均分为 `{current_best_score['mean']:.2f} +/- "
                f"{current_best_score['standard_error']:.2f}`，范围 "
                f"`{current_best_score['min']:.0f}..{current_best_score['max']:.0f}`，"
                f"中位数 `{current_best_score['median']:.1f}`。"
            ),
            "",
            (
                f"当前最强的单一固定 policy 是 `{_policy_label(strongest_fixed)}`，"
                f"均分 `{strongest_fixed_mean:.2f}`；事后逐 seed 选择高 "
                f"`{mean_gain:.2f}` 分。"
            ),
            "",
            "## 获胜次数",
            "",
            "| Policy | 单独最高 | 并列最高 | 最高分出现次数 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in names:
        tied_appearances = winner_appearances[name] - sole_wins[name]
        markdown.append(
            f"| {_policy_label(name)} | {sole_wins[name]} | "
            f"{tied_appearances} | {winner_appearances[name]} |"
        )
    markdown.extend(
        [
            "",
            f"共有 `{tied_seed_count}` 个 seed 出现最高分并列。",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maintained policies on shared seeds"
    )
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="enable both Shared Objectives and Pencil Powers",
    )
    parser.add_argument("--shared-objectives", action="store_true")
    parser.add_argument("--pencil-powers", action="store_true")
    args = parser.parse_args()
    objectives = args.advanced or args.shared_objectives
    powers = args.advanced or args.pencil_powers

    source_manifest = args.seeds_from.resolve()
    seeds = _manifest_seeds(source_manifest, args.games)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "source_manifest": str(source_manifest),
        "game_seeds": seeds,
        "game_seed_semantics": (
            "controls real color order and card shuffles and, when enabled, "
            "objective cards and pencil-power assignments via independent RNG streams"
        ),
        "agent_randomness": "unseeded Random/SystemRandom as defined by each policy",
        "modules": {
            "shared_objectives": objectives,
            "pencil_powers": powers,
        },
        "workers": args.workers,
        "policies": {
            _stored_name(policy, objectives, powers): _configuration(
                policy,
                objectives,
                powers,
            )
            for policy in args.policies
        },
        "completed_policies": [],
    }
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if (
            existing.get("game_seeds") != seeds
            or existing.get("policies") != manifest["policies"]
        ):
            raise ValueError("existing benchmark manifest has different seeds or policies")
        manifest = existing
        manifest["status"] = "running"
        manifest["workers"] = args.workers
    write_json(manifest_path, manifest)

    for policy in args.policies:
        name = _stored_name(policy, objectives, powers)
        output_path = output_dir / "games" / f"{name}.jsonl"
        records = load_jsonl(output_path)
        _validate_records(records, seeds, policy, objectives, powers)
        pending = [
            (index, seed)
            for index, seed in enumerate(seeds)
            if seed not in records
        ]
        print(
            f"policy={name}; completed={len(seeds) - len(pending)}/{len(seeds)}; "
            f"remaining={len(pending)}; workers={args.workers}",
            flush=True,
        )

        if policy == "mcts":
            tasks = [
                (
                    index,
                    seed,
                    DEFAULT_MCTS_SIMULATIONS,
                    DEFAULT_MCTS_EXPLORATION,
                    objectives,
                    powers,
                )
                for index, seed in pending
            ]
            runner = run_mcts_game
        else:
            tasks = [
                (policy, index, seed, objectives, powers)
                for index, seed in pending
            ]
            runner = _run_standard_game

        def accept(result: CompletedGame) -> None:
            records[result.seed] = result.record
            append_jsonl(output_path, result.record)
            print(
                f"{name} {result.index + 1}/{len(seeds)}: seed={result.seed}, "
                f"score={result.score}, elapsed={result.elapsed_seconds:.2f}s",
                flush=True,
            )

        if args.workers == 1:
            for task in tasks:
                accept(runner(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(runner, task) for task in tasks]
                for future in as_completed(futures):
                    accept(future.result())

        selected = [records[seed] for seed in seeds]
        summary = summarize(selected)
        score = summary["score"]
        print(
            f"completed {name}: mean={score['mean']:.2f}, "
            f"SE={score['standard_error']:.2f}, min={score['min']:.0f}, "
            f"median={score['median']:.1f}, max={score['max']:.0f}",
            flush=True,
        )
        completed = manifest.setdefault("completed_policies", [])
        if name not in completed:
            completed.append(name)
        write_json(manifest_path, manifest)

    _write_suite_summary(
        output_dir,
        seeds,
        args.policies,
        objectives,
        powers,
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
