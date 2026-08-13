"""Run resumable cross-tree batched DQN-MCTS benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from engine import GameSession
from solver import DEFAULT_MCTS_EXPLORATION
from solver.RL import (
    DEFAULT_DQN_MCTS_SIMULATIONS,
    DQNMCTSDecision,
    DQNMCTSPolicy,
    DQNMCTSSearch,
    DQNMCTSSession,
)

from .records import (
    append_jsonl,
    build_game_record,
    load_json,
    load_jsonl,
    summarize,
)


DEFAULT_PARALLEL_GAMES = 32


@dataclass(slots=True)
class _ActiveGame:
    index: int
    seed: int
    game: GameSession
    session: DQNMCTSSession
    started: float
    search: DQNMCTSSearch | None = None
    sections: int = 0
    passes: int = 0
    strategic_passes: int = 0
    candidate_actions: int = 0
    selected_errors: list[float] = field(default_factory=list)
    decision_nodes: int = 0
    tree_nodes_max: int = 0
    reuse_attempts: int = 0
    reuse_hits: int = 0
    retained_nodes: int = 0
    pruned_nodes: int = 0
    retained_root_visits: int = 0
    tree_chance_samples: int = 0
    rollout_chance_samples: int = 0
    terminal_rollouts: int = 0
    rollout_decisions: int = 0
    tree_terminal_hits: int = 0
    tree_depth_sum: float = 0.0
    max_tree_depth: int = 0
    forced_rollout_decisions: int = 0
    dqn_network_evaluations: int = 0
    inference_batches_participated: int = 0
    inference_batch_size_sum: float = 0.0
    max_inference_batch_size: int = 0
    root_prepare_seconds: float = 0.0
    subtree_prune_seconds: float = 0.0

    @classmethod
    def create(
        cls,
        index: int,
        seed: int,
        session: DQNMCTSSession,
    ) -> _ActiveGame:
        return cls(
            index=index,
            seed=seed,
            game=GameSession(seed=seed),
            session=session,
            started=perf_counter(),
        )

    def absorb(self, decision: DQNMCTSDecision) -> None:
        stats = decision.stats
        self.candidate_actions += len(decision.estimates)
        self.selected_errors.append(decision.standard_error)
        self.decision_nodes += stats.decision_nodes
        self.tree_nodes_max = max(self.tree_nodes_max, stats.tree_nodes)
        self.reuse_attempts += int(stats.reuse_attempted)
        self.reuse_hits += int(stats.reuse_hit)
        self.retained_nodes += stats.retained_nodes
        self.pruned_nodes += stats.pruned_nodes
        self.retained_root_visits += stats.retained_root_visits
        self.tree_chance_samples += stats.tree_chance_samples
        self.rollout_chance_samples += stats.rollout_chance_samples
        self.terminal_rollouts += stats.terminal_rollouts
        self.rollout_decisions += stats.rollout_decisions
        self.tree_terminal_hits += stats.tree_terminal_hits
        self.tree_depth_sum += stats.mean_tree_depth
        self.max_tree_depth = max(self.max_tree_depth, stats.max_tree_depth)
        self.forced_rollout_decisions += stats.forced_rollout_decisions
        self.dqn_network_evaluations += stats.dqn_network_evaluations
        self.inference_batches_participated += (
            stats.inference_batches_participated
        )
        self.inference_batch_size_sum += (
            stats.mean_inference_batch_size
            * stats.inference_batches_participated
        )
        self.max_inference_batch_size = max(
            self.max_inference_batch_size,
            stats.max_inference_batch_size,
        )
        self.root_prepare_seconds += stats.root_prepare_seconds
        self.subtree_prune_seconds += stats.subtree_prune_seconds


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


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_name(
    simulations: int,
    exploration: float,
    checkpoint_sha256: str,
) -> str:
    exploration_label = f"{exploration:g}".replace(".", "p")
    return (
        f"dqn-mcts-reuse-{simulations}-c{exploration_label}-"
        f"{checkpoint_sha256[:8]}"
    )


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
    *,
    name: str,
    simulations: int,
    exploration: float,
    checkpoint_sha256: str,
) -> None:
    for seed in seeds:
        record = records.get(seed)
        if record is None:
            continue
        algorithm = record.get("algorithm", {})
        if (
            record.get("policy") != name
            or algorithm.get("family") != "chance-sampled-uct"
            or algorithm.get("rollout_policy") != "double-dqn"
            or algorithm.get("simulations_per_decision") != simulations
            or algorithm.get("new_simulations_per_decision") != simulations
            or algorithm.get("checkpoint_sha256") != checkpoint_sha256
            or algorithm.get("in_flight_per_tree") != 1
            or algorithm.get("tree_reuse") is not True
            or algorithm.get("batching") != "asynchronous-cross-tree"
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


def _build_record(
    slot: _ActiveGame,
    *,
    name: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    simulations: int,
    exploration: float,
    parallel_games: int,
) -> dict[str, Any]:
    elapsed = perf_counter() - slot.started
    decisions = len(slot.selected_errors)
    if decisions == 0:
        raise RuntimeError("a completed game has no DQN-MCTS decisions")
    batches = slot.inference_batches_participated
    return build_game_record(
        slot.game,
        policy=name,
        seed_index=slot.index,
        game_seed=slot.seed,
        elapsed_seconds=elapsed,
        actions={
            "sections": slot.sections,
            "passes": slot.passes,
            "strategic_passes": slot.strategic_passes,
        },
        algorithm={
            "family": "chance-sampled-uct",
            "simulations_per_decision": simulations,
            "new_simulations_per_decision": simulations,
            "exploration": exploration,
            "rollout_policy": "double-dqn",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "batching": "asynchronous-cross-tree",
            "parallel_games": parallel_games,
            "in_flight_per_tree": 1,
            "tree_reuse": True,
            "total_simulations": simulations * decisions,
            "candidate_actions": slot.candidate_actions,
            "selected_standard_error_mean": fmean(slot.selected_errors),
            "decision_nodes_total": slot.decision_nodes,
            "decision_nodes_mean": slot.decision_nodes / decisions,
            "tree_nodes_max": slot.tree_nodes_max,
            "reuse_attempts": slot.reuse_attempts,
            "reuse_hits": slot.reuse_hits,
            "reuse_hit_rate": (
                slot.reuse_hits / slot.reuse_attempts
                if slot.reuse_attempts
                else 0.0
            ),
            "retained_nodes_total": slot.retained_nodes,
            "retained_nodes_mean_per_hit": (
                slot.retained_nodes / slot.reuse_hits
                if slot.reuse_hits
                else 0.0
            ),
            "pruned_nodes_total": slot.pruned_nodes,
            "retained_root_visits_total": slot.retained_root_visits,
            "retained_root_visits_mean_per_hit": (
                slot.retained_root_visits / slot.reuse_hits
                if slot.reuse_hits
                else 0.0
            ),
            "tree_chance_samples": slot.tree_chance_samples,
            "rollout_chance_samples": slot.rollout_chance_samples,
            "terminal_rollouts": slot.terminal_rollouts,
            "rollout_decisions": slot.rollout_decisions,
            "tree_terminal_hits": slot.tree_terminal_hits,
            "mean_tree_depth": slot.tree_depth_sum / decisions,
            "max_tree_depth": slot.max_tree_depth,
            "forced_rollout_decisions": slot.forced_rollout_decisions,
            "dqn_network_evaluations": slot.dqn_network_evaluations,
            "inference_batches_participated": batches,
            "mean_inference_batch_size": (
                slot.inference_batch_size_sum / batches if batches else 0.0
            ),
            "max_inference_batch_size": slot.max_inference_batch_size,
            "root_prepare_seconds": slot.root_prepare_seconds,
            "subtree_prune_seconds": slot.subtree_prune_seconds,
        },
    )


def _start_active_game(
    policy: DQNMCTSPolicy,
    index: int,
    seed: int,
) -> _ActiveGame:
    slot = _ActiveGame.create(index, seed, policy.new_session())
    slot.game.draw()
    slot.search = policy.start_search(slot.session, slot.game)
    return slot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chance-sampled UCT with batched Double DQN rollouts"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="first real-game seed when --seeds-from is not used",
    )
    parser.add_argument("--seeds-from", type=Path, help="manifest JSON")
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=DEFAULT_DQN_MCTS_SIMULATIONS,
    )
    parser.add_argument(
        "--exploration", type=_nonnegative_float, default=DEFAULT_MCTS_EXPLORATION
    )
    parser.add_argument(
        "--parallel-games",
        type=_positive_int,
        default=DEFAULT_PARALLEL_GAMES,
        help="independent search trees sharing each DQN inference stream",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect fine-grained CPU phase timings",
    )
    parser.add_argument("--output", type=Path, help="resumable JSONL output")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    checkpoint_sha256 = _checkpoint_sha256(checkpoint)

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

    name = policy_name(args.simulations, args.exploration, checkpoint_sha256)
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
        name=name,
        simulations=args.simulations,
        exploration=args.exploration,
        checkpoint_sha256=checkpoint_sha256,
    )
    pending = [
        (index, seed)
        for index, seed in enumerate(seeds)
        if seed not in records
    ]
    print(
        f"policy={name}; games={game_count}; "
        f"completed={game_count - len(pending)}; remaining={len(pending)}; "
        f"parallel_games={args.parallel_games}; device={args.device}",
        flush=True,
    )

    policy: DQNMCTSPolicy | None = None
    physical_batches = 0
    physical_batch_size_sum = 0.0
    physical_max_batch = 0
    physical_network_states = 0
    physical_inference_seconds = 0.0
    physical_simulations = 0
    physical_rollout_decisions = 0
    physical_forced_decisions = 0
    physical_state_copies = 0
    physical_state_keys = 0
    physical_tree_seconds = 0.0
    physical_copy_seconds = 0.0
    physical_key_seconds = 0.0
    physical_encoding_seconds = 0.0
    physical_rollout_step_seconds = 0.0
    physical_scheduler_seconds = 0.0
    if pending:
        policy = DQNMCTSPolicy.from_checkpoint(
            checkpoint,
            args.simulations,
            exploration=args.exploration,
            device=args.device,
            profile=args.profile,
        )

    cursor = 0
    active: list[_ActiveGame] = []
    while cursor < len(pending) and len(active) < args.parallel_games:
        if policy is None:
            raise RuntimeError("DQN-MCTS policy was not initialized")
        index, seed = pending[cursor]
        active.append(_start_active_game(policy, index, seed))
        cursor += 1

    while active:
        if policy is None:
            raise RuntimeError("DQN-MCTS policy was not initialized")
        searches = tuple(
            slot.search for slot in active if slot.search is not None
        )
        if len(searches) != len(active):
            raise RuntimeError("an active game has no search task")
        slots_by_search = {
            slot.search: slot for slot in active if slot.search is not None
        }
        completed_searches = policy.advance_many(searches)
        batch_stats = policy.last_batch_stats
        physical_batches += batch_stats.inference_batches
        physical_batch_size_sum += (
            batch_stats.mean_inference_batch_size
            * batch_stats.inference_batches
        )
        physical_max_batch = max(
            physical_max_batch,
            batch_stats.max_inference_batch_size,
        )
        physical_network_states += batch_stats.network_evaluations
        physical_inference_seconds += batch_stats.inference_seconds
        physical_simulations += batch_stats.simulations
        physical_rollout_decisions += batch_stats.rollout_decisions
        physical_forced_decisions += batch_stats.forced_rollout_decisions
        physical_state_copies += batch_stats.state_copies
        physical_state_keys += batch_stats.state_keys
        physical_tree_seconds += batch_stats.tree_phase_seconds
        physical_copy_seconds += batch_stats.state_copy_seconds
        physical_key_seconds += batch_stats.state_key_seconds
        physical_encoding_seconds += batch_stats.encoding_seconds
        physical_rollout_step_seconds += batch_stats.rollout_step_seconds
        physical_scheduler_seconds += batch_stats.elapsed_seconds

        finished_slots: list[_ActiveGame] = []
        for search, decision in completed_searches:
            slot = slots_by_search[search]
            slot.search = None
            legal = slot.game.legal_actions()
            slot.absorb(decision)
            action = decision.action
            if action is None:
                slot.passes += 1
                slot.strategic_passes += int(bool(legal))
                slot.game.act()
            else:
                slot.sections += 1
                slot.game.act(action.edge_id, action.source)

            if slot.game.status == "playing":
                slot.game.draw()
                slot.search = policy.start_search(slot.session, slot.game)
                continue

            finished_slots.append(slot)
            record = _build_record(
                slot,
                name=name,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                simulations=args.simulations,
                exploration=args.exploration,
                parallel_games=args.parallel_games,
            )
            records[slot.seed] = record
            if output_path is not None:
                append_jsonl(output_path, record)
            completed = sum(seed in records for seed in seeds)
            print(
                f"game {completed}/{game_count}: seed={slot.seed}, "
                f"score={record['score']['total']}, "
                f"elapsed={record['elapsed_seconds']:.2f}s",
                flush=True,
            )

        for slot in finished_slots:
            active.remove(slot)
        while cursor < len(pending) and len(active) < args.parallel_games:
            index, seed = pending[cursor]
            active.append(_start_active_game(policy, index, seed))
            cursor += 1

    selected = [records[seed] for seed in seeds]
    summary = summarize(selected)
    algorithm_records = [record["algorithm"] for record in selected]
    reuse_attempts = sum(item["reuse_attempts"] for item in algorithm_records)
    reuse_hits = sum(item["reuse_hits"] for item in algorithm_records)
    retained_nodes = sum(
        item["retained_nodes_total"] for item in algorithm_records
    )
    retained_root_visits = sum(
        item["retained_root_visits_total"] for item in algorithm_records
    )
    if output_path is not None:
        print(f"results={output_path}", flush=True)
    if physical_batches:
        print(
            f"current-run inference batches={physical_batches}, "
            f"mean_batch={physical_batch_size_sum / physical_batches:.1f}, "
            f"max_batch={physical_max_batch}, "
            f"network_states={physical_network_states}, "
            f"inference_seconds={physical_inference_seconds:.2f}",
            flush=True,
        )
    if reuse_attempts:
        print(
            f"subtree reuse hits={reuse_hits}/{reuse_attempts} "
            f"({100.0 * reuse_hits / reuse_attempts:.1f}%), "
            f"mean_retained_nodes="
            f"{retained_nodes / reuse_hits if reuse_hits else 0.0:.1f}, "
            f"mean_retained_root_visits="
            f"{retained_root_visits / reuse_hits if reuse_hits else 0.0:.1f}",
            flush=True,
        )
    if args.profile and physical_simulations:
        print(
            f"profile simulations={physical_simulations}, "
            f"rollout_decisions={physical_rollout_decisions}, "
            f"forced={physical_forced_decisions}, "
            f"state_copies={physical_state_copies}, "
            f"state_keys={physical_state_keys}",
            flush=True,
        )
        print(
            f"profile seconds: scheduler={physical_scheduler_seconds:.2f}, "
            f"tree={physical_tree_seconds:.2f}, "
            f"copy={physical_copy_seconds:.2f}, "
            f"key={physical_key_seconds:.2f}, "
            f"encode={physical_encoding_seconds:.2f}, "
            f"inference={physical_inference_seconds:.2f}, "
            f"rollout_step={physical_rollout_step_seconds:.2f}",
            flush=True,
        )
    score = summary["score"]
    print(
        f"score mean={score['mean']:.2f}, SE={score['standard_error']:.2f}, "
        f"min={score['min']:.0f}, median={score['median']:.1f}, "
        f"max={score['max']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
