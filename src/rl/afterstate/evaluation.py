"""Evaluation and paired-comparison helpers for afterstate policies."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from engine_cpp import GameSession

from .environment import AfterstateEnv
from .network import AfterstateValueNetwork, network_from_checkpoint, resolve_device
from .policy import AfterstatePolicy, BellmanImprovedAfterstatePolicy

if TYPE_CHECKING:
    from solver.RL.exact_chance import ExactChanceDQNPolicy


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    scores: tuple[int, ...]
    mean: float
    standard_error: float
    minimum: int
    median: float
    maximum: int
    elapsed_seconds: float

    def summary_dict(self) -> dict[str, float | int]:
        return {
            "games": len(self.scores),
            "mean": self.mean,
            "standard_error": self.standard_error,
            "min": self.minimum,
            "median": self.median,
            "max": self.maximum,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class BackupEvaluation:
    result: EvaluationResult
    diagnostics: dict[str, float | int]


def summarize_scores(
    scores: Sequence[int],
    elapsed_seconds: float = 0.0,
) -> EvaluationResult:
    if not scores:
        raise ValueError("evaluation requires at least one score")
    array = np.asarray(scores, dtype=np.float64)
    return EvaluationResult(
        scores=tuple(int(score) for score in scores),
        mean=float(array.mean()),
        standard_error=(
            float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
        ),
        minimum=int(array.min()),
        median=float(np.median(array)),
        maximum=int(array.max()),
        elapsed_seconds=float(elapsed_seconds),
    )


def evaluate_policy(
    policy: AfterstatePolicy | BellmanImprovedAfterstatePolicy,
    seeds: Sequence[int],
    *,
    num_envs: int = 64,
) -> EvaluationResult:
    """Run an afterstate policy while preserving the input seed order."""

    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    started = perf_counter()
    scores: list[int | None] = [None] * len(seeds)
    for start in range(0, len(seeds), num_envs):
        batch_seeds = tuple(int(seed) for seed in seeds[start : start + num_envs])
        envs = [
            AfterstateEnv(seed, verify=True, initial_game_seed=seed)
            for seed in batch_seeds
        ]
        completed = [False] * len(envs)
        while not all(completed):
            active = [index for index, done in enumerate(completed) if not done]
            groups = tuple(envs[index].candidates() for index in active)
            decisions = policy.select_groups(groups)
            for index, decision in zip(active, decisions):
                episode = envs[index].step(decision.candidate)
                if episode is not None:
                    scores[start + index] = episode.score
                    completed[index] = True
    if any(score is None for score in scores):
        raise RuntimeError("evaluation ended with unfinished games")
    return summarize_scores(
        tuple(int(score) for score in scores),
        perf_counter() - started,
    )


def evaluate_network(
    network: AfterstateValueNetwork,
    seeds: Sequence[int],
    *,
    device: torch.device,
    num_envs: int = 64,
    reward_scale: float = 10.0,
) -> EvaluationResult:
    policy = AfterstatePolicy(
        network,
        device=device,
        reward_scale=reward_scale,
    )
    return evaluate_policy(policy, seeds, num_envs=num_envs)


def evaluate_backup_network(
    network: AfterstateValueNetwork,
    seeds: Sequence[int],
    *,
    device: torch.device,
    num_envs: int,
    reward_scale: float,
    gamma: float,
    inference_batch_size: int,
) -> BackupEvaluation:
    policy = BellmanImprovedAfterstatePolicy(
        network,
        network,
        device=device,
        reward_scale=reward_scale,
        gamma=gamma,
        inference_batch_size=inference_batch_size,
    )
    result = evaluate_policy(policy, seeds, num_envs=num_envs)
    stats = policy.stats
    return BackupEvaluation(
        result=result,
        diagnostics={
            "decisions": stats.decisions,
            "action_agreement": stats.agreement_rate,
            "mean_regret_points": stats.mean_normalized_regret * reward_scale,
            "root_candidates_per_decision": (
                stats.root_candidates / stats.decisions if stats.decisions else 0.0
            ),
        },
    )


def load_manifest_seeds(path: Path, games: int | None = None) -> tuple[int, ...]:
    """Read a UTF-8 seed manifest and preserve its declared order."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in raw.get("game_seeds", ()))
    if not seeds:
        raise ValueError(f"{path} does not contain game_seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} contains duplicate game seeds")
    if games is not None:
        if games < 1 or games > len(seeds):
            raise ValueError(f"requested {games} games from {len(seeds)} seeds")
        seeds = seeds[:games]
    return seeds


def generate_validation_seeds(seed: int, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed ^ 0x5A17_D3C4_91E2_6B0F)
    result: list[int] = []
    seen: set[int] = set()
    while len(result) < count:
        candidate = rng.getrandbits(63) | (1 << 63)
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return tuple(result)


def generate_independent_seeds(
    source_seeds: Sequence[int],
    *,
    seed: int,
    count: int = 200,
) -> tuple[int, ...]:
    """Generate a deterministic seed set disjoint from a source manifest."""

    if count < 1:
        raise ValueError("count must be positive")
    excluded = {int(value) for value in source_seeds}
    rng = random.Random(int(seed) ^ 0xA4F2_71C9_0D55_38B7)
    result: list[int] = []
    while len(result) < count:
        candidate = rng.getrandbits(63) | (1 << 63)
        if candidate not in excluded:
            excluded.add(candidate)
            result.append(candidate)
    return tuple(result)


def _evaluate_exact_depth1_ordered(
    policy: ExactChanceDQNPolicy,
    seeds: Sequence[int],
    *,
    num_envs: int,
) -> EvaluationResult:
    started = perf_counter()
    scores: list[int | None] = [None] * len(seeds)
    next_seed = 0
    slots: list[tuple[int, GameSession]] = []

    def make_slot(index: int) -> tuple[int, GameSession]:
        game = GameSession(seed=int(seeds[index]), advanced=False)
        game.draw()
        return index, game

    while next_seed < len(seeds) and len(slots) < num_envs:
        slots.append(make_slot(next_seed))
        next_seed += 1
    while slots:
        decisions = policy.choose_many(tuple(game for _, game in slots))
        new_slots: list[tuple[int, GameSession]] = []
        for (seed_index, game), decision in zip(slots, decisions):
            game.apply_legal_action(decision.action)
            if game.status == "finished":
                final = game.final_score()
                if final is None:
                    raise RuntimeError("finished baseline game has no score")
                scores[seed_index] = int(final.total)
                if next_seed < len(seeds):
                    new_slots.append(make_slot(next_seed))
                    next_seed += 1
            else:
                game.draw()
                new_slots.append((seed_index, game))
        slots = new_slots
    if any(score is None for score in scores):
        raise RuntimeError("baseline evaluation ended with unfinished games")
    return summarize_scores(
        tuple(int(score) for score in scores),
        perf_counter() - started,
    )


def paired_summary(
    afterstate: EvaluationResult,
    baseline: EvaluationResult,
) -> dict[str, Any]:
    """Return paired difference statistics and win/tie/loss counts."""

    if len(afterstate.scores) != len(baseline.scores):
        raise ValueError("paired evaluations have different game counts")
    differences = np.asarray(afterstate.scores, dtype=np.float64) - np.asarray(
        baseline.scores,
        dtype=np.float64,
    )
    mean = float(differences.mean())
    se = (
        float(differences.std(ddof=1) / np.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    wins = int(np.count_nonzero(differences > 0))
    ties = int(np.count_nonzero(differences == 0))
    losses = int(np.count_nonzero(differences < 0))
    return {
        "games": len(differences),
        "mean_difference": mean,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "differences": [int(value) for value in differences],
    }


def run_final_evaluation(
    checkpoint: Path,
    baseline_checkpoint: Path,
    seeds: Sequence[int],
    output_dir: Path,
    *,
    device_name: str = "auto",
    num_envs: int = 32,
    baseline_num_envs: int = 8,
) -> dict[str, Any]:
    """Run both policies on exactly the same seeds and write a report."""

    from solver.RL.exact_chance import ExactChanceDQNPolicy

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    network = network_from_checkpoint(raw, device)
    config = raw.get("config", {})
    reward_scale = float(config.get("reward_scale", 10.0))
    afterstate_evaluation = evaluate_backup_network(
        network,
        seeds,
        device=device,
        num_envs=num_envs,
        reward_scale=reward_scale,
        gamma=float(config.get("gamma", 1.0)),
        inference_batch_size=int(config.get("inference_batch_size", 8192)),
    )
    afterstate_result = afterstate_evaluation.result
    baseline = ExactChanceDQNPolicy.from_checkpoint(
        baseline_checkpoint,
        depth=1,
        device=str(device),
        leaf_batch_size=8192,
    )
    baseline_result = _evaluate_exact_depth1_ordered(
        baseline,
        seeds,
        num_envs=baseline_num_envs,
    )
    paired = paired_summary(afterstate_result, baseline_result)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "game_seed_semantics": "controls only real color order and card shuffles",
        "game_seeds": [int(seed) for seed in seeds],
        "afterstate_checkpoint": str(checkpoint.resolve()),
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "afterstate_policy": "BellmanImprovedAfterstatePolicy(online/online)",
        "baseline_policy": "ExactChanceDQNPolicy(depth=1)",
        "afterstate": afterstate_result.summary_dict(),
        "afterstate_diagnostics": afterstate_evaluation.diagnostics,
        "baseline": baseline_result.summary_dict(),
        "afterstate_scores": list(afterstate_result.scores),
        "baseline_scores": list(baseline_result.scores),
        "paired": paired,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Final Afterstate Comparison",
        "",
        f"Seeds: {len(seeds)} (paired, identical order)",
        "",
        "| Policy | Mean | SE | Min | Median | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| afterstate online/online exact backup | {afterstate_result.mean:.3f} | {afterstate_result.standard_error:.3f} | {afterstate_result.minimum} | {afterstate_result.median:.1f} | {afterstate_result.maximum} |",
        f"| exact depth-1 baseline | {baseline_result.mean:.3f} | {baseline_result.standard_error:.3f} | {baseline_result.minimum} | {baseline_result.median:.1f} | {baseline_result.maximum} |",
        "",
        f"Paired difference (afterstate - baseline): {paired['mean_difference']:.3f} "
        f"(95% CI {paired['ci95_low']:.3f} to {paired['ci95_high']:.3f}); "
        f"W/T/L = {paired['wins']}/{paired['ties']}/{paired['losses']}.",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def evaluate_afterstate_checkpoint(
    checkpoint: Path,
    seeds: Sequence[int],
    *,
    device_name: str = "auto",
    num_envs: int = 64,
) -> EvaluationResult:
    device = resolve_device(device_name)
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    network = network_from_checkpoint(raw, device)
    reward_scale = float(raw.get("config", {}).get("reward_scale", 10.0))
    return evaluate_network(
        network,
        seeds,
        device=device,
        num_envs=num_envs,
        reward_scale=reward_scale,
    )
