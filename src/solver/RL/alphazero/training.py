"""Self-play collection, optimization, evaluation, and checkpointing."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import secrets
from statistics import median
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from engine import GameSession

from ..codec import ACTION_COUNT, PASS_ACTION_INDEX
from ..dqn import resolve_device
from ..environment import DecisionEnv
from .network import NetworkSpec, PolicyValueNetwork, parameter_count
from .replay import AlphaZeroReplay, ReplayBatch, ReplayRecord
from .search import (
    VALUE_SCALE,
    BatchedPUCT,
    PolicyValueEvaluator,
    SearchBatchStats,
    SearchConfig,
)


_SEARCH_SEED_SALT = 0x9E37_79B9_7F4A_7C15


@dataclass(frozen=True, slots=True)
class TrainConfig:
    run_seed: int
    num_envs: int = 128
    total_positions: int = 1_000_000
    simulations: int = 256
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 12
    replay_capacity: int = 500_000
    replay_ratio: float = 16.0
    batch_size: int = 2048
    learning_rate: float = 3.0e-4
    final_learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    warmup_updates: int = 200
    gradient_clip: float = 5.0
    validation_interval: int = 50_000
    evaluation_envs: int = 128
    evaluation_simulations: int = 256
    max_wall_seconds: float = 27_000.0
    device: str = "auto"
    network_width: int = 1024
    residual_blocks: int = 6
    value_hidden: int = 512
    validation_seeds: tuple[int, ...] = ()

    @classmethod
    def fresh(cls, **overrides: object) -> TrainConfig:
        return cls(run_seed=secrets.randbits(63), **overrides)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> TrainConfig:
        values = dict(raw)
        values.pop("schema_version", None)
        values["validation_seeds"] = tuple(
            int(seed) for seed in values.get("validation_seeds", ())
        )
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["validation_seeds"] = list(self.validation_seeds)
        return raw

    @property
    def network_spec(self) -> NetworkSpec:
        return NetworkSpec(
            width=self.network_width,
            residual_blocks=self.residual_blocks,
            value_hidden=self.value_hidden,
        )

    @property
    def search_config(self) -> SearchConfig:
        return SearchConfig(
            simulations=self.simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )

    def validate(self) -> None:
        self.search_config.validate()
        positive_ints = {
            "num_envs": self.num_envs,
            "total_positions": self.total_positions,
            "temperature_moves": self.temperature_moves,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "warmup_updates": self.warmup_updates,
            "validation_interval": self.validation_interval,
            "evaluation_envs": self.evaluation_envs,
            "evaluation_simulations": self.evaluation_simulations,
            "network_width": self.network_width,
            "residual_blocks": self.residual_blocks,
            "value_hidden": self.value_hidden,
        }
        if any(value < 1 for value in positive_ints.values()):
            raise ValueError("AlphaZero integer configuration values must be positive")
        for name, value in {
            "replay_ratio": self.replay_ratio,
            "learning_rate": self.learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "gradient_clip": self.gradient_clip,
            "max_wall_seconds": self.max_wall_seconds,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must hold at least one batch")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    scores: tuple[int, ...]
    elapsed_seconds: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def standard_error(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return float(np.std(self.scores, ddof=1) / math.sqrt(len(self.scores)))

    def summary_dict(self) -> dict[str, float | int]:
        return {
            "games": len(self.scores),
            "mean": self.mean,
            "standard_error": self.standard_error,
            "minimum": min(self.scores),
            "median": float(median(self.scores)),
            "maximum": max(self.scores),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class _EpisodePosition:
    observation: NDArray[np.uint8]
    action_mask: NDArray[np.bool_]
    visits: NDArray[np.uint16]
    reward: float = 0.0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    records: tuple[ReplayRecord, ...]
    scores: tuple[int, ...]
    searches: int
    simulations: int
    expanded_nodes: int
    network_evaluations: int
    inference_batches: int
    mean_inference_batch_size: float
    max_inference_batch_size: int
    forced_decisions: int
    elapsed_seconds: float


class AlphaZeroLearner:
    def __init__(
        self,
        network: PolicyValueNetwork,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.network = network.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda"
        )
        self.updates = 0

    def _learning_rate(self) -> float:
        if self.updates < self.config.warmup_updates:
            return self.config.learning_rate * (
                (self.updates + 1) / self.config.warmup_updates
            )
        target_updates = max(
            self.config.warmup_updates + 1,
            math.ceil(
                self.config.total_positions
                * self.config.replay_ratio
                / self.config.batch_size
            ),
        )
        progress = min(
            1.0,
            (self.updates - self.config.warmup_updates)
            / (target_updates - self.config.warmup_updates),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.final_learning_rate + cosine * (
            self.config.learning_rate - self.config.final_learning_rate
        )

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        self.network.train()
        learning_rate = self._learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        observations = torch.as_tensor(batch.observations, device=self.device).float()
        masks = torch.as_tensor(batch.action_masks, device=self.device)
        targets = torch.as_tensor(batch.policy_targets, device=self.device)
        values = torch.as_tensor(batch.values, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if amp_enabled else torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits, predicted_values = self.network(observations)
            masked_logits = logits.masked_fill(~masks, -1.0e4)
            log_probabilities = torch.log_softmax(masked_logits, dim=1)
            policy_loss = -(targets * log_probabilities).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_values, values)
            loss = policy_loss + value_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config.gradient_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.updates += 1

        with torch.no_grad():
            value_error = predicted_values.float() - values
            entropy = -(targets * torch.log(targets.clamp_min(1.0e-8))).sum(dim=1)
            agreement = (
                masked_logits.argmax(dim=1) == targets.argmax(dim=1)
            ).float()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "value_rmse_points": float(value_error.square().mean().sqrt() * VALUE_SCALE),
            "value_bias_points": float(value_error.mean() * VALUE_SCALE),
            "target_policy_entropy": float(entropy.mean()),
            "policy_search_agreement": float(agreement.mean()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": learning_rate,
        }


def _new_game_seed(
    stream: random.Random,
    excluded: set[int],
) -> int:
    while True:
        seed = stream.getrandbits(63)
        if seed not in excluded:
            return seed


def _choose_training_action(
    visits: NDArray[np.uint16],
    decision_index: int,
    temperature_moves: int,
    rng: np.random.Generator,
) -> int:
    total = int(visits.sum(dtype=np.uint64))
    if total < 1:
        raise RuntimeError("search returned no root visits")
    if decision_index < temperature_moves:
        return int(rng.choice(ACTION_COUNT, p=visits.astype(np.float64) / total))
    most = visits.max()
    tied = np.flatnonzero(visits == most)
    return int(tied[rng.integers(len(tied))])


def collect_generation(
    network: PolicyValueNetwork,
    config: TrainConfig,
    *,
    device: str,
    game_seed_stream: random.Random,
    collector_rng: np.random.Generator,
    verify: bool = False,
) -> GenerationResult:
    """Generate exactly one frozen-network batch of complete games."""

    started = perf_counter()
    excluded = set(config.validation_seeds)
    envs = [DecisionEnv(0, verify=verify) for _ in range(config.num_envs)]
    for env in envs:
        env.reset(_new_game_seed(game_seed_stream, excluded))
        if env.game.advanced:
            raise RuntimeError("self-play unexpectedly enabled advanced rules")

    search = BatchedPUCT(network, config.search_config, device=device)
    histories: list[list[_EpisodePosition]] = [
        [] for _ in range(config.num_envs)
    ]
    active = list(range(config.num_envs))
    records: list[ReplayRecord] = []
    scores: list[int] = []
    searches = simulations = expanded_nodes = network_evaluations = 0
    inference_batches = inference_size_sum = max_inference_batch_size = 0
    forced_decisions = 0

    while active:
        searched_indices = [
            index for index in active if int(envs[index].action_mask.sum()) > 1
        ]
        result_by_index = {}
        if searched_indices:
            search_results = search.search(
                [envs[index].game for index in searched_indices],
                seeds=[game_seed_stream.getrandbits(128) for _ in searched_indices],
                add_root_noise=True,
            )
            result_by_index = dict(zip(searched_indices, search_results))
            stats = search.last_stats
            searches += stats.searches
            simulations += stats.simulations
            expanded_nodes += stats.expanded_nodes
            network_evaluations += stats.network_evaluations
            inference_batches += stats.inference_batches
            inference_size_sum += int(
                round(stats.mean_inference_batch_size * stats.inference_batches)
            )
            max_inference_batch_size = max(
                max_inference_batch_size, stats.max_inference_batch_size
            )

        next_active: list[int] = []
        for index in active:
            env = envs[index]
            if index in result_by_index:
                visits = result_by_index[index].action_visits
            else:
                visits = np.zeros(ACTION_COUNT, dtype=np.uint16)
                legal = np.flatnonzero(env.action_mask)
                if len(legal) != 1:
                    raise RuntimeError("a skipped search was not a forced decision")
                visits[int(legal[0])] = 1
                forced_decisions += 1
            action = _choose_training_action(
                visits,
                env.decisions,
                config.temperature_moves,
                collector_rng,
            )
            position = _EpisodePosition(
                observation=np.array(env.observation, copy=True),
                action_mask=np.array(env.action_mask, copy=True),
                visits=np.array(visits, copy=True),
            )
            result = env.step(action)
            position.reward = float(result.reward) / VALUE_SCALE
            histories[index].append(position)
            if result.episode is None:
                next_active.append(index)
                continue

            running_return = 0.0
            episode_records: list[ReplayRecord] = []
            for item in reversed(histories[index]):
                running_return = item.reward + running_return
                episode_records.append(
                    ReplayRecord(
                        observation=item.observation,
                        action_mask=item.action_mask,
                        visits=item.visits,
                        value=running_return,
                    )
                )
            records.extend(reversed(episode_records))
            scores.append(result.episode.score)
        active = next_active

    return GenerationResult(
        records=tuple(records),
        scores=tuple(scores),
        searches=searches,
        simulations=simulations,
        expanded_nodes=expanded_nodes,
        network_evaluations=network_evaluations,
        inference_batches=inference_batches,
        mean_inference_batch_size=(
            inference_size_sum / inference_batches if inference_batches else 0.0
        ),
        max_inference_batch_size=max_inference_batch_size,
        forced_decisions=forced_decisions,
        elapsed_seconds=perf_counter() - started,
    )


def _evaluation_slots(
    seeds: Sequence[int], num_envs: int
) -> tuple[list[tuple[int, DecisionEnv]], int]:
    slots: list[tuple[int, DecisionEnv]] = []
    next_index = 0
    for _ in range(min(num_envs, len(seeds))):
        env = DecisionEnv(0, verify=True)
        env.reset(int(seeds[next_index]))
        slots.append((next_index, env))
        next_index += 1
    return slots, next_index


def evaluate_policy(
    network: PolicyValueNetwork,
    seeds: Sequence[int],
    *,
    device: str,
    num_envs: int,
    simulations: int = 0,
    c_puct: float = 1.5,
) -> EvaluationResult:
    """Evaluate either the raw policy head or deterministic visit-count PUCT."""

    if not seeds:
        raise ValueError("evaluation seed list is empty")
    network.eval()
    resolved_device = resolve_device(device)
    evaluator = PolicyValueEvaluator(network.to(resolved_device), resolved_device)
    search = (
        BatchedPUCT(
            network,
            SearchConfig(
                simulations=simulations,
                c_puct=c_puct,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.0,
            ),
            device=str(resolved_device),
        )
        if simulations
        else None
    )
    slots, next_index = _evaluation_slots(seeds, num_envs)
    scores = [0] * len(seeds)
    started = perf_counter()
    while slots:
        games = [env.game for _, env in slots]
        if search is None:
            nodes, _ = evaluator.evaluate(games)
            actions = [
                node.action_indices[int(np.argmax(node.priors))]
                for node in nodes
            ]
        else:
            results = search.search(
                games,
                seeds=[
                    (
                        int(env.game_seed)
                        ^ _SEARCH_SEED_SALT
                        ^ (env.decisions * 0xD1B5_4A32_D192_ED03)
                    )
                    & ((1 << 128) - 1)
                    for _, env in slots
                ],
                add_root_noise=False,
            )
            actions = [int(np.argmax(result.action_visits)) for result in results]

        next_slots: list[tuple[int, DecisionEnv]] = []
        for (seed_index, env), action in zip(slots, actions):
            result = env.step(action)
            if result.episode is None:
                next_slots.append((seed_index, env))
                continue
            scores[seed_index] = result.episode.score
            if next_index < len(seeds):
                env.reset(int(seeds[next_index]))
                next_slots.append((next_index, env))
                next_index += 1
        slots = next_slots
    return EvaluationResult(tuple(scores), perf_counter() - started)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    config: TrainConfig,
    learner: AlphaZeroLearner,
    replay: AlphaZeroReplay,
    game_seed_stream: random.Random,
    collector_rng: np.random.Generator,
    *,
    positions: int,
    generation: int,
    update_credit: float,
    best_search_mean: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "network_spec": learner.network.config(),
        "network_state_dict": learner.network.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "scaler_state_dict": learner.scaler.state_dict(),
        "positions": positions,
        "generation": generation,
        "updates": learner.updates,
        "update_credit": update_credit,
        "best_search_mean": best_search_mean,
        "elapsed_seconds": elapsed_seconds,
        "replay": replay.state_dict(),
        "game_seed_stream_state": game_seed_stream.getstate(),
        "collector_rng_state": collector_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def load_network(
    checkpoint_path: Path,
    *,
    device: str = "auto",
) -> tuple[PolicyValueNetwork, TrainConfig]:
    resolved = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    spec = NetworkSpec.from_dict(checkpoint["network_spec"])
    network = PolicyValueNetwork(spec).to(resolved)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()
    return network, TrainConfig.from_dict(checkpoint["config"])


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {
        key: float(np.mean([record[key] for record in metrics]))
        for key in metrics[0]
    }


def _write_summary(
    path: Path,
    config: TrainConfig,
    *,
    positions: int,
    generations: int,
    updates: int,
    elapsed_seconds: float,
    best_search_mean: float,
    raw_validation: EvaluationResult | None,
    search_validation: EvaluationResult | None,
) -> None:
    lines = [
        "# AlphaZero base-rules experiment",
        "",
        f"- Positions: `{positions}`",
        f"- Complete self-play generations: `{generations}`",
        f"- Gradient updates: `{updates}`",
        f"- Simulations per decision: `{config.simulations}`",
        f"- Parallel games: `{config.num_envs}`",
        f"- Network: `{config.residual_blocks}x{config.network_width}` residual MLP",
        f"- Replay size limit: `{config.replay_capacity}`",
        f"- Elapsed seconds: `{elapsed_seconds:.1f}`",
        f"- Best searched validation mean: `{best_search_mean:.2f}`",
    ]
    if raw_validation is not None:
        lines.append(
            f"- Final raw policy: `{raw_validation.mean:.2f} +/- "
            f"{raw_validation.standard_error:.2f}`"
        )
    if search_validation is not None:
        lines.append(
            f"- Final PUCT-{config.evaluation_simulations}: "
            f"`{search_validation.mean:.2f} +/- "
            f"{search_validation.standard_error:.2f}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def train(
    config: TrainConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    verify: bool = False,
) -> None:
    """Train at complete-generation boundaries until budget or wall deadline."""

    config.validate()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    replay_path = run_dir / "replay.dat"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    if resume:
        if not config_path.exists() or not latest_path.exists() or not replay_path.exists():
            raise FileNotFoundError("resume requires config, checkpoint, and replay")
    else:
        occupied = [
            path.name
            for path in (config_path, metrics_path, replay_path, latest_path, best_path)
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                "run directory already contains AlphaZero data: " + ", ".join(occupied)
            )
        _write_json(config_path, {"schema_version": 1, **config.to_dict()})

    device = resolve_device(config.device)
    torch.manual_seed(config.run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.run_seed)
    game_seed_stream = random.Random(config.run_seed ^ 0x739A_65F1_04C8_2D7B)
    collector_rng = np.random.default_rng(config.run_seed ^ 0x2B91_74E5_C0D3_A68F)
    network = PolicyValueNetwork(config.network_spec)
    learner = AlphaZeroLearner(network, config, device)
    replay = AlphaZeroReplay(
        config.replay_capacity, replay_path, resume=resume
    )

    positions = generation = 0
    update_credit = 0.0
    best_search_mean = -math.inf
    previous_elapsed = 0.0
    if resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        saved_config = TrainConfig.from_dict(checkpoint["config"])
        if saved_config.network_spec != config.network_spec:
            raise ValueError("resume network configuration differs from checkpoint")
        learner.network.load_state_dict(checkpoint["network_state_dict"])
        learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learner.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        learner.updates = int(checkpoint["updates"])
        positions = int(checkpoint["positions"])
        generation = int(checkpoint["generation"])
        update_credit = float(checkpoint["update_credit"])
        best_search_mean = float(checkpoint["best_search_mean"])
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        replay.load_state_dict(checkpoint["replay"])
        game_seed_stream.setstate(checkpoint["game_seed_stream_state"])
        collector_rng.bit_generator.state = checkpoint["collector_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )

    print(
        f"device={device}; parameters={parameter_count(network):,}; "
        f"envs={config.num_envs}; simulations={config.simulations}; "
        f"replay={replay.allocated_bytes / 2**20:.1f} MiB; "
        f"batch={config.batch_size}; target_positions={config.total_positions}; "
        f"wall_limit={config.max_wall_seconds / 3600:.2f}h",
        flush=True,
    )

    started = perf_counter()
    next_validation = (
        positions // config.validation_interval + 1
    ) * config.validation_interval
    recent_generation_times: deque[float] = deque(maxlen=3)
    last_raw: EvaluationResult | None = None
    last_search: EvaluationResult | None = None

    if not resume and config.validation_seeds:
        last_raw = evaluate_policy(
            learner.network,
            config.validation_seeds,
            device=str(device),
            num_envs=config.evaluation_envs,
        )
        _append_jsonl(
            metrics_path,
            {"event": "validation_raw", "positions": 0, **last_raw.summary_dict()},
        )
        print(
            f"validation raw step=0: {last_raw.mean:.2f} "
            f"+/-{last_raw.standard_error:.2f}",
            flush=True,
        )

    while positions < config.total_positions:
        session_elapsed = perf_counter() - started
        generation_reserve = (
            max(300.0, 1.35 * float(np.mean(recent_generation_times)))
            if recent_generation_times
            else 300.0
        )
        if session_elapsed + generation_reserve >= config.max_wall_seconds:
            print("wall-clock reserve reached before the next generation", flush=True)
            break

        generation_result = collect_generation(
            learner.network,
            config,
            device=str(device),
            game_seed_stream=game_seed_stream,
            collector_rng=collector_rng,
            verify=verify,
        )
        recent_generation_times.append(generation_result.elapsed_seconds)
        replay.add_many(generation_result.records)
        added = len(generation_result.records)
        positions += added
        generation += 1
        update_credit += added * config.replay_ratio / config.batch_size
        update_metrics: list[dict[str, float]] = []
        while update_credit >= 1.0 and replay.size >= config.batch_size:
            update_metrics.append(
                learner.update(replay.sample(config.batch_size, collector_rng))
            )
            update_credit -= 1.0

        elapsed = perf_counter() - started
        metrics = _mean_metrics(update_metrics)
        score_mean = float(np.mean(generation_result.scores))
        record: dict[str, Any] = {
            "event": "generation",
            "generation": generation,
            "positions": positions,
            "new_positions": added,
            "updates": learner.updates,
            "replay_size": replay.size,
            "self_play_mean": score_mean,
            "self_play_se": float(
                np.std(generation_result.scores, ddof=1)
                / math.sqrt(len(generation_result.scores))
            ),
            "searches": generation_result.searches,
            "simulations": generation_result.simulations,
            "expanded_nodes": generation_result.expanded_nodes,
            "network_evaluations": generation_result.network_evaluations,
            "inference_batches": generation_result.inference_batches,
            "mean_inference_batch_size": generation_result.mean_inference_batch_size,
            "max_inference_batch_size": generation_result.max_inference_batch_size,
            "forced_decisions": generation_result.forced_decisions,
            "generation_seconds": generation_result.elapsed_seconds,
            "positions_per_second": added / generation_result.elapsed_seconds,
            "elapsed_seconds": elapsed,
            **metrics,
        }
        _append_jsonl(metrics_path, record)
        print(
            f"generation={generation}; positions={positions}; updates={learner.updates}; "
            f"score={score_mean:.2f}; replay={replay.size}; "
            f"search={generation_result.elapsed_seconds:.1f}s; "
            f"rate={record['positions_per_second']:.2f} positions/s; "
            f"loss={metrics.get('loss', math.nan):.4f}",
            flush=True,
        )

        save_best = False
        if config.validation_seeds and positions >= next_validation:
            last_raw = evaluate_policy(
                learner.network,
                config.validation_seeds,
                device=str(device),
                num_envs=config.evaluation_envs,
            )
            last_search = evaluate_policy(
                learner.network,
                config.validation_seeds,
                device=str(device),
                num_envs=config.evaluation_envs,
                simulations=config.evaluation_simulations,
                c_puct=config.c_puct,
            )
            improved = last_search.mean > best_search_mean
            if improved:
                best_search_mean = last_search.mean
                save_best = True
            for label, result in (("raw", last_raw), ("search", last_search)):
                _append_jsonl(
                    metrics_path,
                    {
                        "event": f"validation_{label}",
                        "positions": positions,
                        "updates": learner.updates,
                        **result.summary_dict(),
                        "best": bool(improved and label == "search"),
                    },
                )
            print(
                f"validation step={positions}: raw={last_raw.mean:.2f} "
                f"+/-{last_raw.standard_error:.2f}; "
                f"PUCT-{config.evaluation_simulations}={last_search.mean:.2f} "
                f"+/-{last_search.standard_error:.2f}",
                flush=True,
            )
            while next_validation <= positions:
                next_validation += config.validation_interval

        replay.flush()
        payload = _checkpoint_payload(
            config,
            learner,
            replay,
            game_seed_stream,
            collector_rng,
            positions=positions,
            generation=generation,
            update_credit=update_credit,
            best_search_mean=best_search_mean,
            elapsed_seconds=previous_elapsed + perf_counter() - started,
        )
        _save_checkpoint(latest_path, payload)
        if save_best:
            _save_checkpoint(best_path, payload)

    elapsed = previous_elapsed + perf_counter() - started
    replay.flush()
    payload = _checkpoint_payload(
        config,
        learner,
        replay,
        game_seed_stream,
        collector_rng,
        positions=positions,
        generation=generation,
        update_credit=update_credit,
        best_search_mean=best_search_mean,
        elapsed_seconds=elapsed,
    )
    _save_checkpoint(latest_path, payload)
    _write_summary(
        run_dir / "summary.md",
        config,
        positions=positions,
        generations=generation,
        updates=learner.updates,
        elapsed_seconds=elapsed,
        best_search_mean=best_search_mean,
        raw_validation=last_raw,
        search_validation=last_search,
    )
    replay.close()
    print(
        f"training stopped: positions={positions}; generations={generation}; "
        f"updates={learner.updates}; elapsed={elapsed / 3600:.2f}h",
        flush=True,
    )
