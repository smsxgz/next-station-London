"""Training, evaluation, checkpointing, and diagnostics for masked DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import random
import secrets
from statistics import median
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

from .codec import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    encode_decision,
    observation_feature_counts,
)
from .dqn import (
    ActionValueNetwork,
    NetworkSpec,
    QNetwork,
    masked_argmax,
    resolve_device,
    select_action_indices,
)
from .environment import DecisionEnv, EpisodeResult, VectorDecisionEnv
from .replay import (
    NStepAccumulator,
    PackedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBatch,
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int
    num_envs: int = 128
    total_transitions: int = 10_000_000
    n_steps: int = 1
    target_depth: int = 0
    init_checkpoint: str | None = None
    gamma: float = 1.0
    replay_capacity: int = 1_000_000
    replay_kind: str = "uniform"
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-3
    batch_size: int = 512
    replay_ratio: float = 8.0
    warmup_transitions: int = 25_000
    learning_rate: float = 3e-4
    target_update_interval: int = 1_000
    epsilon_initial: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_transitions: int = 1_000_000
    reward_scale: float = 10.0
    gradient_clip: float = 10.0
    hidden_sizes: tuple[int, ...] = (512, 512)
    validation_seeds: tuple[int, ...] = ()
    validation_interval: int = 250_000
    evaluation_envs: int = 128
    log_interval: int = 100_000
    checkpoint_interval: int = 250_000
    device: str = "auto"

    def validate(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < (1 << 63)
        ):
            raise ValueError("seed must be an integer in [0, 2^63)")
        positive_ints = {
            "num_envs": self.num_envs,
            "total_transitions": self.total_transitions,
            "n_steps": self.n_steps,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "warmup_transitions": self.warmup_transitions,
            "target_update_interval": self.target_update_interval,
            "epsilon_decay_transitions": self.epsilon_decay_transitions,
            "validation_interval": self.validation_interval,
            "evaluation_envs": self.evaluation_envs,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.epsilon_initial <= 1.0:
            raise ValueError("epsilon_initial must be in [0, 1]")
        if not 0.0 <= self.epsilon_final <= 1.0:
            raise ValueError("epsilon_final must be in [0, 1]")
        if self.replay_ratio <= 0.0:
            raise ValueError("replay_ratio must be positive")
        if self.target_depth not in {0, 1, 2}:
            raise ValueError("target_depth must be 0, 1, or 2")
        if self.target_depth and self.n_steps != 1:
            raise ValueError("exact-chance targets require one-step DQN")
        if self.replay_kind not in {"uniform", "prioritized"}:
            raise ValueError("replay_kind must be uniform or prioritized")
        if not 0.0 <= self.priority_alpha <= 1.0:
            raise ValueError("priority_alpha must be in [0, 1]")
        if not 0.0 <= self.priority_beta_start <= 1.0:
            raise ValueError("priority_beta_start must be in [0, 1]")
        if not self.priority_beta_start <= self.priority_beta_end <= 1.0:
            raise ValueError(
                "priority_beta_end must be in [priority_beta_start, 1]"
            )
        if self.priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        if self.learning_rate <= 0.0 or self.reward_scale <= 0.0:
            raise ValueError("learning_rate and reward_scale must be positive")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must hold at least one batch")
        if self.warmup_transitions < self.batch_size:
            raise ValueError("warm-up must contain at least one batch")
        if self.warmup_transitions > self.replay_capacity:
            raise ValueError("warm-up cannot exceed replay capacity")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        result["validation_seeds"] = list(self.validation_seeds)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        values = dict(raw)
        values.pop("schema_version", None)
        algorithm = values.pop("algorithm", "dqn")
        if algorithm != "dqn":
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
        values.pop("atom_count", None)
        values.pop("value_min", None)
        values.pop("value_max", None)
        values["hidden_sizes"] = tuple(int(size) for size in values["hidden_sizes"])
        values["validation_seeds"] = tuple(
            int(seed) for seed in values.get("validation_seeds", ())
        )
        config = cls(**values)
        config.validate()
        return config


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
class LearnerUpdate:
    metrics: dict[str, float]
    td_errors: NDArray[np.float32]


class DQNLearner:
    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        spec = NetworkSpec(hidden_sizes=config.hidden_sizes)
        self.online = QNetwork(spec).to(device)
        self.target = QNetwork(spec).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.updates = 0

    def update(self, batch: ReplayBatch) -> LearnerUpdate:
        observations = torch.as_tensor(
            batch.observations, device=self.device, dtype=torch.float32
        )
        actions = torch.as_tensor(
            batch.actions, device=self.device, dtype=torch.int64
        )
        rewards = torch.as_tensor(
            batch.rewards,
            device=self.device,
            dtype=torch.float32,
        ) / self.config.reward_scale
        next_observations = torch.as_tensor(
            batch.next_observations,
            device=self.device,
            dtype=torch.float32,
        )
        next_masks = torch.as_tensor(
            batch.next_action_masks,
            device=self.device,
            dtype=torch.bool,
        )
        terminated = torch.as_tensor(
            batch.terminated,
            device=self.device,
            dtype=torch.float32,
        )
        steps = torch.as_tensor(
            batch.steps,
            device=self.device,
            dtype=torch.float32,
        )
        weights = torch.as_tensor(
            batch.weights,
            device=self.device,
            dtype=torch.float32,
        )

        predicted = self.online(observations).gather(
            1, actions[:, None]
        ).squeeze(1)
        with torch.no_grad():
            chance_leaves = 0
            if self.config.target_depth == 1:
                from .exact_chance import exact_chance_double_dqn_values

                next_values, chance_leaves = exact_chance_double_dqn_values(
                    self.online,
                    self.target,
                    batch.next_observations,
                    batch.terminated,
                    device=self.device,
                )
            elif self.config.target_depth == 2:
                from .exact_chance import (
                    exact_chance_depth2_double_dqn_values,
                )

                next_values, chance_leaves = (
                    exact_chance_depth2_double_dqn_values(
                        self.online,
                        self.target,
                        batch.next_observations,
                        batch.terminated,
                        device=self.device,
                        reward_scale=self.config.reward_scale,
                        gamma=self.config.gamma,
                    )
                )
            else:
                next_online = self.online(next_observations)
                next_actions = masked_argmax(next_online, next_masks)
                next_values = self.target(next_observations).gather(
                    1, next_actions[:, None]
                ).squeeze(1)
            discounts = torch.pow(
                torch.full_like(steps, self.config.gamma), steps
            )
            targets = rewards + (1.0 - terminated) * discounts * next_values

        td_errors = (targets - predicted).abs()
        element_losses = F.smooth_l1_loss(
            predicted, targets, reduction="none"
        )
        loss = (element_losses * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        scale = self.config.reward_scale
        return LearnerUpdate(
            metrics={
                "loss": float(loss.detach().item()),
                "q_mean": float(predicted.detach().mean().item() * scale),
                "target_mean": float(targets.detach().mean().item() * scale),
                "td_abs_mean": float(
                    td_errors.detach().mean().item() * scale
                ),
                "gradient_norm": float(gradient_norm.detach().item()),
                "importance_weight_mean": float(weights.mean().item()),
                "chance_leaves_per_sample": chance_leaves / len(actions),
            },
            td_errors=np.ascontiguousarray(
                td_errors.detach().cpu().numpy(), dtype=np.float32
            ),
        )


def generate_validation_seeds(seed: int, count: int) -> tuple[int, ...]:
    if count < 1:
        return ()
    rng = random.Random(seed ^ 0x5A17_D3C4_91E2_6B0F)
    result: list[int] = []
    seen: set[int] = set()
    while len(result) < count:
        # Training game streams use 63-bit non-negative seeds.  Setting the
        # next bit makes the validation set provably disjoint.
        candidate = rng.getrandbits(63) | (1 << 63)
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return tuple(result)


def epsilon_at(config: TrainConfig, transitions: int) -> float:
    progress = min(1.0, transitions / config.epsilon_decay_transitions)
    return config.epsilon_initial + progress * (
        config.epsilon_final - config.epsilon_initial
    )


def priority_beta_at(config: TrainConfig, transitions: int) -> float:
    progress = min(1.0, transitions / config.total_transitions)
    return config.priority_beta_start + progress * (
        config.priority_beta_end - config.priority_beta_start
    )


def _score_result(scores: Sequence[int], elapsed: float) -> EvaluationResult:
    if not scores:
        raise ValueError("evaluation requires at least one score")
    array = np.asarray(scores, dtype=np.float64)
    standard_error = (
        float(array.std(ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return EvaluationResult(
        scores=tuple(int(score) for score in scores),
        mean=float(array.mean()),
        standard_error=standard_error,
        minimum=int(array.min()),
        median=float(median(int(score) for score in scores)),
        maximum=int(array.max()),
        elapsed_seconds=elapsed,
    )


def evaluate_network(
    network: ActionValueNetwork,
    seeds: Sequence[int],
    *,
    device: torch.device,
    num_envs: int = 64,
    verify: bool = True,
) -> EvaluationResult:
    """Evaluate a deterministic greedy DQN policy in batched game slots."""

    if not seeds:
        raise ValueError("evaluation seed list is empty")
    if num_envs < 1:
        raise ValueError("evaluation num_envs must be positive")

    was_training = network.training
    network.eval()
    slots: list[tuple[int, DecisionEnv]] = []
    next_index = 0
    for _ in range(min(num_envs, len(seeds))):
        env = DecisionEnv(stream_seed=0, verify=verify)
        env.reset(int(seeds[next_index]))
        slots.append((next_index, env))
        next_index += 1

    scores = [0] * len(seeds)
    rng = np.random.default_rng(0)
    started = perf_counter()
    while slots:
        observations = np.stack([env.observation for _, env in slots], axis=0)
        masks = np.stack([env.action_mask for _, env in slots], axis=0)
        actions = select_action_indices(
            network,
            observations,
            masks,
            device=device,
            epsilon=0.0,
            rng=rng,
        )
        next_slots: list[tuple[int, DecisionEnv]] = []
        for (seed_index, env), action in zip(slots, actions):
            result = env.step(int(action))
            if result.episode is None:
                next_slots.append((seed_index, env))
                continue
            scores[seed_index] = result.episode.score
            if next_index < len(seeds):
                env.reset(int(seeds[next_index]))
                next_slots.append((next_index, env))
                next_index += 1
        slots = next_slots
    elapsed = perf_counter() - started
    if was_training:
        network.train()
    return _score_result(scores, elapsed)


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


def load_config(run_dir: Path) -> TrainConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return TrainConfig.from_dict(raw)


def _checkpoint_payload(
    config: TrainConfig,
    learner: DQNLearner,
    replay: PackedReplayBuffer,
    vector_env: VectorDecisionEnv,
    rng: np.random.Generator,
    *,
    transitions: int,
    update_credit: float,
    best_validation_mean: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "algorithm": "dqn",
        "config": config.to_dict(),
        "network_spec": learner.online.config(),
        "online_state_dict": learner.online.state_dict(),
        "target_state_dict": learner.target.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "transitions": transitions,
        "updates": learner.updates,
        "update_credit": update_credit,
        "best_validation_mean": best_validation_mean,
        "elapsed_seconds": elapsed_seconds,
        "replay": replay.state_dict(),
        "collector_rng_state": rng.bit_generator.state,
        "env_seed_stream_states": vector_env.seed_stream_states(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_summary(
    path: Path,
    config: TrainConfig,
    *,
    transitions: int,
    updates: int,
    elapsed_seconds: float,
    best_validation_mean: float,
    final_validation: EvaluationResult | None,
) -> None:
    best_text = (
        f"{best_validation_mean:.2f}"
        if math.isfinite(best_validation_mean)
        else "not evaluated"
    )
    lines = [
        "# Deep RL training summary",
        "",
        f"- Transitions: `{transitions}`",
        f"- Gradient updates: `{updates}`",
        f"- Environments: `{config.num_envs}`",
        f"- Batch size: `{config.batch_size}`",
        f"- N-step return: `{config.n_steps}`",
        f"- Exact-chance target depth: `{config.target_depth}`",
        f"- Replay: `{config.replay_kind}` (`{config.replay_capacity}` entries)",
        f"- Elapsed seconds: `{elapsed_seconds:.1f}`",
        f"- Best validation mean: `{best_text}`",
    ]
    if config.init_checkpoint is not None:
        lines.append(f"- Initialized from: `{config.init_checkpoint}`")
    if final_validation is not None:
        lines.extend(
            [
                f"- Final validation mean: `{final_validation.mean:.2f} +/- "
                f"{final_validation.standard_error:.2f}`",
                f"- Final validation range: `{final_validation.minimum}.."
                f"{final_validation.maximum}`",
            ]
        )
    if config.replay_kind == "prioritized":
        lines.extend(
            [
                f"- Priority alpha: `{config.priority_alpha}`",
                f"- Priority beta: `{config.priority_beta_start}` -> "
                f"`{config.priority_beta_end}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def train(
    config: TrainConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    verify_env: bool = False,
) -> EvaluationResult | None:
    """Run synchronous batched collection and masked Double DQN updates."""

    config.validate()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    replay_path = run_dir / "replay.dat"
    priority_path = run_dir / "priorities.dat"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"

    if resume:
        if not config_path.exists() or not latest_path.exists():
            raise FileNotFoundError("resume requires config.json and latest.pt")
    else:
        occupied = [
            path.name
            for path in (
                config_path,
                metrics_path,
                replay_path,
                priority_path,
                latest_path,
                best_path,
            )
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                f"run directory already contains training data: {', '.join(occupied)}"
            )
        stored_config = {"schema_version": 1, **config.to_dict()}
        _write_json(config_path, stored_config)

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed ^ 0x2B91_74E5_C0D3_A68F)
    vector_env = VectorDecisionEnv(
        config.num_envs,
        config.seed ^ 0x739A_65F1_04C8_2D7B,
        verify=verify_env,
    )
    accumulator = NStepAccumulator(config.num_envs, config.n_steps, config.gamma)
    if config.replay_kind == "prioritized":
        replay: PackedReplayBuffer = PrioritizedReplayBuffer(
            config.replay_capacity,
            alpha=config.priority_alpha,
            priority_epsilon=config.priority_epsilon,
            path=replay_path,
            priority_path=priority_path,
            resume=resume,
        )
    else:
        replay = PackedReplayBuffer(
            config.replay_capacity,
            path=replay_path,
            resume=resume,
        )
    learner = DQNLearner(config, device)

    if not resume and config.init_checkpoint is not None:
        if not isinstance(learner, DQNLearner):
            raise ValueError("checkpoint initialization currently requires DQN")
        init_path = Path(config.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
        initial = torch.load(init_path, map_location=device, weights_only=False)
        learner.online.load_state_dict(initial["online_state_dict"])
        learner.target.load_state_dict(learner.online.state_dict())
        print(f"initialized_from={init_path}", flush=True)

    transitions = 0
    update_credit = 0.0
    best_validation_mean = -math.inf
    previous_elapsed_seconds = 0.0
    if resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        checkpoint_config = TrainConfig.from_dict(checkpoint["config"])
        structural_current = replace(
            config,
            total_transitions=checkpoint_config.total_transitions,
            device=checkpoint_config.device,
        )
        if structural_current != checkpoint_config:
            raise ValueError(
                "resume configuration differs from the saved run except for "
                "total_transitions"
            )
        learner.online.load_state_dict(checkpoint["online_state_dict"])
        learner.target.load_state_dict(checkpoint["target_state_dict"])
        learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learner.updates = int(checkpoint["updates"])
        transitions = int(checkpoint["transitions"])
        update_credit = float(checkpoint["update_credit"])
        best_validation_mean = float(checkpoint["best_validation_mean"])
        previous_elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        replay.load_state_dict(checkpoint["replay"])
        rng.bit_generator.state = checkpoint["collector_rng_state"]
        vector_env.restore_seed_stream_states(checkpoint["env_seed_stream_states"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )
        if config.total_transitions < transitions:
            raise ValueError("total_transitions cannot be below the checkpoint step")
        _write_json(config_path, {"schema_version": 1, **config.to_dict()})

    print(
        f"device={device}; envs={config.num_envs}; observation={OBSERVATION_DIM}; "
        f"actions={ACTION_COUNT}; n_steps={config.n_steps}; "
        f"target_depth={config.target_depth}; "
        f"replay_kind={config.replay_kind}; "
        f"replay={replay.allocated_bytes / 2**20:.1f} MiB; "
        f"start={transitions}; target={config.total_transitions}",
        flush=True,
    )

    recent_scores: deque[int] = deque(maxlen=200)
    recent_updates: list[dict[str, float]] = []
    started = perf_counter()
    interval_started = started
    interval_transitions = transitions
    next_log = (transitions // config.log_interval + 1) * config.log_interval
    next_validation = (
        transitions // config.validation_interval + 1
    ) * config.validation_interval
    next_checkpoint = (
        transitions // config.checkpoint_interval + 1
    ) * config.checkpoint_interval
    last_validation: EvaluationResult | None = None
    last_validation_step = -1

    if not resume and config.validation_seeds:
        last_validation = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
        )
        last_validation_step = 0
        best_validation_mean = last_validation.mean
        _append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "transitions": 0,
                "updates": 0,
                **last_validation.summary_dict(),
                "best": True,
            },
        )
        print(
            f"validation step=0: mean={last_validation.mean:.2f} "
            f"+/-{last_validation.standard_error:.2f}; "
            f"range={last_validation.minimum}..{last_validation.maximum}; "
            f"elapsed={last_validation.elapsed_seconds:.2f}s",
            flush=True,
        )
        replay.flush()
        _save_checkpoint(
            best_path,
            _checkpoint_payload(
                config,
                learner,
                replay,
                vector_env,
                rng,
                transitions=0,
                update_credit=0.0,
                best_validation_mean=best_validation_mean,
                elapsed_seconds=perf_counter() - started,
            ),
        )

    while transitions < config.total_transitions:
        current_observations = vector_env.observations
        current_masks = vector_env.action_masks
        epsilon = epsilon_at(config, transitions)
        actions = select_action_indices(
            learner.online,
            current_observations,
            current_masks,
            device=device,
            epsilon=epsilon,
            rng=rng,
        )
        step = vector_env.step(actions)
        emitted_transitions = []
        for env_index in range(config.num_envs):
            emitted_transitions.extend(
                accumulator.add(
                    env_index,
                    current_observations[env_index],
                    int(actions[env_index]),
                    float(step.rewards[env_index]),
                    step.next_observations[env_index],
                    step.next_action_masks[env_index],
                    bool(step.terminated[env_index]),
                )
            )
        replay.add_many(emitted_transitions)
        transitions += config.num_envs
        recent_scores.extend(episode.score for episode in step.completed_episodes)

        if replay.size >= config.warmup_transitions:
            update_credit += (
                config.num_envs * config.replay_ratio / config.batch_size
            )
            while update_credit >= 1.0:
                beta = priority_beta_at(config, transitions)
                batch = replay.sample(config.batch_size, rng, beta=beta)
                learner_update = learner.update(batch)
                replay.update_priorities(
                    batch.indices, learner_update.td_errors
                )
                recent_updates.append(learner_update.metrics)
                update_credit -= 1.0

        if transitions >= next_log:
            now = perf_counter()
            rate = (transitions - interval_transitions) / (now - interval_started)
            update_mean = (
                {
                    key: float(
                        np.mean([item[key] for item in recent_updates])
                    )
                    for key in recent_updates[0]
                }
                if recent_updates
                else {}
            )
            score_mean = (
                float(np.mean(recent_scores)) if recent_scores else math.nan
            )
            record: dict[str, Any] = {
                "event": "train",
                "transitions": transitions,
                "updates": learner.updates,
                "replay_size": replay.size,
                "epsilon": epsilon_at(config, transitions),
                "recent_episode_mean": score_mean,
                "recent_episode_count": len(recent_scores),
                "transitions_per_second": rate,
                **update_mean,
            }
            if config.replay_kind == "prioritized":
                record["priority_beta"] = priority_beta_at(
                    config, transitions
                )
            _append_jsonl(metrics_path, record)
            loss_text = (
                f"{update_mean['loss']:.4f}" if update_mean else "warming-up"
            )
            print(
                f"step={transitions}; updates={learner.updates}; "
                f"replay={replay.size}; epsilon={record['epsilon']:.3f}; "
                f"train_score200={score_mean:.2f}; loss={loss_text}; "
                f"rate={rate:.0f} transitions/s",
                flush=True,
            )
            recent_updates.clear()
            interval_started = now
            interval_transitions = transitions
            while next_log <= transitions:
                next_log += config.log_interval

        if config.validation_seeds and transitions >= next_validation:
            last_validation = evaluate_network(
                learner.online,
                config.validation_seeds,
                device=device,
                num_envs=config.evaluation_envs,
            )
            last_validation_step = transitions
            improved = last_validation.mean > best_validation_mean
            if improved:
                best_validation_mean = last_validation.mean
            _append_jsonl(
                metrics_path,
                {
                    "event": "validation",
                    "transitions": transitions,
                    "updates": learner.updates,
                    **last_validation.summary_dict(),
                    "best": improved,
                },
            )
            print(
                f"validation step={transitions}: mean={last_validation.mean:.2f} "
                f"+/-{last_validation.standard_error:.2f}; "
                f"range={last_validation.minimum}..{last_validation.maximum}; "
                f"elapsed={last_validation.elapsed_seconds:.2f}s",
                flush=True,
            )
            if improved:
                replay.flush()
                payload = _checkpoint_payload(
                    config,
                    learner,
                    replay,
                    vector_env,
                    rng,
                    transitions=transitions,
                    update_credit=update_credit,
                    best_validation_mean=best_validation_mean,
                    elapsed_seconds=(
                        previous_elapsed_seconds + perf_counter() - started
                    ),
                )
                _save_checkpoint(best_path, payload)
            while next_validation <= transitions:
                next_validation += config.validation_interval

        if transitions >= next_checkpoint:
            replay.flush()
            payload = _checkpoint_payload(
                config,
                learner,
                replay,
                vector_env,
                rng,
                transitions=transitions,
                update_credit=update_credit,
                best_validation_mean=best_validation_mean,
                elapsed_seconds=(
                    previous_elapsed_seconds + perf_counter() - started
                ),
            )
            _save_checkpoint(latest_path, payload)
            while next_checkpoint <= transitions:
                next_checkpoint += config.checkpoint_interval

    if config.validation_seeds and last_validation_step != transitions:
        last_validation = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
        )
        improved = last_validation.mean > best_validation_mean
        if improved:
            best_validation_mean = last_validation.mean
        _append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "transitions": transitions,
                "updates": learner.updates,
                **last_validation.summary_dict(),
                "best": improved,
            },
        )
        print(
            f"validation step={transitions}: mean={last_validation.mean:.2f} "
            f"+/-{last_validation.standard_error:.2f}; "
            f"range={last_validation.minimum}..{last_validation.maximum}; "
            f"elapsed={last_validation.elapsed_seconds:.2f}s",
            flush=True,
        )

    replay.flush()
    final_payload = _checkpoint_payload(
        config,
        learner,
        replay,
        vector_env,
        rng,
        transitions=transitions,
        update_credit=update_credit,
        best_validation_mean=best_validation_mean,
        elapsed_seconds=previous_elapsed_seconds + perf_counter() - started,
    )
    _save_checkpoint(latest_path, final_payload)
    if last_validation is not None and last_validation.mean >= best_validation_mean:
        _save_checkpoint(best_path, final_payload)
    session_elapsed = perf_counter() - started
    elapsed = previous_elapsed_seconds + session_elapsed
    _write_summary(
        run_dir / "summary.md",
        config,
        transitions=transitions,
        updates=learner.updates,
        elapsed_seconds=elapsed,
        best_validation_mean=best_validation_mean,
        final_validation=last_validation,
    )
    print(
        f"training complete: transitions={transitions}; updates={learner.updates}; "
        f"session_elapsed={session_elapsed:.1f}s; total_elapsed={elapsed:.1f}s",
        flush=True,
    )
    return last_validation


def benchmark_vector_envs(
    env_counts: Iterable[int],
    *,
    transitions_per_count: int,
    device_name: str = "auto",
    seed: int | None = None,
) -> tuple[dict[str, float | int], ...]:
    """Measure end-to-end batched inference plus engine stepping throughput."""

    if transitions_per_count < 1:
        raise ValueError("transitions_per_count must be positive")
    actual_seed = secrets.randbits(63) if seed is None else int(seed)
    device = resolve_device(device_name)
    torch.manual_seed(actual_seed)
    network = QNetwork().to(device)
    network.eval()
    results: list[dict[str, float | int]] = []
    for num_envs in env_counts:
        vector_env = VectorDecisionEnv(num_envs, actual_seed ^ num_envs)
        rng = np.random.default_rng(actual_seed ^ (num_envs << 17))
        warmup_batches = max(2, math.ceil(256 / num_envs))
        for _ in range(warmup_batches):
            actions = select_action_indices(
                network,
                vector_env.observations,
                vector_env.action_masks,
                device=device,
                epsilon=0.05,
                rng=rng,
            )
            vector_env.step(actions)
        completed_scores: list[int] = []
        completed_transitions = 0
        started = perf_counter()
        while completed_transitions < transitions_per_count:
            actions = select_action_indices(
                network,
                vector_env.observations,
                vector_env.action_masks,
                device=device,
                epsilon=0.05,
                rng=rng,
            )
            step = vector_env.step(actions)
            completed_transitions += num_envs
            completed_scores.extend(
                episode.score for episode in step.completed_episodes
            )
        elapsed = perf_counter() - started
        result: dict[str, float | int] = {
            "num_envs": num_envs,
            "transitions": completed_transitions,
            "elapsed_seconds": elapsed,
            "transitions_per_second": completed_transitions / elapsed,
            "completed_games": len(completed_scores),
            "mean_score": (
                float(np.mean(completed_scores)) if completed_scores else math.nan
            ),
        }
        results.append(result)
    return tuple(results)


def run_self_check(
    *,
    games: int = 16,
    device_name: str = "auto",
) -> dict[str, float | int | str]:
    """Exercise encoding, rewards, n-step replay, masking, and one update."""

    if games < 1:
        raise ValueError("games must be positive")
    seed = 0x4D31_8A72_C6F0_195B
    num_envs = 4
    rng = np.random.default_rng(seed)
    vector_env = VectorDecisionEnv(num_envs, seed, verify=True)
    original = encode_decision(vector_env.envs[0].game)
    public_copy = encode_decision(vector_env.envs[0].game.copy_public_state())
    if not np.array_equal(original.observation, public_copy.observation):
        raise RuntimeError("observation unexpectedly depends on hidden deck order")
    if not np.array_equal(original.action_mask, public_copy.action_mask):
        raise RuntimeError("legal mask unexpectedly depends on hidden deck order")
    accumulator = NStepAccumulator(num_envs, 1, 1.0)
    replay = PackedReplayBuffer(4_096)
    prioritized_replay = PrioritizedReplayBuffer(4_096)
    completed: list[EpisodeResult] = []
    while len(completed) < games:
        observations = vector_env.observations
        masks = vector_env.action_masks
        actions = np.asarray(
            [rng.choice(np.flatnonzero(mask)) for mask in masks],
            dtype=np.int64,
        )
        step = vector_env.step(actions)
        emitted_transitions = []
        for env_index in range(num_envs):
            emitted_transitions.extend(
                accumulator.add(
                    env_index,
                    observations[env_index],
                    int(actions[env_index]),
                    float(step.rewards[env_index]),
                    step.next_observations[env_index],
                    step.next_action_masks[env_index],
                    bool(step.terminated[env_index]),
                )
            )
        replay.add_many(emitted_transitions)
        prioritized_replay.add_many(emitted_transitions)
        completed.extend(step.completed_episodes)

    batch_size = min(128, replay.size)
    batch = replay.sample(batch_size, rng)
    config = TrainConfig(
        seed=seed,
        num_envs=num_envs,
        total_transitions=1_000,
        replay_capacity=4_096,
        batch_size=batch_size,
        warmup_transitions=batch_size,
        validation_seeds=(),
        device=device_name,
    )
    config.validate()
    device = resolve_device(device_name)
    torch.manual_seed(seed)
    learner = DQNLearner(config, device)
    initial_actions = select_action_indices(
        learner.online,
        vector_env.observations,
        vector_env.action_masks,
        device=device,
        epsilon=0.0,
        rng=rng,
    )
    if any(
        not vector_env.action_masks[index, action]
        for index, action in enumerate(initial_actions)
    ):
        raise RuntimeError("initial value network selected an illegal action")

    update = learner.update(batch)
    prioritized_batch = prioritized_replay.sample(batch_size, rng, beta=0.4)
    prioritized_update = learner.update(prioritized_batch)
    prioritized_replay.update_priorities(
        prioritized_batch.indices, prioritized_update.td_errors
    )
    resampled = prioritized_replay.sample(batch_size, rng, beta=1.0)
    if not np.isfinite(resampled.weights).all():
        raise RuntimeError("prioritized replay produced non-finite weights")
    selected = select_action_indices(
        learner.online,
        vector_env.observations,
        vector_env.action_masks,
        device=device,
        epsilon=0.0,
        rng=rng,
    )
    if any(
        not vector_env.action_masks[index, action]
        for index, action in enumerate(selected)
    ):
        raise RuntimeError("masked network inference selected an illegal action")
    return {
        "games": len(completed),
        "transitions_in_replay": replay.size,
        "observation_dim": OBSERVATION_DIM,
        "action_count": ACTION_COUNT,
        "replay_record_bytes": replay.dtype.itemsize,
        "mean_score": float(np.mean([episode.score for episode in completed])),
        "loss": update.metrics["loss"],
        "prioritized_loss": prioritized_update.metrics["loss"],
        "feature_groups": len(observation_feature_counts()),
    }
