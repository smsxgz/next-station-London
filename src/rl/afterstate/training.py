"""Scalar afterstate training and checkpointing."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .codec import (
    AFTERSTATE_SCHEMA_VERSION,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_HASH,
    OBSERVATION_DIM,
)
from .environment import VectorAfterstateEnv
from .evaluation import EvaluationResult, evaluate_network
from .native_backend import feature_records
from .network import (
    AfterstateNetworkSpec,
    AfterstateValueNetwork,
    network_from_checkpoint,
    resolve_device,
)
from .policy import AfterstatePolicy
from .replay import (
    AfterstateReplayBuffer,
    PrioritizedAfterstateReplayBuffer,
    ReplayBatch,
)
from .target import CudaFeatureWorkspace, exact_afterstate_targets


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int
    num_envs: int = 128
    total_transitions: int = 10_000_000
    gamma: float = 1.0
    reward_scale: float = 10.0
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
    gradient_clip: float = 10.0
    hidden_sizes: tuple[int, ...] = (512, 512)
    validation_seeds: tuple[int, ...] = ()
    validation_interval: int = 250_000
    evaluation_envs: int = 64
    inference_batch_size: int = 8192
    log_interval: int = 100_000
    checkpoint_interval: int = 250_000
    device: str = "auto"
    baseline_checkpoint: str = "artifacts/dqn/exact_chance_depth1_lr1e4_4m/best.pt"
    warm_start_checkpoint: str | None = None

    def validate(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if self.replay_kind not in {"uniform", "prioritized"}:
            raise ValueError("replay_kind must be uniform or prioritized")
        for name in (
            "num_envs",
            "total_transitions",
            "replay_capacity",
            "batch_size",
            "warmup_transitions",
            "target_update_interval",
            "epsilon_decay_transitions",
            "validation_interval",
            "evaluation_envs",
            "inference_batch_size",
            "log_interval",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.gamma <= 1.0 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite and in (0, 1]")
        if self.reward_scale <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("reward_scale and learning_rate must be positive")
        if self.replay_ratio <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("replay_ratio and gradient_clip must be positive")
        if not 0.0 <= self.priority_alpha <= 1.0:
            raise ValueError("priority_alpha must be in [0, 1]")
        if not 0.0 <= self.priority_beta_start <= 1.0:
            raise ValueError("priority_beta_start must be in [0, 1]")
        if not self.priority_beta_start <= self.priority_beta_end <= 1.0:
            raise ValueError("priority_beta_end must be in [priority_beta_start, 1]")
        if self.priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        if (
            not 0.0 <= self.epsilon_final <= 1.0
            or not 0.0 <= self.epsilon_initial <= 1.0
        ):
            raise ValueError("epsilon values must be in [0, 1]")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must hold one batch")
        if self.warmup_transitions < self.batch_size:
            raise ValueError("warmup must hold one batch")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique")
        if self.warm_start_checkpoint is not None and not self.warm_start_checkpoint:
            raise ValueError("warm_start_checkpoint cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["hidden_sizes"] = list(self.hidden_sizes)
        raw["validation_seeds"] = list(self.validation_seeds)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        values = dict(raw)
        values.pop("schema_version", None)
        algorithm = values.pop("algorithm", "scalar")
        if algorithm != "scalar":
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
        values.pop("atom_count", None)
        values.pop("value_min", None)
        values.pop("value_max", None)
        values["hidden_sizes"] = tuple(int(value) for value in values["hidden_sizes"])
        values["validation_seeds"] = tuple(
            int(value) for value in values.get("validation_seeds", ())
        )
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class LearnerUpdate:
    metrics: dict[str, float]
    td_errors: np.ndarray


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


class AfterstateLearner:
    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        spec = AfterstateNetworkSpec(hidden_sizes=config.hidden_sizes)
        self.online = AfterstateValueNetwork(spec).to(device)
        self.target = AfterstateValueNetwork(spec).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.feature_workspace = (
            CudaFeatureWorkspace(device) if device.type == "cuda" else None
        )
        self.updates = 0

    def update(self, batch: ReplayBatch) -> LearnerUpdate:
        observations = torch.as_tensor(
            feature_records(batch.records),
            device=self.device,
            dtype=torch.float32,
        )
        predicted_values = self.online(observations)
        targets, stats = exact_afterstate_targets(
            self.online,
            self.target,
            batch.records,
            batch.terminated,
            device=self.device,
            reward_scale=self.config.reward_scale,
            gamma=self.config.gamma,
            inference_batch_size=self.config.inference_batch_size,
            feature_workspace=self.feature_workspace,
        )
        target_values = targets
        element_losses = F.smooth_l1_loss(predicted_values, targets, reduction="none")
        weights = torch.as_tensor(
            batch.weights, device=self.device, dtype=torch.float32
        )
        loss = (element_losses * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        td_errors = (
            (target_values - predicted_values)
            .abs()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        scale = self.config.reward_scale
        metrics = {
            "loss": float(loss.detach().item()),
            "w_mean": float(predicted_values.detach().mean().item() * scale),
            "target_mean": float(target_values.detach().mean().item() * scale),
            "td_abs_mean": float(td_errors.mean() * scale),
            "gradient_norm": float(gradient_norm.detach().item()),
            "importance_weight_mean": float(weights.detach().mean().item()),
            "chance_outcomes_per_sample": stats.chance_outcomes / len(batch.records),
            "candidate_actions_per_sample": stats.candidate_actions
            / len(batch.records),
        }
        return LearnerUpdate(
            metrics=metrics,
            td_errors=td_errors,
        )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _checkpoint_payload(
    config: TrainConfig,
    learner: AfterstateLearner,
    replay: AfterstateReplayBuffer | PrioritizedAfterstateReplayBuffer,
    env: VectorAfterstateEnv,
    rng: np.random.Generator,
    *,
    transitions: int,
    update_credit: float,
    best_validation_mean: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_type": "afterstate_value",
        "algorithm": "scalar",
        "afterstate_schema_version": AFTERSTATE_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "observation_dim": OBSERVATION_DIM,
        "replay_schema_version": replay.state_dict()["replay_schema_version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
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
        "env_states": env.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _select_candidates(
    groups: tuple[tuple[Any, ...], ...],
    learner: AfterstateLearner,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    policy = AfterstatePolicy(
        learner.online, device=learner.device, reward_scale=learner.config.reward_scale
    )
    greedy = policy.select_groups(groups)
    selected = []
    for group, decision in zip(groups, greedy):
        if rng.random() < epsilon:
            selected.append(group[int(rng.integers(0, len(group)))])
        else:
            selected.append(decision.candidate)
    return tuple(selected)


def _apply_warm_start(
    config: TrainConfig,
    learner: AfterstateLearner,
    env: VectorAfterstateEnv,
    rng: np.random.Generator,
    *,
    device: torch.device,
) -> None:
    """Restore training state while deliberately starting with an empty replay."""

    if config.warm_start_checkpoint is None:
        return
    path = Path(config.warm_start_checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source = network_from_checkpoint(checkpoint, device)
    if source.spec != learner.online.spec:
        raise ValueError("warm-start checkpoint network shape is incompatible")
    source_config = TrainConfig.from_dict(checkpoint["config"])
    required_matches = (
        "num_envs",
        "gamma",
        "reward_scale",
        "learning_rate",
        "target_update_interval",
        "gradient_clip",
        "hidden_sizes",
    )
    mismatches = [
        name
        for name in required_matches
        if getattr(config, name) != getattr(source_config, name)
    ]
    if mismatches:
        raise ValueError(
            "warm-start configuration differs in: " + ", ".join(mismatches)
        )
    learner.online.load_state_dict(checkpoint["online_state_dict"])
    learner.target.load_state_dict(checkpoint["target_state_dict"])
    learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    learner.updates = int(checkpoint["updates"])
    env.load_state_dict(checkpoint["env_states"])
    rng.bit_generator.state = checkpoint["collector_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_states" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_states"]]
        )
    print(
        f"warm_start={path}; source_transitions={checkpoint['transitions']}; "
        f"source_updates={checkpoint['updates']}; replay=fresh",
        flush=True,
    )


def train(
    config: TrainConfig,
    run_dir: Path,
    *,
    resume: bool = False,
) -> EvaluationResult | None:
    """Run the 10M-transition afterstate learner or resume it."""

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
                f"run directory contains training data: {', '.join(occupied)}"
            )
        _write_json(config_path, {"schema_version": 1, **config.to_dict()})

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed ^ 0x2B91_74E5_C0D3_A68F)
    env = VectorAfterstateEnv(
        config.num_envs, config.seed ^ 0x739A_65F1_04C8_2D7B, verify=False
    )
    if config.replay_kind == "prioritized":
        replay: AfterstateReplayBuffer | PrioritizedAfterstateReplayBuffer = (
            PrioritizedAfterstateReplayBuffer(
                config.replay_capacity,
                alpha=config.priority_alpha,
                priority_epsilon=config.priority_epsilon,
                path=replay_path,
                priority_path=priority_path,
                resume=resume,
            )
        )
    else:
        replay = AfterstateReplayBuffer(
            config.replay_capacity,
            path=replay_path,
            resume=resume,
        )
    learner = AfterstateLearner(config, device)
    transitions = 0
    update_credit = 0.0
    best_validation_mean = -math.inf
    previous_elapsed = 0.0
    if not resume:
        _apply_warm_start(config, learner, env, rng, device=device)
    if resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        checkpoint_config = TrainConfig.from_dict(checkpoint["config"])
        structural = replace(
            config,
            total_transitions=checkpoint_config.total_transitions,
            device=checkpoint_config.device,
        )
        if structural != checkpoint_config:
            raise ValueError("resume configuration differs from saved run")
        learner.online.load_state_dict(checkpoint["online_state_dict"])
        learner.target.load_state_dict(checkpoint["target_state_dict"])
        learner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learner.updates = int(checkpoint["updates"])
        transitions = int(checkpoint["transitions"])
        update_credit = float(checkpoint["update_credit"])
        best_validation_mean = float(checkpoint["best_validation_mean"])
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        replay.load_state_dict(checkpoint["replay"])
        env.load_state_dict(checkpoint["env_states"])
        rng.bit_generator.state = checkpoint["collector_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )
    print(
        f"device={device}; envs={config.num_envs}; observation={OBSERVATION_DIM}; "
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

    if not resume and config.validation_seeds:
        last_validation = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
        )
        best_validation_mean = last_validation.mean
        _append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "transitions": 0,
                "updates": learner.updates,
                **last_validation.summary_dict(),
                "best": True,
            },
        )
        initial_payload = _checkpoint_payload(
            config,
            learner,
            replay,
            env,
            rng,
            transitions=0,
            update_credit=0.0,
            best_validation_mean=best_validation_mean,
            elapsed_seconds=perf_counter() - started,
        )
        _save_checkpoint(best_path, initial_payload)
        _save_checkpoint(latest_path, initial_payload)

    while transitions < config.total_transitions:
        count = min(config.num_envs, config.total_transitions - transitions)
        groups = env.candidate_groups(count)
        selected = _select_candidates(
            groups, learner, epsilon=epsilon_at(config, transitions), rng=rng
        )
        episodes = env.step(selected)
        for candidate, episode in zip(selected, episodes):
            replay.add(
                candidate.record, action=candidate.action_index, reward=candidate.reward
            )
            if episode is not None:
                recent_scores.append(episode.score)
        transitions += count
        if replay.size >= config.warmup_transitions:
            update_credit += count * config.replay_ratio / config.batch_size
            while update_credit >= 1.0:
                beta = priority_beta_at(config, transitions)
                batch = replay.sample(config.batch_size, rng, beta=beta)
                update = learner.update(batch)
                replay.update_priorities(batch.indices, update.td_errors)
                recent_updates.append(update.metrics)
                update_credit -= 1.0
        if transitions >= next_log:
            now = perf_counter()
            rate = (transitions - interval_transitions) / max(
                now - interval_started, 1e-9
            )
            means = (
                {
                    key: float(np.mean([item[key] for item in recent_updates]))
                    for key in recent_updates[0]
                }
                if recent_updates
                else {}
            )
            score_mean = float(np.mean(recent_scores)) if recent_scores else math.nan
            record = {
                "event": "train",
                "transitions": transitions,
                "updates": learner.updates,
                "replay_size": replay.size,
                "epsilon": epsilon_at(config, transitions),
                "recent_episode_mean": score_mean,
                "transitions_per_second": rate,
                **means,
            }
            if config.replay_kind == "prioritized":
                record["priority_beta"] = priority_beta_at(config, transitions)
            _append_jsonl(metrics_path, record)
            loss_text = f"{means['loss']:.4f}" if "loss" in means else "warming-up"
            print(
                f"step={transitions}; updates={learner.updates}; replay={replay.size}; epsilon={record['epsilon']:.3f}; score200={score_mean:.2f}; loss={loss_text}; rate={rate:.1f}/s",
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
                reward_scale=config.reward_scale,
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
                f"validation step={transitions}: mean={last_validation.mean:.2f} +/-{last_validation.standard_error:.2f}",
                flush=True,
            )
            if improved:
                replay.flush()
                _save_checkpoint(
                    best_path,
                    _checkpoint_payload(
                        config,
                        learner,
                        replay,
                        env,
                        rng,
                        transitions=transitions,
                        update_credit=update_credit,
                        best_validation_mean=best_validation_mean,
                        elapsed_seconds=previous_elapsed + perf_counter() - started,
                    ),
                )
            while next_validation <= transitions:
                next_validation += config.validation_interval
        if transitions >= next_checkpoint:
            replay.flush()
            _save_checkpoint(
                latest_path,
                _checkpoint_payload(
                    config,
                    learner,
                    replay,
                    env,
                    rng,
                    transitions=transitions,
                    update_credit=update_credit,
                    best_validation_mean=best_validation_mean,
                    elapsed_seconds=previous_elapsed + perf_counter() - started,
                ),
            )
            while next_checkpoint <= transitions:
                next_checkpoint += config.checkpoint_interval

    if config.validation_seeds:
        last_validation = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
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
        if improved:
            _save_checkpoint(
                best_path,
                _checkpoint_payload(
                    config,
                    learner,
                    replay,
                    env,
                    rng,
                    transitions=transitions,
                    update_credit=update_credit,
                    best_validation_mean=best_validation_mean,
                    elapsed_seconds=previous_elapsed + perf_counter() - started,
                ),
            )
    replay.flush()
    elapsed = previous_elapsed + perf_counter() - started
    payload = _checkpoint_payload(
        config,
        learner,
        replay,
        env,
        rng,
        transitions=transitions,
        update_credit=update_credit,
        best_validation_mean=best_validation_mean,
        elapsed_seconds=elapsed,
    )
    _save_checkpoint(latest_path, payload)
    if not best_path.exists():
        _save_checkpoint(best_path, payload)
    summary = {
        "transitions": transitions,
        "updates": learner.updates,
        "best_validation_mean": best_validation_mean,
        "elapsed_seconds": elapsed,
        "final_validation": (
            None if last_validation is None else last_validation.summary_dict()
        ),
        "replay_kind": config.replay_kind,
        "warm_start_checkpoint": config.warm_start_checkpoint,
    }
    _write_json(run_dir / "summary.json", summary)
    summary_lines = [
        "# Afterstate Value Training Summary",
        "",
        f"- replay_kind: {config.replay_kind}",
        f"- warm_start_checkpoint: {config.warm_start_checkpoint or 'none'}",
        f"- transitions: {transitions}",
        f"- updates: {learner.updates}",
        f"- elapsed_seconds: {elapsed:.3f}",
        f"- best_validation_mean: {best_validation_mean:.3f}",
    ]
    if last_validation is not None:
        summary_lines.extend(
            [
                "",
                "## Final Validation",
                "",
                f"- games: {len(last_validation.scores)}",
                f"- mean: {last_validation.mean:.3f}",
                f"- standard_error: {last_validation.standard_error:.3f}",
                f"- min / median / max: {last_validation.minimum} / {last_validation.median:.1f} / {last_validation.maximum}",
            ]
        )
    (run_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"training complete: transitions={transitions}; updates={learner.updates}; elapsed={elapsed:.1f}s",
        flush=True,
    )
    return last_validation
