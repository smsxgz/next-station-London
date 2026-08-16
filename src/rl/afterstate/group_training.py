"""Decision-group fitted value training for scalar afterstate networks."""

from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .codec import (
    AFTERSTATE_SCHEMA_VERSION,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_HASH,
    OBSERVATION_DIM,
    Candidate,
)
from .environment import VectorAfterstateEnv
from .evaluation import (
    BackupEvaluation,
    EvaluationResult,
    evaluate_backup_network,
    evaluate_network,
)
from .group_replay import DecisionGroupBatch, DecisionGroupReplayBuffer
from .native_backend import feature_records
from .network import (
    AfterstateNetworkSpec,
    AfterstateValueNetwork,
    network_from_checkpoint,
    resolve_device,
)
from .policy import AfterstatePolicy
from .target import (
    CudaFeatureWorkspace,
    PreparedAfterstateTargets,
    evaluate_prepared_afterstate_targets,
    prepare_exact_afterstate_targets,
)


@dataclass(frozen=True, slots=True)
class GroupTrainConfig:
    seed: int
    source_checkpoint: str
    num_envs: int = 128
    total_transitions: int = 5_000_000
    gamma: float = 1.0
    reward_scale: float = 10.0
    group_capacity: int = 1_000_000
    candidate_capacity: int = 5_000_000
    group_batch_size: int = 500
    replay_ratio: float = 8.0
    warmup_groups: int = 25_000
    learning_rate: float = 3e-4
    target_update_interval: int = 1_000
    epsilon: float = 0.05
    gradient_clip: float = 10.0
    hidden_sizes: tuple[int, ...] = (512, 512)
    validation_seeds: tuple[int, ...] = ()
    validation_interval: int = 250_000
    evaluation_envs: int = 64
    inference_batch_size: int = 8192
    log_interval: int = 100_000
    checkpoint_interval: int = 250_000
    device: str = "auto"

    def validate(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not self.source_checkpoint:
            raise ValueError("source_checkpoint cannot be empty")
        for name in (
            "num_envs",
            "total_transitions",
            "group_capacity",
            "candidate_capacity",
            "group_batch_size",
            "warmup_groups",
            "target_update_interval",
            "validation_interval",
            "evaluation_envs",
            "inference_batch_size",
            "log_interval",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_capacity < self.group_batch_size:
            raise ValueError("group replay capacity must hold one batch")
        if self.warmup_groups < self.group_batch_size:
            raise ValueError("warmup must hold one group batch")
        if not 0.0 < self.gamma <= 1.0 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite and in (0, 1]")
        if self.reward_scale <= 0.0 or not np.isfinite(self.reward_scale):
            raise ValueError("reward_scale must be finite and positive")
        if self.replay_ratio <= 0.0 or not np.isfinite(self.replay_ratio):
            raise ValueError("replay_ratio must be finite and positive")
        if self.learning_rate <= 0.0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.gradient_clip <= 0.0 or not np.isfinite(self.gradient_clip):
            raise ValueError("gradient_clip must be finite and positive")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["hidden_sizes"] = list(self.hidden_sizes)
        raw["validation_seeds"] = list(self.validation_seeds)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GroupTrainConfig:
        values = dict(raw)
        values.pop("schema_version", None)
        values["hidden_sizes"] = tuple(int(value) for value in values["hidden_sizes"])
        values["validation_seeds"] = tuple(
            int(value) for value in values.get("validation_seeds", ())
        )
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class GroupLearnerUpdate:
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PreparedDecisionGroupBatch:
    batch: DecisionGroupBatch
    observations: np.ndarray
    candidate_weights: np.ndarray
    targets: PreparedAfterstateTargets


class GroupAfterstateLearner:
    def __init__(self, config: GroupTrainConfig, device: torch.device) -> None:
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

    def initialize_online_only(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        source = network_from_checkpoint(checkpoint, self.device)
        source_config = checkpoint.get("config", {})
        if float(source_config.get("reward_scale", -1.0)) != self.config.reward_scale:
            raise ValueError("source checkpoint uses a different reward scale")
        if float(source_config.get("gamma", -1.0)) != self.config.gamma:
            raise ValueError("source checkpoint uses a different gamma")
        if source.spec != self.online.spec:
            raise ValueError("source checkpoint network shape is incompatible")
        self.online.load_state_dict(source.state_dict())
        self.target.load_state_dict(source.state_dict())
        self.online.eval()
        self.target.eval()
        return checkpoint

    def prepare(self, batch: DecisionGroupBatch) -> PreparedDecisionGroupBatch:
        if batch.group_count != self.config.group_batch_size:
            raise ValueError("sampled group batch has an unexpected size")
        counts = np.diff(batch.group_offsets)
        if np.any(counts < 1) or int(counts.sum()) != batch.candidate_count:
            raise ValueError("decision-group offsets are invalid")
        candidate_weights = np.concatenate(
            [
                np.full(count, 1.0 / (batch.group_count * count), dtype=np.float32)
                for count in counts
            ]
        )
        return PreparedDecisionGroupBatch(
            batch=batch,
            observations=feature_records(batch.records),
            candidate_weights=candidate_weights,
            targets=prepare_exact_afterstate_targets(
                batch.records,
                batch.terminated,
            ),
        )

    def update_prepared(
        self,
        prepared: PreparedDecisionGroupBatch,
    ) -> GroupLearnerUpdate:
        batch = prepared.batch
        observations = torch.as_tensor(
            prepared.observations,
            device=self.device,
            dtype=torch.float32,
        )
        predicted = self.online(observations)
        targets, stats = evaluate_prepared_afterstate_targets(
            self.online,
            self.target,
            prepared.targets,
            device=self.device,
            reward_scale=self.config.reward_scale,
            gamma=self.config.gamma,
            inference_batch_size=self.config.inference_batch_size,
            feature_workspace=self.feature_workspace,
        )
        element_losses = F.smooth_l1_loss(predicted, targets, reduction="none")
        weights = torch.as_tensor(
            prepared.candidate_weights,
            device=self.device,
            dtype=torch.float32,
        )
        loss = (element_losses * weights).sum()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        predicted_values = predicted.detach().cpu().numpy()
        target_values = targets.detach().cpu().numpy()
        rewards = batch.rewards.astype(np.float64) / self.config.reward_scale
        agreements = 0
        normalized_regret = 0.0
        for group in range(batch.group_count):
            start = int(batch.group_offsets[group])
            stop = int(batch.group_offsets[group + 1])
            raw_scores = rewards[start:stop] + predicted_values[start:stop]
            target_scores = rewards[start:stop] + target_values[start:stop]
            raw_choice = int(np.argmax(raw_scores))
            target_choice = int(np.argmax(target_scores))
            agreements += int(raw_choice == target_choice)
            normalized_regret += max(
                0.0,
                float(target_scores[target_choice] - target_scores[raw_choice]),
            )

        scale = self.config.reward_scale
        td_errors = (targets - predicted).abs().detach()
        return GroupLearnerUpdate(
            metrics={
                "loss": float(loss.detach().item()),
                "w_mean": float((predicted.detach() * weights).sum().item() * scale),
                "target_mean": float((targets * weights).sum().item() * scale),
                "td_abs_mean": float((td_errors * weights).sum().item() * scale),
                "gradient_norm": float(gradient_norm.detach().item()),
                "candidates_per_group": batch.candidate_count / batch.group_count,
                "sample_action_agreement": agreements / batch.group_count,
                "sample_regret_points": normalized_regret * scale / batch.group_count,
                "chance_outcomes_per_group": stats.chance_outcomes / batch.group_count,
                "expanded_candidates_per_group": stats.candidate_actions
                / batch.group_count,
            }
        )

    def update(self, batch: DecisionGroupBatch) -> GroupLearnerUpdate:
        return self.update_prepared(self.prepare(batch))

    def update_pipelined(
        self,
        batches: Sequence[DecisionGroupBatch],
        executor: Executor,
    ) -> tuple[GroupLearnerUpdate, ...]:
        if not batches:
            return ()
        pending = executor.submit(self.prepare, batches[0])
        results: list[GroupLearnerUpdate] = []
        for index in range(len(batches)):
            prepared = pending.result()
            if index + 1 < len(batches):
                pending = executor.submit(self.prepare, batches[index + 1])
            results.append(self.update_prepared(prepared))
        return tuple(results)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    config: GroupTrainConfig,
    learner: GroupAfterstateLearner,
    replay: DecisionGroupReplayBuffer,
    env: VectorAfterstateEnv,
    rng: np.random.Generator,
    *,
    transitions: int,
    update_credit: float,
    best_backup_mean: float,
    best_raw_mean: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_type": "afterstate_value",
        "algorithm": "scalar",
        "training_mode": "decision_group",
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
        "best_backup_mean": best_backup_mean,
        "best_raw_mean": best_raw_mean,
        "elapsed_seconds": elapsed_seconds,
        "replay": replay.state_dict(),
        "collector_rng_state": rng.bit_generator.state,
        "env_states": env.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def _choose_candidates(
    groups: Sequence[Sequence[Candidate]],
    policy: AfterstatePolicy,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[Candidate, ...]:
    greedy = policy.select_groups(groups)
    selected: list[Candidate] = []
    for group, decision in zip(groups, greedy):
        if rng.random() < epsilon:
            selected.append(group[int(rng.integers(0, len(group)))])
        else:
            selected.append(decision.candidate)
    return tuple(selected)


def _validation_record(
    transitions: int,
    updates: int,
    raw: EvaluationResult,
    backup: BackupEvaluation,
    *,
    raw_improved: bool,
    backup_improved: bool,
) -> dict[str, object]:
    return {
        "event": "validation",
        "transitions": transitions,
        "updates": updates,
        "raw": raw.summary_dict(),
        "backup": backup.result.summary_dict(),
        "backup_diagnostics": backup.diagnostics,
        "raw_improved": raw_improved,
        "backup_improved": backup_improved,
    }


def train_groups(
    config: GroupTrainConfig,
    run_dir: Path,
    *,
    resume: bool = False,
) -> tuple[EvaluationResult, BackupEvaluation] | None:
    config.validate()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    group_path = run_dir / "replay_groups.dat"
    candidate_path = run_dir / "replay_candidates.dat"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    best_raw_path = run_dir / "best_raw.pt"
    managed_paths = (
        config_path,
        metrics_path,
        group_path,
        candidate_path,
        latest_path,
        best_path,
        best_raw_path,
    )
    if resume:
        if not config_path.exists() or not latest_path.exists():
            raise FileNotFoundError("resume requires config.json and latest.pt")
    else:
        occupied = [path.name for path in managed_paths if path.exists()]
        if occupied:
            raise FileExistsError(
                "run directory contains training data: " + ", ".join(occupied)
            )
        _write_json(config_path, {"schema_version": 1, **config.to_dict()})

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed ^ 0x2B91_74E5_C0D3_A68F)
    env = VectorAfterstateEnv(
        config.num_envs,
        config.seed ^ 0x739A_65F1_04C8_2D7B,
        verify=False,
    )
    replay = DecisionGroupReplayBuffer(
        config.group_capacity,
        config.candidate_capacity,
        group_path=group_path,
        candidate_path=candidate_path,
        resume=resume,
    )
    learner = GroupAfterstateLearner(config, device)
    transitions = 0
    update_credit = 0.0
    best_backup_mean = -math.inf
    best_raw_mean = -math.inf
    previous_elapsed = 0.0
    if resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        checkpoint_config = GroupTrainConfig.from_dict(checkpoint["config"])
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
        best_backup_mean = float(checkpoint["best_backup_mean"])
        best_raw_mean = float(checkpoint["best_raw_mean"])
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        replay.load_state_dict(checkpoint["replay"])
        env.load_state_dict(checkpoint["env_states"])
        rng.bit_generator.state = checkpoint["collector_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )
    else:
        source_path = Path(config.source_checkpoint).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"source checkpoint not found: {source_path}")
        source = learner.initialize_online_only(source_path)
        print(
            f"source={source_path}; source_transitions={source['transitions']}; "
            "loaded=online; target=online; optimizer=fresh; replay=fresh",
            flush=True,
        )

    print(
        f"device={device}; envs={config.num_envs}; group_batch={config.group_batch_size}; "
        f"replay_groups={config.group_capacity}; replay_candidates={config.candidate_capacity}; "
        f"replay={replay.allocated_bytes / 2**20:.1f} MiB; "
        f"start={transitions}; target={config.total_transitions}",
        flush=True,
    )
    collector_policy = AfterstatePolicy(
        learner.online,
        device=device,
        reward_scale=config.reward_scale,
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
    last_raw: EvaluationResult | None = None
    last_backup: BackupEvaluation | None = None
    last_validation_step = -1
    preparation_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="afterstate-prepare",
    )

    if not resume and config.validation_seeds:
        last_raw = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
        )
        last_backup = evaluate_backup_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
            gamma=config.gamma,
            inference_batch_size=config.inference_batch_size,
        )
        best_raw_mean = last_raw.mean
        best_backup_mean = last_backup.result.mean
        _append_jsonl(
            metrics_path,
            _validation_record(
                0,
                learner.updates,
                last_raw,
                last_backup,
                raw_improved=True,
                backup_improved=True,
            ),
        )
        last_validation_step = 0
        initial = _checkpoint_payload(
            config,
            learner,
            replay,
            env,
            rng,
            transitions=0,
            update_credit=0.0,
            best_backup_mean=best_backup_mean,
            best_raw_mean=best_raw_mean,
            elapsed_seconds=perf_counter() - started,
        )
        _save_checkpoint(best_path, initial)
        _save_checkpoint(best_raw_path, initial)
        _save_checkpoint(latest_path, initial)
        print(
            f"validation step=0: raw={last_raw.mean:.3f}; "
            f"backup={last_backup.result.mean:.3f}",
            flush=True,
        )

    while transitions < config.total_transitions:
        count = min(config.num_envs, config.total_transitions - transitions)
        groups = env.candidate_groups(count)
        selected = _choose_candidates(
            groups,
            collector_policy,
            epsilon=config.epsilon,
            rng=rng,
        )
        for group in groups:
            replay.add(group)
        episodes = env.step(selected)
        for episode in episodes:
            if episode is not None:
                recent_scores.append(episode.score)
        transitions += count

        if replay.size >= config.warmup_groups:
            update_credit += count * config.replay_ratio / config.group_batch_size
            batches: list[DecisionGroupBatch] = []
            while update_credit >= 1.0:
                batches.append(replay.sample(config.group_batch_size, rng))
                update_credit -= 1.0
            for update in learner.update_pipelined(batches, preparation_pool):
                recent_updates.append(update.metrics)

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
            record: dict[str, object] = {
                "event": "train",
                "transitions": transitions,
                "updates": learner.updates,
                "replay_groups": replay.size,
                "epsilon": config.epsilon,
                "recent_episode_mean": score_mean,
                "transitions_per_second": rate,
                **means,
            }
            _append_jsonl(metrics_path, record)
            loss_text = f"{means['loss']:.4f}" if "loss" in means else "warming-up"
            print(
                f"step={transitions}; updates={learner.updates}; "
                f"groups={replay.size}; score200={score_mean:.2f}; "
                f"loss={loss_text}; rate={rate:.1f}/s",
                flush=True,
            )
            recent_updates.clear()
            interval_started = now
            interval_transitions = transitions
            while next_log <= transitions:
                next_log += config.log_interval

        if config.validation_seeds and transitions >= next_validation:
            last_raw = evaluate_network(
                learner.online,
                config.validation_seeds,
                device=device,
                num_envs=config.evaluation_envs,
                reward_scale=config.reward_scale,
            )
            last_backup = evaluate_backup_network(
                learner.online,
                config.validation_seeds,
                device=device,
                num_envs=config.evaluation_envs,
                reward_scale=config.reward_scale,
                gamma=config.gamma,
                inference_batch_size=config.inference_batch_size,
            )
            raw_improved = last_raw.mean > best_raw_mean
            backup_improved = last_backup.result.mean > best_backup_mean
            if raw_improved:
                best_raw_mean = last_raw.mean
            if backup_improved:
                best_backup_mean = last_backup.result.mean
            _append_jsonl(
                metrics_path,
                _validation_record(
                    transitions,
                    learner.updates,
                    last_raw,
                    last_backup,
                    raw_improved=raw_improved,
                    backup_improved=backup_improved,
                ),
            )
            last_validation_step = transitions
            print(
                f"validation step={transitions}: raw={last_raw.mean:.3f}; "
                f"backup={last_backup.result.mean:.3f}; "
                f"agreement={last_backup.diagnostics['action_agreement']:.3f}",
                flush=True,
            )
            if raw_improved or backup_improved:
                replay.flush()
                payload = _checkpoint_payload(
                    config,
                    learner,
                    replay,
                    env,
                    rng,
                    transitions=transitions,
                    update_credit=update_credit,
                    best_backup_mean=best_backup_mean,
                    best_raw_mean=best_raw_mean,
                    elapsed_seconds=previous_elapsed + perf_counter() - started,
                )
                if raw_improved:
                    _save_checkpoint(best_raw_path, payload)
                if backup_improved:
                    _save_checkpoint(best_path, payload)
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
                    best_backup_mean=best_backup_mean,
                    best_raw_mean=best_raw_mean,
                    elapsed_seconds=previous_elapsed + perf_counter() - started,
                ),
            )
            while next_checkpoint <= transitions:
                next_checkpoint += config.checkpoint_interval

    preparation_pool.shutdown(wait=True)

    if config.validation_seeds and last_validation_step != transitions:
        last_raw = evaluate_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
        )
        last_backup = evaluate_backup_network(
            learner.online,
            config.validation_seeds,
            device=device,
            num_envs=config.evaluation_envs,
            reward_scale=config.reward_scale,
            gamma=config.gamma,
            inference_batch_size=config.inference_batch_size,
        )
        raw_improved = last_raw.mean > best_raw_mean
        backup_improved = last_backup.result.mean > best_backup_mean
        if raw_improved:
            best_raw_mean = last_raw.mean
        if backup_improved:
            best_backup_mean = last_backup.result.mean
        _append_jsonl(
            metrics_path,
            _validation_record(
                transitions,
                learner.updates,
                last_raw,
                last_backup,
                raw_improved=raw_improved,
                backup_improved=backup_improved,
            ),
        )
        last_validation_step = transitions
        if raw_improved or backup_improved:
            replay.flush()
            payload = _checkpoint_payload(
                config,
                learner,
                replay,
                env,
                rng,
                transitions=transitions,
                update_credit=update_credit,
                best_backup_mean=best_backup_mean,
                best_raw_mean=best_raw_mean,
                elapsed_seconds=previous_elapsed + perf_counter() - started,
            )
            if raw_improved:
                _save_checkpoint(best_raw_path, payload)
            if backup_improved:
                _save_checkpoint(best_path, payload)

    replay.flush()
    elapsed = previous_elapsed + perf_counter() - started
    final_payload = _checkpoint_payload(
        config,
        learner,
        replay,
        env,
        rng,
        transitions=transitions,
        update_credit=update_credit,
        best_backup_mean=best_backup_mean,
        best_raw_mean=best_raw_mean,
        elapsed_seconds=elapsed,
    )
    _save_checkpoint(latest_path, final_payload)
    summary = {
        "transitions": transitions,
        "updates": learner.updates,
        "best_backup_mean": best_backup_mean,
        "best_raw_mean": best_raw_mean,
        "elapsed_seconds": elapsed,
        "source_checkpoint": config.source_checkpoint,
        "final_raw": None if last_raw is None else last_raw.summary_dict(),
        "final_backup": (
            None if last_backup is None else last_backup.result.summary_dict()
        ),
        "final_backup_diagnostics": (
            None if last_backup is None else last_backup.diagnostics
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    lines = [
        "# Decision-Group Afterstate Training",
        "",
        f"- source_checkpoint: {config.source_checkpoint}",
        f"- transitions: {transitions}",
        f"- updates: {learner.updates}",
        f"- group_batch_size: {config.group_batch_size}",
        f"- elapsed_seconds: {elapsed:.3f}",
        f"- best_raw_mean: {best_raw_mean:.3f}",
        f"- best_backup_mean: {best_backup_mean:.3f}",
    ]
    if last_raw is not None and last_backup is not None:
        lines.extend(
            [
                "",
                "## Final Validation",
                "",
                f"- raw: {last_raw.mean:.3f} +/- {last_raw.standard_error:.3f}",
                f"- backup: {last_backup.result.mean:.3f} +/- {last_backup.result.standard_error:.3f}",
                f"- action_agreement: {last_backup.diagnostics['action_agreement']:.6f}",
            ]
        )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"training complete: transitions={transitions}; updates={learner.updates}; "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    if last_raw is None or last_backup is None:
        return None
    return last_raw, last_backup
