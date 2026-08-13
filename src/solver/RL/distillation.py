"""Single-database Q-target storage and supervised DQN distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import secrets
import sqlite3
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
import torch.nn.functional as F

from .codec import ACTION_COUNT, OBSERVATION_DIM
from .dqn import DQNPolicy, masked_argmax, resolve_device


_OBSERVATION_BYTES = (OBSERVATION_DIM + 7) // 8
_MASK_BYTES = (ACTION_COUNT + 7) // 8
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    init_checkpoint: str
    reward_scale: float = 10.0
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 1e-4
    gradient_clip: float = 10.0
    advantage_coefficient: float = 0.0
    policy_coefficient: float = 0.0
    policy_temperature_points: float = 2.0
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        for name, value in (
            ("reward_scale", self.reward_scale),
            ("learning_rate", self.learning_rate),
            ("gradient_clip", self.gradient_clip),
            ("policy_temperature_points", self.policy_temperature_points),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("advantage_coefficient", self.advantage_coefficient),
            ("policy_coefficient", self.policy_coefficient),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "init_checkpoint": self.init_checkpoint,
            "reward_scale": self.reward_scale,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "advantage_coefficient": self.advantage_coefficient,
            "policy_coefficient": self.policy_coefficient,
            "policy_temperature_points": self.policy_temperature_points,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    loss: float
    absolute_q_loss: float
    advantage_loss: float
    policy_kl: float
    action_agreement: float
    mean_teacher_regret: float
    root_mean_squared_error: float
    advantage_root_mean_squared_error: float


class TeacherDatabase:
    """Append complete teacher games and dense legal-action Q labels."""

    def __init__(
        self,
        path: Path,
        *,
        teacher_checkpoint: Path,
        depth: int,
        reward_scale: float,
    ) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()
        expected = {
            "schema_version": str(_SCHEMA_VERSION),
            "teacher_checkpoint": str(teacher_checkpoint.resolve()),
            "teacher_depth": str(depth),
            "reward_scale": repr(float(reward_scale)),
            "observation_dim": str(OBSERVATION_DIM),
            "action_count": str(ACTION_COUNT),
        }
        stored = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if stored:
            mismatched = {
                key: (stored.get(key), value)
                for key, value in expected.items()
                if stored.get(key) != value
            }
            if mismatched:
                raise ValueError(f"teacher database metadata mismatch: {mismatched}")
        else:
            values = {**expected, "created_at": _timestamp()}
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                values.items(),
            )
            self.connection.commit()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                seed INTEGER NOT NULL UNIQUE,
                validation INTEGER NOT NULL,
                score INTEGER NOT NULL,
                positions INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY,
                game_id INTEGER NOT NULL REFERENCES games(id),
                validation INTEGER NOT NULL,
                observation BLOB NOT NULL,
                legal_mask BLOB NOT NULL,
                q_values BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS samples_validation
                ON samples(validation);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> TeacherDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def completed_games(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM games").fetchone()[0])

    def completed_ordinals(self) -> set[int]:
        return {
            int(row[0])
            for row in self.connection.execute("SELECT ordinal FROM games")
        }

    def contains_seed(self, seed: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM games WHERE seed = ?", (int(seed),)
        ).fetchone()
        return row is not None

    def append_game(
        self,
        *,
        ordinal: int,
        seed: int,
        validation: bool,
        score: int,
        samples: Sequence[tuple[bytes, bytes, bytes]],
    ) -> None:
        if not samples:
            raise ValueError("a teacher game must contain at least one sample")
        flag = int(validation)
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO games(ordinal, seed, validation, score, positions)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ordinal, seed, flag, score, len(samples)),
            )
            game_id = int(cursor.lastrowid)
            self.connection.executemany(
                """
                INSERT INTO samples(
                    game_id, validation, observation, legal_mask, q_values
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (game_id, flag, observation, legal_mask, q_values)
                    for observation, legal_mask, q_values in samples
                ),
            )

    def split_counts(self) -> tuple[int, int]:
        counts = dict(
            self.connection.execute(
                "SELECT validation, COUNT(*) FROM samples GROUP BY validation"
            )
        )
        return int(counts.get(0, 0)), int(counts.get(1, 0))

    def game_summary(self) -> dict[str, float | int]:
        row = self.connection.execute(
            """
            SELECT COUNT(*), SUM(positions), AVG(score), MIN(score), MAX(score)
            FROM games
            """
        ).fetchone()
        return {
            "games": int(row[0] or 0),
            "positions": int(row[1] or 0),
            "mean_score": float(row[2] or 0.0),
            "min_score": int(row[3] or 0),
            "max_score": int(row[4] or 0),
        }


def pack_teacher_sample(
    observation: NDArray[np.uint8],
    legal_mask: NDArray[np.bool_],
    q_values: NDArray[np.float32],
) -> tuple[bytes, bytes, bytes]:
    if observation.shape != (OBSERVATION_DIM,) or observation.dtype != np.uint8:
        raise ValueError("teacher observation has an incompatible shape or dtype")
    if legal_mask.shape != (ACTION_COUNT,) or legal_mask.dtype != np.bool_:
        raise ValueError("teacher mask has an incompatible shape or dtype")
    if q_values.shape != (ACTION_COUNT,) or q_values.dtype != np.float32:
        raise ValueError("teacher Q vector has an incompatible shape or dtype")
    if not legal_mask.any() or not np.isfinite(q_values[legal_mask]).all():
        raise ValueError("teacher sample has invalid legal-action Q values")
    return (
        np.packbits(observation, bitorder="little").tobytes(),
        np.packbits(legal_mask, bitorder="little").tobytes(),
        q_values.tobytes(),
    )


@dataclass(slots=True)
class _LoadedSplit:
    observations: NDArray[np.uint8]
    masks: NDArray[np.uint8]
    q_values: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.observations)

    def batches(
        self,
        indices: NDArray[np.int64],
        batch_size: int,
    ) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.bool_], NDArray[np.float32]]]:
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            observations = np.unpackbits(
                self.observations[selected],
                axis=1,
                count=OBSERVATION_DIM,
                bitorder="little",
            )
            masks = np.unpackbits(
                self.masks[selected],
                axis=1,
                count=ACTION_COUNT,
                bitorder="little",
            ).astype(np.bool_, copy=False)
            yield (
                np.ascontiguousarray(observations),
                np.ascontiguousarray(masks),
                np.ascontiguousarray(self.q_values[selected]),
            )


def _load_split(path: Path, validation: bool) -> _LoadedSplit:
    connection = sqlite3.connect(path)
    try:
        flag = int(validation)
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM samples WHERE validation = ?", (flag,)
            ).fetchone()[0]
        )
        if count < 1:
            raise ValueError("teacher database split is empty")
        observations = np.empty((count, _OBSERVATION_BYTES), dtype=np.uint8)
        masks = np.empty((count, _MASK_BYTES), dtype=np.uint8)
        q_values = np.empty((count, ACTION_COUNT), dtype=np.float32)
        cursor = connection.execute(
            """
            SELECT observation, legal_mask, q_values
            FROM samples WHERE validation = ? ORDER BY id
            """,
            (flag,),
        )
        offset = 0
        while True:
            rows = cursor.fetchmany(4096)
            if not rows:
                break
            for observation, legal_mask, values in rows:
                observations[offset] = np.frombuffer(observation, dtype=np.uint8)
                masks[offset] = np.frombuffer(legal_mask, dtype=np.uint8)
                q_values[offset] = np.frombuffer(values, dtype=np.float32)
                offset += 1
        if offset != count:
            raise RuntimeError("teacher database changed while loading")
        return _LoadedSplit(observations, masks, q_values)
    finally:
        connection.close()


@torch.inference_mode()
def _evaluate(
    network: nn.Module,
    split: _LoadedSplit,
    *,
    batch_size: int,
    config: DistillationConfig,
    device: torch.device,
) -> DistillationMetrics:
    network.eval()
    total_states = 0
    loss_sum = 0.0
    absolute_loss_sum = 0.0
    advantage_loss_sum = 0.0
    policy_kl_sum = 0.0
    squared_error_sum = 0.0
    advantage_squared_error_sum = 0.0
    legal_count = 0
    agreements = 0
    regret_sum = 0.0
    indices = np.arange(len(split), dtype=np.int64)
    for observations, masks, targets in split.batches(indices, batch_size):
        observation_tensor = torch.as_tensor(
            observations, device=device, dtype=torch.float32
        )
        mask_tensor = torch.as_tensor(masks, device=device, dtype=torch.bool)
        target_tensor = torch.as_tensor(
            targets, device=device, dtype=torch.float32
        )
        predicted = network.q_values(observation_tensor)
        per_state, absolute_loss, advantage_loss, policy_kl = (
            _distillation_losses(predicted, target_tensor, mask_tensor, config)
        )
        count = len(observations)
        loss_sum += float(per_state.sum().item())
        absolute_loss_sum += float(absolute_loss.sum().item())
        advantage_loss_sum += float(advantage_loss.sum().item())
        policy_kl_sum += float(policy_kl.sum().item())
        differences = (predicted - target_tensor)[mask_tensor]
        squared_error_sum += float(differences.square().sum().item())
        mask_float = mask_tensor.float()
        legal_per_state = mask_float.sum(dim=1)
        predicted_mean = (predicted * mask_float).sum(dim=1) / legal_per_state
        target_mean = (target_tensor * mask_float).sum(dim=1) / legal_per_state
        advantage_difference = (
            predicted
            - predicted_mean[:, None]
            - target_tensor
            + target_mean[:, None]
        )[mask_tensor]
        advantage_squared_error_sum += float(
            advantage_difference.square().sum().item()
        )
        legal_count += int(mask_tensor.sum().item())
        teacher_actions = masked_argmax(target_tensor, mask_tensor)
        student_actions = masked_argmax(predicted, mask_tensor)
        agreements += int((teacher_actions == student_actions).sum().item())
        teacher_best = target_tensor.gather(1, teacher_actions[:, None]).squeeze(1)
        selected = target_tensor.gather(1, student_actions[:, None]).squeeze(1)
        regret_sum += float(
            ((teacher_best - selected) * config.reward_scale).sum().item()
        )
        total_states += count
    return DistillationMetrics(
        loss=loss_sum / total_states,
        absolute_q_loss=absolute_loss_sum / total_states,
        advantage_loss=advantage_loss_sum / total_states,
        policy_kl=policy_kl_sum / total_states,
        action_agreement=agreements / total_states,
        mean_teacher_regret=regret_sum / total_states,
        root_mean_squared_error=math.sqrt(squared_error_sum / legal_count),
        advantage_root_mean_squared_error=math.sqrt(
            advantage_squared_error_sum / legal_count
        ),
    )


def _distillation_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total and component losses, reduced once per state."""

    mask_float = legal_mask.float()
    legal_per_state = mask_float.sum(dim=1)
    absolute_elements = F.smooth_l1_loss(
        predicted, target, reduction="none"
    )
    absolute = (absolute_elements * mask_float).sum(dim=1) / legal_per_state

    predicted_mean = (predicted * mask_float).sum(dim=1) / legal_per_state
    target_mean = (target * mask_float).sum(dim=1) / legal_per_state
    predicted_advantage = predicted - predicted_mean[:, None]
    target_advantage = target - target_mean[:, None]
    advantage_elements = F.smooth_l1_loss(
        predicted_advantage,
        target_advantage,
        reduction="none",
    )
    advantage = (advantage_elements * mask_float).sum(dim=1) / legal_per_state

    temperature = config.policy_temperature_points / config.reward_scale
    teacher_logits = (target / temperature).masked_fill(~legal_mask, -torch.inf)
    student_logits = (predicted / temperature).masked_fill(
        ~legal_mask, -torch.inf
    )
    teacher_log_probabilities = F.log_softmax(teacher_logits, dim=1)
    student_log_probabilities = F.log_softmax(student_logits, dim=1)
    teacher_probabilities = teacher_log_probabilities.exp()
    policy_kl = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).masked_fill(~legal_mask, 0.0).sum(dim=1)

    total = (
        absolute
        + config.advantage_coefficient * advantage
        + config.policy_coefficient * policy_kl
    )
    return total, absolute, advantage, policy_kl


def _checkpoint_payload(
    network: nn.Module,
    config: DistillationConfig,
    *,
    epoch: int,
    metrics: DistillationMetrics,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": _timestamp(),
        "algorithm": "dqn",
        "q_decomposition": "full_q",
        "config": {
            **config.to_dict(),
            "reward_scale": config.reward_scale,
            "training_kind": "exact-chance-q-distillation",
        },
        "network_spec": network.config(),
        "online_state_dict": network.state_dict(),
        "target_state_dict": network.state_dict(),
        "distillation_epoch": epoch,
        "distillation_validation": asdict(metrics),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_metric(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")


def distill_teacher_database(
    database_path: Path,
    run_dir: Path,
    config: DistillationConfig,
) -> Path:
    """Fit all legal student Q values to a fixed exact-chance teacher."""

    config.validate()
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if config_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite distillation run {run_dir}")
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("loading teacher database into packed memory arrays", flush=True)
    train_split = _load_split(database_path, validation=False)
    validation_split = _load_split(database_path, validation=True)
    print(
        f"teacher positions: train={len(train_split)}; "
        f"validation={len(validation_split)}",
        flush=True,
    )

    device = resolve_device(config.device)
    policy = DQNPolicy.from_checkpoint(config.init_checkpoint, device=str(device))
    network = policy.network
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(secrets.randbits(128))

    initial = _evaluate(
        network,
        validation_split,
        batch_size=config.batch_size,
        config=config,
        device=device,
    )
    _append_metric(
        metrics_path,
        {"event": "validation", "epoch": 0, **asdict(initial), "best": True},
    )
    print(
        f"epoch=0; val_loss={initial.loss:.5f}; "
        f"agreement={initial.action_agreement * 100:.2f}%; "
        f"regret={initial.mean_teacher_regret:.3f}",
        flush=True,
    )
    best_key = (
        initial.action_agreement,
        -initial.mean_teacher_regret,
        -initial.loss,
    )
    best_epoch = 0
    best_metrics = initial
    best_path = run_dir / "best.pt"
    _save_checkpoint(
        best_path,
        _checkpoint_payload(network, config, epoch=0, metrics=initial),
    )

    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        network.train()
        order = rng.permutation(len(train_split))
        loss_sum = 0.0
        absolute_loss_sum = 0.0
        advantage_loss_sum = 0.0
        policy_kl_sum = 0.0
        states = 0
        gradient_sum = 0.0
        batches = 0
        for observations, masks, targets in train_split.batches(
            order, config.batch_size
        ):
            observation_tensor = torch.as_tensor(
                observations, device=device, dtype=torch.float32
            )
            mask_tensor = torch.as_tensor(masks, device=device, dtype=torch.bool)
            target_tensor = torch.as_tensor(
                targets, device=device, dtype=torch.float32
            )
            predicted = network.q_values(observation_tensor)
            per_state, absolute_loss, advantage_loss, policy_kl = (
                _distillation_losses(
                    predicted,
                    target_tensor,
                    mask_tensor,
                    config,
                )
            )
            loss = per_state.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(
                network.parameters(), config.gradient_clip
            )
            optimizer.step()
            count = len(observations)
            loss_sum += float(per_state.detach().sum().item())
            absolute_loss_sum += float(absolute_loss.detach().sum().item())
            advantage_loss_sum += float(advantage_loss.detach().sum().item())
            policy_kl_sum += float(policy_kl.detach().sum().item())
            states += count
            gradient_sum += float(gradient.detach().item())
            batches += 1

        validation = _evaluate(
            network,
            validation_split,
            batch_size=config.batch_size,
            config=config,
            device=device,
        )
        key = (
            validation.action_agreement,
            -validation.mean_teacher_regret,
            -validation.loss,
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_metrics = validation
            _save_checkpoint(
                best_path,
                _checkpoint_payload(
                    network, config, epoch=epoch, metrics=validation
                ),
            )
        record = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": loss_sum / states,
            "train_absolute_q_loss": absolute_loss_sum / states,
            "train_advantage_loss": advantage_loss_sum / states,
            "train_policy_kl": policy_kl_sum / states,
            "gradient_norm": gradient_sum / batches,
            "validation_loss": validation.loss,
            "action_agreement": validation.action_agreement,
            "mean_teacher_regret": validation.mean_teacher_regret,
            "root_mean_squared_error": validation.root_mean_squared_error,
            "best": improved,
            "elapsed_seconds": perf_counter() - started,
        }
        _append_metric(metrics_path, record)
        print(
            f"epoch={epoch}/{config.epochs}; "
            f"train_loss={record['train_loss']:.5f}; "
            f"val_loss={validation.loss:.5f}; "
            f"agreement={validation.action_agreement * 100:.2f}%; "
            f"regret={validation.mean_teacher_regret:.3f}; "
            f"best={improved}",
            flush=True,
        )

    latest_metrics = validation
    _save_checkpoint(
        run_dir / "latest.pt",
        _checkpoint_payload(
            network, config, epoch=config.epochs, metrics=latest_metrics
        ),
    )
    elapsed = perf_counter() - started
    summary = [
        "# Exact-Chance Q Distillation",
        "",
        f"- Teacher database: `{database_path}`",
        f"- Initialized from: `{config.init_checkpoint}`",
        f"- Train positions: `{len(train_split)}`",
        f"- Validation positions: `{len(validation_split)}`",
        f"- Epochs: `{config.epochs}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Learning rate: `{config.learning_rate}`",
        f"- Advantage coefficient: `{config.advantage_coefficient}`",
        f"- Policy coefficient: `{config.policy_coefficient}`",
        f"- Policy temperature: `{config.policy_temperature_points}` points",
        f"- Best epoch: `{best_epoch}`",
        f"- Best teacher action agreement: `{best_metrics.action_agreement * 100:.2f}%`",
        f"- Best teacher regret: `{best_metrics.mean_teacher_regret:.3f}` points",
        f"- Best validation loss: `{best_metrics.loss:.6f}`",
        f"- Elapsed seconds: `{elapsed:.1f}`",
    ]
    (run_dir / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8", newline="\n"
    )
    return best_path


__all__ = [
    "DistillationConfig",
    "DistillationMetrics",
    "TeacherDatabase",
    "distill_teacher_database",
    "pack_teacher_sample",
]
