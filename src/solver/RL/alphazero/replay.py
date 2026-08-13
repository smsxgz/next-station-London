"""Compact disk-backed replay storage for AlphaZero training targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ..codec import ACTION_COUNT, OBSERVATION_DIM


_OBSERVATION_BYTES = (OBSERVATION_DIM + 7) // 8
_MASK_BYTES = (ACTION_COUNT + 7) // 8
_RECORD_DTYPE = np.dtype(
    [
        ("observation", np.uint8, (_OBSERVATION_BYTES,)),
        ("action_mask", np.uint8, (_MASK_BYTES,)),
        ("visits", np.uint16, (ACTION_COUNT,)),
        ("value", np.float32),
    ]
)


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    observation: NDArray[np.uint8]
    action_mask: NDArray[np.bool_]
    visits: NDArray[np.uint16]
    value: float


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    observations: NDArray[np.uint8]
    action_masks: NDArray[np.bool_]
    policy_targets: NDArray[np.float32]
    values: NDArray[np.float32]


class AlphaZeroReplay:
    """A uniform ring buffer whose full allocation is persisted locally."""

    def __init__(
        self,
        capacity: int,
        path: Path,
        *,
        resume: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if resume else "w+"
        if resume and not path.exists():
            raise FileNotFoundError(f"missing replay file: {path}")
        self._records = np.memmap(
            path,
            dtype=_RECORD_DTYPE,
            mode=mode,
            shape=(self.capacity,),
        )
        self.size = 0
        self.position = 0

    @property
    def allocated_bytes(self) -> int:
        return self.capacity * _RECORD_DTYPE.itemsize

    @property
    def record_bytes(self) -> int:
        return _RECORD_DTYPE.itemsize

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("checkpoint replay capacity does not match config")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("checkpoint replay metadata is invalid")
        self.size = size
        self.position = position

    def state_dict(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "record_bytes": self.record_bytes,
        }

    def add_many(self, records: Iterable[ReplayRecord]) -> int:
        added = 0
        for record in records:
            observation = np.asarray(record.observation, dtype=np.uint8)
            mask = np.asarray(record.action_mask, dtype=np.bool_)
            visits = np.asarray(record.visits, dtype=np.uint16)
            if observation.shape != (OBSERVATION_DIM,):
                raise ValueError("replay observation has an invalid shape")
            if mask.shape != (ACTION_COUNT,) or visits.shape != (ACTION_COUNT,):
                raise ValueError("replay action target has an invalid shape")
            if not mask.any() or int(visits.sum(dtype=np.uint64)) < 1:
                raise ValueError("replay target must contain at least one visit")
            if np.any(visits[~mask]):
                raise ValueError("replay target assigns visits to illegal actions")
            row = self._records[self.position]
            row["observation"] = np.packbits(observation, bitorder="little")
            row["action_mask"] = np.packbits(mask, bitorder="little")
            row["visits"] = visits
            row["value"] = np.float32(record.value)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            added += 1
        return added

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> ReplayBatch:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("replay does not contain a full training batch")
        indices = rng.integers(0, self.size, size=batch_size)
        rows = self._records[indices]
        observations = np.unpackbits(
            rows["observation"],
            axis=1,
            count=OBSERVATION_DIM,
            bitorder="little",
        )
        masks = np.unpackbits(
            rows["action_mask"],
            axis=1,
            count=ACTION_COUNT,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        visits = rows["visits"].astype(np.float32)
        policy_targets = visits / visits.sum(axis=1, keepdims=True)
        return ReplayBatch(
            observations=observations,
            action_masks=masks,
            policy_targets=policy_targets,
            values=np.asarray(rows["value"], dtype=np.float32),
        )

    def flush(self) -> None:
        self._records.flush()

    def close(self) -> None:
        self.flush()
        mmap = getattr(self._records, "_mmap", None)
        if mmap is not None:
            mmap.close()
