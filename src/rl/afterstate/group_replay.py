"""Uniform replay whose sampling unit is one complete decision group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .codec import AfterstateRecord, Candidate
from .replay_storage import (
    REPLAY_RECORD_DTYPE,
    replay_record_from_row,
    write_replay_record,
)

GROUP_REPLAY_SCHEMA_VERSION = 1
_GROUP_DTYPE = np.dtype([("start", np.uint64), ("count", np.uint16)])


@dataclass(frozen=True, slots=True)
class DecisionGroupBatch:
    group_indices: NDArray[np.int64]
    group_offsets: NDArray[np.int64]
    records: tuple[AfterstateRecord, ...]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]

    @property
    def group_count(self) -> int:
        return len(self.group_offsets) - 1

    @property
    def candidate_count(self) -> int:
        return len(self.records)


class DecisionGroupReplayBuffer:
    """FIFO ring storage with variable-length, uniformly sampled groups."""

    replay_kind = "afterstate_decision_group"

    def __init__(
        self,
        group_capacity: int,
        candidate_capacity: int,
        *,
        group_path: Path | None = None,
        candidate_path: Path | None = None,
        resume: bool = False,
    ) -> None:
        if group_capacity < 1 or candidate_capacity < 1:
            raise ValueError("replay capacities must be positive")
        self.group_capacity = int(group_capacity)
        self.candidate_capacity = int(candidate_capacity)
        self.group_path = group_path
        self.candidate_path = candidate_path
        self._groups = self._allocate(
            group_path,
            _GROUP_DTYPE,
            self.group_capacity,
            resume=resume,
            label="group replay",
        )
        self._candidates = self._allocate(
            candidate_path,
            REPLAY_RECORD_DTYPE,
            self.candidate_capacity,
            resume=resume,
            label="candidate replay",
        )
        self.head = 0
        self.size = 0
        self.candidate_sequence = 0

    @staticmethod
    def _allocate(
        path: Path | None,
        dtype: np.dtype,
        capacity: int,
        *,
        resume: bool,
        label: str,
    ) -> NDArray[np.void]:
        if path is None:
            return np.empty(capacity, dtype=dtype)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = capacity * dtype.itemsize
        if resume:
            if not path.exists() or path.stat().st_size != expected:
                raise ValueError(f"{label} file is missing or has incompatible size")
            mode = "r+"
        else:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            mode = "w+"
        return np.memmap(path, dtype=dtype, mode=mode, shape=(capacity,))

    @property
    def allocated_bytes(self) -> int:
        return (
            self.group_capacity * _GROUP_DTYPE.itemsize
            + self.candidate_capacity * REPLAY_RECORD_DTYPE.itemsize
        )

    def _discard_oldest(self) -> None:
        if self.size < 1:
            raise RuntimeError("cannot discard from an empty group replay")
        self.head = (self.head + 1) % self.group_capacity
        self.size -= 1

    def _write_candidate(self, index: int, candidate: Candidate) -> None:
        write_replay_record(
            self._candidates[index],
            candidate.record,
            action=candidate.action_index,
            reward=candidate.reward,
        )

    def add(self, candidates: Sequence[Candidate]) -> None:
        count = len(candidates)
        if count < 1:
            raise ValueError("decision groups cannot be empty")
        if count > self.candidate_capacity or count > np.iinfo(np.uint16).max:
            raise ValueError("decision group exceeds replay candidate capacity")
        if any(not 0 <= candidate.action_index <= 155 for candidate in candidates):
            raise ValueError("action index is outside the fixed action space")

        new_end = self.candidate_sequence + count
        earliest_valid = new_end - self.candidate_capacity
        while self.size:
            oldest = self._groups[self.head]
            if int(oldest["start"]) >= earliest_valid:
                break
            self._discard_oldest()
        if self.size == self.group_capacity:
            self._discard_oldest()

        start = self.candidate_sequence
        for offset, candidate in enumerate(candidates):
            self._write_candidate(
                (start + offset) % self.candidate_capacity,
                candidate,
            )
        tail = (self.head + self.size) % self.group_capacity
        self._groups[tail]["start"] = start
        self._groups[tail]["count"] = count
        self.size += 1
        self.candidate_sequence = new_end

    def sample(
        self,
        group_batch_size: int,
        rng: np.random.Generator,
    ) -> DecisionGroupBatch:
        if group_batch_size < 1 or self.size < group_batch_size:
            raise ValueError("replay does not contain a full group batch")
        logical = rng.integers(
            0,
            self.size,
            size=group_batch_size,
            dtype=np.int64,
        )
        group_indices = (self.head + logical) % self.group_capacity
        groups = self._groups[group_indices]
        counts = np.asarray(groups["count"], dtype=np.int64)
        offsets = np.empty(group_batch_size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        candidate_indices = np.empty(int(offsets[-1]), dtype=np.int64)
        for index, group in enumerate(groups):
            start = int(group["start"])
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            candidate_indices[begin:end] = (
                np.arange(start, start + int(group["count"]), dtype=np.int64)
                % self.candidate_capacity
            )
        rows = self._candidates[candidate_indices]
        return DecisionGroupBatch(
            group_indices=np.ascontiguousarray(group_indices, dtype=np.int64),
            group_offsets=offsets,
            records=tuple(replay_record_from_row(row) for row in rows),
            actions=np.asarray(rows["action"], dtype=np.int64),
            rewards=np.asarray(rows["reward"], dtype=np.float32),
            terminated=np.asarray(rows["terminated"], dtype=np.bool_),
        )

    def flush(self) -> None:
        if isinstance(self._groups, np.memmap):
            self._groups.flush()
        if isinstance(self._candidates, np.memmap):
            self._candidates.flush()

    def state_dict(self) -> dict[str, object]:
        return {
            "replay_schema_version": GROUP_REPLAY_SCHEMA_VERSION,
            "replay_kind": self.replay_kind,
            "group_capacity": self.group_capacity,
            "candidate_capacity": self.candidate_capacity,
            "group_record_bytes": _GROUP_DTYPE.itemsize,
            "candidate_record_bytes": REPLAY_RECORD_DTYPE.itemsize,
            "head": self.head,
            "size": self.size,
            "candidate_sequence": self.candidate_sequence,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "replay_schema_version": GROUP_REPLAY_SCHEMA_VERSION,
            "replay_kind": self.replay_kind,
            "group_capacity": self.group_capacity,
            "candidate_capacity": self.candidate_capacity,
            "group_record_bytes": _GROUP_DTYPE.itemsize,
            "candidate_record_bytes": REPLAY_RECORD_DTYPE.itemsize,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint group replay metadata is incompatible")
        head = int(state["head"])
        size = int(state["size"])
        candidate_sequence = int(state["candidate_sequence"])
        if not 0 <= head < self.group_capacity:
            raise ValueError("checkpoint group replay head is invalid")
        if not 0 <= size <= self.group_capacity or candidate_sequence < 0:
            raise ValueError("checkpoint group replay cursor is invalid")
        self.head = head
        self.size = size
        self.candidate_sequence = candidate_sequence
        if self.size:
            oldest = self._groups[self.head]
            newest_index = (self.head + self.size - 1) % self.group_capacity
            newest = self._groups[newest_index]
            if int(oldest["start"]) < candidate_sequence - self.candidate_capacity:
                raise ValueError(
                    "checkpoint group replay references overwritten candidates"
                )
            if int(newest["start"]) + int(newest["count"]) > candidate_sequence:
                raise ValueError(
                    "checkpoint group replay extends beyond its candidate cursor"
                )
