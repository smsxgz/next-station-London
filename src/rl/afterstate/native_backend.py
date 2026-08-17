"""Typed access to the native afterstate feature and target-expansion API."""

from __future__ import annotations

import ctypes
import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from engine_cpp import COLORS
from engine_cpp._native import (
    NativePublicState,
    native_library_path as default_native_library_path,
)

from .codec import OBSERVATION_DIM, AfterstateRecord

_ORDERS = tuple(itertools.permutations(COLORS))
_COLOR_TO_INDEX = {color: index for index, color in enumerate(COLORS)}


@dataclass(frozen=True, slots=True)
class NativeExpansion:
    outcome_owners: NDArray[np.int32]
    outcome_probabilities: NDArray[np.float64]
    outcome_candidate_offsets: NDArray[np.int32]
    outcome_candidate_counts: NDArray[np.int32]
    candidate_rewards: NDArray[np.int32]
    candidate_terminated: NDArray[np.uint8]
    candidate_features: NDArray[np.float32]
    owner_count: int
    _owner: object | None = field(default=None, repr=False, compare=False)


class _NativeArray(np.ndarray):
    """A NumPy view that keeps its immutable native storage alive."""

    _native_owner: object | None

    def __array_finalize__(self, source: object | None) -> None:
        self._native_owner = getattr(source, "_native_owner", None)


class _ExpansionOwner:
    __slots__ = ("_library", "handle")

    def __init__(self, library: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        self._library = library
        self.handle = handle

    def close(self) -> None:
        if self.handle.value:
            self._library.ns_expansion_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _native_view(
    pointer: ctypes._Pointer,
    length: int,
    owner: _ExpansionOwner,
) -> np.ndarray:
    if length == 0:
        raise ValueError("zero-length native views must use an empty NumPy array")
    if not pointer:
        raise RuntimeError("native expansion returned a null data pointer")
    result = np.ctypeslib.as_array(pointer, shape=(length,)).view(_NativeArray)
    result._native_owner = owner
    return result


def record_to_native(record: AfterstateRecord) -> NativePublicState:
    state = NativePublicState()
    order = _ORDERS[record.order_code]
    for color in range(4):
        state.line_station_masks[color] = record.line_station_masks[color]
        edge_mask = record.line_edge_masks[color]
        for word in range(3):
            state.line_edge_words[color][word] = (edge_mask >> (word * 64)) & (
                (1 << 64) - 1
            )
        state.order[color] = _COLOR_TO_INDEX[order[color]]
    state.remaining_mask = record.remaining_mask
    state.round_index = record.round_index
    state.underground_count = record.underground_count
    state.draw_count = record.draw_count
    state.terminated = int(record.terminated)
    return state


def _native_states(records: Sequence[AfterstateRecord]) -> ctypes.Array:
    states = (NativePublicState * len(records))()
    for index, record in enumerate(records):
        states[index] = record_to_native(record)
    return states


class NativeFeatureBackend:
    def __init__(self, library_path: Path | None = None) -> None:
        self.library_path = (
            default_native_library_path()
            if library_path is None
            else library_path.expanduser().resolve()
        )
        self._library = ctypes.CDLL(str(self.library_path))
        state_pointer = ctypes.POINTER(NativePublicState)
        int_pointer = ctypes.POINTER(ctypes.c_int32)
        long_pointer = ctypes.POINTER(ctypes.c_int64)
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        float_pointer = ctypes.POINTER(ctypes.c_float)

        self._library.ns_last_error.argtypes = ()
        self._library.ns_last_error.restype = ctypes.c_char_p
        self._library.ns_public_state_size.argtypes = ()
        self._library.ns_public_state_size.restype = ctypes.c_int32
        self._library.ns_observation_dim.argtypes = ()
        self._library.ns_observation_dim.restype = ctypes.c_int32
        self._library.ns_feature_afterstates.argtypes = (
            state_pointer,
            ctypes.c_int32,
            float_pointer,
            ctypes.c_int32,
        )
        self._library.ns_feature_afterstates.restype = ctypes.c_int
        expansion_handle = ctypes.c_void_p
        self._library.ns_expansion_create.argtypes = (
            state_pointer,
            ctypes.c_int32,
            ctypes.POINTER(expansion_handle),
        )
        self._library.ns_expansion_create.restype = ctypes.c_int
        self._library.ns_expansion_destroy.argtypes = (expansion_handle,)
        self._library.ns_expansion_destroy.restype = None
        self._library.ns_expansion_thread_count.argtypes = (ctypes.c_int32,)
        self._library.ns_expansion_thread_count.restype = ctypes.c_int32
        self._library.ns_expansion_outcome_count.argtypes = (expansion_handle,)
        self._library.ns_expansion_outcome_count.restype = ctypes.c_int32
        self._library.ns_expansion_candidate_count.argtypes = (expansion_handle,)
        self._library.ns_expansion_candidate_count.restype = ctypes.c_int32
        self._library.ns_expansion_outcome_owners.argtypes = (expansion_handle,)
        self._library.ns_expansion_outcome_owners.restype = int_pointer
        self._library.ns_expansion_outcome_probabilities.argtypes = (expansion_handle,)
        self._library.ns_expansion_outcome_probabilities.restype = double_pointer
        self._library.ns_expansion_outcome_candidate_offsets.argtypes = (
            expansion_handle,
        )
        self._library.ns_expansion_outcome_candidate_offsets.restype = int_pointer
        self._library.ns_expansion_outcome_candidate_counts.argtypes = (
            expansion_handle,
        )
        self._library.ns_expansion_outcome_candidate_counts.restype = int_pointer
        self._library.ns_expansion_candidate_rewards.argtypes = (expansion_handle,)
        self._library.ns_expansion_candidate_rewards.restype = int_pointer
        self._library.ns_expansion_candidate_terminated.argtypes = (expansion_handle,)
        self._library.ns_expansion_candidate_terminated.restype = byte_pointer
        self._library.ns_expansion_candidate_features.argtypes = (expansion_handle,)
        self._library.ns_expansion_candidate_features.restype = float_pointer
        self._library.ns_expansion_select_candidates.argtypes = (
            expansion_handle,
            float_pointer,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            long_pointer,
            ctypes.c_int32,
        )
        self._library.ns_expansion_select_candidates.restype = ctypes.c_int
        self._library.ns_expansion_reduce_targets.argtypes = (
            expansion_handle,
            long_pointer,
            ctypes.c_int32,
            float_pointer,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            float_pointer,
            ctypes.c_int32,
        )
        self._library.ns_expansion_reduce_targets.restype = ctypes.c_int

        native_state_size = int(self._library.ns_public_state_size())
        if native_state_size != ctypes.sizeof(NativePublicState):
            raise RuntimeError(
                "native public-state size mismatch: "
                f"C++={native_state_size}, ctypes={ctypes.sizeof(NativePublicState)}"
            )
        native_observation_dim = int(self._library.ns_observation_dim())
        if native_observation_dim != OBSERVATION_DIM:
            raise RuntimeError(
                "native feature dimension mismatch: "
                f"C++={native_observation_dim}, Python={OBSERVATION_DIM}"
            )

    def _check(self, result: int) -> None:
        if result == 0:
            return
        message = self._library.ns_last_error()
        detail = message.decode("utf-8") if message else "unknown native error"
        raise RuntimeError(detail)

    def features(
        self,
        records: Sequence[AfterstateRecord],
    ) -> NDArray[np.float32]:
        records = tuple(records)
        result = np.empty((len(records), OBSERVATION_DIM), dtype=np.float32)
        if not records:
            return result
        states = _native_states(records)
        self._check(
            self._library.ns_feature_afterstates(
                states,
                len(records),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(records),
            )
        )
        return result

    def expand(
        self,
        records: Sequence[AfterstateRecord],
    ) -> NativeExpansion:
        records = tuple(records)
        if not records:
            return NativeExpansion(
                outcome_owners=np.empty(0, dtype=np.int32),
                outcome_probabilities=np.empty(0, dtype=np.float64),
                outcome_candidate_offsets=np.empty(0, dtype=np.int32),
                outcome_candidate_counts=np.empty(0, dtype=np.int32),
                candidate_rewards=np.empty(0, dtype=np.int32),
                candidate_terminated=np.empty(0, dtype=np.uint8),
                candidate_features=np.empty((0, OBSERVATION_DIM), dtype=np.float32),
                owner_count=0,
            )
        states = _native_states(records)
        handle = ctypes.c_void_p()
        self._check(
            self._library.ns_expansion_create(
                states,
                len(records),
                ctypes.byref(handle),
            )
        )
        storage_owner = _ExpansionOwner(self._library, handle)
        try:
            outcomes = int(self._library.ns_expansion_outcome_count(handle))
            candidates = int(self._library.ns_expansion_candidate_count(handle))
            if outcomes < 0 or candidates < 0:
                raise RuntimeError("native expansion returned a negative count")
            if outcomes == 0:
                if candidates != 0:
                    raise RuntimeError("native terminal expansion returned candidates")
                return NativeExpansion(
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.uint8),
                    np.empty((0, OBSERVATION_DIM), dtype=np.float32),
                    len(records),
                    storage_owner,
                )
            owners = _native_view(
                self._library.ns_expansion_outcome_owners(handle),
                outcomes,
                storage_owner,
            )
            probabilities = _native_view(
                self._library.ns_expansion_outcome_probabilities(handle),
                outcomes,
                storage_owner,
            )
            offsets = _native_view(
                self._library.ns_expansion_outcome_candidate_offsets(handle),
                outcomes,
                storage_owner,
            )
            counts = _native_view(
                self._library.ns_expansion_outcome_candidate_counts(handle),
                outcomes,
                storage_owner,
            )
            rewards = _native_view(
                self._library.ns_expansion_candidate_rewards(handle),
                candidates,
                storage_owner,
            )
            terminated = _native_view(
                self._library.ns_expansion_candidate_terminated(handle),
                candidates,
                storage_owner,
            )
            features = _native_view(
                self._library.ns_expansion_candidate_features(handle),
                candidates * OBSERVATION_DIM,
                storage_owner,
            ).reshape(candidates, OBSERVATION_DIM)
        except Exception:
            storage_owner.close()
            raise
        expected_offset = 0
        previous_owner = -1
        for index in range(outcomes):
            record_owner = int(owners[index])
            count = int(counts[index])
            if not 0 <= record_owner < len(records) or record_owner < previous_owner:
                raise RuntimeError("native expansion returned an invalid owner order")
            if int(offsets[index]) != expected_offset or count < 1:
                raise RuntimeError(
                    "native expansion returned an invalid candidate range"
                )
            if not np.isfinite(probabilities[index]) or probabilities[index] <= 0.0:
                raise RuntimeError("native expansion returned an invalid probability")
            previous_owner = record_owner
            expected_offset += count
        if expected_offset != candidates:
            raise RuntimeError("native expansion candidate ranges are incomplete")
        return NativeExpansion(
            owners,
            probabilities,
            offsets,
            counts,
            rewards,
            terminated,
            features,
            len(records),
            storage_owner,
        )

    def select_candidates(
        self,
        expansion: NativeExpansion,
        online_values: NDArray[np.float32],
        *,
        reward_scale: float,
        gamma: float,
    ) -> NDArray[np.int64]:
        values = np.ascontiguousarray(online_values, dtype=np.float32)
        if values.ndim != 1 or len(values) != len(expansion.candidate_rewards):
            raise ValueError("online values do not match native candidates")
        selected = np.empty(len(expansion.outcome_owners), dtype=np.int64)
        if expansion.owner_count == 0:
            if len(values) != 0 or len(selected) != 0:
                raise RuntimeError("empty native expansion contains data")
            return selected
        storage_owner = expansion._owner
        if not isinstance(storage_owner, _ExpansionOwner):
            raise RuntimeError("native expansion storage is unavailable")
        self._check(
            self._library.ns_expansion_select_candidates(
                storage_owner.handle,
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(values),
                reward_scale,
                gamma,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                len(selected),
            )
        )
        return selected

    def reduce_targets(
        self,
        expansion: NativeExpansion,
        selected_indices: NDArray[np.int64],
        target_values: NDArray[np.float32],
        *,
        reward_scale: float,
        gamma: float,
    ) -> NDArray[np.float32]:
        selected = np.ascontiguousarray(selected_indices, dtype=np.int64)
        values = np.ascontiguousarray(target_values, dtype=np.float32)
        outcomes = len(expansion.outcome_owners)
        if selected.ndim != 1 or len(selected) != outcomes:
            raise ValueError("selected indices do not match native outcomes")
        if values.ndim != 1 or len(values) != outcomes:
            raise ValueError("target values do not match native outcomes")
        result = np.empty(expansion.owner_count, dtype=np.float32)
        if expansion.owner_count == 0:
            if outcomes != 0:
                raise RuntimeError("empty native expansion contains outcomes")
            return result
        storage_owner = expansion._owner
        if not isinstance(storage_owner, _ExpansionOwner):
            raise RuntimeError("native expansion storage is unavailable")
        self._check(
            self._library.ns_expansion_reduce_targets(
                storage_owner.handle,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                len(selected),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(values),
                reward_scale,
                gamma,
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(result),
            )
        )
        return result

    def expansion_thread_count(self, input_count: int) -> int:
        return int(self._library.ns_expansion_thread_count(input_count))


@lru_cache(maxsize=1)
def native_backend() -> NativeFeatureBackend:
    return NativeFeatureBackend()


def feature_records(
    records: Sequence[AfterstateRecord],
) -> NDArray[np.float32]:
    return native_backend().features(records)


def expand_afterstate_records(
    records: Sequence[AfterstateRecord],
) -> NativeExpansion:
    return native_backend().expand(records)


def select_expansion_candidates(
    expansion: NativeExpansion,
    online_values: NDArray[np.float32],
    *,
    reward_scale: float,
    gamma: float,
) -> NDArray[np.int64]:
    return native_backend().select_candidates(
        expansion,
        online_values,
        reward_scale=reward_scale,
        gamma=gamma,
    )


def reduce_expansion_targets(
    expansion: NativeExpansion,
    selected_indices: NDArray[np.int64],
    target_values: NDArray[np.float32],
    *,
    reward_scale: float,
    gamma: float,
) -> NDArray[np.float32]:
    return native_backend().reduce_targets(
        expansion,
        selected_indices,
        target_values,
        reward_scale=reward_scale,
        gamma=gamma,
    )
