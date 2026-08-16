"""Exact one-chance Double-DQN backups using native state expansion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .codec import AfterstateRecord
from .native_backend import (
    NativeExpansion,
    expand_afterstate_records,
    reduce_expansion_targets,
    select_expansion_candidates,
)
from .network import AfterstateValueNetwork


@dataclass(frozen=True, slots=True)
class TargetStats:
    chance_outcomes: int = 0
    candidate_actions: int = 0
    online_batches: int = 0
    target_batches: int = 0


@dataclass(frozen=True, slots=True)
class PreparedAfterstateTargets:
    expansion: NativeExpansion


class CudaFeatureWorkspace:
    """Grow-only CUDA storage for variable-length native feature matrices."""

    _ROW_ALIGNMENT = 8192

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA feature workspace requires a CUDA device")
        self.device = device
        self._storage: torch.Tensor | None = None

    @property
    def capacity_rows(self) -> int:
        return 0 if self._storage is None else len(self._storage)

    def upload(self, features: NDArray[np.float32]) -> torch.Tensor:
        if features.ndim != 2 or features.dtype != np.float32:
            raise ValueError("workspace features must be a float32 matrix")
        if not features.flags.c_contiguous:
            raise ValueError("workspace features must be contiguous")
        rows, columns = features.shape
        if rows == 0:
            if self._storage is None:
                return torch.empty(
                    (0, columns),
                    device=self.device,
                    dtype=torch.float32,
                )
            if self._storage.shape[1] != columns:
                raise ValueError("workspace feature width changed")
            return self._storage[:0]

        needs_growth = (
            self._storage is None
            or self._storage.shape[1] != columns
            or self.capacity_rows < rows
        )
        if needs_growth:
            previous_capacity = self.capacity_rows
            grown_capacity = max(rows, (previous_capacity * 3 + 1) // 2)
            alignment = self._ROW_ALIGNMENT
            capacity = ((grown_capacity + alignment - 1) // alignment) * alignment
            self._storage = None
            if previous_capacity:
                torch.cuda.empty_cache()
            self._storage = torch.empty(
                (capacity, columns),
                device=self.device,
                dtype=torch.float32,
            )

        destination = self._storage[:rows]
        destination.copy_(torch.from_numpy(features))
        return destination


def _network_values(
    network: AfterstateValueNetwork,
    features: NDArray[np.float32],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.float32], int]:
    if features.ndim != 2:
        raise ValueError("network features must be a matrix")
    if len(features) == 0:
        return np.empty(0, dtype=np.float32), 0
    values = np.empty(len(features), dtype=np.float32)
    batches = 0
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        tensor = torch.as_tensor(
            features[start:stop], device=device, dtype=torch.float32
        )
        values[start:stop] = network(tensor).detach().cpu().numpy()
        batches += 1
    return values, batches


def _continuation_values(
    network: AfterstateValueNetwork,
    features: NDArray[np.float32],
    terminated: NDArray[np.uint8],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.float32], int]:
    values, batches = _network_values(
        network,
        features,
        device=device,
        batch_size=batch_size,
    )
    values[terminated.astype(np.bool_, copy=False)] = 0.0
    return values, batches


def _network_values_from_tensor(
    network: AfterstateValueNetwork,
    features: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    if features.ndim != 2:
        raise ValueError("network features must be a matrix")
    values = torch.empty(
        len(features),
        device=features.device,
        dtype=torch.float32,
    )
    batches = 0
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        values[start:stop] = network(features[start:stop])
        batches += 1
    return values, batches


def _network_values_at_indices(
    network: AfterstateValueNetwork,
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    if features.ndim != 2 or indices.ndim != 1:
        raise ValueError("network features and indices have incompatible shapes")
    values = torch.empty(
        len(indices),
        device=features.device,
        dtype=torch.float32,
    )
    batches = 0
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        selected = torch.index_select(features, 0, indices[start:stop])
        values[start:stop] = network(selected)
        batches += 1
    return values, batches


def prepare_exact_afterstate_targets(
    records: tuple[AfterstateRecord, ...] | list[AfterstateRecord],
    terminated: NDArray[np.bool_],
) -> PreparedAfterstateTargets:
    """Expand a replay batch without consulting either value network."""

    if len(records) != len(terminated):
        raise ValueError("records and terminated flags have different lengths")
    records = tuple(records)
    for owner, record in enumerate(records):
        if bool(terminated[owner]) != bool(record.terminated):
            raise ValueError("replay terminal flag disagrees with canonical record")
    expansion = expand_afterstate_records(records)
    if expansion.owner_count != len(records):
        raise RuntimeError("native expansion owner count is incompatible")
    return PreparedAfterstateTargets(expansion)


@torch.no_grad()
def evaluate_prepared_afterstate_targets(
    online: AfterstateValueNetwork,
    target: AfterstateValueNetwork,
    prepared: PreparedAfterstateTargets,
    *,
    device: torch.device,
    reward_scale: float = 10.0,
    gamma: float = 1.0,
    inference_batch_size: int = 8192,
    feature_workspace: CudaFeatureWorkspace | None = None,
) -> tuple[torch.Tensor, TargetStats]:
    """Evaluate an already expanded exact one-chance target batch."""

    if reward_scale <= 0.0 or not np.isfinite(reward_scale):
        raise ValueError("reward_scale must be finite and positive")
    if not 0.0 < gamma <= 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and in (0, 1]")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")

    expansion = prepared.expansion
    if device.type == "cuda":
        if feature_workspace is not None and feature_workspace.device != device:
            raise ValueError("CUDA feature workspace uses a different device")
        resident_features = (
            torch.as_tensor(
                expansion.candidate_features,
                device=device,
                dtype=torch.float32,
            )
            if feature_workspace is None
            else feature_workspace.upload(expansion.candidate_features)
        )
        online_tensor, online_batches = _network_values_from_tensor(
            online,
            resident_features,
            batch_size=inference_batch_size,
        )
        online_values = online_tensor.cpu().numpy()
        online_values[expansion.candidate_terminated.astype(np.bool_, copy=False)] = 0.0
        selected_indices = select_expansion_candidates(
            expansion,
            online_values,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        device_indices = torch.as_tensor(
            selected_indices,
            device=device,
            dtype=torch.int64,
        )
        target_tensor, target_batches = _network_values_at_indices(
            target,
            resident_features,
            device_indices,
            batch_size=inference_batch_size,
        )
        target_values = target_tensor.cpu().numpy()
        target_values[
            expansion.candidate_terminated[selected_indices].astype(
                np.bool_, copy=False
            )
        ] = 0.0
    else:
        online_values, online_batches = _continuation_values(
            online,
            expansion.candidate_features,
            expansion.candidate_terminated,
            device=device,
            batch_size=inference_batch_size,
        )
        selected_indices = select_expansion_candidates(
            expansion,
            online_values,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        target_values, target_batches = _continuation_values(
            target,
            expansion.candidate_features[selected_indices],
            expansion.candidate_terminated[selected_indices],
            device=device,
            batch_size=inference_batch_size,
        )

    reduced = reduce_expansion_targets(
        expansion,
        selected_indices,
        target_values,
        reward_scale=reward_scale,
        gamma=gamma,
    )
    result = torch.as_tensor(reduced, device=device, dtype=torch.float32)
    return result, TargetStats(
        chance_outcomes=len(expansion.outcome_owners),
        candidate_actions=len(expansion.candidate_rewards),
        online_batches=online_batches,
        target_batches=target_batches,
    )


@torch.no_grad()
def exact_afterstate_targets(
    online: AfterstateValueNetwork,
    target: AfterstateValueNetwork,
    records: tuple[AfterstateRecord, ...] | list[AfterstateRecord],
    terminated: NDArray[np.bool_],
    *,
    device: torch.device,
    reward_scale: float = 10.0,
    gamma: float = 1.0,
    inference_batch_size: int = 8192,
    feature_workspace: CudaFeatureWorkspace | None = None,
) -> tuple[torch.Tensor, TargetStats]:
    """Return exact one-chance targets in normalized score units.

    For every chance outcome, online values select the action and target values
    evaluate that same selected afterstate.  Engine rewards are never learned.
    """

    prepared = prepare_exact_afterstate_targets(records, terminated)
    return evaluate_prepared_afterstate_targets(
        online,
        target,
        prepared,
        device=device,
        reward_scale=reward_scale,
        gamma=gamma,
        inference_batch_size=inference_batch_size,
        feature_workspace=feature_workspace,
    )
