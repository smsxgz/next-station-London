"""Shared binary layout for scalar afterstate replay records."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .codec import NUM_COLORS, AfterstateRecord

REPLAY_RECORD_DTYPE = np.dtype(
    [
        ("station_masks", np.uint64, (NUM_COLORS,)),
        ("edge_masks", np.uint64, (NUM_COLORS, 3)),
        ("remaining_mask", np.uint16),
        ("order_code", np.uint8),
        ("round_index", np.uint8),
        ("underground_count", np.uint8),
        ("draw_count", np.uint8),
        ("terminated", np.bool_),
        ("action", np.uint16),
        ("reward", np.int16),
    ]
)


def _split_mask(mask: int) -> tuple[int, int, int]:
    return tuple((int(mask) >> (64 * index)) & ((1 << 64) - 1) for index in range(3))


def _join_mask(chunks: NDArray[np.uint64]) -> int:
    return sum(int(value) << (64 * index) for index, value in enumerate(chunks))


def write_replay_record(
    row: np.void,
    record: AfterstateRecord,
    *,
    action: int,
    reward: int,
) -> None:
    row["station_masks"] = np.asarray(record.line_station_masks, dtype=np.uint64)
    row["edge_masks"] = np.asarray(
        [_split_mask(mask) for mask in record.line_edge_masks],
        dtype=np.uint64,
    )
    row["remaining_mask"] = record.remaining_mask
    row["order_code"] = record.order_code
    row["round_index"] = record.round_index
    row["underground_count"] = record.underground_count
    row["draw_count"] = record.draw_count
    row["terminated"] = record.terminated
    row["action"] = action
    row["reward"] = reward


def replay_record_from_row(row: np.void) -> AfterstateRecord:
    return AfterstateRecord(
        line_station_masks=tuple(int(value) for value in row["station_masks"]),
        line_edge_masks=tuple(_join_mask(value) for value in row["edge_masks"]),
        remaining_mask=int(row["remaining_mask"]),
        order_code=int(row["order_code"]),
        round_index=int(row["round_index"]),
        underground_count=int(row["underground_count"]),
        draw_count=int(row["draw_count"]),
        terminated=bool(row["terminated"]),
    )
