"""Canonical public afterstates and explicit score-aware features.

The engine remains the source of truth for transitions and scoring.  This
module only serializes a pending-free public state and derives a fixed input
vector from it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from engine_cpp import COLORS, DECK, TOURIST_TRACK, Action, GameError, GameSession
from solver.state import public_event_successors

NUM_COLORS = len(COLORS)
NUM_EDGES = 155
NUM_STATIONS = 53
NUM_CARDS = len(DECK)
ACTION_COUNT = NUM_EDGES + 1
PASS_ACTION_INDEX = NUM_EDGES
AFTERSTATE_SCHEMA_VERSION = 1

_ORDERS = tuple(itertools.permutations(COLORS))
_ORDER_TO_CODE = {order: index for index, order in enumerate(_ORDERS)}


@dataclass(frozen=True, slots=True)
class AfterstateRecord:
    """Fixed, hidden-deck-free representation of one afterstate."""

    line_station_masks: tuple[int, ...]
    line_edge_masks: tuple[int, ...]
    remaining_mask: int
    order_code: int
    round_index: int
    underground_count: int
    draw_count: int
    terminated: bool

    def __post_init__(self) -> None:
        if len(self.line_station_masks) != NUM_COLORS:
            raise ValueError("afterstate must contain one station mask per color")
        if len(self.line_edge_masks) != NUM_COLORS:
            raise ValueError("afterstate must contain one edge mask per color")
        if not 0 <= self.remaining_mask < (1 << NUM_CARDS):
            raise ValueError("remaining card mask is outside the deck")
        if not 0 <= self.order_code < len(_ORDERS):
            raise ValueError("invalid color-order code")
        if not 0 <= self.round_index < NUM_COLORS:
            raise ValueError("round index is outside the game")
        if not 0 <= self.underground_count <= 5:
            raise ValueError("underground count is outside the game")
        if not 0 <= self.draw_count <= NUM_CARDS:
            raise ValueError("draw count is outside the game")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One legal action and its fully copied public afterstate."""

    action_index: int
    action: Action | None
    reward: int
    afterstate: GameSession
    record: AfterstateRecord
    features: NDArray[np.float32]


def _require_afterstate(game: GameSession) -> None:
    if game.shared_objectives_enabled or game.pencil_powers_enabled:
        raise GameError("afterstate RL supports only the base game")
    if game.status not in {"playing", "finished"}:
        raise GameError("unknown game status")
    if game.pending is not None:
        raise GameError("afterstate codec requires a state without pending card")
    if len(game.lines) != NUM_COLORS or len(game.map.edges) != NUM_EDGES:
        raise GameError("afterstate codec requires the standard London map")
    if len(game.map.stations) != NUM_STATIONS:
        raise GameError("afterstate codec requires the standard London map")


def encode_afterstate(game: GameSession) -> AfterstateRecord:
    """Serialize a pending-free public game without deck order or RNG state."""

    _require_afterstate(game)
    return AfterstateRecord(
        line_station_masks=tuple(
            int(game.lines[color].station_mask) for color in COLORS
        ),
        line_edge_masks=tuple(int(game.lines[color].edge_mask) for color in COLORS),
        remaining_mask=int(game.remaining_card_mask),
        order_code=_ORDER_TO_CODE[tuple(game.order)],
        round_index=int(game.round_index),
        underground_count=int(game.underground_count),
        draw_count=int(game.draw_count),
        terminated=game.status == "finished",
    )


def decode_afterstate(record: AfterstateRecord) -> GameSession:
    """Rebuild an engine state from a canonical afterstate record."""

    return GameSession.from_public_state(
        order=_ORDERS[record.order_code],
        line_station_masks=tuple(record.line_station_masks),
        line_edge_masks=tuple(record.line_edge_masks),
        remaining_mask=record.remaining_mask,
        round_index=record.round_index,
        underground_count=record.underground_count,
        draw_count=record.draw_count,
        terminated=record.terminated,
    )


def afterstate_key(game_or_record: GameSession | AfterstateRecord) -> tuple[int, ...]:
    """Return a deterministic public key for diagnostics and local dedup only."""

    record = (
        encode_afterstate(game_or_record)
        if isinstance(game_or_record, GameSession)
        else game_or_record
    )
    return (
        *record.line_station_masks,
        *record.line_edge_masks,
        record.remaining_mask,
        record.order_code,
        record.round_index,
        record.underground_count,
        record.draw_count,
        int(record.terminated),
    )


def feature_vector(game: GameSession) -> NDArray[np.float32]:
    """Encode the full public afterstate plus explicit scoring features."""

    _require_afterstate(game)
    from .native_backend import feature_records

    vector = feature_records((encode_afterstate(game),))[0].copy()
    if not np.isfinite(vector).all():
        raise RuntimeError("feature extractor produced a non-finite value")
    return vector


_FEATURE_GROUPS = {
    "line_edges": NUM_COLORS * NUM_EDGES,
    "line_stations": NUM_COLORS * NUM_STATIONS,
    "remaining_cards": NUM_CARDS,
    "color_order": NUM_COLORS * NUM_COLORS,
    "round_index": NUM_COLORS,
    "active_color": NUM_COLORS,
    "underground_count": 6,
    "draw_count": NUM_CARDS + 1,
    "terminal": 1,
    "line_score_features": NUM_COLORS * (13 + 13 + 5),
    "global_score_features": 4 + len(TOURIST_TRACK) + 5 + 1 + 1 + 1 + 4 + 4,
}
OBSERVATION_DIM = sum(_FEATURE_GROUPS.values())
FEATURE_SCHEMA = {
    "schema_version": AFTERSTATE_SCHEMA_VERSION,
    "groups": _FEATURE_GROUPS,
    "normalization": {
        "station_counts": 13.0,
        "district_count": 13.0,
        "max_stations": 13.0,
        "route_per_line": 169.0,
        "thames_per_line": 20.0,
        "tourist_visits_per_line": 5.0,
        "route_global": 676.0,
        "thames_global": 80.0,
        "tourist_visits_global": 20.0,
        "tourist_points": 25.0,
        "interchange_count": 53.0,
        "interchange_bonus": 477.0,
        "current_total": 1300.0,
        "tourist_gain": 4.0,
        "interchange_gain": 9.0,
        "interchange_gain_mass": 477.0,
    },
}
FEATURE_SCHEMA_JSON = json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":"))
FEATURE_SCHEMA_HASH = hashlib.sha256(FEATURE_SCHEMA_JSON.encode("utf-8")).hexdigest()


def make_candidates(game: GameSession) -> tuple[Candidate, ...]:
    """Create every legal action's full-copy afterstate and feature vector."""

    if game.status != "playing" or game.pending is None:
        raise GameError("candidate generation requires a pending decision")
    if game.shared_objectives_enabled or game.pencil_powers_enabled:
        raise GameError("afterstate RL supports only the base game")
    candidates: list[Candidate] = []
    for action in (None, *game.legal_actions()):
        action_index = PASS_ACTION_INDEX if action is None else int(action.edge_id)
        reward = 0 if action is None else sum(game.score_delta_for_legal_action(action))
        child = game.copy_public_state()
        child.apply_legal_action(action)
        record = encode_afterstate(child)
        candidates.append(
            Candidate(
                action_index=action_index,
                action=action,
                reward=int(reward),
                afterstate=child,
                record=record,
                features=feature_vector(child),
            )
        )
    candidates.sort(key=lambda candidate: candidate.action_index)
    return tuple(candidates)


def public_afterstate_successors(
    game: GameSession,
) -> Iterator[tuple[float, GameSession]]:
    """Yield exact next public card events from a canonical afterstate."""

    _require_afterstate(game)
    if game.status == "finished":
        return
    probability_sum = 0.0
    for probability, child in public_event_successors(game):
        probability_sum += probability
        yield probability, child
    if not np.isclose(probability_sum, 1.0, atol=1e-12):
        raise RuntimeError(f"chance probabilities sum to {probability_sum}")
