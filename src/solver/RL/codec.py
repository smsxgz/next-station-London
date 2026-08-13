"""Public observation and fixed action encoding for the London map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engine import Action, COLORS, DECK, GameError, GameSession, LONDON, SYMBOLS


NUM_COLORS = len(COLORS)
NUM_EDGES = len(LONDON.edges)
NUM_STATIONS = len(LONDON.stations)
NUM_CARDS = len(DECK)
ACTION_COUNT = NUM_EDGES + 1
PASS_ACTION_INDEX = NUM_EDGES

_COLOR_INDEX = {color: index for index, color in enumerate(COLORS)}
_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}


def _span(start: int, size: int) -> tuple[slice, int]:
    return slice(start, start + size), start + size


_cursor = 0
LINE_EDGES, _cursor = _span(_cursor, NUM_COLORS * NUM_EDGES)
LINE_STATIONS, _cursor = _span(_cursor, NUM_COLORS * NUM_STATIONS)
ACTIVE_LEAVES, _cursor = _span(_cursor, NUM_STATIONS)
REMAINING_CARDS, _cursor = _span(_cursor, NUM_CARDS)
CURRENT_CARDS, _cursor = _span(_cursor, NUM_CARDS)
TARGET_SYMBOL, _cursor = _span(_cursor, len(SYMBOLS) + 1)
EVENT_FLAGS, _cursor = _span(_cursor, 3)
COLOR_ORDER, _cursor = _span(_cursor, NUM_COLORS * NUM_COLORS)
ROUND_INDEX, _cursor = _span(_cursor, NUM_COLORS)
ACTIVE_COLOR, _cursor = _span(_cursor, NUM_COLORS)
UNDERGROUND_COUNT, _cursor = _span(_cursor, 6)
DRAW_COUNT, _cursor = _span(_cursor, NUM_CARDS + 1)
OBSERVATION_DIM = _cursor

Observation = NDArray[np.uint8]
ActionMask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    """One public decision state and its exact legal action mapping."""

    observation: Observation
    action_mask: ActionMask
    actions: dict[int, Action]


def _require_london_decision(game: GameSession) -> None:
    if game.map is not LONDON:
        raise GameError("the RL codec currently supports only the London map")
    if game.status != "playing" or game.pending is None:
        raise GameError("the RL codec requires a playing state with a pending card")


def _write_observation(game: GameSession, observation: Observation) -> None:
    _require_london_decision(game)
    observation.fill(0)

    for color_index, color in enumerate(COLORS):
        line = game.lines[color]
        edge_offset = LINE_EDGES.start + color_index * NUM_EDGES
        station_offset = LINE_STATIONS.start + color_index * NUM_STATIONS
        for edge_id in line.edges:
            observation[edge_offset + edge_id] = 1
        for station_id in line.stations:
            observation[station_offset + station_id] = 1

    for station_id in game.line.leaves(game.map):
        observation[ACTIVE_LEAVES.start + station_id] = 1
    for card_id in game.remaining_card_ids():
        observation[REMAINING_CARDS.start + card_id] = 1

    pending = game.pending
    if pending is None:
        raise RuntimeError("pending card disappeared while encoding")
    for card_id in pending.card_ids:
        observation[CURRENT_CARDS.start + card_id] = 1
    symbol_index = (
        len(SYMBOLS)
        if pending.wild
        else _SYMBOL_INDEX[pending.target_symbol]
    )
    observation[TARGET_SYMBOL.start + symbol_index] = 1
    observation[EVENT_FLAGS] = (
        int(pending.wild),
        int(pending.source_any),
        int(pending.final_card),
    )

    for position, color in enumerate(game.order):
        observation[
            COLOR_ORDER.start + position * NUM_COLORS + _COLOR_INDEX[color]
        ] = 1
    observation[ROUND_INDEX.start + game.round_index] = 1
    observation[ACTIVE_COLOR.start + _COLOR_INDEX[game.color]] = 1
    observation[UNDERGROUND_COUNT.start + game.underground_count] = 1
    observation[DRAW_COUNT.start + game.draw_count] = 1


def encode_observation_into(
    game: GameSession,
    observation: Observation,
) -> None:
    """Encode one observation into a caller-owned reusable buffer."""

    if observation.shape != (OBSERVATION_DIM,) or observation.dtype != np.uint8:
        raise ValueError("observation buffer has an incompatible shape or dtype")
    _write_observation(game, observation)


def encode_observation(game: GameSession) -> Observation:
    """Encode all public Markov information without the hidden deck order."""

    observation = np.empty(OBSERVATION_DIM, dtype=np.uint8)
    _write_observation(game, observation)
    return observation


def encode_decision_into(
    game: GameSession,
    observation: Observation,
    action_mask: ActionMask,
) -> dict[int, Action]:
    """Encode a decision into reusable buffers and return legal edge actions."""

    if observation.shape != (OBSERVATION_DIM,) or observation.dtype != np.uint8:
        raise ValueError("observation buffer has an incompatible shape or dtype")
    if action_mask.shape != (ACTION_COUNT,) or action_mask.dtype != np.bool_:
        raise ValueError("action-mask buffer has an incompatible shape or dtype")
    _write_observation(game, observation)
    action_mask.fill(False)
    action_mask[PASS_ACTION_INDEX] = True
    actions: dict[int, Action] = {}
    for action in game.legal_actions():
        if action.edge_id in actions:
            raise RuntimeError("one edge unexpectedly has two legal orientations")
        actions[action.edge_id] = action
        action_mask[action.edge_id] = True
    return actions


def encode_decision(game: GameSession) -> EncodedDecision:
    """Encode a state while generating legal actions exactly once."""

    observation = np.empty(OBSERVATION_DIM, dtype=np.uint8)
    mask = np.empty(ACTION_COUNT, dtype=np.bool_)
    actions = encode_decision_into(game, observation, mask)
    return EncodedDecision(observation, mask, actions)


def observation_feature_counts() -> dict[str, int]:
    """Return stable feature-group sizes for checkpoints and diagnostics."""

    return {
        "line_edges": LINE_EDGES.stop - LINE_EDGES.start,
        "line_stations": LINE_STATIONS.stop - LINE_STATIONS.start,
        "active_leaves": ACTIVE_LEAVES.stop - ACTIVE_LEAVES.start,
        "remaining_cards": REMAINING_CARDS.stop - REMAINING_CARDS.start,
        "current_cards": CURRENT_CARDS.stop - CURRENT_CARDS.start,
        "target_symbol": TARGET_SYMBOL.stop - TARGET_SYMBOL.start,
        "event_flags": EVENT_FLAGS.stop - EVENT_FLAGS.start,
        "color_order": COLOR_ORDER.stop - COLOR_ORDER.start,
        "round_index": ROUND_INDEX.stop - ROUND_INDEX.start,
        "active_color": ACTIVE_COLOR.stop - ACTIVE_COLOR.start,
        "underground_count": UNDERGROUND_COUNT.stop - UNDERGROUND_COUNT.start,
        "draw_count": DRAW_COUNT.stop - DRAW_COUNT.start,
    }
