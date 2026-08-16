"""Shared public-state helpers for search policies."""

from __future__ import annotations

import random
from itertools import permutations
from typing import Iterator

from engine_cpp import (
    COLORS,
    DECK_BY_ID,
    PENCIL_POWERS,
    SYMBOLS,
    GameError,
    GameSession,
)


_ORDER_CODES = {
    order: index for index, order in enumerate(permutations(COLORS))
}
_SYMBOL_CODES = {symbol: index for index, symbol in enumerate(SYMBOLS)}
_POWER_CODES = {power: index for index, power in enumerate(PENCIL_POWERS)}
_STATUS_CODES = {"playing": 0, "finished": 1}


def public_state_key(game: GameSession) -> tuple[int, ...]:
    """Return a hashable key that contains no hidden deck information."""

    pending = game.pending
    lines = game.lines
    purple = lines[COLORS[0]]
    blue = lines[COLORS[1]]
    pink = lines[COLORS[2]]
    green = lines[COLORS[3]]
    return (
        id(game.map),
        _STATUS_CODES[game.status],
        _ORDER_CODES[game.order],
        game.round_index,
        game.underground_count,
        game.draw_count,
        game.remaining_card_mask,
        int(game.shared_objectives_enabled),
        int(game.pencil_powers_enabled),
        game.shared_objective_mask,
        *(
            _POWER_CODES.get(game.pencil_powers.get(color), -1)
            for color in COLORS
        ),
        game.used_power_mask,
        int(game.double_section_pending),
        -1
        if game.double_target_symbol is None
        else _SYMBOL_CODES.get(game.double_target_symbol, len(_SYMBOL_CODES)),
        0
        if pending is None
        else sum(1 << card_id for card_id in pending.card_ids),
        -1
        if pending is None or pending.target_symbol is None
        else _SYMBOL_CODES[pending.target_symbol],
        0
        if pending is None
        else (
            int(pending.wild)
            | (int(pending.source_any) << 1)
            | (int(pending.final_card) << 2)
        ),
        game.board_edge_mask,
        purple.start,
        purple.station_mask,
        purple.edge_mask,
        blue.start,
        blue.station_mask,
        blue.edge_mask,
        pink.start,
        pink.station_mask,
        pink.edge_mask,
        green.start,
        green.station_mask,
        green.edge_mask,
    )


def sample_public_event(game: GameSession, rng: random.Random) -> None:
    """Sample the next visible card event without reading hidden deck order."""

    if game.status != "playing" or game.pending is not None:
        raise GameError("chance sampling requires no pending card")
    remaining = game.remaining_card_ids()
    if not remaining:
        raise GameError("cannot sample from an empty card pile")

    first_id = rng.choice(remaining)
    card_ids = [first_id]
    if DECK_BY_ID[first_id].switch:
        following = tuple(card_id for card_id in remaining if card_id != first_id)
        if not following:
            raise GameError("the switch card has no following card")
        card_ids.append(rng.choice(following))

    game.draw_known_cards(card_ids)


def public_event_successors(
    game: GameSession,
) -> Iterator[tuple[float, GameSession]]:
    """Enumerate the next visible card event and its exact probability."""

    if game.status != "playing" or game.pending is not None:
        raise GameError("chance expansion requires no pending card")
    remaining = game.remaining_card_ids()
    if not remaining:
        raise GameError("cannot expand an empty card pile")

    first_probability = 1.0 / len(remaining)
    for first_id in remaining:
        first = DECK_BY_ID[first_id]
        if not first.switch:
            child = game.copy_for_search_draw()
            child.draw_known_cards((first_id,))
            yield first_probability, child
            continue

        following = tuple(
            card_id for card_id in remaining if card_id != first_id
        )
        if not following:
            raise GameError("the switch card has no following card")
        probability = first_probability / len(following)
        for second_id in following:
            child = game.copy_for_search_draw()
            child.draw_known_cards((first_id, second_id))
            yield probability, child
