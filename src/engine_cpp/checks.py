"""Executable checks for the native engine's Python adapter."""

from __future__ import annotations

import pickle

from . import (
    COLORS,
    DECK,
    LONDON,
    PENCIL_POWER_CIRCLE,
    PENCIL_POWER_DOUBLE,
    PENCIL_POWER_SWITCH,
    PENCIL_POWER_WILD,
    Action,
    GameSession,
    legal_edge_mask,
)


_ORDER = COLORS
_OBJECTIVES = ("eight-interchanges", "six-thames-crossings")
_POWERS = {
    "purple": PENCIL_POWER_DOUBLE,
    "blue": PENCIL_POWER_WILD,
    "pink": PENCIL_POWER_SWITCH,
    "green": PENCIL_POWER_CIRCLE,
}
_ROUND_EVENTS = ((10, 5), (0,), (1,), (2,), (3,), (4,))


def _line_signature(game: GameSession) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            color,
            game.lines[color].start,
            game.lines[color].station_mask,
            game.lines[color].edge_mask,
            tuple(sorted(game.lines[color].leaves(game.map))),
        )
        for color in COLORS
    )


def _state_signature(game: GameSession) -> tuple[object, ...]:
    return (
        game.status,
        game.order,
        game.round_index,
        game.underground_count,
        game.draw_count,
        game.remaining_card_mask,
        game.board_edge_mask,
        game.pending,
        game.shared_objectives_enabled,
        game.pencil_powers_enabled,
        game.shared_objectives,
        tuple(sorted(game.pencil_powers.items())),
        game.used_power_mask,
        game.double_section_pending,
        game.double_target_symbol,
        game.completed_objectives,
        _line_signature(game),
        game.partial_score_components(),
        tuple(game.round_scores),
        game.score_summary(),
    )


def _assert_equal(left: object, right: object, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label} differs:\nleft={left!r}\nright={right!r}")


def _choose(actions: tuple[Action, ...], decision_index: int) -> Action | None:
    powered = tuple(action for action in actions if action.power is not None)
    candidates = powered or actions
    if not candidates:
        return None
    return candidates[decision_index % len(candidates)]


def _check_metadata() -> None:
    if len(LONDON.stations) != 53 or len(LONDON.edges) != 155:
        raise AssertionError("native London metadata has unexpected dimensions")
    if LONDON.district_count != 13 or len(LONDON.district_names) != 13:
        raise AssertionError("native London district metadata is incomplete")
    if len(DECK) != 11 or not DECK[-1].switch:
        raise AssertionError("native deck metadata is inconsistent")


def _check_legal_mask(game: GameSession) -> None:
    pending = game.pending
    if pending is None:
        raise AssertionError("legal-mask check requires a pending event")
    lines = game.lines
    expected = sum(1 << action.edge_id for action in game.legal_actions())
    actual = legal_edge_mask(
        order=game.order,
        line_station_masks=tuple(lines[color].station_mask for color in COLORS),
        line_edge_masks=tuple(lines[color].edge_mask for color in COLORS),
        remaining_mask=game.remaining_card_mask,
        round_index=game.round_index,
        underground_count=game.underground_count,
        draw_count=game.draw_count,
        target_symbol=pending.target_symbol,
        wild=pending.wild,
        source_any=pending.source_any,
    )
    _assert_equal(expected, actual, "native legal-edge mask")


def _check_seeded_legal_masks() -> None:
    decision = 0
    for seed in range(32):
        game = GameSession(seed=seed)
        while game.status == "playing":
            game.draw()
            while game.pending is not None:
                _check_legal_mask(game)
                actions = game.legal_actions()
                action = actions[(seed + decision) % len(actions)] if actions else None
                game.apply_legal_action(action)
                decision += 1


def _check_known_game(*, advanced: bool) -> None:
    kwargs: dict[str, object] = {
        "order": _ORDER,
        "seed": 31,
        "advanced": advanced,
    }
    if advanced:
        kwargs["objective_cards"] = _OBJECTIVES
        kwargs["power_assignments"] = _POWERS
    game = GameSession(**kwargs)
    initial_total = sum(
        game.partial_score_components()[index] for index in (0, 1, 3, 4, 5)
    )
    reward_total = 0
    decision_index = 0
    while game.status == "playing":
        for card_ids in _ROUND_EVENTS:
            if game.status == "finished":
                break
            game.draw_known_cards(card_ids)
            while game.pending is not None:
                if not advanced:
                    _check_legal_mask(game)
                actions = game.legal_actions()
                action = _choose(actions, decision_index)
                if action is not None:
                    reward_total += sum(
                        game.score_delta_for_legal_action(action)
                    )
                game.apply_legal_action(action)
                decision_index += 1

                public_copy = game.copy_public_state()
                _assert_equal(
                    _state_signature(game),
                    _state_signature(public_copy),
                    "public copy",
                )

    final = game.final_score()
    if final is None:
        raise AssertionError("known game did not finish")
    if initial_total + reward_total != final.total:
        raise AssertionError("dense rewards do not telescope to the final score")
    if advanced:
        unused = tuple(power for power in _POWERS.values() if not game.power_used(power))
        if unused:
            raise AssertionError(f"advanced powers were not exercised: {unused}")
        if final.objective_bonus != final.objectives_completed * 10:
            raise AssertionError("objective score summary is inconsistent")


def _check_pickle_continuation() -> None:
    original = GameSession(seed=7123, advanced=True)
    for _ in range(3):
        original.draw()
        original.apply_legal_action(next(iter(original.legal_actions()), None))
    restored = pickle.loads(pickle.dumps(original))
    _assert_equal(_state_signature(original), _state_signature(restored), "pickle")
    while original.status == "playing":
        _assert_equal(original.draw(), restored.draw(), "restored random draw")
        while original.pending is not None:
            actions = original.legal_actions()
            _assert_equal(actions, restored.legal_actions(), "restored legal actions")
            action = next(iter(actions), None)
            original.apply_legal_action(action)
            restored.apply_legal_action(action)
    _assert_equal(_state_signature(original), _state_signature(restored), "pickle final")


def main() -> None:
    _check_metadata()
    _check_seeded_legal_masks()
    _check_known_game(advanced=False)
    _check_known_game(advanced=True)
    _check_pickle_continuation()
    print("engine_cpp checks: ok")


if __name__ == "__main__":
    main()
