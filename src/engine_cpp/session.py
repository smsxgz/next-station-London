"""Python object interface backed by the native London engine."""

from __future__ import annotations

import ctypes
from typing import Any

from ._native import (
    NativeGameAction,
    NativeGameOptions,
    NativeGameSnapshot,
    NativePublicState,
    _native,
    native_legal_edge_mask,
)
from .types import (
    Action,
    FinalScore,
    GameError,
    LineScore,
    LineState,
    MoveRecord,
    PendingEvent,
    PENCIL_POWERS,
    SHARED_OBJECTIVES,
)
from .london import COLORS, LONDON, SYMBOLS, LondonMap


_COLOR_TO_INDEX = {color: index for index, color in enumerate(COLORS)}
_SYMBOL_TO_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}
_INDEX_TO_SYMBOL = (*SYMBOLS, "central")
_POWER_TO_INDEX = {power: index for index, power in enumerate(PENCIL_POWERS)}
_STARTS = {
    color: next(
        station.id
        for station in LONDON.stations
        if station.departure_color == color
    )
    for color in COLORS
}
_MASK64 = (1 << 64) - 1


def _edge_mask(snapshot: NativeGameSnapshot, color_index: int) -> int:
    return sum(
        int(snapshot.state.line_edge_words[color_index][word]) << (word * 64)
        for word in range(3)
    )


def _ids_from_mask(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(mask.bit_length()) if mask & (1 << index))


def _make_public_state(
    *,
    order: tuple[str, ...],
    line_station_masks: tuple[int, ...],
    line_edge_masks: tuple[int, ...],
    remaining_mask: int,
    round_index: int,
    underground_count: int,
    draw_count: int,
    terminated: bool,
) -> NativePublicState:
    if len(order) != len(COLORS) or set(order) != set(COLORS):
        raise GameError(f"order must contain each color once: {COLORS}")
    if len(line_station_masks) != len(COLORS) or len(line_edge_masks) != len(COLORS):
        raise GameError("public state must contain one station and edge mask per color")
    state = NativePublicState()
    for color_index in range(len(COLORS)):
        state.line_station_masks[color_index] = int(line_station_masks[color_index])
        edge_mask = int(line_edge_masks[color_index])
        for word in range(3):
            state.line_edge_words[color_index][word] = (
                edge_mask >> (word * 64)
            ) & _MASK64
    for index, color in enumerate(order):
        state.order[index] = _COLOR_TO_INDEX[color]
    state.remaining_mask = remaining_mask
    state.round_index = round_index
    state.underground_count = underground_count
    state.draw_count = draw_count
    state.terminated = int(terminated)
    return state


def legal_edge_mask(
    *,
    order: tuple[str, ...],
    line_station_masks: tuple[int, ...],
    line_edge_masks: tuple[int, ...],
    remaining_mask: int,
    round_index: int,
    underground_count: int,
    draw_count: int,
    target_symbol: str | None,
    wild: bool,
    source_any: bool,
) -> int:
    """Return standard-game legal edges using the native rule implementation."""

    if target_symbol is not None and target_symbol not in _SYMBOL_TO_INDEX:
        raise GameError(f"unknown target symbol: {target_symbol}")
    state = _make_public_state(
        order=order,
        line_station_masks=line_station_masks,
        line_edge_masks=line_edge_masks,
        remaining_mask=remaining_mask,
        round_index=round_index,
        underground_count=underground_count,
        draw_count=draw_count,
        terminated=False,
    )
    return native_legal_edge_mask(
        state,
        target_symbol=(
            -1 if target_symbol is None else _SYMBOL_TO_INDEX[target_symbol]
        ),
        wild=wild,
        source_any=source_any,
    )


class GameSession:
    """A London game whose mutable rules state is owned entirely by C++."""

    map = LONDON

    def __init__(
        self,
        order: tuple[str, ...] | list[str] | None = None,
        seed: int | None = None,
        game_map: LondonMap = LONDON,
        advanced: bool = False,
        shared_objectives_enabled: bool | None = None,
        pencil_powers_enabled: bool | None = None,
        objective_cards: tuple[str, ...] | list[str] | None = None,
        power_assignments: dict[str, str] | None = None,
    ) -> None:
        if game_map is not LONDON:
            raise GameError("the C++ engine currently supports only the London map")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise GameError("seed must be an integer or null")
        if not isinstance(advanced, bool):
            raise GameError("advanced must be a boolean")
        for label, value in (
            ("shared_objectives_enabled", shared_objectives_enabled),
            ("pencil_powers_enabled", pencil_powers_enabled),
        ):
            if value is not None and not isinstance(value, bool):
                raise GameError(f"{label} must be a boolean or null")

        objectives_enabled = (
            advanced
            if shared_objectives_enabled is None
            else shared_objectives_enabled
        )
        powers_enabled = (
            advanced if pencil_powers_enabled is None else pencil_powers_enabled
        )
        if objective_cards is not None:
            if shared_objectives_enabled is False:
                raise GameError(
                    "objective_cards cannot be supplied when objectives are disabled"
                )
            objectives_enabled = True
        if power_assignments is not None:
            if pencil_powers_enabled is False:
                raise GameError(
                    "power_assignments cannot be supplied when powers are disabled"
                )
            powers_enabled = True

        options = NativeGameOptions()
        if seed is not None:
            options.seed = seed & _MASK64
            options.has_seed = 1
        if order is not None:
            validated_order = self._validate_order(tuple(order))
            options.has_order = 1
            for index, color in enumerate(validated_order):
                options.order[index] = _COLOR_TO_INDEX[color]
        options.shared_objectives_enabled = int(objectives_enabled)
        options.pencil_powers_enabled = int(powers_enabled)
        if objective_cards is not None:
            selected = tuple(objective_cards)
            if len(selected) != 2 or len(set(selected)) != 2:
                raise GameError("objective_cards must contain two distinct cards")
            if any(item not in SHARED_OBJECTIVES for item in selected):
                raise GameError("objective_cards contains an unknown objective")
            options.objective_count = 2
            for index, objective in enumerate(selected):
                options.objective_cards[index] = SHARED_OBJECTIVES.index(objective)
        if power_assignments is not None:
            assignments = dict(power_assignments)
            if set(assignments) != set(COLORS):
                raise GameError("power_assignments must contain every color")
            if set(assignments.values()) != set(PENCIL_POWERS):
                raise GameError("power_assignments must use every power once")
            options.has_power_assignments = 1
            for color_index, color in enumerate(COLORS):
                options.power_assignments[color_index] = _POWER_TO_INDEX[
                    assignments[color]
                ]

        self.seed = seed
        self._record_history = True
        self.last_move: MoveRecord | None = None
        self.move_history: list[MoveRecord] = []
        self._turn_actions: tuple[Action, ...] = ()
        handle = ctypes.c_void_p()
        native = _native()
        native.check(
            native.library.ns_game_create_configured(options, ctypes.byref(handle))
        )
        self._handle = handle
        self._refresh()

    @staticmethod
    def _validate_order(order: tuple[str, ...]) -> tuple[str, ...]:
        if len(order) != len(COLORS) or set(order) != set(COLORS):
            raise GameError(f"order must contain each color once: {COLORS}")
        return order

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or not handle.value:
            return
        try:
            _native().library.ns_game_destroy(handle)
        except Exception:
            pass
        self._handle = ctypes.c_void_p()

    def _refresh(self) -> None:
        snapshot = NativeGameSnapshot()
        native = _native()
        native.check(native.library.ns_game_export(self._handle, snapshot))
        self._snapshot = snapshot
        state = snapshot.state
        self.order = tuple(COLORS[int(state.order[index])] for index in range(4))
        self.round_index = int(state.round_index)
        self.underground_count = int(state.underground_count)
        self.draw_count = int(state.draw_count)
        self.status = "finished" if state.terminated else "playing"
        self.shared_objectives_enabled = bool(snapshot.shared_objectives_enabled)
        self.pencil_powers_enabled = bool(snapshot.pencil_powers_enabled)
        self.advanced = (
            self.shared_objectives_enabled or self.pencil_powers_enabled
        )
        self.shared_objectives = (
            tuple(
                SHARED_OBJECTIVES[int(snapshot.objective_cards[index])]
                for index in range(2)
            )
            if self.shared_objectives_enabled
            else ()
        )
        self.shared_objective_mask = int(snapshot.shared_objective_mask)
        self.pencil_powers = (
            {
                color: PENCIL_POWERS[int(snapshot.power_assignments[index])]
                for index, color in enumerate(COLORS)
            }
            if self.pencil_powers_enabled
            else {}
        )
        self.pending = self._pending_from_snapshot(snapshot)
        self._lines_cache: dict[str, LineState] | None = None
        self._board_edges_cache: set[int] | None = None
        self._legal_cache: tuple[Action, ...] | None = None
        self._reward_cache: dict[Action, tuple[int, int, int, int, int]] = {}

    @staticmethod
    def _pending_from_snapshot(
        snapshot: NativeGameSnapshot,
    ) -> PendingEvent | None:
        if not snapshot.has_pending:
            return None
        count = int(snapshot.pending_card_count)
        target_index = int(snapshot.pending_target_symbol)
        target_symbol = None if target_index < 0 else _INDEX_TO_SYMBOL[target_index]
        return PendingEvent(
            card_ids=tuple(int(snapshot.pending_card_ids[index]) for index in range(count)),
            target_symbol=target_symbol,
            wild=bool(snapshot.pending_wild),
            source_any=bool(snapshot.pending_source_any),
            final_card=bool(snapshot.pending_final_card),
        )

    @property
    def color(self) -> str:
        return self.order[self.round_index]

    @property
    def lines(self) -> dict[str, LineState]:
        cached = self._lines_cache
        if cached is not None:
            return cached
        result: dict[str, LineState] = {}
        for color_index, color in enumerate(COLORS):
            station_mask = int(self._snapshot.state.line_station_masks[color_index])
            edge_mask = _edge_mask(self._snapshot, color_index)
            line = LineState(
                color=color,
                start=_STARTS[color],
                stations=set(_ids_from_mask(station_mask)),
                edges=set(_ids_from_mask(edge_mask)),
            )
            line.station_mask = station_mask
            line.edge_mask = edge_mask
            line._leaf_stations = set(
                _ids_from_mask(int(self._snapshot.line_leaf_masks[color_index]))
            )
            result[color] = line
        self._lines_cache = result
        return result

    @property
    def line(self) -> LineState:
        return self.lines[self.color]

    @property
    def remaining(self) -> set[int]:
        return set(self.remaining_card_ids())

    @property
    def remaining_card_mask(self) -> int:
        return int(self._snapshot.state.remaining_mask)

    def remaining_card_ids(self) -> tuple[int, ...]:
        return _ids_from_mask(self.remaining_card_mask)

    @property
    def board_edge_mask(self) -> int:
        return _edge_mask(self._snapshot, 0) | _edge_mask(self._snapshot, 1) \
            | _edge_mask(self._snapshot, 2) | _edge_mask(self._snapshot, 3)

    @property
    def board_edges(self) -> set[int]:
        cached = self._board_edges_cache
        if cached is None:
            cached = set(_ids_from_mask(self.board_edge_mask))
            self._board_edges_cache = cached
        return cached

    @property
    def active_power(self) -> str | None:
        return self.pencil_powers.get(self.color)

    @property
    def double_section_pending(self) -> bool:
        return bool(self._snapshot.double_section_pending)

    @property
    def double_target_symbol(self) -> str | None:
        if not self.double_section_pending:
            return None
        index = int(self._snapshot.double_target_symbol)
        return None if index < 0 else _INDEX_TO_SYMBOL[index]

    @property
    def used_power_mask(self) -> int:
        return int(self._snapshot.used_power_mask)

    def power_used(self, power: str) -> bool:
        if power not in _POWER_TO_INDEX:
            raise GameError(f"unknown pencil power: {power}")
        return bool(self.used_power_mask & (1 << _POWER_TO_INDEX[power]))

    @property
    def completed_objectives(self) -> tuple[str, ...]:
        mask = int(self._snapshot.completed_objective_mask)
        return tuple(
            objective
            for objective in self.shared_objectives
            if mask & (1 << SHARED_OBJECTIVES.index(objective))
        )

    @property
    def tourist_visits(self) -> int:
        return int(self._snapshot.score_summary[1])

    @property
    def round_scores(self) -> list[LineScore]:
        return [
            LineScore(*map(int, self._snapshot.round_scores[index]))
            for index in range(int(self._snapshot.round_score_count))
        ]

    def reset(self) -> None:
        native = _native()
        native.check(native.library.ns_game_reset(self._handle))
        self.last_move = None
        self.move_history = []
        self._turn_actions = ()
        self._refresh()

    def draw(self) -> PendingEvent:
        native = _native()
        native.check(native.library.ns_game_draw(self._handle))
        self._refresh()
        if self.pending is None:
            raise RuntimeError("native draw did not create a pending event")
        return self.pending

    def draw_known_cards(
        self,
        card_ids: tuple[int, ...] | list[int],
    ) -> PendingEvent:
        selected = tuple(card_ids)
        values = (ctypes.c_uint8 * len(selected))(*selected)
        native = _native()
        native.check(
            native.library.ns_game_draw_known(
                self._handle,
                values,
                len(selected),
            )
        )
        self._refresh()
        if self.pending is None:
            raise RuntimeError("native known draw did not create a pending event")
        return self.pending

    def _load_legal_actions(self) -> tuple[Action, ...]:
        if self._legal_cache is not None:
            return self._legal_cache
        if self.pending is None:
            self._legal_cache = ()
            return ()
        native = _native()
        count = ctypes.c_int32()
        native.check(
            native.library.ns_game_legal_actions(
                self._handle,
                None,
                0,
                ctypes.byref(count),
            )
        )
        if count.value == 0:
            self._legal_cache = ()
            return ()
        values = (NativeGameAction * count.value)()
        native.check(
            native.library.ns_game_legal_actions(
                self._handle,
                values,
                count.value,
                ctypes.byref(count),
            )
        )
        actions: list[Action] = []
        rewards: dict[Action, tuple[int, int, int, int, int]] = {}
        for value in values:
            power_index = int(value.power)
            action = Action(
                edge_id=int(value.edge_id),
                source=int(value.source),
                target=int(value.target),
                power=None if power_index < 0 else PENCIL_POWERS[power_index],
            )
            actions.append(action)
            rewards[action] = tuple(map(int, value.reward_components))
        self._legal_cache = tuple(actions)
        self._reward_cache = rewards
        return self._legal_cache

    def legal_actions(self) -> tuple[Action, ...]:
        return self._load_legal_actions()

    def score_delta_for_legal_action(
        self,
        action: Action,
    ) -> tuple[int, int, int, int, int]:
        self._load_legal_actions()
        try:
            return self._reward_cache[action]
        except KeyError as exc:
            raise GameError("the action must be legal for the pending card") from exc

    def _record_completed_turn(
        self,
        event: PendingEvent,
        color: str,
        round_number: int,
        actions: tuple[Action, ...],
    ) -> None:
        if not self._record_history:
            return
        move = MoveRecord(
            round_number=round_number,
            color=color,
            event=event,
            action=actions[0] if actions else None,
            second_action=actions[1] if len(actions) == 2 else None,
        )
        self.last_move = move
        self.move_history.append(move)

    def apply_legal_action(self, chosen: Action | None) -> None:
        if self.status != "playing" or self.pending is None:
            raise GameError("cannot resolve an action in the current state")
        event = self.pending
        color = self.color
        round_number = self.round_index + 1
        was_second_phase = self.double_section_pending
        native_action: NativeGameAction | None = None
        if chosen is not None:
            native_action = NativeGameAction()
            native_action.edge_id = chosen.edge_id
            native_action.source = chosen.source
            native_action.target = chosen.target
            native_action.power = (
                -1 if chosen.power is None else _POWER_TO_INDEX[chosen.power]
            )
        native = _native()
        native.check(
            native.library.ns_game_apply_action(
                self._handle,
                native_action,
            )
        )
        if was_second_phase:
            actions = self._turn_actions
            if chosen is not None:
                actions = (*actions, chosen)
        else:
            actions = () if chosen is None else (chosen,)
        self._refresh()
        if self.pending is not None and self.double_section_pending:
            self._turn_actions = actions
            return
        self._record_completed_turn(event, color, round_number, actions)
        self._turn_actions = ()

    def act(
        self,
        edge_id: int | None = None,
        source: int | None = None,
        power: str | None = None,
    ) -> None:
        if self.status != "playing":
            raise GameError("the game is finished")
        if self.pending is None:
            raise GameError("draw a card first")
        chosen: Action | None = None
        if edge_id is not None:
            for action in self.legal_actions():
                if (
                    action.edge_id == edge_id
                    and (source is None or action.source == source)
                    and action.power == power
                ):
                    chosen = action
                    break
            else:
                raise GameError("that section is not legal for the current card")
        self.apply_legal_action(chosen)

    def partial_score_components(self) -> tuple[int, int, int, int, int, int]:
        return tuple(map(int, self._snapshot.partial_components))

    def score_summary(self) -> FinalScore:
        return FinalScore(*map(int, self._snapshot.score_summary))

    def final_score(self) -> FinalScore | None:
        if self.status != "finished":
            return None
        return self.score_summary()

    def _public_clone(self) -> GameSession:
        handle = ctypes.c_void_p()
        native = _native()
        native.check(native.library.ns_game_clone(self._handle, ctypes.byref(handle)))
        child = object.__new__(type(self))
        child.seed = self.seed
        child._handle = handle
        child._record_history = False
        child.last_move = None
        child.move_history = []
        child._turn_actions = ()
        child._refresh()
        return child

    def copy_public_state(self) -> GameSession:
        return self._public_clone()

    def copy_for_search_action(self) -> GameSession:
        return self._public_clone()

    def copy_for_search_draw(self) -> GameSession:
        return self._public_clone()

    def _serialize_native(self) -> bytes:
        native = _native()
        size = ctypes.c_int32()
        native.check(
            native.library.ns_game_serialize(
                self._handle,
                None,
                0,
                ctypes.byref(size),
            )
        )
        data = (ctypes.c_uint8 * size.value)()
        native.check(
            native.library.ns_game_serialize(
                self._handle,
                data,
                size.value,
                ctypes.byref(size),
            )
        )
        return bytes(data)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "native": self._serialize_native(),
            "seed": self.seed,
            "record_history": self._record_history,
            "last_move": self.last_move,
            "move_history": self.move_history,
            "turn_actions": self._turn_actions,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        raw = bytes(state["native"])
        data = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        handle = ctypes.c_void_p()
        native = _native()
        native.check(
            native.library.ns_game_deserialize(
                data,
                len(raw),
                ctypes.byref(handle),
            )
        )
        old_handle = getattr(self, "_handle", None)
        if old_handle is not None and old_handle.value:
            native.library.ns_game_destroy(old_handle)
        self._handle = handle
        self.seed = state["seed"]
        self._record_history = bool(state["record_history"])
        self.last_move = state["last_move"]
        self.move_history = list(state["move_history"])
        self._turn_actions = tuple(state["turn_actions"])
        self._refresh()

    @classmethod
    def from_public_state(
        cls,
        *,
        order: tuple[str, ...],
        line_station_masks: tuple[int, ...],
        line_edge_masks: tuple[int, ...],
        remaining_mask: int,
        round_index: int,
        underground_count: int,
        draw_count: int,
        terminated: bool = False,
    ) -> GameSession:
        snapshot = NativeGameSnapshot()
        validated_order = cls._validate_order(tuple(order))
        snapshot.state = _make_public_state(
            order=validated_order,
            line_station_masks=line_station_masks,
            line_edge_masks=line_edge_masks,
            remaining_mask=remaining_mask,
            round_index=round_index,
            underground_count=underground_count,
            draw_count=draw_count,
            terminated=terminated,
        )
        for index in range(4):
            snapshot.power_assignments[index] = -1
        handle = ctypes.c_void_p()
        native = _native()
        native.check(
            native.library.ns_game_create_from_snapshot(
                snapshot,
                ctypes.byref(handle),
            )
        )
        child = object.__new__(cls)
        child.seed = None
        child._handle = handle
        child._record_history = False
        child.last_move = None
        child.move_history = []
        child._turn_actions = ()
        child._refresh()
        return child
