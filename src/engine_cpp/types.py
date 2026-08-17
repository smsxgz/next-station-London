"""Python value objects exposed by the native engine adapter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from ._native import GameError, NativeCardInfo, _native
from .london import LONDON, LondonMap, SYMBOLS


SHARED_OBJECTIVES = (
    "eight-interchanges",
    "all-districts",
    "all-tourist-sites",
    "all-central-stations",
    "six-thames-crossings",
)
PENCIL_POWER_DOUBLE = "double-section"
PENCIL_POWER_WILD = "wild-card"
PENCIL_POWER_SWITCH = "railroad-switch"
PENCIL_POWER_CIRCLE = "circle-station"
PENCIL_POWERS = (
    PENCIL_POWER_DOUBLE,
    PENCIL_POWER_WILD,
    PENCIL_POWER_SWITCH,
    PENCIL_POWER_CIRCLE,
)
_POWER_INDEX = {power: index for index, power in enumerate(PENCIL_POWERS)}


@dataclass(frozen=True, slots=True)
class Card:
    id: int
    name: str
    symbol: str | None
    underground: bool
    switch: bool = False


def _load_deck() -> tuple[Card, ...]:
    native = _native()
    library = native.library
    cards: list[Card] = []
    for card_id in range(int(library.ns_card_count())):
        info = NativeCardInfo()
        native.check(library.ns_card_get(card_id, ctypes.byref(info)))
        if int(info.id) != card_id:
            raise RuntimeError("native card metadata is not ordered by id")
        symbol_index = int(info.symbol)
        symbol = None if symbol_index < 0 else SYMBOLS[symbol_index]
        underground = bool(info.underground)
        switch = bool(info.is_switch)
        prefix = "underground" if underground else "street"
        name = (
            "street-switch"
            if switch
            else f"{prefix}-{symbol if symbol is not None else 'joker'}"
        )
        cards.append(Card(card_id, name, symbol, underground, switch))
    return tuple(cards)


DECK = _load_deck()
DECK_BY_ID = {card.id: card for card in DECK}


@dataclass
class LineState:
    color: str
    start: int
    stations: set[int] = field(default_factory=set)
    edges: set[int] = field(default_factory=set)
    station_mask: int = field(init=False, repr=False, compare=False)
    edge_mask: int = field(init=False, repr=False, compare=False)
    _leaf_stations: set[int] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.stations.add(self.start)
        self.station_mask = sum(1 << station_id for station_id in self.stations)
        self.edge_mask = sum(1 << edge_id for edge_id in self.edges)

    def leaves(self, game_map: LondonMap = LONDON) -> set[int]:
        if game_map is not LONDON:
            raise GameError("the C++ engine currently supports only the London map")
        leaves = self._leaf_stations
        if leaves is None:
            raise RuntimeError("line leaf data was not supplied by the native engine")
        return set(leaves)


@dataclass(frozen=True, slots=True)
class Action:
    edge_id: int
    source: int
    target: int
    power: str | None = None

    def __post_init__(self) -> None:
        if self.power is not None and self.power not in PENCIL_POWERS:
            raise ValueError(f"unknown pencil power: {self.power}")

    @property
    def sort_key(self) -> tuple[int, ...]:
        return (
            -1 if self.power is None else _POWER_INDEX[self.power],
            self.edge_id,
            self.source,
            self.target,
        )


@dataclass(frozen=True, slots=True)
class PendingEvent:
    card_ids: tuple[int, ...]
    target_symbol: str | None
    wild: bool
    source_any: bool
    final_card: bool


@dataclass(frozen=True, slots=True)
class LineScore:
    districts: int
    max_stations: int
    thames_crossings: int
    route: int
    thames: int
    total: int


@dataclass(frozen=True, slots=True)
class MoveRecord:
    round_number: int
    color: str
    event: PendingEvent
    action: Action | None
    second_action: Action | None = None

    @property
    def actions(self) -> tuple[Action, ...]:
        if self.action is None:
            return ()
        if self.second_action is None:
            return (self.action,)
        return self.action, self.second_action


@dataclass(frozen=True, slots=True)
class FinalScore:
    line_total: int
    tourist_visits: int
    tourist_bonus: int
    two_line_stations: int
    three_line_stations: int
    four_line_stations: int
    interchange_bonus: int
    objectives_completed: int
    objective_bonus: int
    total: int
