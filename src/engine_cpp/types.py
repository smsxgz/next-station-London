"""Python value objects exposed by the native engine adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .london import LONDON, LondonMap, SYMBOLS


TOURIST_TRACK = (0, 1, 2, 4, 6, 8, 11, 14, 17, 21, 25)
INTERCHANGE_POINTS = {2: 2, 3: 5, 4: 9}
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


def _build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    next_id = 0
    for underground, prefix in ((True, "underground"), (False, "street")):
        for symbol in SYMBOLS:
            cards.append(
                Card(next_id, f"{prefix}-{symbol}", symbol, underground)
            )
            next_id += 1
        cards.append(Card(next_id, f"{prefix}-joker", None, underground))
        next_id += 1
        if not underground:
            cards.append(
                Card(next_id, "street-switch", None, False, switch=True)
            )
            next_id += 1
    return tuple(cards)


DECK = _build_deck()
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
        if not self.edges:
            self._leaf_stations = {self.start}

    def _cached_leaves(self, game_map: LondonMap) -> set[int]:
        leaves = self._leaf_stations
        if leaves is None:
            degree: Counter[int] = Counter()
            for edge_id in self.edges:
                edge = game_map.edge(edge_id)
                degree[edge.u] += 1
                degree[edge.v] += 1
            leaves = {
                station_id
                for station_id in self.stations
                if degree[station_id] <= 1
            }
            self._leaf_stations = leaves
        return leaves

    def leaves(self, game_map: LondonMap = LONDON) -> set[int]:
        return set(self._cached_leaves(game_map))


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


class GameError(ValueError):
    """A user action that violates the native game rules."""
