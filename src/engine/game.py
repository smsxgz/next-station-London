"""Rules and mutable game state for Next Station: London."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random

from .london import COLORS, LONDON, SYMBOLS, LondonMap


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
_OBJECTIVE_INDEX = {
    objective: index for index, objective in enumerate(SHARED_OBJECTIVES)
}
_OBJECTIVE_SEED_SALT = 0x9E3779B97F4A7C15
_POWER_SEED_SALT = 0xD1B54A32D192ED03
_INTERCHANGE_TRACK = tuple(
    INTERCHANGE_POINTS.get(count, 0)
    for count in range(len(COLORS) + 1)
)


@dataclass(frozen=True, slots=True)
class Card:
    id: int
    name: str
    symbol: str | None
    underground: bool
    switch: bool = False


def build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    next_id = 0
    for underground, prefix in ((True, "underground"), (False, "street")):
        for symbol in SYMBOLS:
            cards.append(Card(next_id, f"{prefix}-{symbol}", symbol, underground))
            next_id += 1
        cards.append(Card(next_id, f"{prefix}-joker", None, underground))
        next_id += 1
        if not underground:
            cards.append(Card(next_id, "street-switch", None, False, switch=True))
            next_id += 1
    return tuple(cards)


DECK = build_deck()
DECK_BY_ID = {card.id: card for card in DECK}
_FULL_DECK_MASK = (1 << len(DECK)) - 1
_CARD_IDS_BY_MASK = tuple(
    tuple(card_id for card_id in range(len(DECK)) if mask & (1 << card_id))
    for mask in range(1 << len(DECK))
)


def _district_data(
    game_map: LondonMap,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if (
        game_map.district_count > 0
        and len(game_map.station_district_indices) == len(game_map.stations)
        and len(game_map.edge_district_masks) == len(game_map.edges)
    ):
        return (
            game_map.station_district_indices,
            game_map.edge_district_masks,
            game_map.district_count,
        )

    names = tuple(
        sorted(
            {station.district for station in game_map.stations}
            | {
                district
                for edge in game_map.edges
                for district in edge.districts
            }
        )
    )
    indices = {district: index for index, district in enumerate(names)}
    station_indices = tuple(
        indices[station.district] for station in game_map.stations
    )
    edge_masks = tuple(
        sum(
            1 << indices[district]
            for district in (
                set(edge.districts)
                | {
                    game_map.station(edge.u).district,
                    game_map.station(edge.v).district,
                }
            )
        )
        for edge in game_map.edges
    )
    return station_indices, edge_masks, len(names)


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


@dataclass(slots=True)
class _LineMetrics:
    district_mask: int
    station_counts: list[int]
    max_stations: int
    route: int
    thames_crossings: int = 0
    tourist_visits: int = 0


class _PublicStateRng:
    """Reject accidental sampling when a clone has no hidden deck order."""

    @staticmethod
    def shuffle(values: list[int]) -> None:
        return None

    @staticmethod
    def choice(values: list[int]) -> int:
        raise RuntimeError("search attempted to sample a hidden card")


_PUBLIC_STATE_RNG = _PublicStateRng()


class GameError(ValueError):
    """A user action that violates the construction rules."""


class GameSession:
    """A solo game, reproducible whenever an explicit seed is supplied."""

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
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
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
        if (objectives_enabled or powers_enabled) and game_map is not LONDON:
            raise GameError("advanced rules currently require the London map")
        self.map = game_map
        (
            self._station_district_indices,
            self._edge_district_masks,
            self._district_count,
        ) = _district_data(game_map)
        self.seed = seed
        self.rng = random.Random(seed)
        if order is None:
            shuffled_order = list(COLORS)
            self.rng.shuffle(shuffled_order)
            self.order = tuple(shuffled_order)
        else:
            self.order = self._validate_order(tuple(order))
        self.shared_objectives_enabled = objectives_enabled
        self.pencil_powers_enabled = powers_enabled
        self.advanced = objectives_enabled or powers_enabled
        self._all_district_mask = (1 << self._district_count) - 1
        self._tourist_station_mask = sum(
            1 << station.id for station in self.map.stations if station.tourist
        )
        self._central_station_mask = sum(
            1 << station.id
            for station in self.map.stations
            if station.district == "middle_central"
        )
        if objectives_enabled:
            if objective_cards is None:
                objective_rng = random.Random(
                    None if seed is None else seed ^ _OBJECTIVE_SEED_SALT
                )
                self.shared_objectives = tuple(
                    objective_rng.sample(SHARED_OBJECTIVES, 2)
                )
            else:
                selected = tuple(objective_cards)
                if len(selected) != 2 or len(set(selected)) != 2:
                    raise GameError("objective_cards must contain two distinct cards")
                if any(item not in SHARED_OBJECTIVES for item in selected):
                    raise GameError("objective_cards contains an unknown objective")
                self.shared_objectives = selected
        else:
            self.shared_objectives = ()
        if powers_enabled:
            if power_assignments is None:
                power_rng = random.Random(
                    None if seed is None else seed ^ _POWER_SEED_SALT
                )
                powers = list(PENCIL_POWERS)
                power_rng.shuffle(powers)
                self.pencil_powers = dict(zip(COLORS, powers))
            else:
                assignments = dict(power_assignments)
                if set(assignments) != set(COLORS):
                    raise GameError("power_assignments must contain every color")
                if set(assignments.values()) != set(PENCIL_POWERS):
                    raise GameError("power_assignments must use every power once")
                self.pencil_powers = assignments
        else:
            self.pencil_powers = {}
        self.shared_objective_mask = sum(
            1 << _OBJECTIVE_INDEX[objective]
            for objective in self.shared_objectives
        )
        self._record_history = True
        self.reset()

    @staticmethod
    def _validate_order(order: tuple[str, ...]) -> tuple[str, ...]:
        if len(order) != len(COLORS) or set(order) != set(COLORS):
            raise GameError(f"order must contain each color once: {COLORS}")
        return order

    def reset(self) -> None:
        self.round_index = 0
        self._used_power_mask = 0
        self._double_section_pending = False
        self._double_target_symbol: str | None = None
        self._turn_actions: tuple[Action, ...] = ()
        self.lines: dict[str, LineState] = {}
        self._line_metrics: dict[str, _LineMetrics] = {}
        self._network_station_mask = 0
        self._network_district_mask = 0
        for color in COLORS:
            starts = [s.id for s in self.map.stations if s.departure_color == color]
            if len(starts) != 1:
                raise GameError(f"expected one departure station for {color}")
            start = starts[0]
            self.lines[color] = LineState(color=color, start=start)
            district_index = self._station_district_indices[start]
            station_counts = [0] * self._district_count
            station_counts[district_index] = 1
            self._line_metrics[color] = _LineMetrics(
                district_mask=1 << district_index,
                station_counts=station_counts,
                max_stations=1,
                route=1,
                tourist_visits=int(self.map.stations[start].tourist),
            )
            self._network_station_mask |= 1 << start
            self._network_district_mask |= 1 << district_index
        self.board_edges: set[int] = set()
        self._board_mask_value = 0
        self.round_scores: list[LineScore] = []
        self.tourist_visits = 0
        self._route_total = len(COLORS)
        self._thames_total = 0
        self._partial_tourist_visits = 0
        self._partial_tourist_points = 0
        self._lines_per_station = [0] * len(self.map.stations)
        self._interchange_counts = [0] * (len(COLORS) + 1)
        for line in self.lines.values():
            station_id = line.start
            before = self._lines_per_station[station_id]
            if before:
                self._interchange_counts[before] -= 1
            after = before + 1
            self._lines_per_station[station_id] = after
            self._interchange_counts[after] += 1
            self._partial_tourist_visits += int(
                self.map.station(station_id).tourist
            )
        self._partial_tourist_points = TOURIST_TRACK[
            min(self._partial_tourist_visits, len(TOURIST_TRACK) - 1)
        ]
        self._interchange_total = sum(
            self._interchange_counts[count] * points
            for count, points in INTERCHANGE_POINTS.items()
        )
        self._interchange_station_total = sum(self._interchange_counts[2:])
        self._completed_objective_mask = self._achieved_objective_mask(
            network_station_mask=self._network_station_mask,
            network_district_mask=self._network_district_mask,
            interchange_stations=self._interchange_station_total,
            thames_crossings=0,
        ) & self.shared_objective_mask
        self.remaining: set[int] = set()
        self._remaining_mask = 0
        self.underground_count = 0
        self.draw_count = 0
        self.pending: PendingEvent | None = None
        self.last_move: MoveRecord | None = None
        self.move_history: list[MoveRecord] = []
        self.status = "playing"
        self._start_round()

    @staticmethod
    def _copy_line(line: LineState) -> LineState:
        child = object.__new__(LineState)
        child.__dict__ = line.__dict__.copy()
        child.stations = set(line.stations)
        child.edges = set(line.edges)
        if line._leaf_stations is not None:
            child._leaf_stations = set(line._leaf_stations)
        return child

    @staticmethod
    def _copy_metrics(metrics: _LineMetrics) -> _LineMetrics:
        return _LineMetrics(
            district_mask=metrics.district_mask,
            station_counts=list(metrics.station_counts),
            max_stations=metrics.max_stations,
            route=metrics.route,
            thames_crossings=metrics.thames_crossings,
            tourist_visits=metrics.tourist_visits,
        )

    def _public_copy_base(self) -> GameSession:
        child = object.__new__(GameSession)
        child.__dict__ = self.__dict__.copy()
        child.rng = _PUBLIC_STATE_RNG
        child.deck_order = []
        child.last_move = None
        child.move_history = []
        child._record_history = False
        return child

    def copy_public_state(self) -> GameSession:
        """Return an independent clone containing no hidden deck order."""

        child = self._public_copy_base()
        child.lines = {
            color: self._copy_line(line)
            for color, line in self.lines.items()
        }
        child._line_metrics = {
            color: self._copy_metrics(metrics)
            for color, metrics in self._line_metrics.items()
        }
        child.board_edges = set(self.board_edges)
        child.round_scores = list(self.round_scores)
        child.remaining = set(self.remaining)
        child._lines_per_station = list(self._lines_per_station)
        child._interchange_counts = list(self._interchange_counts)
        return child

    def copy_for_search_action(self) -> GameSession:
        """Copy only state that resolving the pending action can mutate."""

        child = self._public_copy_base()
        child.lines = dict(self.lines)
        child.lines[self.color] = self._copy_line(self.line)
        child._line_metrics = dict(self._line_metrics)
        child._line_metrics[self.color] = self._copy_metrics(
            self._line_metrics[self.color]
        )
        child.board_edges = set(self.board_edges)
        child.round_scores = list(self.round_scores)
        child._lines_per_station = list(self._lines_per_station)
        child._interchange_counts = list(self._interchange_counts)
        return child

    def copy_for_search_draw(self) -> GameSession:
        """Copy only state that drawing the next public event can mutate."""

        child = self._public_copy_base()
        child.remaining = set(self.remaining)
        return child

    @property
    def color(self) -> str:
        return self.order[self.round_index]

    @property
    def line(self) -> LineState:
        return self.lines[self.color]

    @property
    def active_power(self) -> str | None:
        return self.pencil_powers.get(self.color)

    @property
    def double_section_pending(self) -> bool:
        return self._double_section_pending

    @property
    def double_target_symbol(self) -> str | None:
        return self._double_target_symbol

    @property
    def used_power_mask(self) -> int:
        return self._used_power_mask

    def power_used(self, power: str) -> bool:
        if power not in _POWER_INDEX:
            raise GameError(f"unknown pencil power: {power}")
        return bool(self._used_power_mask & (1 << _POWER_INDEX[power]))

    @property
    def completed_objectives(self) -> tuple[str, ...]:
        return tuple(
            objective
            for objective in self.shared_objectives
            if self._completed_objective_mask
            & (1 << _OBJECTIVE_INDEX[objective])
        )

    def _power_available(self, power: str) -> bool:
        return self.active_power == power and not self.power_used(power)

    def _mark_power_used(self, power: str) -> None:
        if not self._power_available(power):
            raise GameError("that pencil power is not available")
        self._used_power_mask |= 1 << _POWER_INDEX[power]

    def _start_round(self) -> None:
        self.remaining = set(range(len(DECK)))
        self._remaining_mask = _FULL_DECK_MASK
        self.deck_order = list(self.remaining)
        self.rng.shuffle(self.deck_order)

    @property
    def remaining_card_mask(self) -> int:
        """Bit mask of cards not yet consumed in the current round."""

        return self._remaining_mask

    @property
    def board_edge_mask(self) -> int:
        """Bit mask of every section already occupied on the map."""

        return self._board_mask_value

    def remaining_card_ids(self) -> tuple[int, ...]:
        """Card ids not yet consumed in the current round."""

        return _CARD_IDS_BY_MASK[self._remaining_mask]

    def _draw_one(self) -> Card:
        if not self.remaining:
            raise GameError("the card pile is empty")
        # A shuffled order gives deterministic replay for a supplied seed.
        while self.deck_order and self.deck_order[0] not in self.remaining:
            self.deck_order.pop(0)
        if not self.deck_order:
            choices = sorted(self.remaining)
            card_id = self.rng.choice(choices)
        else:
            card_id = self.deck_order.pop(0)
        self.remaining.remove(card_id)
        self._remaining_mask &= ~(1 << card_id)
        card = DECK_BY_ID[card_id]
        self.draw_count += 1
        if card.underground:
            self.underground_count += 1
        return card

    def draw(self) -> PendingEvent:
        if self.status != "playing":
            raise GameError("the game is finished")
        if self.pending is not None:
            raise GameError("resolve the current card first")

        first = self._draw_one()
        if first.switch:
            # The switch always consumes the immediately following card and
            # permits starting from any station already on the current line.
            # In the base game this has no extra effect during the first two
            # turns because the line does not yet contain an internal station.
            second = self._draw_one()
            event = PendingEvent(
                card_ids=(first.id, second.id),
                target_symbol=second.symbol,
                wild=second.symbol is None,
                source_any=True,
                final_card=self.underground_count >= 5,
            )
        else:
            event = PendingEvent(
                card_ids=(first.id,),
                target_symbol=first.symbol,
                wild=first.symbol is None,
                source_any=False,
                final_card=self.underground_count >= 5,
            )
        self.pending = event
        return event

    def draw_known_cards(self, card_ids: tuple[int, ...] | list[int]) -> PendingEvent:
        """Reveal one explicitly selected public event from the remaining pile.

        Search policies use this to enumerate or sample chance outcomes without
        consulting the hidden order of a real game.
        """

        selected = tuple(card_ids)
        if not selected or any(card_id not in DECK_BY_ID for card_id in selected):
            raise GameError("known draw contains an invalid card id")
        expected = 2 if DECK_BY_ID[selected[0]].switch else 1
        if len(selected) != expected or len(set(selected)) != len(selected):
            raise GameError("known draw does not form one public card event")
        if any(card_id not in self.remaining for card_id in selected):
            raise GameError("known draw contains a card outside the remaining pile")
        self.deck_order = list(selected)
        return self.draw()

    def _target_matches(self, target_id: int, event: PendingEvent) -> bool:
        station = self.map.station(target_id)
        return (
            station.symbol == "central"
            or event.wild
            or station.symbol == event.target_symbol
        )

    def _achieved_objective_mask(
        self,
        *,
        network_station_mask: int,
        network_district_mask: int,
        interchange_stations: int,
        thames_crossings: int,
    ) -> int:
        mask = 0
        if interchange_stations >= 8:
            mask |= 1 << _OBJECTIVE_INDEX["eight-interchanges"]
        if network_district_mask == self._all_district_mask:
            mask |= 1 << _OBJECTIVE_INDEX["all-districts"]
        if (
            network_station_mask & self._tourist_station_mask
            == self._tourist_station_mask
        ):
            mask |= 1 << _OBJECTIVE_INDEX["all-tourist-sites"]
        if (
            network_station_mask & self._central_station_mask
            == self._central_station_mask
        ):
            mask |= 1 << _OBJECTIVE_INDEX["all-central-stations"]
        if thames_crossings >= 6:
            mask |= 1 << _OBJECTIVE_INDEX["six-thames-crossings"]
        return mask

    def _edge_is_open(self, edge_id: int, board_mask: int | None = None) -> bool:
        if board_mask is None:
            board_mask = self._board_mask_value
        occupied = board_mask & (1 << edge_id)
        return not occupied and (self.map.conflict_masks[edge_id] & board_mask) == 0

    def _section_actions(
        self,
        *,
        target_symbol: str | None,
        wild: bool,
        source_any: bool,
        power: str | None,
    ) -> tuple[Action, ...]:
        line = self.line
        sources = line.stations if source_any else line._cached_leaves(self.map)
        board_mask = self._board_mask_value
        station_mask = line.station_mask
        conflict_masks = self.map.conflict_masks
        stations = self.map.stations
        oriented_adjacency = self.map.oriented_adjacency
        actions: list[Action] = []
        for source in sorted(sources):
            if oriented_adjacency:
                adjacent = oriented_adjacency[source]
            else:
                adjacent = tuple(
                    (
                        edge_id,
                        self.map.edges[edge_id].v
                        if self.map.edges[edge_id].u == source
                        else self.map.edges[edge_id].u,
                    )
                    for edge_id in self.map.adjacency[source]
                )
            for edge_id, target in adjacent:
                if (
                    board_mask & (1 << edge_id)
                    or conflict_masks[edge_id] & board_mask
                ):
                    continue
                if station_mask & (1 << target):
                    continue
                station = stations[target]
                if (
                    station.symbol != "central"
                    and not wild
                    and station.symbol != target_symbol
                ):
                    continue
                actions.append(
                    Action(
                        edge_id=edge_id,
                        source=source,
                        target=target,
                        power=power,
                    )
                )
        # A given edge can only have one legal orientation because the target
        # must be a new station in the current color.
        return tuple(actions)

    def legal_actions(self) -> tuple[Action, ...]:
        event = self.pending
        if event is None:
            return ()

        if self._double_section_pending:
            # BGA applies a revealed Railroad Switch to the optional second
            # section even when its first-section privilege was ignored early
            # in the round.
            switch_revealed = any(
                DECK_BY_ID[card_id].switch for card_id in event.card_ids
            )
            return self._section_actions(
                target_symbol=self._double_target_symbol,
                wild=False,
                source_any=event.source_any or switch_revealed,
                power=PENCIL_POWER_DOUBLE,
            )

        base = self._section_actions(
            target_symbol=event.target_symbol,
            wild=event.wild,
            source_any=event.source_any,
            power=None,
        )
        power = self.active_power
        if power not in (PENCIL_POWER_WILD, PENCIL_POWER_SWITCH):
            return base
        if not self._power_available(power):
            return base
        if power == PENCIL_POWER_WILD:
            powered = self._section_actions(
                target_symbol=None,
                wild=True,
                source_any=event.source_any,
                power=power,
            )
        else:
            # The pencil card says to treat the station card as accompanied by
            # a Railroad Switch, including the first-two-draw restriction.
            if self.draw_count <= 2:
                return base
            powered = self._section_actions(
                target_symbol=event.target_symbol,
                wild=event.wild,
                source_any=True,
                power=power,
            )

        base_geometry = {
            (action.edge_id, action.source, action.target) for action in base
        }
        useful_powered = tuple(
            action
            for action in powered
            if (action.edge_id, action.source, action.target) not in base_geometry
        )
        return (*base, *useful_powered)

    def _find_action(
        self,
        edge_id: int,
        source: int | None = None,
        power: str | None = None,
    ) -> Action:
        for action in self.legal_actions():
            if (
                action.edge_id == edge_id
                and (source is None or action.source == source)
                and action.power == power
            ):
                return action
        raise GameError("that section is not legal for the current card")

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
            chosen = self._find_action(edge_id, source, power)
        self.apply_legal_action(chosen)

    def _circle_route_bonus(self) -> int:
        if not self.advanced:
            return 0
        return sum(
            self._line_metrics[color].district_mask.bit_count()
            for color, power in self.pencil_powers.items()
            if power == PENCIL_POWER_CIRCLE
        )

    def partial_score_components(self) -> tuple[int, int, int, int, int, int]:
        """Return all dense, final-score-equivalent score components."""

        return (
            self._route_total + self._circle_route_bonus(),
            self._thames_total,
            self._partial_tourist_visits,
            self._partial_tourist_points,
            self._interchange_total,
            self._completed_objective_mask.bit_count() * 10,
        )

    def score_delta_for_legal_action(
        self,
        action: Action,
    ) -> tuple[int, int, int, int, int]:
        """Return the exact dense score delta for an already legal action."""

        metrics = self._line_metrics[self.order[self.round_index]]
        edge = self.map.edges[action.edge_id]
        target = self.map.stations[action.target]
        districts_after = (
            metrics.district_mask | self._edge_district_masks[action.edge_id]
        ).bit_count()
        target_district = self._station_district_indices[action.target]
        max_stations_after = max(
            metrics.max_stations,
            metrics.station_counts[target_district] + 1,
        )
        route = districts_after * max_stations_after - metrics.route
        if self.active_power == PENCIL_POWER_CIRCLE:
            route += districts_after - metrics.district_mask.bit_count()
        thames = int(edge.crosses_thames) * 2

        tourist = 0
        if target.tourist:
            visits_after = self._partial_tourist_visits + 1
            tourist = (
                TOURIST_TRACK[min(visits_after, len(TOURIST_TRACK) - 1)]
                - self._partial_tourist_points
            )

        lines_before = self._lines_per_station[action.target]
        lines_after = lines_before + 1
        interchange = (
            _INTERCHANGE_TRACK[lines_after]
            - _INTERCHANGE_TRACK[lines_before]
        )
        network_station_mask = self._network_station_mask | (1 << action.target)
        network_district_mask = self._network_district_mask | (
            1 << target_district
        )
        interchange_stations = self._interchange_station_total + int(
            lines_before == 1
        )
        thames_crossings = self._thames_total // 2 + int(edge.crosses_thames)
        achieved = self._achieved_objective_mask(
            network_station_mask=network_station_mask,
            network_district_mask=network_district_mask,
            interchange_stations=interchange_stations,
            thames_crossings=thames_crossings,
        ) & self.shared_objective_mask
        objective = (
            achieved & ~self._completed_objective_mask
        ).bit_count() * 10
        return route, thames, tourist, interchange, objective

    def _update_score_caches(self, action: Action) -> None:
        _route, thames, tourist, interchange, _objective = (
            self.score_delta_for_legal_action(action)
        )
        metrics = self._line_metrics[self.color]
        edge = self.map.edge(action.edge_id)
        target = self.map.station(action.target)
        target_district = self._station_district_indices[action.target]
        districts_after = (
            metrics.district_mask | self._edge_district_masks[action.edge_id]
        ).bit_count()
        max_stations_after = max(
            metrics.max_stations,
            metrics.station_counts[target_district] + 1,
        )
        base_route_delta = (
            districts_after * max_stations_after - metrics.route
        )

        metrics.district_mask |= self._edge_district_masks[action.edge_id]
        metrics.station_counts[target_district] += 1
        metrics.max_stations = max_stations_after
        metrics.route += base_route_delta
        metrics.thames_crossings += int(edge.crosses_thames)
        self._route_total += base_route_delta
        self._thames_total += thames

        if target.tourist:
            metrics.tourist_visits += 1
            self._partial_tourist_visits += 1
            self._partial_tourist_points += tourist

        lines_before = self._lines_per_station[action.target]
        if lines_before:
            self._interchange_counts[lines_before] -= 1
        lines_after = lines_before + 1
        self._lines_per_station[action.target] = lines_after
        self._interchange_counts[lines_after] += 1
        self._interchange_total += interchange
        self._interchange_station_total += int(lines_before == 1)

        self._network_station_mask |= 1 << action.target
        self._network_district_mask |= 1 << target_district
        self._completed_objective_mask = self._achieved_objective_mask(
            network_station_mask=self._network_station_mask,
            network_district_mask=self._network_district_mask,
            interchange_stations=self._interchange_station_total,
            thames_crossings=self._thames_total // 2,
        ) & self.shared_objective_mask

    def _apply_section(self, action: Action) -> None:
        line = self.line
        leaves = line._cached_leaves(self.map)
        self._update_score_caches(action)
        if line.edges:
            leaves.discard(action.source)
        leaves.add(action.target)
        line.edges.add(action.edge_id)
        line.stations.add(action.target)
        line.edge_mask |= 1 << action.edge_id
        line.station_mask |= 1 << action.target
        self.board_edges.add(action.edge_id)
        self._board_mask_value |= 1 << action.edge_id

    def _record_turn(
        self,
        event: PendingEvent,
        active_color: str,
    ) -> None:
        if not self._record_history:
            return
        move = MoveRecord(
            round_number=self.round_index + 1,
            color=active_color,
            event=event,
            action=self._turn_actions[0] if self._turn_actions else None,
            second_action=(
                self._turn_actions[1] if len(self._turn_actions) == 2 else None
            ),
        )
        self.last_move = move
        self.move_history.append(move)

    def _complete_turn(
        self,
        event: PendingEvent,
        active_color: str,
    ) -> None:
        self._record_turn(event, active_color)
        self._turn_actions = ()
        self._double_section_pending = False
        self._double_target_symbol = None
        self.pending = None
        if event.final_card:
            self._finish_round()

    def apply_legal_action(self, chosen: Action | None) -> None:
        """Commit an action already produced for this exact state.

        This avoids regenerating legal actions in search code. Public callers
        that have only an edge id should use :meth:`act`, which validates it.
        """

        if self.status != "playing" or self.pending is None:
            raise GameError("cannot resolve an action in the current state")
        event = self.pending
        active_color = self.color
        if self._double_section_pending:
            if chosen is not None:
                if chosen.power != PENCIL_POWER_DOUBLE:
                    raise GameError("the optional second section must use its power")
                self._apply_section(chosen)
                self._mark_power_used(PENCIL_POWER_DOUBLE)
                self._turn_actions = (*self._turn_actions, chosen)
            # Stop keeps the power available for a later turn.
            self._complete_turn(event, active_color)
            return

        self._turn_actions = ()
        if chosen is None:
            self._complete_turn(event, active_color)
            return

        if chosen.power == PENCIL_POWER_DOUBLE:
            raise GameError("a second-section action is not legal in the main phase")
        self._apply_section(chosen)
        self._turn_actions = (chosen,)
        if chosen.power is not None:
            self._mark_power_used(chosen.power)

        if self._power_available(PENCIL_POWER_DOUBLE):
            self._double_target_symbol = (
                self.map.station(chosen.target).symbol
                if event.wild
                else event.target_symbol
            )
            self._double_section_pending = True
            if self.legal_actions():
                return
            self._double_section_pending = False
            self._double_target_symbol = None

        self._complete_turn(event, active_color)

    def _finish_round(self) -> None:
        metrics = self._line_metrics[self.order[self.round_index]]
        circle_bonus = 0
        max_stations = metrics.max_stations
        if self._power_available(PENCIL_POWER_CIRCLE):
            circle_bonus = metrics.district_mask.bit_count()
            max_stations += 1
            self._mark_power_used(PENCIL_POWER_CIRCLE)
        thames = metrics.thames_crossings * 2
        breakdown = LineScore(
            districts=metrics.district_mask.bit_count(),
            max_stations=max_stations,
            thames_crossings=metrics.thames_crossings,
            route=metrics.route + circle_bonus,
            thames=thames,
            total=metrics.route + circle_bonus + thames,
        )
        self.round_scores.append(breakdown)
        self.tourist_visits += metrics.tourist_visits
        if self.round_index == len(self.order) - 1:
            self.status = "finished"
            return
        self.round_index += 1
        self.underground_count = 0
        self.draw_count = 0
        self.pending = None
        self._start_round()

    def score_summary(self) -> FinalScore:
        """Return the score represented by the current score sheet.

        Route and tourist scores are recorded only when their color round has
        ended. Interchanges, however, reflect all sections drawn so far.
        """

        line_total = sum(item.total for item in self.round_scores)
        tourist_index = min(self.tourist_visits, len(TOURIST_TRACK) - 1)
        tourist_bonus = TOURIST_TRACK[tourist_index]
        interchange_bonus = self._interchange_total
        objectives_completed = self._completed_objective_mask.bit_count()
        objective_bonus = objectives_completed * 10
        total = (
            line_total
            + tourist_bonus
            + interchange_bonus
            + objective_bonus
        )
        return FinalScore(
            line_total=line_total,
            tourist_visits=self.tourist_visits,
            tourist_bonus=tourist_bonus,
            two_line_stations=self._interchange_counts[2],
            three_line_stations=self._interchange_counts[3],
            four_line_stations=self._interchange_counts[4],
            interchange_bonus=interchange_bonus,
            objectives_completed=objectives_completed,
            objective_bonus=objective_bonus,
            total=total,
        )

    def final_score(self) -> FinalScore | None:
        if self.status != "finished":
            return None
        return self.score_summary()


def line_score(game_map: LondonMap, line: LineState) -> LineScore:
    """Score one constructed line according to the printed route formula.

    A departure station is an ordinary station on its line and therefore
    counts toward both the districts visited and the largest station count in
    one district, even when the player draws no section during the round.
    """

    districts = {game_map.station(station_id).district for station_id in line.stations}
    for edge_id in line.edges:
        edge = game_map.edge(edge_id)
        if edge.districts:
            districts.update(edge.districts)
        else:
            # Keep hand-built/toy maps compatible with the original endpoint
            # representation when they do not provide edge metadata.
            districts.add(game_map.station(edge.u).district)
            districts.add(game_map.station(edge.v).district)
    per_district = Counter(
        game_map.station(station_id).district for station_id in line.stations
    )
    max_stations = max(per_district.values(), default=0)
    crossings = sum(game_map.edge(edge_id).crosses_thames for edge_id in line.edges)
    route = len(districts) * max_stations
    thames = crossings * 2
    return LineScore(
        districts=len(districts),
        max_stations=max_stations,
        thames_crossings=crossings,
        route=route,
        thames=thames,
        total=route + thames,
    )
