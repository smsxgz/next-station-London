"""Static Python view of the London map used by encoders and policies.

The printed map uses a ten by ten lattice.  A potential section connects two
stations on one of the eight compass directions when no other station lies
between them.  This reproduces the 53 stations and 155 grey potential lines
on the supplied official map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


SYMBOLS = ("circle", "triangle", "square", "pentagon")
COLORS = ("purple", "blue", "pink", "green")


@dataclass(frozen=True, slots=True)
class Station:
    id: int
    x: int
    y: int
    symbol: str
    district: str
    tourist: bool = False
    departure_color: str | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    id: int
    u: int
    v: int
    crosses_thames: bool
    districts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LondonMap:
    stations: tuple[Station, ...]
    edges: tuple[Edge, ...]
    adjacency: tuple[tuple[int, ...], ...]
    conflict_masks: tuple[int, ...]
    edge_lookup: dict[tuple[int, int], int]
    oriented_adjacency: tuple[tuple[tuple[int, int], ...], ...] = field(
        default=(), repr=False, compare=False
    )
    station_district_indices: tuple[int, ...] = field(
        default=(), repr=False, compare=False
    )
    edge_district_masks: tuple[int, ...] = field(
        default=(), repr=False, compare=False
    )
    district_count: int = field(default=0, repr=False, compare=False)

    def station(self, station_id: int) -> Station:
        return self.stations[station_id]

    def edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]


def district_for(x: int, y: int) -> str:
    """Return the district printed behind a lattice coordinate."""

    # The four inset corner districts each contain exactly one station.  The
    # stations on the outer column at y=2/7 belong to the adjacent main
    # district, as shown by the yellow boundaries on the official map.
    if (x, y) == (0, 0):
        return "northwest"
    if (x, y) == (9, 0):
        return "northeast"
    if (x, y) == (0, 9):
        return "southwest"
    if (x, y) == (9, 9):
        return "southeast"

    column = "west" if x <= 2 else "central" if x <= 6 else "east"
    row = "north" if y <= 2 else "middle" if y <= 6 else "south"
    return f"{row}_{column}"


def _district_at_point(x: float, y: float) -> str:
    """Classify an interior point using the printed yellow boundaries."""

    if x < 0.5 and y < 0.5:
        return "northwest"
    if x > 8.5 and y < 0.5:
        return "northeast"
    if x < 0.5 and y > 8.5:
        return "southwest"
    if x > 8.5 and y > 8.5:
        return "southeast"
    column = "west" if x < 2.5 else "central" if x < 6.5 else "east"
    row = "north" if y < 2.5 else "middle" if y < 6.5 else "south"
    return f"{row}_{column}"


def _edge_districts(first: Station, second: Station) -> tuple[str, ...]:
    """Return all districts touched by a straight candidate section."""

    dx = second.x - first.x
    dy = second.y - first.y
    parameters = [0.0, 1.0]
    for boundary in (0.5, 2.5, 6.5, 8.5):
        if dx:
            t = (boundary - first.x) / dx
            if 0.0 < t < 1.0:
                parameters.append(t)
        if dy:
            t = (boundary - first.y) / dy
            if 0.0 < t < 1.0:
                parameters.append(t)
    parameters = sorted(set(parameters))
    districts = {first.district, second.district}
    for left, right in zip(parameters, parameters[1:]):
        middle = (left + right) / 2.0
        districts.add(_district_at_point(first.x + dx * middle, first.y + dy * middle))
    return tuple(sorted(districts))


# (x, y, symbol, tourist, departure color)
_RAW_STATIONS: tuple[tuple[int, int, str, bool, str | None], ...] = (
    (0, 0, "pentagon", False, None),
    (1, 0, "triangle", False, None),
    (2, 0, "square", False, None),
    (4, 0, "triangle", False, None),
    (5, 0, "circle", False, None),
    (7, 0, "triangle", False, None),
    (9, 0, "circle", False, None),
    (1, 1, "pentagon", False, None),
    (3, 1, "square", False, None),
    (6, 1, "pentagon", True, None),
    (8, 1, "square", False, None),
    (9, 1, "pentagon", False, None),
    (0, 2, "circle", False, None),
    (3, 2, "triangle", False, "green"),
    (6, 2, "square", False, None),
    (9, 2, "triangle", False, None),
    (0, 3, "square", True, None),
    (2, 3, "pentagon", False, None),
    (4, 3, "triangle", False, None),
    (5, 3, "central", True, None),
    (6, 3, "circle", False, None),
    (7, 3, "circle", False, "pink"),
    (9, 3, "square", False, None),
    (1, 4, "triangle", False, None),
    (2, 4, "square", False, None),
    (4, 4, "pentagon", False, None),
    (5, 4, "square", False, None),
    (8, 4, "pentagon", False, None),
    (0, 5, "pentagon", False, None),
    (2, 5, "square", False, "purple"),
    (4, 5, "circle", False, None),
    (7, 5, "circle", False, None),
    (3, 6, "pentagon", False, None),
    (4, 6, "triangle", False, None),
    (6, 6, "square", False, None),
    (7, 6, "triangle", False, None),
    (9, 6, "triangle", True, None),
    (0, 7, "circle", False, None),
    (2, 7, "square", False, None),
    (3, 7, "circle", False, None),
    (5, 7, "pentagon", False, "blue"),
    (8, 7, "circle", False, None),
    (9, 7, "pentagon", False, None),
    (1, 8, "circle", False, None),
    (6, 8, "pentagon", False, None),
    (8, 8, "triangle", False, None),
    (0, 9, "triangle", False, None),
    (1, 9, "square", False, None),
    (3, 9, "pentagon", False, None),
    (4, 9, "circle", True, None),
    (5, 9, "triangle", False, None),
    (7, 9, "circle", False, None),
    (9, 9, "square", False, None),
)


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-9
    return (
        abs(_orientation(a, b, p)) <= eps
        and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    eps = 1e-9
    if ((o1 > eps) != (o2 > eps)) and ((o3 > eps) != (o4 > eps)):
        return True
    return any(
        abs(value) <= eps and point_on
        for value, point_on in (
            (o1, _on_segment(a, b, c)),
            (o2, _on_segment(a, b, d)),
            (o3, _on_segment(c, d, a)),
            (o4, _on_segment(c, d, b)),
        )
    )


_RIVER_CENTERLINE = (
    ((-1.0, 3.35), (2.05, 3.35)),
    ((2.05, 3.35), (3.65, 5.25)),
    ((3.65, 5.25), (5.05, 5.25)),
    ((5.05, 5.25), (6.45, 4.25)),
    ((6.45, 4.25), (10.0, 4.25)),
)


def _crosses_thames(a: tuple[int, int], b: tuple[int, int]) -> bool:
    p, q = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    return any(_segments_intersect(p, q, r, s) for r, s in _RIVER_CENTERLINE)


def _candidate_pairs(stations: tuple[Station, ...]) -> Iterable[tuple[int, int]]:
    coords = {(station.x, station.y): station.id for station in stations}
    for i, first in enumerate(stations):
        for second in stations[i + 1 :]:
            dx = second.x - first.x
            dy = second.y - first.y
            if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
                continue
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            x, y = first.x + sx, first.y + sy
            blocked = False
            while (x, y) != (second.x, second.y):
                if (x, y) in coords:
                    blocked = True
                    break
                x += sx
                y += sy
            if not blocked:
                yield first.id, second.id


def _edge_conflict(
    first: Edge,
    second: Edge,
    stations: tuple[Station, ...],
) -> bool:
    if first.u in (second.u, second.v) or first.v in (second.u, second.v):
        return False
    a = stations[first.u]
    b = stations[first.v]
    c = stations[second.u]
    d = stations[second.v]
    return _segments_intersect((a.x, a.y), (b.x, b.y), (c.x, c.y), (d.x, d.y))


def build_london_map() -> LondonMap:
    stations = tuple(
        Station(
            id=index,
            x=x,
            y=y,
            symbol=symbol,
            district=district_for(x, y),
            tourist=tourist,
            departure_color=departure_color,
        )
        for index, (x, y, symbol, tourist, departure_color) in enumerate(_RAW_STATIONS)
    )
    pairs = list(_candidate_pairs(stations))
    edges = tuple(
        Edge(
            id=index,
            u=u,
            v=v,
            crosses_thames=_crosses_thames(
                (stations[u].x, stations[u].y), (stations[v].x, stations[v].y)
            ),
            districts=_edge_districts(stations[u], stations[v]),
        )
        for index, (u, v) in enumerate(pairs)
    )
    adjacency: list[list[int]] = [[] for _ in stations]
    oriented_adjacency: list[list[tuple[int, int]]] = [[] for _ in stations]
    for edge in edges:
        adjacency[edge.u].append(edge.id)
        adjacency[edge.v].append(edge.id)
        oriented_adjacency[edge.u].append((edge.id, edge.v))
        oriented_adjacency[edge.v].append((edge.id, edge.u))
    conflicts = [0] * len(edges)
    for i, edge in enumerate(edges):
        for j in range(i):
            if _edge_conflict(edge, edges[j], stations):
                conflicts[i] |= 1 << j
                conflicts[j] |= 1 << i
    lookup = {(min(edge.u, edge.v), max(edge.u, edge.v)): edge.id for edge in edges}
    district_names = tuple(
        sorted(
            {station.district for station in stations}
            | {
                district
                for edge in edges
                for district in edge.districts
            }
        )
    )
    district_indices = {
        district: index for index, district in enumerate(district_names)
    }
    station_district_indices = tuple(
        district_indices[station.district] for station in stations
    )
    edge_district_masks = tuple(
        sum(
            1 << district_indices[district]
            for district in (
                set(edge.districts)
                | {stations[edge.u].district, stations[edge.v].district}
            )
        )
        for edge in edges
    )
    return LondonMap(
        stations=stations,
        edges=edges,
        adjacency=tuple(tuple(items) for items in adjacency),
        conflict_masks=tuple(conflicts),
        edge_lookup=lookup,
        oriented_adjacency=tuple(tuple(items) for items in oriented_adjacency),
        station_district_indices=station_district_indices,
        edge_district_masks=edge_district_masks,
        district_count=len(district_names),
    )


LONDON = build_london_map()


def station_id_at(x: int, y: int) -> int:
    for station in LONDON.stations:
        if station.x == x and station.y == y:
            return station.id
    raise KeyError((x, y))
