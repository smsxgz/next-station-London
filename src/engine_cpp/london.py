"""Read-only Python view of the native London map."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from ._native import NativeEdgeInfo, NativeStationInfo, _native


SYMBOLS = ("circle", "triangle", "square", "pentagon")
COLORS = ("purple", "blue", "pink", "green")
_STATION_SYMBOLS = (*SYMBOLS, "central")


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
    edge_district_masks: tuple[int, ...] = field(default=(), repr=False, compare=False)
    district_count: int = field(default=0, repr=False, compare=False)
    district_names: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def station(self, station_id: int) -> Station:
        return self.stations[station_id]

    def edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]


def _load_london_map() -> LondonMap:
    native = _native()
    library = native.library
    district_names: list[str] = []
    for district_id in range(int(library.ns_district_count())):
        raw_name = library.ns_district_name(district_id)
        if raw_name is None:
            native.check(1)
        district_names.append(raw_name.decode("utf-8"))
    district_names_tuple = tuple(district_names)

    stations: list[Station] = []
    station_district_indices: list[int] = []
    for station_id in range(int(library.ns_station_count())):
        info = NativeStationInfo()
        native.check(library.ns_station_get(station_id, ctypes.byref(info)))
        if int(info.id) != station_id:
            raise RuntimeError("native station metadata is not ordered by id")
        symbol_index = int(info.symbol)
        district_index = int(info.district)
        departure_index = int(info.departure_color)
        stations.append(
            Station(
                id=station_id,
                x=int(info.x),
                y=int(info.y),
                symbol=_STATION_SYMBOLS[symbol_index],
                district=district_names_tuple[district_index],
                tourist=bool(info.tourist),
                departure_color=(
                    None if departure_index < 0 else COLORS[departure_index]
                ),
            )
        )
        station_district_indices.append(district_index)
    stations_tuple = tuple(stations)

    edges: list[Edge] = []
    edge_district_masks: list[int] = []
    conflict_masks: list[int] = []
    adjacency: list[list[int]] = [[] for _ in stations_tuple]
    oriented_adjacency: list[list[tuple[int, int]]] = [[] for _ in stations_tuple]
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_id in range(int(library.ns_edge_count())):
        info = NativeEdgeInfo()
        native.check(library.ns_edge_get(edge_id, ctypes.byref(info)))
        if int(info.id) != edge_id:
            raise RuntimeError("native edge metadata is not ordered by id")
        u = int(info.u)
        v = int(info.v)
        district_mask = int(info.district_mask)
        edge = Edge(
            id=edge_id,
            u=u,
            v=v,
            crosses_thames=bool(info.crosses_thames),
            districts=tuple(
                name
                for district_index, name in enumerate(district_names_tuple)
                if district_mask & (1 << district_index)
            ),
        )
        edges.append(edge)
        edge_district_masks.append(district_mask)
        conflict_masks.append(
            sum(int(info.conflict_words[word]) << (word * 64) for word in range(3))
        )
        adjacency[u].append(edge_id)
        adjacency[v].append(edge_id)
        oriented_adjacency[u].append((edge_id, v))
        oriented_adjacency[v].append((edge_id, u))
        edge_lookup[(min(u, v), max(u, v))] = edge_id

    return LondonMap(
        stations=stations_tuple,
        edges=tuple(edges),
        adjacency=tuple(tuple(items) for items in adjacency),
        conflict_masks=tuple(conflict_masks),
        edge_lookup=edge_lookup,
        oriented_adjacency=tuple(tuple(items) for items in oriented_adjacency),
        station_district_indices=tuple(station_district_indices),
        edge_district_masks=tuple(edge_district_masks),
        district_count=len(district_names_tuple),
        district_names=district_names_tuple,
    )


LONDON = _load_london_map()
