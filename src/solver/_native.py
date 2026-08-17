"""ctypes boundary for the native search kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache

from engine_cpp import GameError
from engine_cpp._native import _native as _engine_native


_MAX_ACTIONS = 156


class _NativeActionEstimate(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("edge_id", ctypes.c_int32),
        ("source", ctypes.c_int32),
        ("target", ctypes.c_int32),
        ("power", ctypes.c_int8),
        ("reward_components", ctypes.c_int32 * 5),
        ("value", ctypes.c_double),
        ("visits", ctypes.c_int64),
        ("standard_error", ctypes.c_double),
    )


class _NativeLookaheadStats(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("decision_nodes", ctypes.c_int64),
        ("chance_nodes", ctypes.c_int64),
        ("chance_outcomes", ctypes.c_int64),
        ("cache_hits", ctypes.c_int64),
    )


class _NativeMCTSStats(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("simulations", ctypes.c_int64),
        ("decision_nodes", ctypes.c_int64),
        ("tree_chance_samples", ctypes.c_int64),
        ("rollout_chance_samples", ctypes.c_int64),
        ("terminal_rollouts", ctypes.c_int64),
        ("rollout_decisions", ctypes.c_int64),
        ("tree_terminal_hits", ctypes.c_int64),
        ("max_tree_depth", ctypes.c_int32),
        ("mean_tree_depth", ctypes.c_double),
        ("elapsed_seconds", ctypes.c_double),
    )


@dataclass(frozen=True, slots=True)
class NativeActionEstimate:
    edge_id: int
    source: int
    target: int
    power: int
    reward_components: tuple[int, int, int, int, int]
    value: float
    visits: int
    standard_error: float


@dataclass(frozen=True, slots=True)
class NativeLookaheadStats:
    decision_nodes: int
    chance_nodes: int
    chance_outcomes: int
    cache_hits: int


@dataclass(frozen=True, slots=True)
class NativeMCTSStats:
    simulations: int
    decision_nodes: int
    tree_chance_samples: int
    rollout_chance_samples: int
    terminal_rollouts: int
    rollout_decisions: int
    tree_terminal_hits: int
    max_tree_depth: int
    mean_tree_depth: float
    elapsed_seconds: float


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    library = _engine_native().library
    estimate_pointer = ctypes.POINTER(_NativeActionEstimate)
    count_pointer = ctypes.POINTER(ctypes.c_int32)
    library.ns_solver_last_error.argtypes = ()
    library.ns_solver_last_error.restype = ctypes.c_char_p
    library.ns_solver_lookahead.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint8,
        estimate_pointer,
        ctypes.c_int32,
        count_pointer,
        ctypes.POINTER(_NativeLookaheadStats),
    )
    library.ns_solver_lookahead.restype = ctypes.c_int
    library.ns_solver_mcts.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_int32,
        estimate_pointer,
        ctypes.c_int32,
        count_pointer,
        ctypes.POINTER(_NativeMCTSStats),
    )
    library.ns_solver_mcts.restype = ctypes.c_int
    for name in (
        "ns_solver_action_estimate_size",
        "ns_solver_lookahead_stats_size",
        "ns_solver_mcts_stats_size",
    ):
        function = getattr(library, name)
        function.argtypes = ()
        function.restype = ctypes.c_int32
    sizes = (
        (
            "action estimate",
            _NativeActionEstimate,
            library.ns_solver_action_estimate_size,
        ),
        (
            "lookahead stats",
            _NativeLookaheadStats,
            library.ns_solver_lookahead_stats_size,
        ),
        ("MCTS stats", _NativeMCTSStats, library.ns_solver_mcts_stats_size),
    )
    for label, structure, native_size in sizes:
        actual = ctypes.sizeof(structure)
        expected = int(native_size())
        if actual != expected:
            raise RuntimeError(
                f"native solver {label} size mismatch: C++={expected}, ctypes={actual}"
            )
    return library


def _check(result: int) -> None:
    if result == 0:
        return
    raw = _library().ns_solver_last_error()
    detail = raw.decode("utf-8") if raw else "unknown native solver error"
    raise GameError(detail)


def _copy_estimates(
    values: ctypes.Array[_NativeActionEstimate],
    count: int,
) -> tuple[NativeActionEstimate, ...]:
    if not 1 <= count <= _MAX_ACTIONS:
        raise RuntimeError(f"native solver returned an invalid action count: {count}")
    return tuple(
        NativeActionEstimate(
            edge_id=int(values[index].edge_id),
            source=int(values[index].source),
            target=int(values[index].target),
            power=int(values[index].power),
            reward_components=tuple(
                int(values[index].reward_components[component])
                for component in range(5)
            ),
            value=float(values[index].value),
            visits=int(values[index].visits),
            standard_error=float(values[index].standard_error),
        )
        for index in range(count)
    )


def native_lookahead(
    handle: ctypes.c_void_p,
    *,
    depth: int,
    specialized_depth_two: bool,
) -> tuple[tuple[NativeActionEstimate, ...], NativeLookaheadStats]:
    values = (_NativeActionEstimate * _MAX_ACTIONS)()
    count = ctypes.c_int32()
    stats = _NativeLookaheadStats()
    _check(
        _library().ns_solver_lookahead(
            handle,
            depth,
            int(specialized_depth_two),
            values,
            _MAX_ACTIONS,
            ctypes.byref(count),
            ctypes.byref(stats),
        )
    )
    return _copy_estimates(values, int(count.value)), NativeLookaheadStats(
        decision_nodes=int(stats.decision_nodes),
        chance_nodes=int(stats.chance_nodes),
        chance_outcomes=int(stats.chance_outcomes),
        cache_hits=int(stats.cache_hits),
    )


def native_mcts(
    handle: ctypes.c_void_p,
    *,
    simulations: int,
    exploration: float,
    seed: int,
    rollout_policy: str = "greedy",
) -> tuple[tuple[NativeActionEstimate, ...], NativeMCTSStats]:
    try:
        rollout_policy_code = {
            "greedy": 0,
            "lookahead-2": 1,
            "simple-random": 2,
        }[rollout_policy]
    except KeyError as exc:
        raise ValueError(f"unknown MCTS rollout policy: {rollout_policy}") from exc
    values = (_NativeActionEstimate * _MAX_ACTIONS)()
    count = ctypes.c_int32()
    stats = _NativeMCTSStats()
    _check(
        _library().ns_solver_mcts(
            handle,
            simulations,
            exploration,
            seed,
            rollout_policy_code,
            values,
            _MAX_ACTIONS,
            ctypes.byref(count),
            ctypes.byref(stats),
        )
    )
    return _copy_estimates(values, int(count.value)), NativeMCTSStats(
        simulations=int(stats.simulations),
        decision_nodes=int(stats.decision_nodes),
        tree_chance_samples=int(stats.tree_chance_samples),
        rollout_chance_samples=int(stats.rollout_chance_samples),
        terminal_rollouts=int(stats.terminal_rollouts),
        rollout_decisions=int(stats.rollout_decisions),
        tree_terminal_hits=int(stats.tree_terminal_hits),
        max_tree_depth=int(stats.max_tree_depth),
        mean_tree_depth=float(stats.mean_tree_depth),
        elapsed_seconds=float(stats.elapsed_seconds),
    )
