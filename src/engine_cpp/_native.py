"""Low-level ctypes bindings for the native engine."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys


class GameError(ValueError):
    """A user action or state that violates the native game rules."""


class NativePublicState(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("line_station_masks", ctypes.c_uint64 * 4),
        ("line_edge_words", (ctypes.c_uint64 * 3) * 4),
        ("remaining_mask", ctypes.c_uint16),
        ("order", ctypes.c_uint8 * 4),
        ("round_index", ctypes.c_uint8),
        ("underground_count", ctypes.c_uint8),
        ("draw_count", ctypes.c_uint8),
        ("terminated", ctypes.c_uint8),
    )


class NativeStationInfo(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("id", ctypes.c_int32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("symbol", ctypes.c_int8),
        ("district", ctypes.c_int8),
        ("tourist", ctypes.c_uint8),
        ("departure_color", ctypes.c_int8),
    )


class NativeEdgeInfo(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("id", ctypes.c_int32),
        ("u", ctypes.c_int32),
        ("v", ctypes.c_int32),
        ("crosses_thames", ctypes.c_uint8),
        ("district_mask", ctypes.c_uint16),
        ("conflict_words", ctypes.c_uint64 * 3),
    )


class NativeCardInfo(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("id", ctypes.c_int32),
        ("symbol", ctypes.c_int8),
        ("underground", ctypes.c_uint8),
        ("is_switch", ctypes.c_uint8),
    )


class NativeGameSnapshot(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("state", NativePublicState),
        ("line_leaf_masks", ctypes.c_uint64 * 4),
        ("shared_objectives_enabled", ctypes.c_uint8),
        ("pencil_powers_enabled", ctypes.c_uint8),
        ("objective_cards", ctypes.c_uint8 * 2),
        ("shared_objective_mask", ctypes.c_uint8),
        ("power_assignments", ctypes.c_int8 * 4),
        ("used_power_mask", ctypes.c_uint8),
        ("completed_objective_mask", ctypes.c_uint8),
        ("double_section_pending", ctypes.c_uint8),
        ("double_target_symbol", ctypes.c_int8),
        ("has_pending", ctypes.c_uint8),
        ("pending_card_ids", ctypes.c_uint8 * 2),
        ("pending_card_count", ctypes.c_uint8),
        ("pending_target_symbol", ctypes.c_int8),
        ("pending_wild", ctypes.c_uint8),
        ("pending_source_any", ctypes.c_uint8),
        ("pending_final_card", ctypes.c_uint8),
        ("partial_components", ctypes.c_int32 * 6),
        ("score_summary", ctypes.c_int32 * 10),
        ("round_score_count", ctypes.c_uint8),
        ("round_scores", (ctypes.c_int32 * 6) * 4),
    )


class NativeGameAction(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("edge_id", ctypes.c_int32),
        ("source", ctypes.c_int32),
        ("target", ctypes.c_int32),
        ("power", ctypes.c_int8),
        ("reward_components", ctypes.c_int32 * 5),
    )


class NativeGameOptions(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("seed", ctypes.c_uint64),
        ("has_seed", ctypes.c_uint8),
        ("has_order", ctypes.c_uint8),
        ("order", ctypes.c_uint8 * 4),
        ("shared_objectives_enabled", ctypes.c_uint8),
        ("pencil_powers_enabled", ctypes.c_uint8),
        ("objective_count", ctypes.c_uint8),
        ("objective_cards", ctypes.c_uint8 * 2),
        ("has_power_assignments", ctypes.c_uint8),
        ("power_assignments", ctypes.c_int8 * 4),
    )


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("next_station_engine_capi.dll",)
    if sys.platform == "darwin":
        return ("libnext_station_engine_capi.dylib",)
    return ("libnext_station_engine_capi.so",)


def native_library_path() -> Path:
    configured = os.environ.get("NEXT_STATION_NATIVE_LIBRARY")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"native engine library does not exist: {path}")
        return path

    root = Path(__file__).resolve().parents[2]
    directories = (
        Path(__file__).resolve().parent,
        root / "build" / "engine-cpp",
        root / "build" / "engine-cpp" / "Release",
    )
    candidates = tuple(
        directory / name for directory in directories for name in _library_names()
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError("native engine is not built; searched " + searched)


class _NativeLibrary:
    def __init__(self) -> None:
        self.path = native_library_path()
        self.library = ctypes.CDLL(str(self.path))
        handle = ctypes.c_void_p
        handle_pointer = ctypes.POINTER(handle)
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        int_pointer = ctypes.POINTER(ctypes.c_int32)
        word_pointer = ctypes.POINTER(ctypes.c_uint64)

        self.library.ns_last_error.argtypes = ()
        self.library.ns_last_error.restype = ctypes.c_char_p

        for name in (
            "ns_station_count",
            "ns_edge_count",
            "ns_card_count",
            "ns_district_count",
            "ns_public_state_size",
            "ns_station_info_size",
            "ns_edge_info_size",
            "ns_card_info_size",
            "ns_game_snapshot_size",
            "ns_game_action_size",
            "ns_game_options_size",
        ):
            function = getattr(self.library, name)
            function.argtypes = ()
            function.restype = ctypes.c_int32

        self.library.ns_station_get.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(NativeStationInfo),
        )
        self.library.ns_station_get.restype = ctypes.c_int
        self.library.ns_edge_get.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(NativeEdgeInfo),
        )
        self.library.ns_edge_get.restype = ctypes.c_int
        self.library.ns_card_get.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(NativeCardInfo),
        )
        self.library.ns_card_get.restype = ctypes.c_int
        self.library.ns_district_name.argtypes = (ctypes.c_int32,)
        self.library.ns_district_name.restype = ctypes.c_char_p
        self.library.ns_legal_edge_mask.argtypes = (
            ctypes.POINTER(NativePublicState),
            ctypes.c_int8,
            ctypes.c_uint8,
            ctypes.c_uint8,
            word_pointer,
        )
        self.library.ns_legal_edge_mask.restype = ctypes.c_int

        self.library.ns_game_create_configured.argtypes = (
            ctypes.POINTER(NativeGameOptions),
            handle_pointer,
        )
        self.library.ns_game_create_configured.restype = ctypes.c_int
        self.library.ns_game_create_from_snapshot.argtypes = (
            ctypes.POINTER(NativeGameSnapshot),
            handle_pointer,
        )
        self.library.ns_game_create_from_snapshot.restype = ctypes.c_int
        self.library.ns_game_clone.argtypes = (handle, handle_pointer)
        self.library.ns_game_clone.restype = ctypes.c_int
        self.library.ns_game_destroy.argtypes = (handle,)
        self.library.ns_game_destroy.restype = None
        self.library.ns_game_reset.argtypes = (handle,)
        self.library.ns_game_reset.restype = ctypes.c_int
        self.library.ns_game_export.argtypes = (
            handle,
            ctypes.POINTER(NativeGameSnapshot),
        )
        self.library.ns_game_export.restype = ctypes.c_int
        self.library.ns_game_draw.argtypes = (handle,)
        self.library.ns_game_draw.restype = ctypes.c_int
        self.library.ns_game_draw_known.argtypes = (
            handle,
            byte_pointer,
            ctypes.c_int32,
        )
        self.library.ns_game_draw_known.restype = ctypes.c_int
        self.library.ns_game_legal_actions.argtypes = (
            handle,
            ctypes.POINTER(NativeGameAction),
            ctypes.c_int32,
            int_pointer,
        )
        self.library.ns_game_legal_actions.restype = ctypes.c_int
        self.library.ns_game_apply_action.argtypes = (
            handle,
            ctypes.POINTER(NativeGameAction),
        )
        self.library.ns_game_apply_action.restype = ctypes.c_int
        self.library.ns_game_serialize.argtypes = (
            handle,
            byte_pointer,
            ctypes.c_int32,
            int_pointer,
        )
        self.library.ns_game_serialize.restype = ctypes.c_int
        self.library.ns_game_deserialize.argtypes = (
            byte_pointer,
            ctypes.c_int32,
            handle_pointer,
        )
        self.library.ns_game_deserialize.restype = ctypes.c_int

        sizes = (
            ("public state", NativePublicState, self.library.ns_public_state_size),
            ("station info", NativeStationInfo, self.library.ns_station_info_size),
            ("edge info", NativeEdgeInfo, self.library.ns_edge_info_size),
            ("card info", NativeCardInfo, self.library.ns_card_info_size),
            ("game snapshot", NativeGameSnapshot, self.library.ns_game_snapshot_size),
            ("game action", NativeGameAction, self.library.ns_game_action_size),
            ("game options", NativeGameOptions, self.library.ns_game_options_size),
        )
        for label, structure, native_size in sizes:
            actual = ctypes.sizeof(structure)
            expected = int(native_size())
            if actual != expected:
                raise RuntimeError(
                    f"native {label} size mismatch: C++={expected}, ctypes={actual}"
                )

    def check(self, result: int) -> None:
        if result == 0:
            return
        raw = self.library.ns_last_error()
        detail = raw.decode("utf-8") if raw else "unknown native engine error"
        raise GameError(detail)


@lru_cache(maxsize=1)
def _native() -> _NativeLibrary:
    return _NativeLibrary()


def native_available() -> bool:
    try:
        _native()
    except (AttributeError, OSError, RuntimeError):
        return False
    return True


def native_legal_edge_mask(
    state: NativePublicState,
    *,
    target_symbol: int,
    wild: bool,
    source_any: bool,
) -> int:
    words = (ctypes.c_uint64 * 3)()
    native = _native()
    native.check(
        native.library.ns_legal_edge_mask(
            state,
            target_symbol,
            int(wild),
            int(source_any),
            words,
        )
    )
    return sum(int(words[word]) << (word * 64) for word in range(3))
