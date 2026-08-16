"""UTF-8 result records and compact benchmark summaries."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Sequence

from engine_cpp import PENCIL_POWERS, GameSession


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            seed = int(record["game_seed"])
            if seed in records:
                raise ValueError(f"duplicate game seed {seed} in {path}")
            records[seed] = record
    return records


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_game_record(
    game: GameSession,
    *,
    policy: str,
    seed_index: int,
    game_seed: int,
    elapsed_seconds: float,
    actions: dict[str, int],
    algorithm: dict[str, Any],
) -> dict[str, Any]:
    final = game.final_score()
    if final is None:
        raise RuntimeError("experiment ended before final scoring")
    decision_count = (
        actions["sections"]
        + actions["passes"]
        + actions.get("second_section_stops", 0)
    )
    card_sequences = [
        {
            "round": round_number,
            "color": color,
            "events": [
                list(move.event.card_ids)
                for move in game.move_history
                if move.round_number == round_number
            ],
        }
        for round_number, color in enumerate(game.order, start=1)
    ]
    return {
        "schema_version": 2,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": policy,
        "seed_index": seed_index,
        "game_seed": game_seed,
        "color_order": list(game.order),
        "modules": {
            "shared_objectives": game.shared_objectives_enabled,
            "pencil_powers": game.pencil_powers_enabled,
            "objective_cards": list(game.shared_objectives),
            "power_assignments": dict(game.pencil_powers),
            "powers_used": [
                power for power in PENCIL_POWERS if game.power_used(power)
            ],
        },
        "score": asdict(final),
        "rounds": [
            {"round": index + 1, "color": color, **asdict(score)}
            for index, (color, score) in enumerate(
                zip(game.order, game.round_scores)
            )
        ],
        "card_sequences": card_sequences,
        "actions": {
            **actions,
            "decisions": decision_count,
            "cards_consumed": sum(
                len(move.event.card_ids) for move in game.move_history
            ),
        },
        "algorithm": algorithm,
        "elapsed_seconds": elapsed_seconds,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Sequence[int | float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0}
    deviation = stdev(numeric) if len(numeric) > 1 else 0.0
    return {
        "count": len(numeric),
        "mean": fmean(numeric),
        "standard_error": deviation / math.sqrt(len(numeric)),
        "standard_deviation": deviation,
        "min": min(numeric),
        "p10": _percentile(numeric, 0.10),
        "median": median(numeric),
        "p90": _percentile(numeric, 0.90),
        "max": max(numeric),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty experiment")
    return {
        "games": len(records),
        "score": describe([record["score"]["total"] for record in records]),
        "components": {
            key: describe([record["score"].get(key, 0) for record in records])
            for key in (
                "line_total",
                "tourist_bonus",
                "interchange_bonus",
                "objective_bonus",
            )
        },
        "actions": {
            key: describe([record["actions"][key] for record in records])
            for key in ("sections", "passes", "strategic_passes", "decisions")
        },
        "elapsed_seconds": describe(
            [record["elapsed_seconds"] for record in records]
        ),
    }
