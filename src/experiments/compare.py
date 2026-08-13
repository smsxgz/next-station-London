"""Paired comparison of two JSONL policy runs on shared game seeds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean, stdev

from .records import load_jsonl, write_json


def _standard_error(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired game results")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON summary")
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    seeds = sorted(set(baseline) & set(candidate))
    if not seeds:
        parser.error("the files have no shared game seeds")

    differences = [
        float(candidate[seed]["score"]["total"])
        - float(baseline[seed]["score"]["total"])
        for seed in seeds
    ]
    mean = fmean(differences)
    error = _standard_error(differences)
    components = {}
    for key in ("line_total", "tourist_bonus", "interchange_bonus"):
        values = [
            float(candidate[seed]["score"][key])
            - float(baseline[seed]["score"][key])
            for seed in seeds
        ]
        component_mean = fmean(values)
        component_error = _standard_error(values)
        components[key] = {
            "mean_difference": component_mean,
            "standard_error": component_error,
            "ci95": [
                component_mean - 1.96 * component_error,
                component_mean + 1.96 * component_error,
            ],
        }

    result = {
        "games": len(seeds),
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_mean": fmean(
            float(baseline[seed]["score"]["total"]) for seed in seeds
        ),
        "candidate_mean": fmean(
            float(candidate[seed]["score"]["total"]) for seed in seeds
        ),
        "mean_difference": mean,
        "standard_error": error,
        "ci95": [mean - 1.96 * error, mean + 1.96 * error],
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "color_order_mismatches": sum(
            candidate[seed].get("color_order") != baseline[seed].get("color_order")
            for seed in seeds
        ),
        "card_sequence_mismatches": sum(
            candidate[seed].get("card_sequences")
            != baseline[seed].get("card_sequences")
            for seed in seeds
        ),
        "components": components,
    }
    if args.output is not None:
        write_json(args.output, result)
    print(
        f"games={len(seeds)}; candidate={result['candidate_mean']:.2f}; "
        f"baseline={result['baseline_mean']:.2f}; difference={mean:+.2f} "
        f"+/- {error:.2f}; 95% CI [{mean - 1.96 * error:+.2f}, "
        f"{mean + 1.96 * error:+.2f}]"
    )
    print(
        f"wins/ties/losses={result['wins']}/{result['ties']}/"
        f"{result['losses']}; color mismatches="
        f"{result['color_order_mismatches']}; card mismatches="
        f"{result['card_sequence_mismatches']}"
    )


if __name__ == "__main__":
    main()
