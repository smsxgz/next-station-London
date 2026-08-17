"""Discover and summarize benchmark result files on shared seed sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Sequence


RESULTS_ROOT = Path(__file__).resolve().parent

KNOWN_RESULTS = {
    "simple-random.jsonl": ("Search", "Random"),
    "greedy.jsonl": ("Search", "Greedy"),
    "lookahead-2.jsonl": ("Search", "Lookahead-2"),
    "lookahead-3.jsonl": ("Search", "Lookahead-3"),
    "lookahead-4.jsonl": ("Search", "Lookahead-4"),
    "mcts-uct-5120-c22p5.jsonl": ("Search", "MCTS-5120"),
    "raw-dqn.jsonl": ("Full-Q", "Raw DQN"),
    "fullq-depth-1.jsonl": ("Full-Q", "Full-Q depth-1"),
    "fullq-depth-2.jsonl": ("Full-Q", "Full-Q depth-2"),
}

AFTERSTATE_MODES = {
    "greedy": "raw",
    "online-online": "online-online",
    "online-target": "online-target",
}


@dataclass(frozen=True)
class ScoreStats:
    mean: float
    standard_error: float
    minimum: float
    median: float
    maximum: float


@dataclass(frozen=True)
class SolverResult:
    file_name: str
    family: str
    label: str
    checkpoint_key: str | None
    checkpoint_label: str | None
    inference: str | None
    scores: dict[int, int]
    stats: ScoreStats


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _score_stats(values: Sequence[int]) -> ScoreStats:
    if not values:
        raise ValueError("cannot summarize an empty score sequence")
    error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return ScoreStats(
        mean=fmean(values),
        standard_error=error,
        minimum=min(values),
        median=median(values),
        maximum=max(values),
    )


def _load_jsonl_scores(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    scores: dict[int, int] = {}
    first_record: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                seed = int(record["game_seed"])
                score = int(record["score"]["total"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid result at {path}:{line_number}") from exc
            if seed in scores:
                raise ValueError(f"duplicate game seed {seed} in {path}")
            scores[seed] = score
            if first_record is None:
                first_record = record
    return scores, first_record or {}


def _load_json_scores(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    value = _load_json(path)
    seeds = value.get("game_seeds")
    values = value.get("scores")
    if not isinstance(seeds, list) or not isinstance(values, list):
        raise ValueError(f"missing game_seeds or scores in {path}")
    if len(seeds) != len(values):
        raise ValueError(f"game_seeds and scores have different lengths in {path}")
    scores = {int(seed): int(score) for seed, score in zip(seeds, values)}
    if len(scores) != len(seeds):
        raise ValueError(f"duplicate game seed in {path}")
    return scores, value


def _pretty_checkpoint(value: str) -> str:
    try:
        prefix, transitions = value.rsplit("-", 1)
    except ValueError:
        return value
    if not transitions.endswith("m"):
        return value
    transitions = transitions.removesuffix("m").replace("p", ".") + "M"
    return f"{prefix}-{transitions}"


def _afterstate_label(stem: str) -> tuple[str, str] | None:
    if not stem.startswith("afterstate-"):
        return None
    value = stem.removeprefix("afterstate-")
    for mode, mode_label in AFTERSTATE_MODES.items():
        suffix = f"-{mode}"
        if not value.endswith(suffix):
            continue
        checkpoint = value[: -len(suffix)]
        return "Afterstate", f"{_pretty_checkpoint(checkpoint)} / {mode_label}"
    return "Afterstate", value


def _result_label(path: Path, metadata: dict[str, Any]) -> tuple[str, str]:
    known = KNOWN_RESULTS.get(path.name)
    if known is not None:
        return known
    for suffix in ("-advanced", "-shared-objectives", "-pencil-powers"):
        if not path.stem.endswith(suffix):
            continue
        base_name = path.name[: -len(suffix) - len(path.suffix)] + path.suffix
        known = KNOWN_RESULTS.get(base_name)
        if known is not None:
            return known
    afterstate = _afterstate_label(path.stem)
    if afterstate is not None:
        return afterstate
    if "checkpoint" in metadata and "mode" in metadata and "scores" in metadata:
        checkpoint = _pretty_checkpoint(str(metadata.get("label", path.stem)))
        mode = AFTERSTATE_MODES.get(
            str(metadata.get("mode", "")),
            str(metadata.get("mode", "unknown")),
        )
        return "Afterstate", f"{checkpoint} / {mode}"
    algorithm = metadata.get("algorithm")
    if isinstance(algorithm, dict):
        family = str(algorithm.get("family", ""))
        policy = str(metadata.get("policy", path.stem))
        for suffix in ("-advanced", "-shared-objectives", "-pencil-powers"):
            policy = policy.removesuffix(suffix)
        if "dqn" in family:
            return "Full-Q", policy
        if family == "lookahead":
            return "Search", policy.title()
        if family == "chance-sampled-uct":
            simulations = algorithm.get("simulations_per_decision", "?")
            rollout = str(algorithm.get("rollout_policy", "greedy"))
            if rollout == "greedy":
                label = f"MCTS-{simulations}"
            else:
                label = f"MCTS-{simulations} + {rollout.title()} rollout"
            return "Search", label
        if family in {"greedy", "simple-random"}:
            return "Search", policy.replace("-", " ").title()
    return "Other", path.stem


def _short_checkpoint(path: str) -> str:
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 1 else path


def _checkpoint_metadata(
    family: str,
    path: Path,
    metadata: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    if family == "Afterstate":
        checkpoint = str(metadata.get("checkpoint", path.stem))
        key = str(metadata.get("checkpoint_sha256") or checkpoint)
        label = _pretty_checkpoint(str(metadata.get("label", path.stem)))
        mode = AFTERSTATE_MODES.get(
            str(metadata.get("mode", "")),
            str(metadata.get("mode", "unknown")),
        )
        return key, label, mode
    if family == "Full-Q":
        algorithm = metadata.get("algorithm")
        algorithm = algorithm if isinstance(algorithm, dict) else {}
        checkpoint = str(algorithm.get("checkpoint", path.stem))
        inference = {
            "raw-dqn.jsonl": "raw",
            "fullq-depth-1.jsonl": "depth-1",
            "fullq-depth-2.jsonl": "depth-2",
        }.get(path.name, str(metadata.get("policy", path.stem)))
        return checkpoint, _short_checkpoint(checkpoint), inference
    return None, None, None


def _load_result(path: Path, expected_seeds: set[int]) -> SolverResult:
    if path.suffix == ".jsonl":
        scores, metadata = _load_jsonl_scores(path)
    elif path.suffix == ".json":
        scores, metadata = _load_json_scores(path)
    else:
        raise ValueError(f"unsupported result format: {path}")
    if not scores:
        raise ValueError(f"empty result file: {path}")
    extra_seeds = set(scores) - expected_seeds
    if extra_seeds:
        raise ValueError(f"{path} has {len(extra_seeds)} seeds outside the manifest")
    family, label = _result_label(path, metadata)
    checkpoint_key, checkpoint_label, inference = _checkpoint_metadata(
        family,
        path,
        metadata,
    )
    return SolverResult(
        file_name=path.name,
        family=family,
        label=label,
        checkpoint_key=checkpoint_key,
        checkpoint_label=checkpoint_label,
        inference=inference,
        scores=scores,
        stats=_score_stats(list(scores.values())),
    )


def _table_row(
    result: SolverResult,
    sole_best: int,
    tied_best: int,
) -> str:
    stats = result.stats
    return (
        f"| {result.family} | {result.label} | {stats.mean:.3f} | "
        f"{stats.standard_error:.3f} | {stats.minimum:.0f} | "
        f"{stats.median:.1f} | {stats.maximum:.0f} | {sole_best} | "
        f"{tied_best} | {sole_best + tied_best} |"
    )


def _stats_row(family: str, label: str, stats: ScoreStats) -> str:
    return (
        f"| {family} | {label} | {stats.mean:.3f} | "
        f"{stats.standard_error:.3f} | {stats.minimum:.0f} | "
        f"{stats.median:.1f} | {stats.maximum:.0f} | - | - | - |"
    )


def _select_main_results(
    results: list[SolverResult],
) -> tuple[list[SolverResult], dict[str, str]]:
    selected: dict[str, str] = {}
    main = [
        result
        for result in results
        if result.family not in {"Afterstate", "Full-Q"}
    ]
    for family in ("Afterstate", "Full-Q"):
        groups: dict[str, list[SolverResult]] = {}
        for result in results:
            if result.family != family:
                continue
            key = result.checkpoint_key or result.file_name
            groups.setdefault(key, []).append(result)
        if not groups:
            continue
        key, group = max(
            groups.items(),
            key=lambda item: (
                max(result.stats.mean for result in item[1]),
                item[0],
            ),
        )
        selected[family] = key
        main.extend(group)
    return main, selected


def _checkpoint_table(
    family: str,
    results: list[SolverResult],
    selected_key: str | None,
) -> list[str]:
    family_results = [result for result in results if result.family == family]
    if not family_results:
        return []
    groups: dict[str, list[SolverResult]] = {}
    for result in family_results:
        key = result.checkpoint_key or result.file_name
        groups.setdefault(key, []).append(result)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: -max(result.stats.mean for result in item[1]),
    )
    mode_order = {
        "raw": 0,
        "online-online": 1,
        "online-target": 2,
        "depth-1": 1,
        "depth-2": 2,
    }
    lines = [
        "",
        f"## {family} Checkpoints",
        "",
        (
            "The checkpoint whose best inference mode has the highest mean is "
            "selected for the main table."
        ),
        "",
        "| Checkpoint | Inference | Mean | SE | Min | Median | Max | Main table |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, group in ordered_groups:
        for result in sorted(
            group,
            key=lambda item: (mode_order.get(item.inference or "", 99), item.label),
        ):
            stats = result.stats
            lines.append(
                f"| {result.checkpoint_label or result.file_name} | "
                f"{result.inference or result.label} | {stats.mean:.3f} | "
                f"{stats.standard_error:.3f} | {stats.minimum:.0f} | "
                f"{stats.median:.1f} | {stats.maximum:.0f} | "
                f"{'yes' if key == selected_key else ''} |"
            )
    return lines


def _render_summary(
    name: str,
    seeds: list[int],
    complete: list[SolverResult],
    incomplete: list[SolverResult],
) -> str:
    main_results, selected_checkpoints = _select_main_results(complete)
    sole_best = {result.file_name: 0 for result in main_results}
    tied_best = {result.file_name: 0 for result in main_results}
    per_seed_best: list[int] = []
    tied_seed_count = 0
    for seed in seeds:
        best = max(result.scores[seed] for result in main_results)
        winners = [result for result in main_results if result.scores[seed] == best]
        per_seed_best.append(best)
        if len(winners) == 1:
            sole_best[winners[0].file_name] += 1
        else:
            tied_seed_count += 1
            for winner in winners:
                tied_best[winner.file_name] += 1

    oracle = _score_stats(per_seed_best)
    ranked = sorted(main_results, key=lambda item: (-item.stats.mean, item.label))
    strongest = ranked[0]
    oracle_gain = oracle.mean - strongest.stats.mean

    lines = [
        f"# {name} Evaluation Summary",
        "",
        f"Games: `{len(seeds)}`  ",
        f"Complete result files: `{len(complete)}`  ",
        f"Main-table solver results: `{len(main_results)}`  ",
        f"Incomplete result files: `{len(incomplete)}`",
        "",
        "Complete results are aligned by the same manifest seed set.",
        "",
        (
            "For Afterstate and Full-Q, the main table keeps all inference modes "
            "from only the checkpoint with the highest single-mode mean."
        ),
        "",
        (
            "> **Per-seed best** is the post-hoc maximum across the solver rows "
            "in the main table. It is not an executable solver or a theoretical "
            "upper bound."
        ),
        "",
        "## Main Solver Comparison",
        "",
        (
            "| Family | Solver | Mean | SE | Min | Median | Max | Sole best | "
            "Tied best | Best total |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _stats_row("Post-hoc", "**Per-seed best**", oracle),
    ]
    for result in ranked:
        lines.append(
            _table_row(
                result,
                sole_best[result.file_name],
                tied_best[result.file_name],
            )
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            (
                f"The strongest single solver is **{strongest.label}** at "
                f"`{strongest.stats.mean:.3f} +/- "
                f"{strongest.stats.standard_error:.3f}`."
            ),
            "",
            (
                f"Per-seed best is `{oracle.mean:.3f} +/- "
                f"{oracle.standard_error:.3f}`, with range "
                f"`{oracle.minimum:.0f}..{oracle.maximum:.0f}` and median "
                f"`{oracle.median:.1f}`. Its post-hoc gain over the strongest "
                f"single solver is `{oracle_gain:.3f}` points."
            ),
            "",
            (
                f"`{tied_seed_count}` seeds have more than one highest-scoring "
                "solver. Sole best counts an unshared maximum; tied best counts "
                "an appearance in a shared maximum; best total is their sum."
            ),
        ]
    )

    lines.extend(
        _checkpoint_table(
            "Afterstate",
            complete,
            selected_checkpoints.get("Afterstate"),
        )
    )
    lines.extend(
        _checkpoint_table(
            "Full-Q",
            complete,
            selected_checkpoints.get("Full-Q"),
        )
    )

    if incomplete:
        lines.extend(
            [
                "",
                "## Incomplete Results",
                "",
                (
                    "These provisional results are reported separately and are "
                    "excluded from ranking, per-seed best, and best counts."
                ),
                "",
                "| Family | Solver | Games | Expected | Mean | SE | Missing |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in sorted(incomplete, key=lambda item: item.label):
            lines.append(
                f"| {result.family} | {result.label} | {len(result.scores)} | "
                f"{len(seeds)} | {result.stats.mean:.3f} | "
                f"{result.stats.standard_error:.3f} | "
                f"{len(seeds) - len(result.scores)} |"
            )
    return "\n".join(lines) + "\n"


def summarize_directory(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    seeds = [int(seed) for seed in manifest.get("game_seeds", ())]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"manifest has no unique game seeds: {manifest_path}")

    layout = manifest.setdefault("layout", {})
    if not isinstance(layout, dict):
        raise ValueError(f"manifest layout is not an object: {manifest_path}")
    games_dir = root / str(layout.get("games", "games"))
    paths = sorted(
        (
            path
            for path in games_dir.iterdir()
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"no result files found in {games_dir}")

    expected_seeds = set(seeds)
    results = [_load_result(path, expected_seeds) for path in paths]
    complete = [result for result in results if len(result.scores) == len(seeds)]
    incomplete = [result for result in results if len(result.scores) < len(seeds)]
    if not complete:
        raise ValueError(f"no complete result files found in {games_dir}")

    layout["summary"] = str(layout.get("summary", "summary.md"))
    layout["games"] = str(layout.get("games", "games/"))
    layout["result_files"] = [path.name for path in paths]
    manifest["completed_policies"] = [result.file_name for result in complete]

    try:
        display_name = root.relative_to(RESULTS_ROOT).as_posix()
    except ValueError:
        display_name = root.name
    summary_path = root / layout["summary"]
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        _render_summary(display_name, seeds, complete, incomplete),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(summary_path)
    _write_json(manifest_path, manifest)
    print(
        f"{root}: complete={len(complete)}, incomplete={len(incomplete)}, "
        f"seeds={len(seeds)}",
        flush=True,
    )


def _default_directories() -> list[Path]:
    return sorted(
        (manifest.parent for manifest in RESULTS_ROOT.glob("**/manifest.json")),
        key=lambda path: path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and summarize benchmark result files"
    )
    parser.add_argument("directories", nargs="*", type=Path)
    args = parser.parse_args()
    directories = (
        [directory.resolve() for directory in args.directories]
        if args.directories
        else _default_directories()
    )
    if not directories:
        parser.error("no benchmark directories found")
    for directory in directories:
        summarize_directory(directory)


if __name__ == "__main__":
    main()
