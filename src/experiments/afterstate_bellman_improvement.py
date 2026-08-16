"""Evaluate one exact Bellman improvement of scalar afterstate policies."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from rl.afterstate.evaluation import (
    EvaluationResult,
    evaluate_network,
    evaluate_policy,
    load_manifest_seeds,
    paired_summary,
    summarize_scores,
)
from rl.afterstate.network import (
    AfterstateValueNetwork,
    network_from_checkpoint,
    resolve_device,
)
from rl.afterstate.policy import BellmanImprovedAfterstatePolicy

from .records import load_json, write_json

MODES = ("greedy", "online-online", "online-target")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _checkpoint_argument(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in label
    ):
        raise argparse.ArgumentTypeError("checkpoint label must be filename-safe")
    return label, Path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scalar_networks(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], AfterstateValueNetwork, AfterstateValueNetwork]:
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    online = network_from_checkpoint(raw, device)
    target = network_from_checkpoint(raw, device)
    target.load_state_dict(raw["target_state_dict"])
    online.eval()
    target.eval()
    return raw, online, target


def _result_payload(
    result: EvaluationResult,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": result.summary_dict(),
        "scores": list(result.scores),
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def _diagnostics(
    policy: BellmanImprovedAfterstatePolicy,
) -> dict[str, int | float]:
    stats = policy.stats
    return {
        **asdict(stats),
        "agreement_rate": stats.agreement_rate,
        "mean_normalized_regret": stats.mean_normalized_regret,
        "mean_regret_score_points": (
            stats.mean_normalized_regret * policy.reward_scale
        ),
        "root_candidates_per_decision": (
            stats.root_candidates / stats.decisions if stats.decisions else 0.0
        ),
        "chance_outcomes_per_decision": (
            stats.chance_outcomes / stats.decisions if stats.decisions else 0.0
        ),
        "expanded_candidates_per_decision": (
            stats.expanded_candidates / stats.decisions if stats.decisions else 0.0
        ),
    }


def _validate_cached(
    cached: dict[str, Any],
    *,
    seeds: Sequence[int],
    checkpoint_sha256: str,
    mode: str,
) -> None:
    if cached.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("cached result belongs to a different checkpoint")
    if cached.get("mode") != mode:
        raise ValueError("cached result belongs to a different inference mode")
    if cached.get("game_seeds") != list(seeds):
        raise ValueError("cached result uses different game seeds")


def _run_mode(
    label: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    mode: str,
    seeds: Sequence[int],
    output_dir: Path,
    *,
    device: torch.device,
    num_envs: int,
    inference_batch_size: int,
) -> dict[str, Any]:
    result_path = output_dir / "results" / f"{label}-{mode}.json"
    if result_path.exists():
        cached = load_json(result_path)
        _validate_cached(
            cached,
            seeds=seeds,
            checkpoint_sha256=checkpoint_sha256,
            mode=mode,
        )
        print(f"{label}/{mode}: cached", flush=True)
        return cached

    raw, online, target = _load_scalar_networks(checkpoint_path, device)
    reward_scale = float(raw.get("config", {}).get("reward_scale", 10.0))
    gamma = float(raw.get("config", {}).get("gamma", 1.0))
    print(f"{label}/{mode}: evaluating {len(seeds)} games", flush=True)
    if mode == "greedy":
        result = evaluate_network(
            online,
            seeds,
            device=device,
            num_envs=num_envs,
            reward_scale=reward_scale,
        )
        payload = _result_payload(result)
    else:
        backup_target = online if mode == "online-online" else target
        policy = BellmanImprovedAfterstatePolicy(
            online,
            backup_target,
            device=device,
            reward_scale=reward_scale,
            gamma=gamma,
            inference_batch_size=inference_batch_size,
        )
        result = evaluate_policy(policy, seeds, num_envs=num_envs)
        payload = _result_payload(result, diagnostics=_diagnostics(policy))
    payload.update(
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "label": label,
            "mode": mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_transitions": int(raw["transitions"]),
            "checkpoint_updates": int(raw["updates"]),
            "game_seeds": list(seeds),
        }
    )
    write_json(result_path, payload)
    summary = payload["summary"]
    print(
        f"{label}/{mode}: mean={summary['mean']:.3f} "
        f"+/-{summary['standard_error']:.3f}; "
        f"elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def _paired_payload(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    candidate_result = summarize_scores(candidate["scores"])
    reference_result = summarize_scores(reference["scores"])
    paired = paired_summary(candidate_result, reference_result)
    paired.pop("differences")
    return paired


def _write_summary(
    output_dir: Path,
    labels: Sequence[str],
    results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    lines = [
        "# Afterstate Exact Bellman Improvement",
        "",
        "All policies use the same ordered manifest seeds.",
        "",
        "| Checkpoint | Inference | Mean | SE | Paired vs greedy | 95% CI | W/T/L | Agreement | Mean regret (points) | Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        for mode in MODES:
            item = results[label][mode]
            summary = item["summary"]
            if mode == "greedy":
                paired_text = "-"
                interval_text = "-"
                record_text = "-"
                agreement_text = "-"
                regret_text = "-"
            else:
                paired = item["paired_vs_greedy"]
                diagnostics = item["diagnostics"]
                paired_text = f"{paired['mean_difference']:+.3f}"
                interval_text = (
                    f"[{paired['ci95_low']:+.3f}, {paired['ci95_high']:+.3f}]"
                )
                record_text = f"{paired['wins']}/{paired['ties']}/{paired['losses']}"
                agreement_text = f"{diagnostics['agreement_rate']:.3%}"
                regret_text = f"{diagnostics['mean_regret_score_points']:.3f}"
            lines.append(
                f"| {label} | {mode} | {summary['mean']:.3f} | "
                f"{summary['standard_error']:.3f} | {paired_text} | "
                f"{interval_text} | {record_text} | {agreement_text} | "
                f"{regret_text} | "
                f"{summary['elapsed_seconds']:.1f} |"
            )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one exact Bellman improvement for scalar afterstate checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_argument,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=_positive_int, default=64)
    parser.add_argument("--inference-batch-size", type=_positive_int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels):
        parser.error("checkpoint labels must be unique")
    checkpoints = [(label, path.resolve()) for label, path in args.checkpoint]
    for _label, path in checkpoints:
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    seeds = load_manifest_seeds(args.seeds_from.resolve(), args.games)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_manifest": str(args.seeds_from.resolve()),
        "game_seeds": list(seeds),
        "modes": list(MODES),
        "checkpoints": [],
    }
    checkpoint_hashes: dict[str, str] = {}
    for label, path in checkpoints:
        checkpoint_hash = _sha256(path)
        checkpoint_hashes[label] = checkpoint_hash
        manifest["checkpoints"].append(
            {"label": label, "path": str(path), "sha256": checkpoint_hash}
        )
    write_json(output_dir / "manifest.json", manifest)

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in checkpoints:
        label_results: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            item = _run_mode(
                label,
                path,
                checkpoint_hashes[label],
                mode,
                seeds,
                output_dir,
                device=device,
                num_envs=args.num_envs,
                inference_batch_size=args.inference_batch_size,
            )
            label_results[mode] = item
            if mode != "greedy":
                item["paired_vs_greedy"] = _paired_payload(
                    item, label_results["greedy"]
                )
                write_json(output_dir / "results" / f"{label}-{mode}.json", item)
        results[label] = label_results
        _write_summary(output_dir, labels[: len(results)], results)

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(output_dir / "manifest.json", manifest)
    _write_summary(output_dir, labels, results)


if __name__ == "__main__":
    main()
