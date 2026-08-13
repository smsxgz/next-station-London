"""Command line entry point for base-rules AlphaZero experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .diagnostics import benchmark_generation, run_self_check
from .training import TrainConfig, evaluate_policy, load_network, train


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _load_seeds(path: Path, games: int | None = None) -> tuple[int, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in raw.get("game_seeds", ()))
    if not seeds:
        raise ValueError(f"{path} does not contain game_seeds")
    if games is not None:
        if games > len(seeds):
            raise ValueError("requested more games than the manifest contains")
        seeds = seeds[:games]
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Base-rules AlphaZero")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--device", default="auto")

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--num-envs", type=_positive_int, default=128)
    benchmark.add_argument("--simulations", type=_positive_int, default=256)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--width", type=_positive_int, default=1024)
    benchmark.add_argument("--blocks", type=_positive_int, default=6)

    training = commands.add_parser("train")
    training.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/alphazero/base_v1")
    )
    training.add_argument("--resume", action="store_true")
    training.add_argument("--num-envs", type=_positive_int, default=128)
    training.add_argument("--simulations", type=_positive_int, default=256)
    training.add_argument("--total-positions", type=_positive_int, default=1_000_000)
    training.add_argument("--batch-size", type=_positive_int, default=2048)
    training.add_argument("--replay-capacity", type=_positive_int, default=500_000)
    training.add_argument("--validation-interval", type=_positive_int, default=50_000)
    training.add_argument("--eval-simulations", type=_positive_int, default=256)
    training.add_argument("--eval-seeds-from", type=Path)
    training.add_argument("--validation-games", type=_positive_int, default=200)
    training.add_argument("--time-limit-hours", type=_positive_float, default=7.5)
    training.add_argument("--device", default="auto")
    training.add_argument("--verify", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seeds-from", type=Path, required=True)
    evaluate.add_argument("--games", type=_positive_int)
    evaluate.add_argument("--num-envs", type=_positive_int, default=128)
    evaluate.add_argument("--simulations", type=int, default=256)
    evaluate.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "check":
        print(json.dumps(run_self_check(args.device), indent=2), flush=True)
        return
    if args.command == "benchmark":
        result = benchmark_generation(
            num_envs=args.num_envs,
            simulations=args.simulations,
            device=args.device,
            width=args.width,
            residual_blocks=args.blocks,
        )
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.command == "train":
        if args.resume:
            raw = json.loads(
                (args.run_dir / "config.json").read_text(encoding="utf-8")
            )
            config = TrainConfig.from_dict(raw)
            config = TrainConfig.from_dict(
                {
                    **config.to_dict(),
                    "total_positions": args.total_positions,
                    "max_wall_seconds": args.time_limit_hours * 3600.0,
                    "device": args.device,
                }
            )
        else:
            validation_seeds = (
                _load_seeds(args.eval_seeds_from, args.validation_games)
                if args.eval_seeds_from is not None
                else ()
            )
            config = TrainConfig.fresh(
                num_envs=args.num_envs,
                simulations=args.simulations,
                total_positions=args.total_positions,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                validation_interval=args.validation_interval,
                evaluation_simulations=args.eval_simulations,
                max_wall_seconds=args.time_limit_hours * 3600.0,
                device=args.device,
                validation_seeds=validation_seeds,
            )
        print(f"run_seed={config.run_seed}", flush=True)
        train(config, args.run_dir, resume=args.resume, verify=args.verify)
        return
    if args.command == "evaluate":
        if args.simulations < 0:
            raise ValueError("evaluation simulations must be non-negative")
        network, _ = load_network(args.checkpoint, device=args.device)
        result = evaluate_policy(
            network,
            _load_seeds(args.seeds_from, args.games),
            device=args.device,
            num_envs=args.num_envs,
            simulations=args.simulations,
        )
        print(json.dumps(result.summary_dict(), indent=2), flush=True)
        return
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
