"""Locate where exact-chance depth-2 changes depth-1 decisions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable

from engine import DECK_BY_ID, GameSession
from solver.RL import ExactChanceDQNPolicy

from .records import append_jsonl, describe, load_json, write_json


@dataclass(slots=True)
class _ActiveGame:
    seed_index: int
    seed: int
    game: GameSession
    decision_index: int = 0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _load_seeds(path: Path, games: int | None) -> list[int]:
    available = [int(seed) for seed in load_json(path).get("game_seeds", ())]
    if not available:
        raise ValueError(f"{path} has no game_seeds")
    if len(set(available)) != len(available):
        raise ValueError(f"{path} contains duplicate seeds")
    count = len(available) if games is None else games
    if count > len(available):
        raise ValueError(f"requested {count} games from {len(available)} seeds")
    return available[:count]


def _start_game(seed_index: int, seed: int) -> _ActiveGame:
    game = GameSession(seed=seed)
    game.draw()
    return _ActiveGame(seed_index, seed, game)


def _gap(decision: Any) -> float | None:
    if len(decision.estimates) < 2:
        return None
    return float(
        decision.estimates[0].expected_gain
        - decision.estimates[1].expected_gain
    )


def _estimate_for_action(decision: Any, action_index: int) -> float:
    for estimate in decision.estimates:
        index = -1 if estimate.action is None else estimate.action.edge_id
        if index == action_index:
            return float(estimate.expected_gain)
    raise RuntimeError("one exact-chance policy omitted a legal root action")


def _card_kind(game: GameSession) -> str:
    pending = game.pending
    if pending is None:
        raise RuntimeError("disagreement record requires a pending card")
    if DECK_BY_ID[pending.card_ids[0]].switch:
        return "switch"
    target = DECK_BY_ID[pending.card_ids[-1]]
    return "underground" if target.underground else "street"


def _gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "single-action"
    for upper in (0.25, 0.5, 1.0, 2.0, 5.0):
        if gap <= upper:
            return f"<= {upper:g}"
    return "> 5"


def _action_bucket(actions: int) -> str:
    if actions == 1:
        return "1"
    if actions <= 3:
        return "2-3"
    if actions <= 6:
        return "4-6"
    if actions <= 10:
        return "7-10"
    return "11+"


def _group_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    disagreements = [item for item in items if not item["actions_agree"]]
    regrets = [float(item["depth2_regret_of_depth1"]) for item in items]
    disagreement_regrets = [
        float(item["depth2_regret_of_depth1"]) for item in disagreements
    ]
    return {
        "states": len(items),
        "disagreements": len(disagreements),
        "disagreement_rate": len(disagreements) / len(items) if items else 0.0,
        "mean_regret_all": fmean(regrets) if regrets else 0.0,
        "mean_regret_when_different": (
            fmean(disagreement_regrets) if disagreement_regrets else 0.0
        ),
        "regret_sum": sum(regrets),
    }


def _groups(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({str(record[key]) for record in records})
    return {
        value: _group_summary(
            record for record in records if str(record[key]) == value
        )
        for value in values
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Exact-Chance Depth-1 / Depth-2 Disagreements",
        "",
        f"Checkpoint: `{summary['checkpoint']}`  ",
        f"Games: `{summary['games']}`  ",
        f"Depth-2 trajectory score: `{summary['scores']['mean']:.2f} +/- {summary['scores']['standard_error']:.2f}`",
        "",
        (
            f"Depth-1 and depth-2 disagree on `{overall['disagreements']}` of "
            f"`{overall['states']}` states (`{overall['disagreement_rate'] * 100:.1f}%`). "
            f"Mean depth-2 regret of the depth-1 action is "
            f"`{overall['mean_regret_all']:.3f}` points/state overall and "
            f"`{overall['mean_regret_when_different']:.3f}` when different."
        ),
        "",
    ]
    for title, key in (
        ("By round", "by_round"),
        ("By draw count", "by_draw_count"),
        ("By legal-action count", "by_action_bucket"),
        ("By depth-2 top-two gap", "by_gap_bucket"),
        ("By card kind", "by_card_kind"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Group | States | Different | Rate | Mean regret | Regret when different | Regret share |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        total_regret = max(float(overall["regret_sum"]), 1e-12)
        for group, item in summary[key].items():
            lines.append(
                f"| {group} | {item['states']} | {item['disagreements']} | "
                f"{item['disagreement_rate'] * 100:.1f}% | "
                f"{item['mean_regret_all']:.3f} | "
                f"{item['mean_regret_when_different']:.3f} | "
                f"{item['regret_sum'] / total_regret * 100:.1f}% |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=_positive_int)
    parser.add_argument("--parallel-games", type=_positive_int, default=16)
    parser.add_argument("--leaf-batch-size", type=_positive_int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    seeds = _load_seeds(args.seeds_from.resolve(), args.games)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "states.jsonl"
    if states_path.exists():
        raise FileExistsError(f"refusing to overwrite {states_path}")

    depth1 = ExactChanceDQNPolicy.from_checkpoint(
        checkpoint,
        1,
        device=args.device,
        leaf_batch_size=args.leaf_batch_size,
    )
    depth2 = ExactChanceDQNPolicy.from_checkpoint(
        checkpoint,
        2,
        device=args.device,
        leaf_batch_size=args.leaf_batch_size,
    )

    pending = list(enumerate(seeds))
    cursor = 0
    active: list[_ActiveGame] = []
    while cursor < len(pending) and len(active) < args.parallel_games:
        active.append(_start_game(*pending[cursor]))
        cursor += 1

    records: list[dict[str, Any]] = []
    scores: list[int] = []
    total_leaf_states = {1: 0, 2: 0}
    total_build_seconds = {1: 0.0, 2: 0.0}
    started = perf_counter()
    completed = 0
    while active:
        games = [slot.game for slot in active]
        depth1_decisions = depth1.choose_many(games)
        total_leaf_states[1] += depth1.last_batch_stats.unique_leaf_states
        total_build_seconds[1] += depth1.last_batch_stats.build_seconds
        depth2_decisions = depth2.choose_many(games)
        total_leaf_states[2] += depth2.last_batch_stats.unique_leaf_states
        total_build_seconds[2] += depth2.last_batch_stats.build_seconds

        finished: list[_ActiveGame] = []
        for slot, decision1, decision2 in zip(
            active, depth1_decisions, depth2_decisions
        ):
            legal_actions = len(slot.game.legal_actions()) + 1
            estimate1_under_depth2 = _estimate_for_action(
                decision2,
                -1 if decision1.action is None else decision1.action.edge_id,
            )
            regret = max(
                0.0,
                float(decision2.expected_gain) - estimate1_under_depth2,
            )
            gap1 = _gap(decision1)
            gap2 = _gap(decision2)
            record = {
                "seed_index": slot.seed_index,
                "game_seed": slot.seed,
                "decision_index": slot.decision_index,
                "round": slot.game.round_index + 1,
                "draw_count": slot.game.draw_count,
                "underground_count": slot.game.underground_count,
                "remaining_cards": len(slot.game.remaining_card_ids()),
                "card_kind": _card_kind(slot.game),
                "legal_actions": legal_actions,
                "action_bucket": _action_bucket(legal_actions),
                "depth1_action": decision1.action_index,
                "depth2_action": decision2.action_index,
                "actions_agree": decision1.action_index == decision2.action_index,
                "depth1_gap": gap1,
                "depth2_gap": gap2,
                "gap_bucket": _gap_bucket(gap2),
                "depth2_regret_of_depth1": regret,
            }
            records.append(record)
            append_jsonl(states_path, record)

            slot.game.apply_legal_action(decision2.action)
            slot.decision_index += 1
            if slot.game.status == "playing":
                slot.game.draw()
                continue
            final = slot.game.final_score()
            if final is None:
                raise RuntimeError("depth-2 trajectory ended without a score")
            scores.append(final.total)
            finished.append(slot)
            completed += 1
            if completed == len(seeds) or completed % 10 == 0:
                print(f"games={completed}/{len(seeds)}", flush=True)

        for slot in finished:
            active.remove(slot)
        while cursor < len(pending) and len(active) < args.parallel_games:
            active.append(_start_game(*pending[cursor]))
            cursor += 1

    overall = _group_summary(records)
    highest_regret = sorted(
        (record for record in records if not record["actions_agree"]),
        key=lambda item: float(item["depth2_regret_of_depth1"]),
        reverse=True,
    )[:20]
    summary = {
        "checkpoint": str(checkpoint),
        "games": len(seeds),
        "elapsed_seconds": perf_counter() - started,
        "scores": describe(scores),
        "overall": overall,
        "by_round": _groups(records, "round"),
        "by_draw_count": _groups(records, "draw_count"),
        "by_action_bucket": _groups(records, "action_bucket"),
        "by_gap_bucket": _groups(records, "gap_bucket"),
        "by_card_kind": _groups(records, "card_kind"),
        "search_totals": {
            "depth1_unique_leaf_states": total_leaf_states[1],
            "depth2_unique_leaf_states": total_leaf_states[2],
            "depth1_build_seconds": total_build_seconds[1],
            "depth2_build_seconds": total_build_seconds[2],
        },
        "highest_regret_states": highest_regret,
    }
    write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary)
    print(
        f"states={overall['states']}; disagreements={overall['disagreements']} "
        f"({overall['disagreement_rate'] * 100:.1f}%); "
        f"mean_regret={overall['mean_regret_all']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
