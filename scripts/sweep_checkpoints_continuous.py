"""Sweep SAC/TD3 checkpoints with fresh PR11a eval metrics.

This is a post-training selection helper: it evaluates every checkpoint matched
by a glob, writes per-checkpoint metrics JSONs, builds a summary table, and
selects the best checkpoint by target-funnel metrics.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from agents.checkpointing import SUPPORTED_AGENT_TYPES
from eval.checkpoint_sweep import (
    SUPPORTED_CHECKPOINT_SWEEP_RANK_MODES,
    SweepCandidate,
    best_fresh_eval_payload,
    discover_checkpoints,
    dry_run_summary_row,
    failure_summary_row,
    promote_checkpoint,
    rank_key_for_metrics,
    select_best_summary_row,
    summary_row_from_metrics,
    write_output_manifest,
    write_summary_files,
)
from scripts import eval_checkpoint_continuous


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaac", "fake"], default="isaac")
    parser.add_argument("--agent-type", "--agent_type", dest="agent_type", choices=SUPPORTED_AGENT_TYPES, required=True)
    parser.add_argument("--checkpoint-glob", "--checkpoint_glob", dest="checkpoint_glob", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument("--num-envs", "--num_envs", "--num-parallel-envs", dest="num_envs", type=int, default=1)
    parser.add_argument("--num-episodes", "--num_episodes", dest="num_episodes", type=int, default=100)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=230)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle-steps", "--settle_steps", dest="settle_steps", type=int, default=550)
    parser.add_argument(
        "--rank-by",
        "--rank_by",
        dest="rank_by",
        choices=SUPPORTED_CHECKPOINT_SWEEP_RANK_MODES,
        default="target_funnel",
    )
    parser.add_argument("--promote-best", "--promote_best", dest="promote_best", action="store_true")
    parser.add_argument("--overwrite-promoted", "--overwrite_promoted", dest="overwrite_promoted", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-threshold-m", dest="success_threshold_m", type=float, default=0.02)
    parser.add_argument("--target-hold-consecutive-steps", dest="target_hold_consecutive_steps", type=int, default=5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.target_hold_consecutive_steps <= 0:
        raise ValueError("--target-hold-consecutive-steps must be positive")
    if args.success_threshold_m <= 0.0:
        raise ValueError("--success-threshold-m must be positive")


def _eval_metrics_path(out_dir: Path, candidate: SweepCandidate, num_episodes: int) -> Path:
    return out_dir / f"{candidate.label}_eval_{num_episodes}eps.json"


def _eval_command_file(out_dir: Path, candidate: SweepCandidate, num_episodes: int) -> Path:
    return out_dir / f"{candidate.label}_eval_{num_episodes}eps_command.txt"


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _eval_cli_args(args: argparse.Namespace, candidate: SweepCandidate, metrics_path: Path) -> list[str]:
    return [
        "--backend",
        args.backend,
        "--agent-type",
        args.agent_type,
        "--checkpoint",
        str(candidate.checkpoint),
        "--save-metrics",
        str(metrics_path),
        "--num-episodes",
        str(args.num_episodes),
        "--max-steps",
        str(args.max_steps),
        "--num-envs",
        str(args.num_envs),
        "--settle-steps",
        str(args.settle_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--success-threshold-m",
        str(args.success_threshold_m),
        "--target-hold-consecutive-steps",
        str(args.target_hold_consecutive_steps),
        _bool_flag("deterministic", args.deterministic),
        _bool_flag("headless", args.headless),
        _bool_flag("progress", args.progress),
    ]


def format_eval_command(args: argparse.Namespace, candidate: SweepCandidate, metrics_path: Path) -> str:
    argv = ["python", "-u", "-m", "scripts.eval_checkpoint_continuous"]
    argv.extend(_eval_cli_args(args, candidate, metrics_path))
    return " ".join(shlex.quote(part) for part in argv)


def _run_eval(args: argparse.Namespace, candidate: SweepCandidate, metrics_path: Path) -> dict[str, Any]:
    eval_args = eval_checkpoint_continuous.parse_args(_eval_cli_args(args, candidate, metrics_path))
    if eval_args.backend == "fake":
        return eval_checkpoint_continuous.run_fake_backend(eval_args)
    if eval_args.backend == "isaac":
        return eval_checkpoint_continuous.run_isaac_backend(eval_args)
    raise ValueError(f"unsupported backend {eval_args.backend!r}")


def _read_metrics(metrics_path: Path) -> dict[str, Any]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _failed_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if row.get("status") == "failed":
            failures.append(
                {
                    "label": row.get("label"),
                    "checkpoint": row.get("checkpoint"),
                    "metrics_path": row.get("metrics_path"),
                    "error": row.get("error"),
                }
            )
    return failures


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_checkpoints(args.checkpoint_glob)
    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    selected_row: dict[str, Any] | None = None
    promoted_checkpoint: Path | None = None
    selection_error: str | None = None

    for candidate in candidates:
        metrics_path = _eval_metrics_path(out_dir, candidate, args.num_episodes)
        command = format_eval_command(args, candidate, metrics_path)
        command_file = _eval_command_file(out_dir, candidate, args.num_episodes)
        command_file.write_text(command + "\n", encoding="utf-8")
        commands.append(f"# {candidate.label}\n{command}\n")

        if args.dry_run:
            rows.append(dry_run_summary_row(candidate, metrics_path=metrics_path, rank_by=args.rank_by))
            continue

        try:
            _run_eval(args, candidate, metrics_path)
            metrics = _read_metrics(metrics_path)
            rank_key_for_metrics(metrics, args.rank_by)
            rows.append(
                summary_row_from_metrics(
                    candidate,
                    metrics_path=metrics_path,
                    metrics=metrics,
                    rank_by=args.rank_by,
                )
            )
        except Exception as exc:  # noqa: BLE001 - sweep should continue and report bad candidates.
            rows.append(
                failure_summary_row(
                    candidate,
                    metrics_path=metrics_path,
                    rank_by=args.rank_by,
                    error=exc,
                )
            )

    if not args.dry_run:
        try:
            selected_row = select_best_summary_row(rows)
            if args.promote_best:
                promoted_checkpoint = promote_checkpoint(
                    selected_row["checkpoint"],
                    overwrite=args.overwrite_promoted,
                )
        except ValueError as exc:
            selection_error = str(exc)

    summary_csv, summary_json = write_summary_files(out_dir, rows, num_episodes=args.num_episodes)
    commands_used_path = out_dir / "commands_used.txt"
    commands_used_path.write_text("\n".join(commands) + ("\n" if commands else ""), encoding="utf-8")
    command_summary = (
        f"python -m scripts.sweep_checkpoints_continuous "
        f"--backend {args.backend} --agent-type {args.agent_type} "
        f"--checkpoint-glob {shlex.quote(args.checkpoint_glob)} --out-dir {shlex.quote(str(out_dir))} "
        f"--num-envs {args.num_envs} --num-episodes {args.num_episodes} "
        f"--max-steps {args.max_steps} --seed {args.seed} --device {args.device} "
        f"--settle-steps {args.settle_steps} --rank-by {args.rank_by}"
    )
    if args.promote_best:
        command_summary += " --promote-best"
    if args.dry_run:
        command_summary += " --dry-run"
    best_payload = best_fresh_eval_payload(
        selected_row=selected_row,
        rank_by=args.rank_by,
        candidate_count=len(candidates),
        failed_candidates=_failed_candidates(rows),
        command=command_summary,
        summary_csv=summary_csv,
        summary_json=summary_json,
        promoted_checkpoint=promoted_checkpoint,
    )
    best_path = out_dir / "best_fresh_eval.json"
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = write_output_manifest(out_dir)

    result = {
        "status": "dry_run" if args.dry_run else ("ok" if selected_row is not None else "failed"),
        "candidate_count": len(candidates),
        "summary_csv": str(summary_csv.resolve()),
        "summary_json": str(summary_json.resolve()),
        "best_fresh_eval": str(best_path.resolve()),
        "commands_used": str(commands_used_path.resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "selected_checkpoint": best_payload["selected_checkpoint"],
        "promoted_checkpoint": best_payload["promoted_checkpoint"],
        "failed_candidates": best_payload["failed_candidates"],
        "selection_error": selection_error,
    }
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_sweep(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
