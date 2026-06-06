"""Helpers for post-training SAC/TD3 checkpoint eval sweeps.

The sweep CLI reuses PR11a checkpoint eval metrics, then ranks checkpoint JSONs
with a small, explicit target-funnel policy-selection rule.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Mapping

RANK_TARGET_FUNNEL = "target_funnel"
RANK_SUCCESS_RETURN = "success_return"
SUPPORTED_CHECKPOINT_SWEEP_RANK_MODES = (RANK_TARGET_FUNNEL, RANK_SUCCESS_RETURN)

SUMMARY_FIELDS = (
    "label",
    "checkpoint",
    "metrics_path",
    "command_path",
    "log_path",
    "returncode",
    "num_env_steps",
    "num_episodes",
    "mean_return",
    "success_rate",
    "target_hold_episode_rate",
    "target_20cm_episode_rate",
    "target_10cm_episode_rate",
    "target_5cm_episode_rate",
    "target_2cm_episode_rate",
    "mean_cube_to_target_m",
    "p50_cube_to_target_m",
    "final_cube_to_target_m",
    "min_cube_to_target_m",
    "max_cube_lift_m",
    "min_ee_to_cube_m",
    "gripper_close_near_cube_rate",
    "mean_action_jerk",
    "rank_by",
    "rank_key",
    "selected",
    "status",
    "error",
)

METRIC_SUMMARY_FIELDS = tuple(
    field
    for field in SUMMARY_FIELDS
    if field
    not in {
        "label",
        "checkpoint",
        "metrics_path",
        "command_path",
        "log_path",
        "returncode",
        "num_episodes",
        "rank_by",
        "rank_key",
        "selected",
        "status",
        "error",
    }
)

_STEP_RE = re.compile(r"(?:^|[_-])step[_-]?(\d+)(?:$|[_-])")
_BEST_RE = re.compile(r"(?:^|[_-])best(?:$|[_-])")
_FINAL_RE = re.compile(r"(?:^|[_-])final(?:$|[_-])")


@dataclass(frozen=True)
class SweepCandidate:
    """Checkpoint file selected by a glob."""

    label: str
    checkpoint: Path


def checkpoint_label(path: str | Path) -> str:
    return Path(path).stem


def checkpoint_sort_key(path: str | Path) -> tuple[int, int, str]:
    """Sort checkpoints deterministically by kind, training step, then name."""

    checkpoint = Path(path)
    stem = checkpoint.stem
    match = _STEP_RE.search(stem)
    if match:
        return (1, int(match.group(1)), stem)
    if _BEST_RE.search(stem):
        return (0, -1, stem)
    if _FINAL_RE.search(stem):
        return (2, 2**63 - 1, stem)
    return (1, 2**62, stem)


def discover_checkpoints(pattern: str) -> list[SweepCandidate]:
    """Expand a checkpoint glob into sorted candidates."""

    matches = [Path(path) for path in glob(pattern)]
    if not matches:
        raise FileNotFoundError(f"checkpoint glob matched no files: {pattern}")
    candidates = [
        SweepCandidate(label=checkpoint_label(path), checkpoint=path)
        for path in sorted(matches, key=checkpoint_sort_key)
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"checkpoint glob matched no files: {pattern}")
    return candidates


def required_finite_metric(metrics: Mapping[str, Any], field: str) -> float:
    """Return a finite metric value or raise a readable error."""

    if field not in metrics:
        raise ValueError(f"missing required metric: {field}")
    value = metrics[field]
    if value is None:
        raise ValueError(f"missing required metric: {field}")
    try:
        metric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric required metric {field}: {value!r}") from exc
    if not math.isfinite(metric):
        raise ValueError(f"nonfinite required metric {field}: {value!r}")
    return metric


def rank_key_for_metrics(metrics: Mapping[str, Any], rank_by: str) -> tuple[float, ...]:
    """Build the ranking tuple for one PR11a metrics payload."""

    if rank_by == RANK_TARGET_FUNNEL:
        return (
            required_finite_metric(metrics, "success_rate"),
            required_finite_metric(metrics, "target_hold_episode_rate"),
            required_finite_metric(metrics, "target_2cm_episode_rate"),
            required_finite_metric(metrics, "target_5cm_episode_rate"),
            required_finite_metric(metrics, "target_10cm_episode_rate"),
            required_finite_metric(metrics, "target_20cm_episode_rate"),
            -required_finite_metric(metrics, "p50_cube_to_target_m"),
            -required_finite_metric(metrics, "mean_action_jerk"),
            required_finite_metric(metrics, "mean_return"),
        )
    if rank_by == RANK_SUCCESS_RETURN:
        return (
            required_finite_metric(metrics, "success_rate"),
            required_finite_metric(metrics, "mean_return"),
            -required_finite_metric(metrics, "mean_action_jerk"),
        )
    raise ValueError(
        f"rank_by must be one of {SUPPORTED_CHECKPOINT_SWEEP_RANK_MODES}; got {rank_by!r}"
    )


def summary_row_from_metrics(
    candidate: SweepCandidate,
    *,
    metrics_path: str | Path,
    metrics: Mapping[str, Any],
    rank_by: str,
) -> dict[str, Any]:
    """Convert one metrics JSON payload into a summary row."""

    rank_key = rank_key_for_metrics(metrics, rank_by)
    row: dict[str, Any] = {
        "label": candidate.label,
        "checkpoint": str(Path(metrics.get("checkpoint") or candidate.checkpoint).resolve()),
        "metrics_path": str(Path(metrics_path).resolve()),
        "command_path": "",
        "log_path": "",
        "returncode": "",
        "num_episodes": metrics.get("num_eval_episodes"),
        "rank_by": rank_by,
        "rank_key": list(rank_key),
        "_rank_key_tuple": rank_key,
        "selected": False,
        "status": "ok",
        "error": "",
    }
    for field in METRIC_SUMMARY_FIELDS:
        row[field] = metrics.get(field)
    return row


def failure_summary_row(
    candidate: SweepCandidate,
    *,
    metrics_path: str | Path | None,
    rank_by: str,
    error: BaseException | str,
    command_path: str | Path | None = None,
    log_path: str | Path | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    """Build a row for a candidate that failed eval or ranking."""

    message = str(error)
    row: dict[str, Any] = {
        "label": candidate.label,
        "checkpoint": str(candidate.checkpoint.resolve()),
        "metrics_path": "" if metrics_path is None else str(Path(metrics_path).resolve()),
        "command_path": "" if command_path is None else str(Path(command_path).resolve()),
        "log_path": "" if log_path is None else str(Path(log_path).resolve()),
        "returncode": "" if returncode is None else int(returncode),
        "num_episodes": "",
        "rank_by": rank_by,
        "rank_key": [],
        "selected": False,
        "status": "failed",
        "error": message,
    }
    for field in METRIC_SUMMARY_FIELDS:
        row[field] = ""
    return row


def dry_run_summary_row(
    candidate: SweepCandidate,
    *,
    metrics_path: str | Path,
    rank_by: str,
) -> dict[str, Any]:
    row = failure_summary_row(candidate, metrics_path=metrics_path, rank_by=rank_by, error="")
    row["status"] = "dry_run"
    row["error"] = ""
    return row


def select_best_summary_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mark and return the best successful row."""

    row_list = list(rows)
    candidates = [row for row in row_list if row.get("status") == "ok" and "_rank_key_tuple" in row]
    if not candidates:
        raise ValueError("no successful checkpoint metrics available for selection")
    best = max(candidates, key=lambda row: row["_rank_key_tuple"])
    for row in row_list:
        row["selected"] = row is best
    return best


def public_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        value = row.get(field, "")
        if field == "rank_key" and not isinstance(value, str):
            value = json.dumps(value, separators=(",", ":"))
        public[field] = value
    return public


def write_summary_files(
    out_dir: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    num_episodes: int,
) -> tuple[Path, Path]:
    """Write sweep summary CSV and JSON files."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [public_summary_row(row) for row in rows]
    csv_path = output_dir / f"summary_eval_{num_episodes}eps.csv"
    json_path = output_dir / f"summary_eval_{num_episodes}eps.json"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(public_rows)
    json_path.write_text(json.dumps(public_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def best_fresh_eval_payload(
    *,
    selected_row: Mapping[str, Any] | None,
    rank_by: str,
    candidate_count: int,
    failed_candidates: list[dict[str, Any]],
    command: str,
    summary_csv: str | Path,
    summary_json: str | Path,
    promoted_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build the top-level selection artifact."""

    rank_key = None
    selected_checkpoint = None
    selected_metrics_path = None
    selected_public_row = None
    if selected_row is not None:
        selected_public_row = public_summary_row(selected_row)
        selected_checkpoint = selected_public_row["checkpoint"]
        selected_metrics_path = selected_public_row["metrics_path"]
        rank_key = selected_public_row["rank_key"]
    return {
        "selected_checkpoint": selected_checkpoint,
        "selected_metrics_path": selected_metrics_path,
        "selected_summary_row": selected_public_row,
        "rank_by": rank_by,
        "rank_key": rank_key,
        "candidate_count": int(candidate_count),
        "failed_candidates": failed_candidates,
        "command": command,
        "summary_csv": str(Path(summary_csv).resolve()),
        "summary_json": str(Path(summary_json).resolve()),
        "promoted_checkpoint": None if promoted_checkpoint is None else str(Path(promoted_checkpoint).resolve()),
    }


def promotion_path_for_checkpoint(checkpoint: str | Path) -> Path:
    """Return the explicit report-only opt-in promotion path for a checkpoint."""

    path = Path(checkpoint)
    stem = path.stem
    step_match = _STEP_RE.search(stem)
    if step_match:
        prefix = stem[: step_match.start()].rstrip("_-") or stem
    elif _BEST_RE.search(stem):
        prefix = _BEST_RE.sub("_", stem).strip("_-") or stem
    elif _FINAL_RE.search(stem):
        prefix = _FINAL_RE.sub("_", stem).strip("_-") or stem
    else:
        prefix = stem
    return path.with_name(f"{prefix}_best_fresh_eval{path.suffix}")


def promote_checkpoint(
    checkpoint: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy the selected checkpoint to the fresh-eval-best promotion path."""

    source = Path(checkpoint)
    destination = promotion_path_for_checkpoint(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"promoted checkpoint already exists: {destination}; pass overwrite=True to replace it"
        )
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def write_output_manifest(out_dir: str | Path) -> Path:
    """Write a simple file manifest for upload/reproducibility."""

    output_dir = Path(out_dir)
    manifest_path = output_dir / "output_manifest.txt"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    lines = [str(path.resolve()) for path in files if path.name != manifest_path.name]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest_path


__all__ = [
    "RANK_SUCCESS_RETURN",
    "RANK_TARGET_FUNNEL",
    "SUMMARY_FIELDS",
    "SUPPORTED_CHECKPOINT_SWEEP_RANK_MODES",
    "SweepCandidate",
    "best_fresh_eval_payload",
    "checkpoint_label",
    "checkpoint_sort_key",
    "discover_checkpoints",
    "dry_run_summary_row",
    "failure_summary_row",
    "promote_checkpoint",
    "promotion_path_for_checkpoint",
    "public_summary_row",
    "rank_key_for_metrics",
    "required_finite_metric",
    "select_best_summary_row",
    "summary_row_from_metrics",
    "write_output_manifest",
    "write_summary_files",
]
