"""Tests for PR6.18 fresh checkpoint eval sweep and selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import pytest

from agents.fake_checkpoints import make_fake_sac_checkpoint, make_fake_td3_checkpoint
from eval.checkpoint_sweep import (
    RANK_SUCCESS_RETURN,
    RANK_TARGET_FUNNEL,
    SUMMARY_FIELDS,
    SweepCandidate,
    discover_checkpoints,
    failure_summary_row,
    promotion_path_for_checkpoint,
    rank_key_for_metrics,
    select_best_summary_row,
    summary_row_from_metrics,
    write_summary_files,
)
from scripts import sweep_checkpoints_continuous


def _metrics(**overrides) -> dict[str, object]:
    metrics: dict[str, object] = {
        "agent_type": "sac",
        "checkpoint": "/tmp/fake.pt",
        "num_env_steps": 100,
        "num_eval_episodes": 100,
        "mean_return": 10.0,
        "success_rate": 0.5,
        "target_hold_episode_rate": 0.25,
        "target_20cm_episode_rate": 0.8,
        "target_10cm_episode_rate": 0.6,
        "target_5cm_episode_rate": 0.4,
        "target_2cm_episode_rate": 0.2,
        "mean_cube_to_target_m": 0.12,
        "p50_cube_to_target_m": 0.10,
        "final_cube_to_target_m": 0.08,
        "min_cube_to_target_m": 0.02,
        "max_cube_lift_m": 0.05,
        "min_ee_to_cube_m": 0.01,
        "gripper_close_near_cube_rate": 0.7,
        "mean_action_jerk": 0.2,
    }
    metrics.update(overrides)
    return metrics


def _touch(path: Path, text: str = "checkpoint") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _patch_short_fake_env(monkeypatch, terminal_step: int = 3) -> None:
    from scripts import train_sac_continuous
    from scripts.train_sac_continuous import _FakeSACEnv

    def _short_env(*, num_envs: int = 1, seed: int = 0):
        return _FakeSACEnv(num_envs=num_envs, seed=seed, terminal_step=terminal_step)

    monkeypatch.setattr(train_sac_continuous, "_build_fake_env", _short_env)


def test_discover_checkpoints_sorts_by_kind_step_and_name(tmp_path):
    final = _touch(tmp_path / "run_final.pt")
    step_200 = _touch(tmp_path / "run_step_000200.pt")
    step_100 = _touch(tmp_path / "run_step_000100.pt")
    best = _touch(tmp_path / "run_best.pt")

    candidates = discover_checkpoints(str(tmp_path / "run_*.pt"))

    assert [candidate.checkpoint for candidate in candidates] == [best, step_100, step_200, final]


def test_dry_run_writes_commands_and_summary_without_eval(tmp_path):
    _touch(tmp_path / "run_step_000100.pt")
    _touch(tmp_path / "run_step_000200.pt")
    out_dir = tmp_path / "out"

    args = sweep_checkpoints_continuous.parse_args(
        [
            "--backend",
            "fake",
            "--agent_type",
            "sac",
            "--checkpoint_glob",
            str(tmp_path / "run_*.pt"),
            "--out_dir",
            str(out_dir),
            "--num_envs",
            "1",
            "--num_episodes",
            "4",
            "--max_steps",
            "8",
            "--settle_steps",
            "0",
            "--device",
            "cpu",
            "--rank_by",
            RANK_TARGET_FUNNEL,
            "--dry_run",
            "--no-progress",
            "--no-headless",
        ]
    )
    result = sweep_checkpoints_continuous.run_sweep(args)

    assert result["status"] == "dry_run"
    assert result["candidate_count"] == 2
    assert result["selected_checkpoint"] is None
    assert (out_dir / "commands_used.txt").exists()
    assert (out_dir / "output_manifest.txt").exists()
    assert not list(out_dir.glob("run_step_*_eval_4eps.json"))

    rows = json.loads((out_dir / "summary_eval_4eps.json").read_text())
    assert {row["status"] for row in rows} == {"dry_run"}
    best_payload = json.loads((out_dir / "best_fresh_eval.json").read_text())
    assert best_payload["selected_checkpoint"] is None


def test_empty_checkpoint_glob_fails_readably(tmp_path):
    args = sweep_checkpoints_continuous.parse_args(
        [
            "--backend",
            "fake",
            "--agent-type",
            "sac",
            "--checkpoint-glob",
            str(tmp_path / "missing_*.pt"),
            "--out-dir",
            str(tmp_path / "out"),
            "--num-envs",
            "1",
            "--num-episodes",
            "1",
            "--max-steps",
            "1",
            "--settle-steps",
            "0",
            "--device",
            "cpu",
            "--dry-run",
            "--no-progress",
            "--no-headless",
        ]
    )
    with pytest.raises(FileNotFoundError, match="checkpoint glob matched no files"):
        sweep_checkpoints_continuous.run_sweep(args)


def test_target_funnel_ranking_prefers_success_over_return():
    high_return = _metrics(success_rate=0.40, mean_return=999.0)
    high_success = _metrics(success_rate=0.41, mean_return=1.0)

    assert rank_key_for_metrics(high_success, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        high_return, RANK_TARGET_FUNNEL
    )


def test_target_funnel_ranking_prefers_hold_when_success_ties():
    low_hold = _metrics(success_rate=0.5, target_hold_episode_rate=0.2)
    high_hold = _metrics(success_rate=0.5, target_hold_episode_rate=0.3)

    assert rank_key_for_metrics(high_hold, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        low_hold, RANK_TARGET_FUNNEL
    )


@pytest.mark.parametrize(
    "field",
    [
        "target_2cm_episode_rate",
        "target_5cm_episode_rate",
        "target_10cm_episode_rate",
        "target_20cm_episode_rate",
    ],
)
def test_target_funnel_ranking_prefers_funnel_rates_in_order(field):
    low = _metrics(
        success_rate=0.5,
        target_hold_episode_rate=0.5,
        target_2cm_episode_rate=0.5,
        target_5cm_episode_rate=0.5,
        target_10cm_episode_rate=0.5,
        target_20cm_episode_rate=0.5,
    )
    high = dict(low)
    high[field] = float(high[field]) + 0.1

    assert rank_key_for_metrics(high, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        low, RANK_TARGET_FUNNEL
    )


def test_target_funnel_ranking_prefers_lower_distance_then_lower_jerk_then_return():
    base = _metrics(success_rate=0.5, target_hold_episode_rate=0.5, mean_return=10.0)
    closer = _metrics(success_rate=0.5, target_hold_episode_rate=0.5, p50_cube_to_target_m=0.09)
    smoother = _metrics(
        success_rate=0.5,
        target_hold_episode_rate=0.5,
        p50_cube_to_target_m=0.10,
        mean_action_jerk=0.1,
    )
    higher_return = _metrics(
        success_rate=0.5,
        target_hold_episode_rate=0.5,
        p50_cube_to_target_m=0.10,
        mean_action_jerk=0.2,
        mean_return=11.0,
    )

    assert rank_key_for_metrics(closer, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        base, RANK_TARGET_FUNNEL
    )
    assert rank_key_for_metrics(smoother, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        base, RANK_TARGET_FUNNEL
    )
    assert rank_key_for_metrics(higher_return, RANK_TARGET_FUNNEL) > rank_key_for_metrics(
        base, RANK_TARGET_FUNNEL
    )


def test_success_return_ranking_is_simpler_fallback():
    high_return = _metrics(success_rate=0.5, mean_return=12.0, mean_action_jerk=2.0)
    low_return = _metrics(success_rate=0.5, mean_return=11.0, mean_action_jerk=0.1)

    assert rank_key_for_metrics(high_return, RANK_SUCCESS_RETURN) > rank_key_for_metrics(
        low_return, RANK_SUCCESS_RETURN
    )


def test_missing_or_nonfinite_required_metric_is_rejected():
    missing = _metrics()
    missing.pop("target_2cm_episode_rate")
    with pytest.raises(ValueError, match="missing required metric"):
        rank_key_for_metrics(missing, RANK_TARGET_FUNNEL)

    nonfinite = _metrics(p50_cube_to_target_m=float("nan"))
    with pytest.raises(ValueError, match="nonfinite required metric"):
        rank_key_for_metrics(nonfinite, RANK_TARGET_FUNNEL)


def test_summary_files_include_required_fields_and_selected_row(tmp_path):
    checkpoint_a = _touch(tmp_path / "run_step_000100.pt")
    checkpoint_b = _touch(tmp_path / "run_step_000200.pt")
    row_a = summary_row_from_metrics(
        SweepCandidate("a", checkpoint_a),
        metrics_path=tmp_path / "a.json",
        metrics=_metrics(checkpoint=str(checkpoint_a), success_rate=0.1),
        rank_by=RANK_TARGET_FUNNEL,
    )
    row_b = summary_row_from_metrics(
        SweepCandidate("b", checkpoint_b),
        metrics_path=tmp_path / "b.json",
        metrics=_metrics(checkpoint=str(checkpoint_b), success_rate=0.2),
        rank_by=RANK_TARGET_FUNNEL,
    )
    failed = failure_summary_row(
        SweepCandidate("bad", tmp_path / "bad.pt"),
        metrics_path=tmp_path / "bad.json",
        rank_by=RANK_TARGET_FUNNEL,
        error="missing metric",
    )
    rows = [row_a, row_b, failed]
    best = select_best_summary_row(rows)
    csv_path, json_path = write_summary_files(tmp_path / "summary", rows, num_episodes=100)

    assert best is row_b
    parsed_rows = json.loads(json_path.read_text())
    assert set(SUMMARY_FIELDS).issubset(parsed_rows[0].keys())
    assert [row["selected"] for row in parsed_rows] == [False, True, False]

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert csv_rows[1]["label"] == "b"
    assert csv_rows[1]["selected"] == "True"
    assert csv_rows[2]["status"] == "failed"


def test_run_sweep_records_bad_metrics_and_selects_good_candidate(tmp_path, monkeypatch):
    good = _touch(tmp_path / "run_step_000100.pt")
    bad = _touch(tmp_path / "run_step_000200.pt")
    out_dir = tmp_path / "out"

    def _fake_run_eval(_args, candidate: SweepCandidate, metrics_path: Path):
        metrics = _metrics(checkpoint=str(candidate.checkpoint), success_rate=0.8)
        if candidate.checkpoint == bad:
            metrics["p50_cube_to_target_m"] = float("inf")
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return {"status": "ok"}

    monkeypatch.setattr(sweep_checkpoints_continuous, "_run_eval", _fake_run_eval)
    args = sweep_checkpoints_continuous.parse_args(
        [
            "--backend",
            "fake",
            "--agent-type",
            "sac",
            "--checkpoint-glob",
            str(tmp_path / "run_*.pt"),
            "--out-dir",
            str(out_dir),
            "--num-envs",
            "1",
            "--num-episodes",
            "8",
            "--max-steps",
            "8",
            "--settle-steps",
            "0",
            "--device",
            "cpu",
            "--no-progress",
            "--no-headless",
        ]
    )
    result = sweep_checkpoints_continuous.run_sweep(args)

    assert result["status"] == "ok"
    assert result["selected_checkpoint"] == str(good.resolve())
    assert result["failed_candidates"][0]["checkpoint"] == str(bad.resolve())
    best_payload = json.loads((out_dir / "best_fresh_eval.json").read_text())
    assert best_payload["selected_checkpoint"] == str(good.resolve())
    assert best_payload["failed_candidates"][0]["error"]


def test_promote_best_copies_selected_checkpoint_only_when_requested(tmp_path, monkeypatch):
    checkpoint = _touch(tmp_path / "run_step_000100.pt", text="selected-weights")
    out_dir = tmp_path / "out"

    def _fake_run_eval(_args, candidate: SweepCandidate, metrics_path: Path):
        metrics_path.write_text(
            json.dumps(_metrics(checkpoint=str(candidate.checkpoint), success_rate=1.0)),
            encoding="utf-8",
        )
        return {"status": "ok"}

    monkeypatch.setattr(sweep_checkpoints_continuous, "_run_eval", _fake_run_eval)

    base_args = [
        "--backend",
        "fake",
        "--agent-type",
        "sac",
        "--checkpoint-glob",
        str(tmp_path / "run_*.pt"),
        "--out-dir",
        str(out_dir),
        "--num-envs",
        "1",
        "--num-episodes",
        "8",
        "--max-steps",
        "8",
        "--settle-steps",
        "0",
        "--device",
        "cpu",
        "--no-progress",
        "--no-headless",
    ]
    report_only = sweep_checkpoints_continuous.run_sweep(
        sweep_checkpoints_continuous.parse_args(base_args)
    )
    assert report_only["promoted_checkpoint"] is None
    assert not promotion_path_for_checkpoint(checkpoint).exists()

    promoted = sweep_checkpoints_continuous.run_sweep(
        sweep_checkpoints_continuous.parse_args(
            base_args + ["--out-dir", str(tmp_path / "out_promote"), "--promote-best"]
        )
    )
    promoted_path = Path(promoted["promoted_checkpoint"])
    assert promoted_path.exists()
    assert promoted_path.read_text(encoding="utf-8") == "selected-weights"


@pytest.mark.parametrize(
    ("agent_type", "factory"),
    [
        ("sac", make_fake_sac_checkpoint),
        ("td3", make_fake_td3_checkpoint),
    ],
)
def test_fake_sac_and_td3_checkpoints_run_through_sweep(
    tmp_path,
    monkeypatch,
    agent_type: str,
    factory: Callable[..., Path],
):
    _patch_short_fake_env(monkeypatch, terminal_step=3)
    checkpoint = factory(tmp_path / f"{agent_type}_step_000001.pt", seed=0, num_env_steps=123)
    out_dir = tmp_path / f"{agent_type}_out"
    args = sweep_checkpoints_continuous.parse_args(
        [
            "--backend",
            "fake",
            "--agent-type",
            agent_type,
            "--checkpoint-glob",
            str(tmp_path / f"{agent_type}_*.pt"),
            "--out-dir",
            str(out_dir),
            "--num-envs",
            "1",
            "--num-episodes",
            "2",
            "--max-steps",
            "5",
            "--settle-steps",
            "0",
            "--device",
            "cpu",
            "--rank-by",
            RANK_SUCCESS_RETURN,
            "--no-progress",
            "--no-headless",
        ]
    )
    result = sweep_checkpoints_continuous.run_sweep(args)

    assert result["status"] == "ok"
    assert result["selected_checkpoint"] == str(checkpoint.resolve())
    metrics_path = out_dir / f"{checkpoint.stem}_eval_2eps.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["agent_type"] == agent_type
    assert metrics["num_eval_episodes"] == 2
    assert metrics["num_env_steps"] == 123


def test_parser_aliases_and_invalid_values(tmp_path):
    _touch(tmp_path / "run_step_000100.pt")
    args = sweep_checkpoints_continuous.parse_args(
        [
            "--backend",
            "fake",
            "--agent_type",
            "sac",
            "--checkpoint_glob",
            str(tmp_path / "run_*.pt"),
            "--out_dir",
            str(tmp_path / "out"),
            "--num_envs",
            "1",
            "--num_episodes",
            "1",
            "--max_steps",
            "1",
            "--settle_steps",
            "0",
            "--rank_by",
            RANK_SUCCESS_RETURN,
        ]
    )
    assert args.agent_type == "sac"
    assert args.rank_by == RANK_SUCCESS_RETURN

    with pytest.raises(SystemExit):
        sweep_checkpoints_continuous.parse_args(
            [
                "--agent-type",
                "sac",
                "--checkpoint-glob",
                str(tmp_path / "run_*.pt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--rank-by",
                "not_a_rank_mode",
            ]
        )
    invalid_cases = [
        ("--num-envs", "0", "--num-envs must be positive"),
        ("--num-episodes", "0", "--num-episodes must be positive"),
        ("--max-steps", "0", "--max-steps must be positive"),
    ]
    for flag, value, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            sweep_checkpoints_continuous.parse_args(
                [
                    "--agent-type",
                    "sac",
                    "--checkpoint-glob",
                    str(tmp_path / "run_*.pt"),
                    "--out-dir",
                    str(tmp_path / "out"),
                    flag,
                    value,
                ]
            )
