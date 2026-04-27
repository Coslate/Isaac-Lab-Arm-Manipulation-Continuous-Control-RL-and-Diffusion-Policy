"""Tests for the auto-reset visual diagnostic script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.inspect_auto_reset_visuals import parse_args, run_fake_backend


def _base_args(tmp_path: Path, **overrides) -> list[str]:
    args = [
        "--backend",
        "fake",
        "--num-envs",
        "2",
        "--capture-lane",
        "1",
        "--max-steps",
        "20",
        "--after-reset-steps",
        "0,1,3",
        "--save-dir",
        str(tmp_path / "auto_reset_visuals"),
    ]
    for key, value in overrides.items():
        args.extend([f"--{key}", str(value)])
    return args


def test_auto_reset_visual_diagnostic_writes_before_after_frames_and_summary(tmp_path) -> None:
    args = parse_args(_base_args(tmp_path))

    result = run_fake_backend(args)

    assert result["status"] == "ok"
    assert result["backend"] == "fake"
    assert result["num_events"] == 1
    assert result["capture_lane"] == 1
    assert result["after_reset_steps"] == [0, 1, 3]
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    event = payload["events"][0]
    assert event["terminated"] is True
    assert event["truncated"] is False
    assert set(event["snapshots"]) == {
        "before_done",
        "after_reset_step000",
        "after_reset_step001",
        "after_reset_step003",
    }
    for snapshot in event["snapshots"].values():
        assert Path(snapshot["policy_png"]).exists()
        assert Path(snapshot["debug_png"]).exists()
        assert set(snapshot["proprio"]) == {"cube_pos_base", "target_pos_base", "cube_to_target"}

    before = event["snapshots"]["before_done"]
    after0 = event["snapshots"]["after_reset_step000"]
    with Image.open(before["debug_png"]) as before_debug, Image.open(after0["debug_png"]) as after_debug:
        assert before_debug.size == (128, 96)
        assert after_debug.size == (128, 96)
        assert before_debug.convert("RGB").getpixel((0, 0)) != after_debug.convert("RGB").getpixel((0, 0))
    with Image.open(before["policy_png"]) as before_policy, Image.open(after0["policy_png"]) as after_policy:
        assert before_policy.size == (224, 224)
        assert after_policy.size == (224, 224)
        assert before_policy.convert("RGB").getpixel((0, 0)) != after_policy.convert("RGB").getpixel((0, 0))


def test_auto_reset_visual_diagnostic_records_nested_done_during_long_after_window(tmp_path) -> None:
    args = parse_args(_base_args(tmp_path, **{"after-reset-steps": "0,1,5"}))

    result = run_fake_backend(args)
    event = result["events"][0]

    assert event["nested_done_after_reset_steps"] == [4]
    assert Path(event["snapshots"]["after_reset_step005"]["debug_png"]).exists()


def test_auto_reset_visual_diagnostic_rejects_bad_after_reset_steps(tmp_path) -> None:
    args = _base_args(tmp_path)
    index = args.index("--after-reset-steps")
    args[index] = "--after-reset-steps=-1,2"
    del args[index + 1]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_args(args)


def test_auto_reset_visual_diagnostic_raises_when_no_reset_seen(tmp_path) -> None:
    args = parse_args(_base_args(tmp_path, **{"max-steps": "2"}))

    with pytest.raises(RuntimeError, match="no done/truncated event"):
        run_fake_backend(args)
