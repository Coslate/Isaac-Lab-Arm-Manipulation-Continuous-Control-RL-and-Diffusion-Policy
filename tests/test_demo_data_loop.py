"""Tests for the one-command demo data loop."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from PIL import Image

from dataset import list_episode_keys
from scripts.demo_data_loop import (
    TARGET_RETICLE_COLOR,
    _draw_target_overlay,
    parse_args,
    run_demo_data_loop,
)


def _run_fake_demo(
    tmp_path,
    *,
    policy: str = "random",
    replay_dataset=None,
    settle_steps: int = 0,
    target_overlay: str = "none",
    save_mp4: bool = False,
):
    args = [
        "--backend",
        "fake",
        "--policy",
        policy,
        "--num_episodes",
        "1",
        "--max-steps",
        "5",
        "--settle-steps",
        str(settle_steps),
        "--save_dataset",
        str(tmp_path / f"{policy}_rollouts.h5"),
        "--save_metrics",
        str(tmp_path / "logs" / f"{policy}_metrics.json"),
        "--save_gif",
        str(tmp_path / "gifs" / f"{policy}.gif"),
        "--save-debug-frames-dir",
        str(tmp_path / "debug_frames"),
        "--target-overlay",
        target_overlay,
        "--no-progress",
    ]
    if save_mp4:
        args.extend(["--save_mp4", str(tmp_path / "videos" / f"{policy}.mp4")])
    if replay_dataset is not None:
        args.extend(["--replay_dataset", str(replay_dataset)])
    return run_demo_data_loop(parse_args(args))


def test_cli_accepts_required_demo_options(tmp_path) -> None:
    args = parse_args(
        [
            "--backend",
            "fake",
            "--policy",
            "random",
            "--num_episodes",
            "1",
            "--save_dataset",
            str(tmp_path / "rollouts.h5"),
            "--save_metrics",
            str(tmp_path / "metrics.json"),
            "--save_gif",
            str(tmp_path / "demo.gif"),
        ]
    )

    assert args.backend == "fake"
    assert args.policy == "random"
    assert args.num_episodes == 1
    assert args.settle_steps == 0
    assert args.target_overlay == "none"
    assert str(args.save_dataset).endswith("rollouts.h5")
    assert str(args.save_metrics).endswith("metrics.json")
    assert str(args.save_gif).endswith("demo.gif")
    assert args.save_mp4 is None


@pytest.mark.parametrize("policy", ["random", "heuristic"])
def test_demo_data_loop_creates_dataset_metrics_and_gif(tmp_path, policy: str) -> None:
    result = _run_fake_demo(tmp_path, policy=policy)

    assert result["status"] == "ok"
    assert result["backend"] == "fake"
    assert result["policy"] == policy
    assert result["episode_keys"] == ["episode_000"]
    assert result["gif_num_frames"] > 1
    assert list_episode_keys(result["save_dataset"]) == ["episode_000"]

    with h5py.File(result["save_dataset"], "r") as h5_file:
        assert int(h5_file.attrs["num_episodes"]) == 1
        assert h5_file["episode_000"]["images"].shape[1:] == (3, 224, 224)

    metrics = json.loads(open(result["save_metrics"], encoding="utf-8").read())
    assert {
        "policy_name",
        "env_backend",
        "num_episodes",
        "mean_return",
        "success_rate",
        "mean_episode_length",
        "mean_action_jerk",
        "success_threshold_m",
        "consecutive_success_steps",
        "success_source",
        "success_distance_metric",
        "success_distance_source",
        "episode_successes",
        "closest_target_approach_by_episode",
        "target_position_base_m",
        "target_position_base_m_source",
        "target_positions_base_m_by_episode",
        "target_position_constant_by_episode",
        "target_debug_camera_name",
        "target_debug_pixel",
        "target_debug_pixel_by_episode",
        "target_debug_pixel_visible",
        "target_debug_pixel_visible_by_episode",
        "target_debug_pixel_source",
    } <= set(metrics)
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["episode_successes"] == {"episode_000": True}
    assert metrics["closest_target_approach_by_episode"]["episode_000"]["success"] is True
    assert metrics["closest_target_approach_by_episode"]["episode_000"]["closest_step"] >= 0
    assert metrics["closest_target_approach_by_episode"]["episode_000"]["closest_distance_m"] >= 0.0
    assert metrics["target_position_base_m"] == pytest.approx([0.45, 0.0, 0.2])
    assert metrics["target_position_base_m_source"] == "episode_000_step_000_proprio_24_27"
    assert metrics["target_position_constant_by_episode"] is True
    assert metrics["target_debug_pixel"] == [100, 32]
    assert metrics["target_debug_pixel_by_episode"] == {"episode_000": [100, 32]}
    assert metrics["target_debug_pixel_visible"] is True
    assert metrics["target_debug_pixel_visible_by_episode"] == {"episode_000": True}
    assert metrics["target_debug_pixel_source"] == "debug_camera_projection"

    with Image.open(result["save_gif"]) as gif:
        assert gif.n_frames == result["gif_num_frames"]
        assert gif.n_frames > 1
    assert len(result["sampled_debug_frames"]) > 0


def test_demo_data_loop_can_save_mp4(tmp_path) -> None:
    pytest.importorskip("imageio")
    result = _run_fake_demo(tmp_path, policy="heuristic", target_overlay="text-reticle", save_mp4=True)

    assert result["save_mp4"].endswith("heuristic.mp4")
    with open(result["save_mp4"], "rb") as mp4_file:
        assert len(mp4_file.read(16)) > 0


def test_target_overlay_draws_reticle_and_coordinate_text_on_debug_frames(tmp_path) -> None:
    result = _run_fake_demo(tmp_path, policy="heuristic", target_overlay="text-reticle")

    assert result["target_overlay"] == "text-reticle"
    with Image.open(result["sampled_debug_frames"][0]) as frame:
        rgb = frame.convert("RGB")
        assert rgb.getpixel((100, 32)) == TARGET_RETICLE_COLOR
        pixels = np.asarray(rgb)
        label_mask = (pixels[..., 0] > 200) & (pixels[..., 1] > 180) & (pixels[..., 2] < 100)
        assert np.any(label_mask)


def test_target_overlay_skips_out_of_view_projection() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    projection = {"pixel": [20, 10], "visible": False}

    annotated = _draw_target_overlay(frame, projection, mode="text-reticle")

    np.testing.assert_array_equal(annotated, frame)


def test_settle_steps_are_warmed_up_before_recording(tmp_path) -> None:
    result = _run_fake_demo(tmp_path, policy="heuristic", settle_steps=2)

    assert result["settle_steps"] == 2
    with h5py.File(result["save_dataset"], "r") as h5_file:
        images = h5_file["episode_000"]["images"]
        assert images.shape[0] == 5
        assert int(images[0, 1, 0, 0]) == 40


def test_replay_policy_requires_replay_dataset(tmp_path) -> None:
    with pytest.raises(ValueError, match="--replay-dataset is required"):
        parse_args(
            [
                "--backend",
                "fake",
                "--policy",
                "replay",
                "--save_dataset",
                str(tmp_path / "rollouts.h5"),
                "--save_metrics",
                str(tmp_path / "metrics.json"),
                "--save_gif",
                str(tmp_path / "demo.gif"),
            ]
        )


def test_replay_policy_creates_outputs_from_saved_dataset(tmp_path) -> None:
    source = _run_fake_demo(tmp_path / "source", policy="heuristic")
    replay = _run_fake_demo(tmp_path / "replay", policy="replay", replay_dataset=source["save_dataset"])

    assert replay["policy"] == "replay"
    assert list_episode_keys(replay["save_dataset"]) == ["episode_000"]
    assert replay["gif_num_frames"] > 1
    with Image.open(replay["save_gif"]) as gif:
        assert gif.n_frames == replay["gif_num_frames"]
    metrics = json.loads(open(replay["save_metrics"], encoding="utf-8").read())
    assert metrics["policy_name"] == "replay"
