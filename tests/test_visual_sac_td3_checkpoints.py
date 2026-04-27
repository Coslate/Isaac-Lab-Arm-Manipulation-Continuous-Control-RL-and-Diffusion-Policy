"""Tests for PR12a SAC/TD3 checkpoint visual rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from agents.fake_checkpoints import make_fake_sac_checkpoint, make_fake_td3_checkpoint
from eval import TARGET_RETICLE_COLOR
from scripts import record_gif_continuous
from scripts.record_gif_continuous import (
    _FakeVisualCheckpointEnv,
    parse_args,
    run_fake_backend,
)


def _base_args(tmp_path: Path, *, agent_type: str, checkpoint: Path, **overrides) -> list[str]:
    args = [
        "--backend",
        "fake",
        "--agent-type",
        agent_type,
        "--checkpoint",
        str(checkpoint),
        "--save-gif",
        str(tmp_path / "gifs" / f"{agent_type}.gif"),
        "--save-debug-frames-dir",
        str(tmp_path / "debug_frames" / agent_type),
        "--save-metrics",
        str(tmp_path / "logs" / f"{agent_type}_visual_metrics.json"),
        "--gif-max-steps",
        "5",
        "--settle-steps",
        "0",
        "--device",
        "cpu",
        "--seed",
        "0",
        "--no-headless",
    ]
    for key, value in overrides.items():
        args.extend([f"--{key}", str(value)])
    return args


def test_cli_accepts_pr12a_required_options(tmp_path) -> None:
    checkpoint = tmp_path / "sac.pt"
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=checkpoint))

    assert args.backend == "fake"
    assert args.agent_type == "sac"
    assert args.checkpoint == str(checkpoint)
    assert str(args.save_gif).endswith("sac.gif")
    assert str(args.save_metrics).endswith("sac_visual_metrics.json")
    assert args.gif_max_steps == 5
    assert args.settle_steps == 0


@pytest.mark.parametrize(
    ("agent_type", "factory"),
    [("sac", make_fake_sac_checkpoint), ("td3", make_fake_td3_checkpoint)],
)
def test_record_checkpoint_visuals_create_gif_png_and_metrics(tmp_path, agent_type, factory) -> None:
    checkpoint = factory(tmp_path / f"{agent_type}.pt", seed=0, num_env_steps=123)
    args = parse_args(_base_args(tmp_path, agent_type=agent_type, checkpoint=checkpoint))

    result = run_fake_backend(args)

    assert result["status"] == "ok"
    assert result["agent_type"] == agent_type
    assert result["gif_num_frames"] == 5
    assert Path(result["save_gif"]).exists()
    assert Path(result["save_metrics"]).exists()
    assert len(result["sampled_debug_frames"]) == 3
    assert all(Path(path).exists() for path in result["sampled_debug_frames"])

    payload = json.loads(Path(result["save_metrics"]).read_text(encoding="utf-8"))
    assert payload["agent_type"] == agent_type
    assert payload["num_eval_episodes"] == 1
    assert payload["num_env_steps"] == 123
    assert payload["visual_rollout_source"] == "fresh_env_reset"
    assert payload["overlay_metrics_source"] == "visual_rollout"
    assert payload["gif_num_frames"] == 5
    assert payload["target_debug_pixel_source"] == "debug_camera_projection"
    assert payload["target_debug_pixel_by_episode"] == {"episode_000": [100, 32]}
    assert {"mean_return", "success_rate", "mean_action_jerk"} <= set(payload)
    assert len(payload["visual_rollout_reward_trace"]) == payload["visual_rollout_reward_num_steps"] == 5
    assert payload["visual_rollout_reward_sum"] == pytest.approx(payload["mean_return"])
    assert payload["visual_rollout_reward_min"] <= payload["visual_rollout_reward_mean"] <= payload["visual_rollout_reward_max"]

    with Image.open(result["save_gif"]) as gif:
        assert gif.n_frames == result["gif_num_frames"]


def test_record_checkpoint_visuals_can_save_mp4(tmp_path) -> None:
    pytest.importorskip("imageio")
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    args = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
        + ["--save-mp4", str(tmp_path / "videos" / "sac.mp4")]
    )

    result = run_fake_backend(args)

    assert result["save_mp4"] is not None
    assert Path(result["save_mp4"]).exists()
    assert Path(result["save_mp4"]).stat().st_size > 0


def test_visual_debug_pngs_come_from_debug_camera_not_policy_image(tmp_path) -> None:
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=checkpoint))

    result = run_fake_backend(args)

    with Image.open(result["sampled_debug_frames"][0]) as frame:
        # Fake policy images are dark [8, 8, 8] or wrist tensors with red=11.
        # The fixed debug camera background is intentionally different.
        assert frame.convert("RGB").getpixel((0, 0)) == (36, 39, 42)


def test_target_overlay_draws_reticle_on_sampled_debug_frames(tmp_path) -> None:
    checkpoint = make_fake_td3_checkpoint(tmp_path / "td3.pt", seed=0, num_env_steps=10)
    args = parse_args(
        _base_args(tmp_path, agent_type="td3", checkpoint=checkpoint)
        + ["--target-overlay", "text-reticle"]
    )

    result = run_fake_backend(args)

    with Image.open(result["sampled_debug_frames"][0]) as frame:
        assert frame.convert("RGB").getpixel((100, 32)) == TARGET_RETICLE_COLOR


def test_target_overlay_falls_back_when_projection_api_missing(tmp_path, monkeypatch) -> None:
    class NoProjectionFakeEnv(_FakeVisualCheckpointEnv):
        project_base_point_to_debug_pixel = None

    def no_projection_env(*, num_envs: int = 1, seed: int = 0, terminal_step: int = 6):
        return NoProjectionFakeEnv(num_envs=num_envs, seed=seed, terminal_step=terminal_step)

    monkeypatch.setattr(record_gif_continuous, "_build_fake_visual_env", no_projection_env)
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    args = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
        + ["--target-overlay", "text-reticle"]
    )

    result = run_fake_backend(args)
    payload = json.loads(Path(result["save_metrics"]).read_text(encoding="utf-8"))

    assert Path(result["save_gif"]).exists()
    assert payload["target_debug_pixel_source"] == "not_available_no_debug_camera_projection_api"
    assert payload["target_debug_pixel_by_episode"] == {}


def test_record_visuals_reject_missing_checkpoint(tmp_path) -> None:
    args = parse_args(_base_args(tmp_path, agent_type="sac", checkpoint=tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_fake_backend(args)


def test_record_visuals_reject_unknown_agent_type(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "fake",
                "--agent-type",
                "ppo",
                "--checkpoint",
                str(tmp_path / "x.pt"),
                "--save-gif",
                str(tmp_path / "x.gif"),
            ]
        )


def test_metrics_payload_mismatch_guard_rejects_wrong_seed(tmp_path) -> None:
    checkpoint = make_fake_sac_checkpoint(tmp_path / "sac.pt", seed=0, num_env_steps=10)
    metrics_path = tmp_path / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "agent_type": "sac",
                "checkpoint": str(checkpoint.resolve()),
                "seed": 99,
                "settle_steps": 0,
                "deterministic": True,
                "mean_return": 1.0,
                "success_rate": 0.0,
                "mean_action_jerk": 0.0,
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(
        _base_args(tmp_path, agent_type="sac", checkpoint=checkpoint)
        + ["--metrics-payload", str(metrics_path)]
    )

    with pytest.raises(ValueError, match="seed"):
        run_fake_backend(args)


def test_metrics_payload_can_drive_overlay_and_saved_top_level_metrics(tmp_path) -> None:
    checkpoint = make_fake_td3_checkpoint(tmp_path / "td3.pt", seed=0, num_env_steps=10)
    metrics_path = tmp_path / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "agent_type": "td3",
                "checkpoint": str(checkpoint.resolve()),
                "seed": 0,
                "settle_steps": 0,
                "deterministic": True,
                "mean_return": 123.0,
                "success_rate": 0.5,
                "mean_action_jerk": 0.25,
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(
        _base_args(tmp_path, agent_type="td3", checkpoint=checkpoint)
        + ["--metrics-payload", str(metrics_path)]
    )

    result = run_fake_backend(args)
    payload = json.loads(Path(result["save_metrics"]).read_text(encoding="utf-8"))

    assert payload["mean_return"] == 123.0
    assert payload["success_rate"] == 0.5
    assert payload["mean_action_jerk"] == 0.25
    assert payload["overlay_metrics_source"] == str(metrics_path.resolve())
    assert len(payload["visual_rollout_reward_trace"]) == 5
    assert payload["visual_rollout_metrics"]["agent_type"] == "td3"
    assert payload["visual_rollout_metrics"]["num_eval_episodes"] == 1
    assert len(payload["visual_rollout_metrics"]["visual_rollout_reward_trace"]) == 5
