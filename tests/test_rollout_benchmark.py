"""Tests for rollout collection benchmark command/CSV plumbing."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import benchmark_rollout_collection
from scripts.benchmark_rollout_collection import (
    BenchmarkConfig,
    CommandRunResult,
    _subprocess_env,
    benchmark,
    build_collect_command,
    dataset_path_for,
    parse_parallel_envs,
    summarize_dataset,
)


def _config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    defaults = dict(
        parallel_envs=[1, 2],
        repeats=1,
        num_episodes=2,
        max_steps=3,
        policy="random",
        output_csv=tmp_path / "logs" / "bench.csv",
        dataset_dir=tmp_path / "data",
        dataset_prefix="bench",
        seed=7,
        device="cuda:0",
        timeout_s=30.0,
        include_raw_policy_images=True,
        include_debug_images=True,
        headless=True,
        collect_progress=False,
        display=":1",
        xauthority=None,
        python_executable="/usr/bin/python",
        fail_fast=False,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def test_parse_parallel_envs_accepts_comma_separated_positive_values() -> None:
    assert parse_parallel_envs("1, 2,4") == [1, 2, 4]

    with pytest.raises(ValueError, match="positive"):
        parse_parallel_envs("1,0")

    with pytest.raises(ValueError, match="at least one"):
        parse_parallel_envs(" , ")


def test_build_collect_command_matches_current_collect_rollouts_cli(tmp_path) -> None:
    config = _config(tmp_path)
    dataset_path = dataset_path_for(config, num_parallel_envs=2, repeat=0)

    command = build_collect_command(config, num_parallel_envs=2, repeat=0, dataset_path=dataset_path)

    assert command[:3] == ["/usr/bin/python", "-m", "scripts.collect_rollouts"]
    assert "--num-parallel-envs" in command
    assert command[command.index("--num-parallel-envs") + 1] == "2"
    assert command[command.index("--num-episodes") + 1] == "2"
    assert command[command.index("--max-steps") + 1] == "3"
    assert command[command.index("--seed") + 1] == "7"
    assert "--include-raw-policy-images" in command
    assert "--include-debug-images" in command
    assert "--no-progress" in command


def test_summarize_dataset_counts_episode_groups_and_steps(tmp_path) -> None:
    dataset_path = tmp_path / "rollouts.h5"
    with h5py.File(dataset_path, "w") as h5_file:
        for episode_index, length in enumerate((2, 3)):
            group = h5_file.create_group(f"episode_{episode_index:03d}")
            group.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))

    summary = summarize_dataset(dataset_path)

    assert summary.episodes_written == 2
    assert summary.actual_steps == 5
    assert summary.dataset_size_bytes > 0


def test_benchmark_writes_csv_rows_with_fake_runner(tmp_path) -> None:
    config = _config(tmp_path, parallel_envs=[1, 2], repeats=1)

    def fake_runner(command: list[str], *, timeout_s: float, env: dict[str, str]) -> CommandRunResult:
        assert timeout_s == config.timeout_s
        assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"
        assert env["PRIVACY_CONSENT"] == "Y"
        assert env["DISPLAY"] == ":1"
        dataset_path = Path(command[command.index("--save-dataset") + 1])
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dataset_path, "w") as h5_file:
            for episode_index in range(config.num_episodes):
                group = h5_file.create_group(f"episode_{episode_index:03d}")
                group.create_dataset("actions", data=np.zeros((config.max_steps, 7), dtype=np.float32))
        return CommandRunResult(returncode=0, wall_time_s=2.0, status="ok")

    rows = benchmark(config, runner=fake_runner)

    assert len(rows) == 2
    assert config.output_csv.exists()
    csv_text = config.output_csv.read_text()
    assert "num_parallel_envs" in csv_text
    assert "episodes_per_s" in csv_text
    assert rows[0]["episodes_written"] == 2
    assert rows[0]["actual_steps"] == 6
    assert rows[0]["episodes_per_s"] == "1.000000"
    assert rows[0]["steps_per_s"] == "3.000000"


def test_subprocess_env_auto_discovers_sddm_xauthority(tmp_path, monkeypatch) -> None:
    """If shell XAUTHORITY is empty, _subprocess_env falls back to /var/run/sddm/{*}."""

    monkeypatch.delenv("XAUTHORITY", raising=False)
    fake_cookie = tmp_path / "{fake-cookie}"
    fake_cookie.write_bytes(b"")
    monkeypatch.setattr(benchmark_rollout_collection, "_discover_sddm_xauthority", lambda: fake_cookie)

    env = _subprocess_env(_config(tmp_path))

    assert env["XAUTHORITY"] == str(fake_cookie)


def test_subprocess_env_respects_explicit_xauthority_override(tmp_path, monkeypatch) -> None:
    """--xauthority PATH wins over shell env and discovery."""

    monkeypatch.setenv("XAUTHORITY", "/shell/cookie")
    monkeypatch.setattr(
        benchmark_rollout_collection,
        "_discover_sddm_xauthority",
        lambda: tmp_path / "{should-not-win}",
    )

    env = _subprocess_env(_config(tmp_path, xauthority="/explicit/cookie"))

    assert env["XAUTHORITY"] == "/explicit/cookie"


def test_subprocess_env_preserves_existing_xauthority(tmp_path, monkeypatch) -> None:
    """If shell already has XAUTHORITY and no explicit override, discovery is skipped."""

    monkeypatch.setenv("XAUTHORITY", "/shell/cookie")

    def _must_not_be_called() -> None:
        raise AssertionError("discovery should not run when XAUTHORITY is already set")

    monkeypatch.setattr(benchmark_rollout_collection, "_discover_sddm_xauthority", _must_not_be_called)

    env = _subprocess_env(_config(tmp_path))

    assert env["XAUTHORITY"] == "/shell/cookie"
