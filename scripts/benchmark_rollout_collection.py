"""Benchmark rollout collection throughput across Isaac vectorized env counts."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py

from dataset.episode_dataset import EPISODE_KEY_PREFIX


CSV_COLUMNS = [
    "timestamp_utc",
    "status",
    "returncode",
    "policy",
    "num_parallel_envs",
    "repeat",
    "seed",
    "num_episodes_requested",
    "episodes_written",
    "max_steps",
    "actual_steps",
    "wall_time_s",
    "episodes_per_s",
    "steps_per_s",
    "include_raw_policy_images",
    "include_debug_images",
    "dataset_path",
    "dataset_size_bytes",
    "command",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    parallel_envs: list[int]
    repeats: int
    num_episodes: int
    max_steps: int
    policy: str
    output_csv: Path
    dataset_dir: Path
    dataset_prefix: str
    seed: int
    device: str
    timeout_s: float
    include_raw_policy_images: bool
    include_debug_images: bool
    headless: bool
    collect_progress: bool
    display: str | None
    xauthority: str | None
    python_executable: str
    fail_fast: bool


@dataclass(frozen=True)
class CommandRunResult:
    returncode: int
    wall_time_s: float
    status: str


@dataclass(frozen=True)
class DatasetSummary:
    episodes_written: int = 0
    actual_steps: int = 0
    dataset_size_bytes: int = 0


def parse_parallel_envs(value: str) -> list[int]:
    """Parse a comma-separated list of positive vectorized-env counts."""

    parsed: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        count = int(stripped)
        if count <= 0:
            raise ValueError("parallel env counts must be positive")
        parsed.append(count)
    if not parsed:
        raise ValueError("at least one parallel env count is required")
    return parsed


def dataset_path_for(config: BenchmarkConfig, num_parallel_envs: int, repeat: int) -> Path:
    """Return the per-condition HDF5 path for one benchmark run."""

    image_suffix = []
    if config.include_raw_policy_images:
        image_suffix.append("raw")
    if config.include_debug_images:
        image_suffix.append("debug")
    suffix = "_" + "_".join(image_suffix) if image_suffix else ""
    filename = (
        f"{config.dataset_prefix}_{config.policy}_eps{config.num_episodes}_steps{config.max_steps}"
        f"_envs{num_parallel_envs}_rep{repeat:02d}{suffix}.h5"
    )
    return config.dataset_dir / filename


def build_collect_command(config: BenchmarkConfig, num_parallel_envs: int, repeat: int, dataset_path: Path) -> list[str]:
    """Build the subprocess command for one `scripts.collect_rollouts` run."""

    command = [
        config.python_executable,
        "-m",
        "scripts.collect_rollouts",
        "--backend",
        "isaac",
        "--policy",
        config.policy,
        "--num-parallel-envs",
        str(num_parallel_envs),
        "--num-episodes",
        str(config.num_episodes),
        "--max-steps",
        str(config.max_steps),
        "--save-dataset",
        str(dataset_path),
        "--seed",
        str(config.seed + repeat),
        "--device",
        config.device,
    ]
    command.append("--headless" if config.headless else "--no-headless")
    command.append("--progress" if config.collect_progress else "--no-progress")
    if config.include_raw_policy_images:
        command.append("--include-raw-policy-images")
    if config.include_debug_images:
        command.append("--include-debug-images")
    return command


def run_subprocess(command: list[str], *, timeout_s: float, env: dict[str, str]) -> CommandRunResult:
    """Run one collection subprocess and measure end-to-end wall time."""

    start = time.perf_counter()
    try:
        completed = subprocess.run(command, env=env, timeout=timeout_s, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        return CommandRunResult(
            returncode=int(completed.returncode),
            wall_time_s=time.perf_counter() - start,
            status=status,
        )
    except subprocess.TimeoutExpired:
        return CommandRunResult(returncode=124, wall_time_s=time.perf_counter() - start, status="timeout")


def summarize_dataset(path: Path) -> DatasetSummary:
    """Summarize an HDF5 rollout dataset if it exists and is readable."""

    if not path.exists():
        return DatasetSummary()

    with h5py.File(path, "r") as h5_file:
        episode_keys = sorted(key for key in h5_file.keys() if key.startswith(EPISODE_KEY_PREFIX))
        actual_steps = 0
        for episode_key in episode_keys:
            if "actions" in h5_file[episode_key]:
                actual_steps += int(h5_file[episode_key]["actions"].shape[0])
    return DatasetSummary(
        episodes_written=len(episode_keys),
        actual_steps=actual_steps,
        dataset_size_bytes=int(path.stat().st_size),
    )


def result_row(
    *,
    config: BenchmarkConfig,
    num_parallel_envs: int,
    repeat: int,
    dataset_path: Path,
    command: list[str],
    run_result: CommandRunResult,
    dataset_summary: DatasetSummary,
) -> dict[str, Any]:
    """Build one CSV row."""

    episodes_per_s = (
        dataset_summary.episodes_written / run_result.wall_time_s if run_result.wall_time_s > 0.0 else 0.0
    )
    steps_per_s = dataset_summary.actual_steps / run_result.wall_time_s if run_result.wall_time_s > 0.0 else 0.0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": run_result.status,
        "returncode": run_result.returncode,
        "policy": config.policy,
        "num_parallel_envs": num_parallel_envs,
        "repeat": repeat,
        "seed": config.seed + repeat,
        "num_episodes_requested": config.num_episodes,
        "episodes_written": dataset_summary.episodes_written,
        "max_steps": config.max_steps,
        "actual_steps": dataset_summary.actual_steps,
        "wall_time_s": f"{run_result.wall_time_s:.6f}",
        "episodes_per_s": f"{episodes_per_s:.6f}",
        "steps_per_s": f"{steps_per_s:.6f}",
        "include_raw_policy_images": config.include_raw_policy_images,
        "include_debug_images": config.include_debug_images,
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": dataset_summary.dataset_size_bytes,
        "command": shlex.join(command),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write benchmark rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def benchmark(config: BenchmarkConfig, runner: Callable[..., CommandRunResult] = run_subprocess) -> list[dict[str, Any]]:
    """Run the benchmark matrix and return CSV-ready rows."""

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    env = _subprocess_env(config)
    rows: list[dict[str, Any]] = []
    for num_parallel_envs in config.parallel_envs:
        for repeat in range(config.repeats):
            dataset_path = dataset_path_for(config, num_parallel_envs, repeat)
            command = build_collect_command(config, num_parallel_envs, repeat, dataset_path)
            print(
                f"[benchmark] envs={num_parallel_envs} repeat={repeat} "
                f"episodes={config.num_episodes} max_steps={config.max_steps}",
                flush=True,
            )
            run_result = runner(command, timeout_s=config.timeout_s, env=env)
            dataset_summary = summarize_dataset(dataset_path)
            row = result_row(
                config=config,
                num_parallel_envs=num_parallel_envs,
                repeat=repeat,
                dataset_path=dataset_path,
                command=command,
                run_result=run_result,
                dataset_summary=dataset_summary,
            )
            rows.append(row)
            write_csv(config.output_csv, rows)
            print(
                f"[benchmark] status={run_result.status} wall_time_s={row['wall_time_s']} "
                f"episodes_per_s={row['episodes_per_s']} csv={config.output_csv}",
                flush=True,
            )
            if config.fail_fast and run_result.returncode != 0:
                raise SystemExit(run_result.returncode)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-envs", default="1,2", help="Comma-separated vectorized env counts, e.g. 1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--policy", choices=["random", "heuristic"], default="random")
    parser.add_argument("--output-csv", default="logs/rollout_collection_benchmark.csv")
    parser.add_argument("--dataset-dir", default="data/benchmark_rollouts")
    parser.add_argument("--dataset-prefix", default="bench_rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--include-raw-policy-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-debug-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collect-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward progress display to child collect_rollouts runs. Disabled by default for cleaner timing.",
    )
    parser.add_argument("--display", default=None, help="DISPLAY value for Isaac/Kit. Omit to inherit the environment.")
    parser.add_argument(
        "--xauthority",
        default=None,
        help=(
            "XAUTHORITY cookie path. If omitted, the benchmark uses the shell-exported "
            "XAUTHORITY if set, else auto-discovers a /var/run/sddm/{...} cookie on Linux "
            "(vast.ai bare-metal convention). Isaac Sim needs a valid cookie even in headless "
            "mode because the RTX / Vulkan WSI layer opens an X display surface."
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        parallel_envs=parse_parallel_envs(args.parallel_envs),
        repeats=args.repeats,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        policy=args.policy,
        output_csv=Path(args.output_csv),
        dataset_dir=Path(args.dataset_dir),
        dataset_prefix=args.dataset_prefix,
        seed=args.seed,
        device=args.device,
        timeout_s=args.timeout_s,
        include_raw_policy_images=args.include_raw_policy_images,
        include_debug_images=args.include_debug_images,
        headless=args.headless,
        collect_progress=args.collect_progress,
        display=args.display,
        xauthority=args.xauthority,
        python_executable=args.python_executable,
        fail_fast=args.fail_fast,
    )
    rows = benchmark(config)
    print(f"[benchmark] wrote {len(rows)} rows to {config.output_csv}", flush=True)


def _subprocess_env(config: BenchmarkConfig) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("PRIVACY_CONSENT", "Y")
    if config.display is not None:
        env["DISPLAY"] = config.display
    # X11 auth precedence:
    #   explicit --xauthority > shell-exported XAUTHORITY > SDDM cookie auto-discovery.
    # Without a cookie Isaac's RTX / Vulkan WSI layer will fail with
    # "Authorization required" even in --headless mode, because it still needs
    # an X display surface to enumerate the GPU.
    if config.xauthority is not None:
        env["XAUTHORITY"] = config.xauthority
    elif not env.get("XAUTHORITY"):
        discovered = _discover_sddm_xauthority()
        if discovered is not None:
            env["XAUTHORITY"] = str(discovered)
    return env


def _discover_sddm_xauthority() -> Path | None:
    """Return one SDDM X-cookie path if present (vast.ai bare-metal convention)."""

    sddm_dir = Path("/var/run/sddm")
    if not sddm_dir.is_dir():
        return None
    cookies = sorted(sddm_dir.glob("{*}"))
    return cookies[0] if cookies else None


if __name__ == "__main__":
    main()
