"""Inspect an HDF5 rollout dataset and optionally export sample frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from dataset.episode_dataset import list_episode_keys, read_episode_metadata


def summarize_rollout_dataset(path: str | Path) -> dict[str, Any]:
    """Return a JSON-friendly summary of episode groups and metadata."""

    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as h5_file:
        episodes: dict[str, Any] = {}
        for episode_key in list_episode_keys(dataset_path):
            group = h5_file[episode_key]
            datasets = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in group.items()
                if isinstance(value, h5py.Dataset)
            }
            episodes[episode_key] = {
                "datasets": datasets,
                "metadata": read_episode_metadata(group),
            }
        return {
            "path": str(dataset_path),
            "schema_version": h5_file.attrs.get("schema_version", ""),
            "num_episodes": int(h5_file.attrs.get("num_episodes", len(episodes))),
            "episodes": episodes,
        }


def export_policy_frame(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    episode: str = "episode_000",
    step: int = 0,
) -> Path:
    """Export one wrist policy image from `images` as a PNG."""

    with h5py.File(dataset_path, "r") as h5_file:
        frame_chw = _read_frame(h5_file, episode, "images", step)
    if frame_chw.shape != (3, 224, 224):
        raise ValueError(f"policy frame must have shape (3, 224, 224), got {frame_chw.shape}")
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
    return _save_rgb_png(frame_hwc, output_path)


def export_raw_policy_frame(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    episode: str = "episode_000",
    step: int = 0,
) -> Path:
    """Export one native-resolution wrist policy image from `raw_policy_images` as a PNG."""

    with h5py.File(dataset_path, "r") as h5_file:
        frame_hwc = _read_frame(h5_file, episode, "raw_policy_images", step)
    if frame_hwc.ndim != 3 or frame_hwc.shape[-1] != 3:
        raise ValueError(f"raw policy frame must have shape (H, W, 3), got {frame_hwc.shape}")
    return _save_rgb_png(frame_hwc, output_path)


def export_debug_frame(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    episode: str = "episode_000",
    step: int = 0,
) -> Path:
    """Export one fixed-camera debug image from `debug_images` as a PNG."""

    with h5py.File(dataset_path, "r") as h5_file:
        frame_hwc = _read_frame(h5_file, episode, "debug_images", step)
    if frame_hwc.ndim != 3 or frame_hwc.shape[-1] != 3:
        raise ValueError(f"debug frame must have shape (H, W, 3), got {frame_hwc.shape}")
    return _save_rgb_png(frame_hwc, output_path)


def inspect_rollout_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect a dataset, save requested frames, and return a summary."""

    summary = summarize_rollout_dataset(args.dataset)
    saved_frames: dict[str, str] = {}
    if args.save_policy_frame:
        saved_frames["policy"] = str(
            export_policy_frame(args.dataset, args.save_policy_frame, episode=args.episode, step=args.step)
        )
    if args.save_raw_policy_frame:
        saved_frames["raw_policy"] = str(
            export_raw_policy_frame(args.dataset, args.save_raw_policy_frame, episode=args.episode, step=args.step)
        )
    if args.save_debug_frame:
        saved_frames["debug"] = str(
            export_debug_frame(args.dataset, args.save_debug_frame, episode=args.episode, step=args.step)
        )
    summary["saved_frames"] = saved_frames
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", default="episode_000")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--save-policy-frame")
    parser.add_argument("--save-raw-policy-frame")
    parser.add_argument("--save-debug-frame")
    return parser.parse_args()


def main() -> None:
    summary = inspect_rollout_dataset(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_frame(h5_file: h5py.File, episode: str, dataset_name: str, step: int) -> np.ndarray:
    if episode not in h5_file:
        raise KeyError(f"episode group {episode!r} not found")
    group = h5_file[episode]
    if dataset_name not in group:
        raise KeyError(f"{episode}/{dataset_name} not found")
    dataset = group[dataset_name]
    if step < 0 or step >= dataset.shape[0]:
        raise IndexError(f"step {step} outside valid range [0, {dataset.shape[0]}) for {episode}/{dataset_name}")
    return np.asarray(dataset[step])


def _save_rgb_png(frame_hwc: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(frame_hwc)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(output)
    return output


if __name__ == "__main__":
    main()
