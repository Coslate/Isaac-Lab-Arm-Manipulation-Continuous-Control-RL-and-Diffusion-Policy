"""Filesystem helpers for standard project outputs."""

from __future__ import annotations

from pathlib import Path

from configs.project_config import ProjectPaths


def ensure_output_dirs(
    root: str | Path = ".",
    paths: ProjectPaths | None = None,
) -> dict[str, Path]:
    """Create standard output directories and return their absolute paths."""

    root_path = Path(root)
    project_paths = paths or ProjectPaths()
    created: dict[str, Path] = {}

    for name, relative_path in project_paths.as_dict().items():
        output_path = root_path / relative_path
        output_path.mkdir(parents=True, exist_ok=True)
        created[name] = output_path.resolve()

    return created

