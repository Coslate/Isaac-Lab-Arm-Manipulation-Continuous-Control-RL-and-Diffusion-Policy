"""Fast checks for the Isaac Sim / Isaac Lab installation.

These tests intentionally avoid importing ``isaaclab`` itself because the first
runtime import may bootstrap Omniverse Kit and ask for EULA acceptance.
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util


REQUIRED_DISTRIBUTIONS = (
    "isaacsim",
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "gymnasium",
    "torch",
)


def test_isaac_runtime_distributions_are_installed() -> None:
    versions = {name: metadata.version(name) for name in REQUIRED_DISTRIBUTIONS}

    assert versions["isaacsim"].startswith("5.")
    assert versions["isaaclab"] == "2.3.2.post1"
    assert versions["isaaclab_assets"]
    assert versions["isaaclab_tasks"]
    assert versions["gymnasium"]
    assert versions["torch"]


def test_isaac_lab_extension_modules_are_discoverable_without_booting_kit() -> None:
    assert importlib.util.find_spec("isaaclab_assets") is not None
    assert importlib.util.find_spec("isaaclab_tasks") is not None
