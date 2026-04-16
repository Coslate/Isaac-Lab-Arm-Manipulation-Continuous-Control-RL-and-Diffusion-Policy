"""Reproducibility helpers used by scripts and tests."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed.

    PyTorch is intentionally imported lazily so PR0 remains usable in a light
    CPU-only environment.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed

