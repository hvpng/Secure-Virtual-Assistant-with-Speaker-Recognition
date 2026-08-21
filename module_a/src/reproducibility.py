"""Reproducibility helpers that do not require PyTorch or a GPU in A1."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python and NumPy; later milestones may additionally seed PyTorch."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
