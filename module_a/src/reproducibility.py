"""Reproducibility helpers for deterministic data and A2 model sanity."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch without requiring CUDA."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
