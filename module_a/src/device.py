"""Device and mixed-precision policy shared by A2 smoke and future A3."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cpu, or cuda")


def autocast_context(
    device: torch.device, enabled: bool
) -> ContextManager[object]:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
