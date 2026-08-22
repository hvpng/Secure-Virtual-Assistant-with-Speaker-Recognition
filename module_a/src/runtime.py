"""Deployment export and stable ECAPA speaker-embedding runtime ABI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from module_a.src.config import save_json
from module_a.src.data import load_waveform, repeat_or_crop
from module_a.src.ecapa import SpeakerEmbeddingModel, build_embedding_model


class RuntimeModelError(RuntimeError):
    """Raised when the deployment artifact or inference output is invalid."""


@dataclass
class RuntimeModel:
    model: SpeakerEmbeddingModel
    config: dict[str, Any]
    device: torch.device


def _device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeModelError("CUDA was requested but is unavailable.")
    if normalized not in {"cpu", "cuda"}:
        raise RuntimeModelError("Device must be auto, cpu, or cuda.")
    return torch.device(normalized)


def export_artifact(
    model: SpeakerEmbeddingModel,
    config: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if thresholds.get("threshold_source") != "validation":
        raise RuntimeModelError("Deployment thresholds must come from validation.")
    temporary = output / "model.pt.tmp"
    torch.save(
        {
            "artifact_version": 1,
            "architecture": "ecapa_tdnn",
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(output / "model.pt")
    save_json(output / "config.json", dict(config))
    save_json(output / "thresholds.json", dict(thresholds))
    save_json(output / "metadata.json", dict(metadata))
    return output


def load_model(model_dir: str | Path, device: str = "auto") -> RuntimeModel:
    """Load a deployment directory produced by Phase 2."""

    root = Path(model_dir).expanduser().resolve()
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeModelError(f"Cannot load ECAPA deployment artifact: {root}") from exc
    if (
        not isinstance(config, dict)
        or not isinstance(payload, dict)
        or payload.get("architecture") != "ecapa_tdnn"
        or "model_state_dict" not in payload
    ):
        raise RuntimeModelError("Deployment artifact is malformed.")
    resolved_device = _device(device)
    model = build_embedding_model(config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeModelError("Deployment model state is incompatible with config.json.") from exc
    model.to(resolved_device).eval()
    return RuntimeModel(model=model, config=config, device=resolved_device)


def extract_embedding(model: RuntimeModel, audio_path: str | Path) -> np.ndarray:
    """Return one finite, float32, normalized 192-D ECAPA embedding."""

    sample_rate = int(model.config["audio"]["sample_rate"])
    target_samples = round(sample_rate * float(model.config["audio"]["segment_seconds"]))
    waveform = repeat_or_crop(
        load_waveform(audio_path, sample_rate), target_samples, random_crop=False
    )
    with torch.no_grad():
        vector = (
            model.model.extract_embedding(waveform.unsqueeze(0).to(model.device))[0]
            .float()
            .cpu()
            .numpy()
        )
    embedding = np.asarray(vector, dtype=np.float32)
    if (
        embedding.shape != (192,)
        or not np.isfinite(embedding).all()
        or not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)
    ):
        raise RuntimeModelError("Runtime embedding violates the stable ABI.")
    return embedding
