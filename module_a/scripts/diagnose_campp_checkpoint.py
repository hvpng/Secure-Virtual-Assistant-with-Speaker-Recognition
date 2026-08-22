"""Inspect CAM++ BatchNorm state and one real eval utterance without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from module_a.src.audio_batch import load_waveform, prepare_deterministic_segment
from module_a.src.device import resolve_device
from module_a.src.evaluation import EvaluationError, load_evaluation_model
from module_a.src.models.campp import batch_norm_running_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect CAM++ BN running stats and one FP32 eval embedding."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--audio-path", required=True, help="Path relative to dataset root.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    bundle = load_evaluation_model(
        args.checkpoint,
        device=device,
        local_files_only=args.local_files_only,
    )
    root = Path(args.dataset_root).expanduser().resolve()
    audio_path = (root / Path(args.audio_path)).resolve()
    try:
        audio_path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError("Diagnostic audio path escapes dataset root.") from exc
    waveform = load_waveform(
        audio_path,
        target_sample_rate=bundle.config.audio.target_sample_rate,
    )
    segment = prepare_deterministic_segment(
        waveform,
        sample_rate=bundle.config.audio.target_sample_rate,
        segment_seconds=bundle.config.audio.segment_seconds,
    )
    batch = segment.unsqueeze(0).to(device)
    attention_mask = torch.ones(batch.shape, dtype=torch.long, device=device)
    bundle.model.eval()
    with torch.no_grad():
        embedding = bundle.model.extract_embedding(batch, attention_mask=attention_mask)

    all_statistics = batch_norm_running_statistics(bundle.model)
    bottleneck_statistics = [
        item for item in all_statistics if ".bottleneck.0" in str(item["name"])
    ]
    report = {
        "checkpoint": str(bundle.checkpoint_path),
        "device": str(device),
        "audio_path": args.audio_path,
        "original_samples": waveform.numel(),
        "evaluation_samples": segment.numel(),
        "batch_norm_1d_count": len(all_statistics),
        "bottleneck_batch_norm_count": len(bottleneck_statistics),
        "all_running_stats_finite": all(bool(item["finite"]) for item in all_statistics),
        "bottleneck_running_var_min": min(
            float(item["running_var_min"]) for item in bottleneck_statistics
        ),
        "lowest_bottleneck_running_vars": sorted(
            bottleneck_statistics, key=lambda item: float(item["running_var_min"])
        )[:10],
        "embedding_shape": list(embedding.shape),
        "embedding_finite": bool(torch.isfinite(embedding).all()),
        "embedding_norm": float(embedding.norm(dim=1).item()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        raise SystemExit(f"CAM++ diagnostic failed: {exc}") from exc
