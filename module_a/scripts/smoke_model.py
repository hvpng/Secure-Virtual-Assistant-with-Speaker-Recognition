"""Optional real Hugging Face WavLM integration smoke; may download weights."""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch
from torch.nn import functional as F

from module_a.src.audio_batch import collate_fixed_waveforms, load_waveform
from module_a.src.config import load_config
from module_a.src.device import autocast_context, resolve_device
from module_a.src.model_factory import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load real microsoft/wavlm-base-plus and run one frozen forward pass."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--audio-path", help="Optional real audio path; otherwise use a sine wave.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download; require the Hugging Face model in local cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config()
        device = resolve_device(args.device)
        model = build_model(
            config,
            num_classes=2,
            local_files_only=args.local_files_only,
        ).to(device)
        model.eval()
        if args.audio_path:
            waveform = load_waveform(
                args.audio_path,
                target_sample_rate=config.audio.target_sample_rate,
            )
            waveforms, attention_mask = collate_fixed_waveforms(
                [waveform],
                sample_rate=config.audio.target_sample_rate,
                segment_seconds=config.audio.segment_seconds,
            )
        else:
            if args.duration_seconds <= 0:
                raise ValueError("duration-seconds must be positive.")
            sample_count = round(config.audio.target_sample_rate * args.duration_seconds)
            time = torch.arange(sample_count, dtype=torch.float32) / config.audio.target_sample_rate
            waveforms = (0.1 * torch.sin(2 * math.pi * 220 * time)).unsqueeze(0)
            attention_mask = torch.ones_like(waveforms, dtype=torch.long)
        waveforms = waveforms.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad(), autocast_context(device, config.training.mixed_precision):
            hidden, adapted = model.encoder.extract_features(waveforms, attention_mask)
            raw_embedding = model.encoder.campp(adapted.transpose(1, 2).contiguous())
            embedding = F.normalize(raw_embedding.float(), dim=1, eps=1e-8)
        report = {
            "status": "ok",
            "model": config.model.wavlm_model_name,
            "device": str(device),
            "waveform_shape": list(waveforms.shape),
            "waveform_duration_seconds": round(
                waveforms.shape[1] / config.audio.target_sample_rate, 3
            ),
            "wavlm_shape": list(hidden.shape),
            "adapter_shape": list(adapted.shape),
            "campp_input_shape": list(adapted.transpose(1, 2).shape),
            "embedding_shape": list(embedding.shape),
            "embedding_norm": round(float(embedding.norm(dim=1).item()), 6),
            "wavlm_frozen": all(
                not parameter.requires_grad
                for parameter in model.encoder.frontend.parameters()
            ),
        }
        if device.type == "cuda":
            report["cuda_peak_memory_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 2
            )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f"Real WavLM smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
