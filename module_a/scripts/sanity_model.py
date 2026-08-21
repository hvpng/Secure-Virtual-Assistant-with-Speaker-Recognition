"""Offline forward/backward/optimizer/checkpoint sanity for A2."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from module_a.src.checkpoint import load_checkpoint, save_checkpoint
from module_a.src.config import config_to_dict, load_config
from module_a.src.device import resolve_device
from module_a.src.model_factory import build_model, build_optimizer
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM
from module_a.src.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run A2 model sanity with a deterministic fake WavLM; no download."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser


def _sanity_config():
    config = load_config()
    return replace(
        config,
        model=replace(
            config.model,
            campp_growth_rate=4,
            campp_block_layers=(1, 1, 1),
            campp_init_channels=8,
            campp_bottleneck_channels=8,
            campp_fcm_channels=4,
            campp_segment_frames=4,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _sanity_config()
    seed_everything(config.seed)
    device = resolve_device(args.device)
    num_classes = 3
    frontend = DeterministicFakeWavLM(
        config.model.wavlm_hidden_dimension, frame_count=12
    )
    model = build_model(config, num_classes=num_classes, frontend=frontend).to(device)
    optimizer = build_optimizer(model, config.training.learning_rate)
    waveforms = torch.linspace(-0.5, 0.5, steps=1_600, device=device).repeat(2, 1)
    waveforms[1] = waveforms[1].flip(0)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    model.train()
    hidden, adapted = model.encoder.extract_features(waveforms)
    tracked_parameter = model.encoder.projection.weight.detach().clone()
    output = model(waveforms, labels)
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()

    campp_gradients = [
        parameter.grad
        for parameter in model.encoder.campp.parameters()
        if parameter.requires_grad
    ]
    gradient_sanity = (
        model.encoder.projection.weight.grad is not None
        and any(gradient is not None for gradient in campp_gradients)
        and model.aam_head.weight.grad is not None
        and all(parameter.grad is None for parameter in model.encoder.frontend.parameters())
        and all(
            torch.isfinite(gradient).all()
            for gradient in campp_gradients
            if gradient is not None
        )
    )
    optimizer.step()
    optimizer_changed_backend = not torch.equal(
        tracked_parameter, model.encoder.projection.weight.detach()
    )

    model.eval()
    with torch.no_grad():
        reference_embedding = model.extract_embedding(waveforms)

    with tempfile.TemporaryDirectory(prefix="module-a-a2-") as temporary:
        checkpoint_path = Path(temporary) / "sanity.pt"
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            step=1,
            config=config_to_dict(config),
            num_classes=num_classes,
            speaker_to_index={"synthetic_a": 0, "synthetic_b": 1, "synthetic_c": 2},
        )
        reloaded = build_model(
            config,
            num_classes=num_classes,
            frontend=DeterministicFakeWavLM(
                config.model.wavlm_hidden_dimension, frame_count=12
            ),
        ).to(device)
        reloaded_optimizer = build_optimizer(reloaded, config.training.learning_rate)
        payload = load_checkpoint(
            checkpoint_path,
            model=reloaded,
            optimizer=reloaded_optimizer,
            map_location=device,
        )
        reloaded.eval()
        with torch.no_grad():
            reloaded_embedding = reloaded.extract_embedding(waveforms)
        checkpoint_roundtrip = torch.allclose(
            reference_embedding, reloaded_embedding, atol=1e-6, rtol=1e-5
        )

    report = {
        "status": "ok",
        "frontend": "deterministic_fake_wavlm_no_download",
        "device": str(device),
        "batch_shape": list(waveforms.shape),
        "frontend_shape": list(hidden.shape),
        "adapter_shape": list(adapted.shape),
        "embedding_shape": list(output.raw_embedding.shape),
        "loss": round(float(output.loss.detach().cpu()), 6),
        "gradient_sanity": bool(gradient_sanity),
        "optimizer_changed_backend": optimizer_changed_backend,
        "wavlm_frozen": all(
            not parameter.requires_grad for parameter in model.encoder.frontend.parameters()
        ),
        "checkpoint_fields": sorted(payload.keys()),
        "checkpoint_roundtrip": bool(checkpoint_roundtrip),
        "embedding_norms": [
            round(float(value), 6)
            for value in reference_embedding.norm(dim=1).detach().cpu()
        ],
    }
    if not all(
        (
            gradient_sanity,
            optimizer_changed_backend,
            checkpoint_roundtrip,
            report["wavlm_frozen"],
        )
    ):
        raise RuntimeError(f"A2 synthetic sanity failed: {report}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
