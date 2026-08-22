"""Offline CPU A3 dataset/sampler/train/monitor/checkpoint/resume sanity."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import wave
from array import array
from dataclasses import replace
from pathlib import Path

import torch

from module_a.src.config import load_config
from module_a.src.model_factory import build_model
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM
from module_a.src.reproducibility import seed_everything
from module_a.src.trainer import prepare_training_data, train_model


def _write_wav(path: Path, *, phase: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array(
        "h",
        (
            int(2_000 * math.sin(2 * math.pi * 220 * index / 16_000 + phase))
            for index in range(1_600)
        ),
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(samples.tobytes())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="module-a-a3-") as temporary:
        root = Path(temporary)
        dataset_root = root / "dataset"
        manifest_path = root / "train_manifest.csv"
        rows: list[dict[str, object]] = []
        for speaker_index in range(4):
            for utterance_index in range(4):
                relative = Path(f"speaker_{speaker_index}") / f"utt_{utterance_index}.wav"
                _write_wav(
                    dataset_root / relative,
                    phase=speaker_index + utterance_index / 10,
                )
                rows.append(
                    {
                        "path": relative.as_posix(),
                        "speaker_id": f"speaker_{speaker_index}",
                        "split": "train",
                    }
                )
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "speaker_id", "split"))
            writer.writeheader()
            writer.writerows(rows)

        config = load_config(dataset_root=dataset_root, output_root=root / "run")
        config = replace(
            config,
            audio=replace(config.audio, segment_seconds=0.05),
            model=replace(
                config.model,
                campp_growth_rate=4,
                campp_block_layers=(1, 1, 1),
                campp_init_channels=8,
                campp_bottleneck_channels=8,
                campp_fcm_channels=4,
                campp_segment_frames=4,
            ),
            training=replace(
                config.training,
                epochs=1,
                max_steps=2,
                mixed_precision=True,
                speakers_per_batch=2,
                utterances_per_speaker=2,
                num_workers=0,
                log_every_steps=1,
                max_train_speakers=4,
                max_monitor_speakers=2,
                monitor_holdout_ratio=0.25,
            ),
        )
        seed_everything(config.seed)
        data = prepare_training_data(
            config,
            train_manifest=manifest_path,
            dataset_root=dataset_root,
        )
        model = build_model(
            config,
            num_classes=len(data.speaker_to_index),
            frontend=DeterministicFakeWavLM(768, frame_count=12),
        )
        first = train_model(
            model=model,
            config=config,
            data=data,
            device=torch.device("cpu"),
            output_dir=root / "run",
            stop_after_steps=1,
        )
        resumed_model = build_model(
            config,
            num_classes=len(data.speaker_to_index),
            frontend=DeterministicFakeWavLM(768, frame_count=12),
        )
        resumed = train_model(
            model=resumed_model,
            config=config,
            data=data,
            device=torch.device("cpu"),
            output_dir=root / "run",
            resume=first.checkpoint_path,
        )
        report = {
            "status": resumed.status,
            "frontend": "deterministic_fake_wavlm_no_download",
            "device": resumed.device,
            "train_speakers": resumed.train_speakers,
            "fit_utterances": resumed.fit_utterances,
            "monitor_utterances": resumed.monitor_utterances,
            "first_global_step": first.global_step,
            "resumed_global_step": resumed.global_step,
            "final_train_loss": resumed.final_train_loss,
            "final_monitor_loss": resumed.final_monitor_loss,
            "checkpoint_exists": Path(resumed.checkpoint_path).is_file(),
            "resume_compatible": resumed.resumed_from == first.checkpoint_path,
        }
        if not (
            report["status"] == "ok"
            and report["first_global_step"] == 1
            and report["resumed_global_step"] == 2
            and math.isfinite(float(report["final_train_loss"]))
            and math.isfinite(float(report["final_monitor_loss"]))
            and report["checkpoint_exists"]
            and report["resume_compatible"]
        ):
            raise RuntimeError(f"A3 synthetic sanity failed: {report}")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
