"""Offline synthetic end-to-end sanity check for A0/A1 only."""

from __future__ import annotations

import json
import math
import tempfile
import wave
from array import array
from pathlib import Path

from module_a.src.config import load_config
from module_a.src.pipeline import prepare_manifests


def _write_wav(path: Path, *, sample_rate: int = 16_000, duration_sec: float = 0.12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array(
        "h",
        (
            int(2_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(round(sample_rate * duration_sec))
        ),
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="module-a-a1-") as temporary:
        workspace = Path(temporary)
        dataset_root = workspace / "dataset"
        for speaker_index in range(10):
            for utterance_index in range(2):
                _write_wav(
                    dataset_root
                    / f"speaker_{speaker_index:02d}"
                    / f"utterance_{utterance_index}.wav"
                )
        corrupt_path = dataset_root / "speaker_00" / "corrupt.wav"
        corrupt_path.write_bytes(b"not a wav file")

        first_output = workspace / "first"
        second_output = workspace / "second"
        first_config = load_config(
            dataset_root=dataset_root,
            output_root=first_output,
        )
        second_config = load_config(
            dataset_root=dataset_root,
            output_root=second_output,
        )
        first = prepare_manifests(first_config)
        second = prepare_manifests(second_config)

        for split in ("train", "val", "test"):
            first_csv = (first_output / f"{split}_manifest.csv").read_text(encoding="utf-8")
            second_csv = (second_output / f"{split}_manifest.csv").read_text(encoding="utf-8")
            if first_csv != second_csv:
                raise RuntimeError(f"Determinism check failed for {split} manifest.")

        report = {
            "status": "ok",
            "speakers_created": 10,
            "total_discovered_files": first.inspection.summary["total_discovered_files"],
            "usable_files": first.inspection.summary["usable_files"],
            "corrupt_files": first.inspection.summary["corrupt_files"],
            "train_speakers": len(first.allocation.train),
            "val_speakers": len(first.allocation.val),
            "test_speakers": len(first.allocation.test),
            "manifest_records": len(first.manifests),
            "speaker_disjoint": first.split_summary["speaker_disjoint"],
            "deterministic": True,
        }
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
