"""Offline synthetic A4 cache/protocol/calibration/test sanity check."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from module_a.src.config import load_config
from module_a.src.evaluation import (
    EvaluationModelBundle,
    get_or_create_embedding_cache,
    write_json_atomic,
)
from module_a.src.evaluation_pipeline import run_test_protocols, run_validation_protocols
from module_a.src.training_data import TrainingRecord


def _records_and_embeddings(
    split: str, *, speaker_count: int = 6, utterances: int = 6, dimension: int = 192
) -> tuple[list[TrainingRecord], dict[str, np.ndarray]]:
    records: list[TrainingRecord] = []
    embeddings: dict[str, np.ndarray] = {}
    split_offset = 0 if split == "val" else 24
    for speaker_index in range(speaker_count):
        base = np.zeros(dimension, dtype=np.float32)
        base[split_offset + speaker_index] = 1.0
        for utterance_index in range(utterances):
            path = f"{split}/speaker_{speaker_index:02d}/utt_{utterance_index:02d}.wav"
            vector = base.copy()
            vector[80 + utterance_index] = 0.02
            vector /= np.linalg.norm(vector)
            records.append(TrainingRecord(path, f"{split}_speaker_{speaker_index:02d}", split))
            embeddings[path] = vector.astype(np.float32)
    return records, embeddings


def main() -> int:
    config = load_config()
    with tempfile.TemporaryDirectory(prefix="module-a-a4-sanity-") as temporary:
        root = Path(temporary)
        output = root / "module_a_a4"
        bundle = EvaluationModelBundle(
            model=nn.Identity(),
            config=config,
            checkpoint_path=root / "synthetic-checkpoint.pt",
            checkpoint_sha256="synthetic-checkpoint-sha256",
            num_classes=2,
            speaker_to_index={"train_a": 0, "train_b": 1},
            device=torch.device("cpu"),
        )
        val_records, val_embeddings = _records_and_embeddings("val")
        test_records, test_embeddings = _records_and_embeddings("test")
        val_cache_path = output / "embeddings" / "val_embeddings.npz"
        val_cache = get_or_create_embedding_cache(
            val_cache_path,
            bundle=bundle,
            records=val_records,
            split="val",
            dataset_root=root,
            extractor=lambda: val_embeddings,
        )
        cached_again = get_or_create_embedding_cache(
            val_cache_path,
            bundle=bundle,
            records=val_records,
            split="val",
            dataset_root=root,
        )
        if set(val_cache.embeddings) != set(cached_again.embeddings):
            raise RuntimeError("Synthetic embedding cache roundtrip failed.")
        validation = run_validation_protocols(
            val_records,
            val_cache.embeddings,
            output_dir=output,
            seed=42,
            max_sv_positive_per_speaker=4,
            sid_known_ratio=0.8,
            sid_max_enrollment=5,
        )
        test_cache = get_or_create_embedding_cache(
            output / "embeddings" / "test_embeddings.npz",
            bundle=bundle,
            records=test_records,
            split="test",
            dataset_root=root,
            extractor=lambda: test_embeddings,
        )
        test = run_test_protocols(
            test_records,
            test_cache.embeddings,
            output_dir=output,
            seed=42,
            max_sv_positive_per_speaker=4,
            sid_known_ratio=0.8,
            sid_max_enrollment=5,
        )
        write_json_atomic(
            output / "run_config.json",
            {"mode": "synthetic", "seed": 42, "device": "cpu"},
        )
        write_json_atomic(
            output / "evaluation_summary.json",
            {
                "status": "passed",
                "device": "cpu",
                "sv": {"validation": validation["sv"], "test": test["sv"]},
                "sid": {"validation": validation["sid"], "test": test["sid"]},
            },
        )
        required = (
            "embeddings/val_embeddings.npz",
            "embeddings/test_embeddings.npz",
            "trials/sv_val_trials.csv",
            "trials/sv_test_trials.csv",
            "protocols/sid_val_protocol.json",
            "protocols/sid_test_protocol.json",
            "calibration/sv_calibration.json",
            "calibration/sid_calibration.json",
            "scores/sv_val_scores.csv",
            "scores/sv_test_scores.csv",
            "scores/sid_val_scores.csv",
            "scores/sid_test_scores.csv",
            "metrics/sv_val_metrics.json",
            "metrics/sv_test_metrics.json",
            "metrics/sid_val_metrics.json",
            "metrics/sid_test_metrics.json",
            "run_config.json",
            "evaluation_summary.json",
        )
        if not all((output / relative).is_file() for relative in required):
            raise RuntimeError("Synthetic A4 did not create every required artifact.")
        report = {
            "device": "cpu",
            "validation_sv_eer": validation["sv"]["metrics"]["eer"],
            "validation_sv_eer_threshold": validation["sv"]["calibration"]["eer_threshold"],
            "validation_sv_deployment_threshold": validation["sv"]["calibration"]["deployment_threshold"],
            "validation_sid_threshold": validation["sid"]["calibration"]["selected_threshold"],
            "test_sv_frozen_threshold": test["sv"]["metrics"]["frozen_validation_sv_deployment_threshold"],
            "test_sid_frozen_threshold": test["sid"]["metrics"]["frozen_validation_sid_threshold"],
            "artifacts_verified": len(required),
            "status": "passed",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
