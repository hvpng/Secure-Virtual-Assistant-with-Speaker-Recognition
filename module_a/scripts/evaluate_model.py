"""A4 validation calibration and frozen-threshold test evaluation CLI."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from module_a.src.config import config_to_dict
from module_a.src.device import resolve_device
from module_a.src.evaluation import (
    EvaluationError,
    fingerprint_records,
    get_or_create_embedding_cache,
    load_evaluation_model,
    load_split_manifest,
    read_json_object,
    validate_evaluation_isolation,
    write_json_atomic,
)
from module_a.src.evaluation_pipeline import (
    A4_CALIBRATION_CONTRACT_VERSION,
    require_frozen_calibrations,
    run_test_protocols,
    run_validation_protocols,
)
from module_a.src.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an A3 frozen Stage-1 checkpoint. Validation calibrates and "
            "persists thresholds; test requires and consumes those frozen artifacts."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-sv-positive-per-speaker", type=int)
    parser.add_argument("--sv-target-far", type=float)
    parser.add_argument("--sid-known-ratio", type=float)
    parser.add_argument("--sid-max-enrollment", type=int)
    parser.add_argument("--sid-target-unknown-far", type=float)
    parser.add_argument("--recompute-embeddings", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.phase in {"test", "all"} and not args.test_manifest:
        parser.error("--test-manifest is required for --phase test/all")
    if args.max_sv_positive_per_speaker is not None and args.max_sv_positive_per_speaker <= 0:
        parser.error("--max-sv-positive-per-speaker must be positive")
    if args.sv_target_far is not None and (
        not math.isfinite(args.sv_target_far) or not 0 <= args.sv_target_far <= 1
    ):
        parser.error("--sv-target-far must be finite and between 0 and 1")
    if args.sid_known_ratio is not None and not 0 < args.sid_known_ratio < 1:
        parser.error("--sid-known-ratio must be between 0 and 1")
    if args.sid_max_enrollment is not None and args.sid_max_enrollment <= 0:
        parser.error("--sid-max-enrollment must be positive")
    if args.sid_target_unknown_far is not None and (
        not math.isfinite(args.sid_target_unknown_far)
        or not 0 <= args.sid_target_unknown_far <= 1
    ):
        parser.error("--sid-target-unknown-far must be finite and between 0 and 1")


def _base_summary(bundle: Any, *, device: Any, seed: int) -> dict[str, Any]:
    return {
        "status": "running",
        "device": str(device),
        "model": {
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "architecture": bundle.config.model.architecture,
            "embedding_dimension": bundle.config.model.embedding_dimension,
            "wavlm_frozen": bundle.config.model.wavlm_frozen,
        },
        "protocol": {"seed": seed},
        "sv": {},
        "sid": {},
    }


def _load_phase_manifests(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any] | None]:
    """Load test records only when the requested phase explicitly includes test."""

    validation_records = load_split_manifest(args.val_manifest, "val")
    test_records = (
        load_split_manifest(args.test_manifest, "test")
        if args.phase in {"test", "all"}
        else None
    )
    return validation_records, test_records


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    started = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Test-only mode must prove calibration exists before loading a model or test audio.
    if args.phase == "test":
        require_frozen_calibrations(output_dir)

    device = resolve_device(args.device)
    bundle = load_evaluation_model(
        args.checkpoint,
        device=device,
        local_files_only=args.local_files_only,
    )
    seed = bundle.config.seed if args.seed is None else args.seed
    evaluation = replace(
        bundle.config.evaluation,
        max_sv_positive_per_speaker=(
            bundle.config.evaluation.max_sv_positive_per_speaker
            if args.max_sv_positive_per_speaker is None
            else args.max_sv_positive_per_speaker
        ),
        sv_target_far=(
            bundle.config.evaluation.sv_target_far
            if args.sv_target_far is None
            else args.sv_target_far
        ),
        sid_known_ratio=(
            bundle.config.evaluation.sid_known_ratio
            if args.sid_known_ratio is None
            else args.sid_known_ratio
        ),
        sid_max_enrollment=(
            bundle.config.evaluation.sid_max_enrollment
            if args.sid_max_enrollment is None
            else args.sid_max_enrollment
        ),
        sid_target_unknown_far=(
            bundle.config.evaluation.sid_target_unknown_far
            if args.sid_target_unknown_far is None
            else args.sid_target_unknown_far
        ),
    )
    bundle = replace(
        bundle,
        config=replace(bundle.config, seed=seed, evaluation=evaluation),
    )
    seed_everything(seed)

    validation_records, test_records = _load_phase_manifests(args)
    validate_evaluation_isolation(
        train_speakers=tuple(bundle.speaker_to_index),
        validation_records=validation_records,
        test_records=test_records,
    )
    calibration_provenance = {
        "calibration_contract_version": A4_CALIBRATION_CONTRACT_VERSION,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "validation_manifest_sha256": fingerprint_records(validation_records),
        "seed": seed,
        "max_sv_positive_per_speaker": evaluation.max_sv_positive_per_speaker,
        "sv_target_far": evaluation.sv_target_far,
        "sid_known_ratio": evaluation.sid_known_ratio,
        "sid_max_enrollment": evaluation.sid_max_enrollment,
        "sid_target_unknown_far": evaluation.sid_target_unknown_far,
    }

    run_config = {
        "phase": args.phase,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "val_manifest": str(Path(args.val_manifest).expanduser().resolve()),
        "test_manifest": (
            str(Path(args.test_manifest).expanduser().resolve()) if args.test_manifest else None
        ),
        "checkpoint": str(bundle.checkpoint_path),
        "device": str(device),
        "seed": seed,
        "recompute_embeddings": args.recompute_embeddings,
        "config": config_to_dict(bundle.config),
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    summary_path = output_dir / "evaluation_summary.json"
    summary = _base_summary(bundle, device=device, seed=seed)
    if args.phase == "test" and summary_path.exists():
        previous = read_json_object(summary_path)
        if isinstance(previous.get("protocol"), dict):
            summary["protocol"].update(previous["protocol"])
        if isinstance(previous.get("sv"), dict):
            summary["sv"].update(previous["sv"])
        if isinstance(previous.get("sid"), dict):
            summary["sid"].update(previous["sid"])

    if args.phase in {"validation", "all"}:
        val_cache = get_or_create_embedding_cache(
            output_dir / "embeddings" / "val_embeddings.npz",
            bundle=bundle,
            records=validation_records,
            split="val",
            dataset_root=args.dataset_root,
            recompute=args.recompute_embeddings,
        )
        validation = run_validation_protocols(
            validation_records,
            val_cache.embeddings,
            output_dir=output_dir,
            seed=seed,
            max_sv_positive_per_speaker=evaluation.max_sv_positive_per_speaker,
            sid_known_ratio=evaluation.sid_known_ratio,
            sid_max_enrollment=evaluation.sid_max_enrollment,
            sv_target_far=evaluation.sv_target_far,
            sid_target_unknown_far=evaluation.sid_target_unknown_far,
            calibration_provenance=calibration_provenance,
        )
        summary["protocol"].update(
            {
                "val_speakers": len({record.speaker_id for record in validation_records}),
                "sv_val_positive_trials": validation["sv"]["metrics"]["positive_trials"],
                "sv_val_negative_trials": validation["sv"]["metrics"]["negative_trials"],
                "sid_val_known_speakers": validation["sid"]["known_speakers"],
                "sid_val_unknown_speakers": validation["sid"]["unknown_speakers"],
                "sid_max_enrollment_utterances": evaluation.sid_max_enrollment,
            }
        )
        summary["sv"]["validation"] = validation["sv"]
        summary["sid"]["validation"] = validation["sid"]

    if args.phase in {"test", "all"}:
        # Even in all mode, reload persisted artifacts to enforce the same test contract.
        require_frozen_calibrations(
            output_dir, expected_provenance=calibration_provenance
        )
        assert test_records is not None
        test_cache = get_or_create_embedding_cache(
            output_dir / "embeddings" / "test_embeddings.npz",
            bundle=bundle,
            records=test_records,
            split="test",
            dataset_root=args.dataset_root,
            recompute=args.recompute_embeddings,
        )
        test = run_test_protocols(
            test_records,
            test_cache.embeddings,
            output_dir=output_dir,
            seed=seed,
            max_sv_positive_per_speaker=evaluation.max_sv_positive_per_speaker,
            sid_known_ratio=evaluation.sid_known_ratio,
            sid_max_enrollment=evaluation.sid_max_enrollment,
            expected_calibration_provenance=calibration_provenance,
        )
        summary["protocol"].update(
            {
                "test_speakers": len({record.speaker_id for record in test_records}),
                "sv_test_positive_trials": test["sv"]["metrics"]["positive_trials"],
                "sv_test_negative_trials": test["sv"]["metrics"]["negative_trials"],
                "sid_test_known_speakers": test["sid"]["known_speakers"],
                "sid_test_unknown_speakers": test["sid"]["unknown_speakers"],
                "sid_max_enrollment_utterances": evaluation.sid_max_enrollment,
            }
        )
        summary["sv"]["test"] = test["sv"]
        summary["sid"]["test"] = test["sid"]

    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["status"] = "completed"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        raise SystemExit(f"A4 evaluation failed: {exc}") from exc
