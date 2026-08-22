"""Phase 2: one frozen-calibration VoxVietnam-O evaluation and ECAPA export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from module_a.src.config import ConfigError, save_json
from module_a.src.data import (
    DataError,
    discover_records,
    hf_token_from_environment,
    resolve_dataset_subset,
)
from module_a.src.evaluation import (
    EvaluationError,
    build_sid_protocol,
    build_sid_prototypes,
    build_sv_trials,
    compute_sv_metrics,
    extract_embeddings,
    load_frozen_calibrations,
    load_official_sv_trials,
    score_sid,
    score_sv_trials,
    sha256_file,
    sid_metrics,
    sv_rates,
    write_sv_scores,
)
from module_a.src.runtime import RuntimeModelError, export_artifact
from module_a.src.training import TrainingError, load_checkpoint_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: load best.pt and frozen validation calibration, evaluate "
            "VoxVietnam-O once, then export the ECAPA runtime artifact."
        )
    )
    parser.add_argument("--dataset-root", help="Mounted dataset root or VoxVietnam-O directory")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Defaults to OUTPUT_DIR/checkpoints/best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--official-trials")
    parser.add_argument("--sv-protocol", choices=("auto", "official", "custom"), default="auto")
    return parser


def _find_official_trials(root: Path) -> Path | None:
    matches = sorted(
        {
            *root.rglob("*trial*.txt"),
            *root.rglob("*trials*.csv"),
        }
    )
    if len(matches) > 1:
        raise EvaluationError(
            "Multiple possible official trial files found; pass --official-trials explicitly."
        )
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    output = Path(args.output_dir).expanduser().resolve()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else output / "checkpoints" / "best.pt"
    )
    device = resolve_device(args.device)
    model, config, _ = load_checkpoint_model(checkpoint, device=device)
    checkpoint_hash = sha256_file(checkpoint)
    # Fail before resolving or scanning VoxVietnam-O if validation calibration is absent.
    sv_calibration, sid_calibration = load_frozen_calibrations(
        output, checkpoint_sha256=checkpoint_hash
    )

    dataset = config["dataset"]
    test_root = resolve_dataset_subset(
        subset_name=str(dataset["test_subset"]),
        local_root=args.dataset_root,
        hf_repo_id=args.hf_repo_id or str(dataset["hf_repo_id"]),
        hf_token=hf_token_from_environment(args.hf_token_env),
        hf_cache_dir=args.hf_cache_dir,
    )
    records = discover_records(
        test_root,
        split="test",
        audio_extensions=dataset["audio_extensions"],
        speaker_component_from_end=int(dataset["speaker_id_component_from_end"]),
    )
    embeddings = extract_embeddings(model, records, test_root, config, device=device)

    configured_trials = config["evaluation"].get("official_trials")
    explicit_trials = args.official_trials or configured_trials
    official_path = Path(explicit_trials).expanduser().resolve() if explicit_trials else None
    if args.sv_protocol == "auto" and official_path is None:
        official_path = _find_official_trials(test_root)
    if args.sv_protocol == "official" and official_path is None:
        raise EvaluationError(
            "Official VoxVietnam-O SV protocol requested but no trial file was supplied/found."
        )
    use_official = args.sv_protocol != "custom" and official_path is not None
    if use_official:
        sv_trials = load_official_sv_trials(official_path, set(embeddings))
        sv_protocol_label = (
            "official_voxvietnam_o_trials_user_asserted"
            if args.sv_protocol == "official"
            else "discovered_voxvietnam_o_trial_list_equivalence_unverified"
        )
    else:
        sv_trials = build_sv_trials(
            records,
            seed=int(config["seed"]),
            max_positive_per_speaker=int(
                config["training"]["max_positive_trials_per_speaker"]
            ),
        )
        sv_protocol_label = "custom_balanced_voxvietnam_o_sv_v1_not_official"
    sv_scores = score_sv_trials(sv_trials, embeddings)
    write_sv_scores(output / "metrics" / "sv_test_scores.csv", sv_scores)
    sv_intrinsic = compute_sv_metrics(sv_scores)
    frozen_sv = sv_rates(sv_scores, float(sv_calibration["threshold"]))
    sv_test = {
        "dataset": "VoxVietnam-O",
        "protocol": sv_protocol_label,
        **sv_intrinsic,
        "frozen_validation_threshold": float(sv_calibration["threshold"]),
        "far_at_frozen_threshold": frozen_sv["far"],
        "frr_at_frozen_threshold": frozen_sv["frr"],
        "tar_at_frozen_threshold": frozen_sv["tpr"],
        "threshold_recalibrated_on_test": False,
    }
    save_json(output / "metrics" / "sv_test_metrics.json", sv_test)

    sid_protocol = build_sid_protocol(
        records,
        seed=int(config["seed"]),
        known_ratio=float(config["calibration"]["sid_known_ratio"]),
        max_enrollment=int(config["calibration"]["sid_max_enrollment"]),
    )
    sid_scores = score_sid(
        sid_protocol, embeddings, build_sid_prototypes(sid_protocol, embeddings)
    )
    sid_test = {
        "dataset": "VoxVietnam-O",
        "protocol": "custom_open_set_sid_v1_not_official_voxvietnam_benchmark",
        **sid_metrics(sid_scores, float(sid_calibration["threshold"])),
        "frozen_validation_threshold": float(sid_calibration["threshold"]),
        "threshold_recalibrated_on_test": False,
    }
    save_json(output / "metrics" / "sid_test_metrics.json", sid_test)

    thresholds = {
        "embedding_dimension": int(config["model"]["embedding_dim"]),
        "sv_threshold": float(sv_calibration["threshold"]),
        "sid_threshold": float(sid_calibration["threshold"]),
        "threshold_source": "validation",
        "sv_target_far": float(sv_calibration["target_far"]),
        "sid_target_unknown_far": float(sid_calibration["target_unknown_far"]),
    }
    metadata = {
        "artifact_version": 1,
        "architecture": "ECAPA-TDNN",
        "training_dataset": "VoxVietnam-T",
        "test_dataset": "VoxVietnam-O",
        "checkpoint_sha256": checkpoint_hash,
        "sv_test_protocol": sv_protocol_label,
        "sid_test_protocol": "custom_open_set_sid_v1",
    }
    export_dir = export_artifact(
        model, config, thresholds, metadata, output / "module_a_export"
    )
    summary = {
        "status": "completed",
        "phase": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "sv": sv_test,
        "sid": sid_test,
        "export_dir": str(export_dir),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output / "evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, DataError, EvaluationError, TrainingError, RuntimeModelError) as exc:
        raise SystemExit(f"Phase 2 failed: {exc}") from exc
