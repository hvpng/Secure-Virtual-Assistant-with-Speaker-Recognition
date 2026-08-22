"""Phase 1: train ECAPA on VoxVietnam-T and calibrate on its validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from module_a.src.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, save_json
from module_a.src.data import (
    DataError,
    hf_token_from_environment,
    prepare_train_validation_manifests,
    resolve_dataset_subset,
)
from module_a.src.training import TrainingError, resolve_device, train_phase1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: train ECAPA-TDNN on VoxVietnam-T, select best.pt by "
            "speaker-disjoint validation SV EER, then freeze validation calibration."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset-root", help="Mounted dataset root or VoxVietnam-T directory")
    parser.add_argument("--hf-repo-id", help="Override the configured Hugging Face dataset ID")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Explicit engineering smoke limit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    config = load_config(args.config)
    dataset = config["dataset"]
    # This is intentionally the only dataset subset resolved anywhere in Phase 1.
    train_root = resolve_dataset_subset(
        subset_name=str(dataset["train_subset"]),
        local_root=args.dataset_root,
        hf_repo_id=args.hf_repo_id or str(dataset["hf_repo_id"]),
        hf_token=hf_token_from_environment(args.hf_token_env),
        hf_cache_dir=args.hf_cache_dir,
    )
    train, validation = prepare_train_validation_manifests(
        train_root,
        args.output_dir,
        audio_extensions=dataset["audio_extensions"],
        speaker_component_from_end=int(dataset["speaker_id_component_from_end"]),
        seed=int(config["seed"]),
        validation_ratio=float(dataset["validation_ratio"]),
    )
    run_config = {
        "phase": 1,
        "dataset": "VoxVietnam-T",
        "dataset_root": str(train_root),
        "config": config,
        "resume": args.resume,
        "max_steps": args.max_steps,
    }
    save_json(Path(args.output_dir) / "run_config_phase1.json", run_config)
    summary = train_phase1(
        train,
        validation,
        train_root,
        config,
        args.output_dir,
        device=resolve_device(args.device),
        resume=args.resume,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, DataError, TrainingError) as exc:
        raise SystemExit(f"Phase 1 failed: {exc}") from exc

