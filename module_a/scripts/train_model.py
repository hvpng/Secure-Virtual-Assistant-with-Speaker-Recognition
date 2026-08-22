"""Bounded real-data A3 mini-training CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from module_a.src.config import load_config
from module_a.src.device import resolve_device
from module_a.src.model_factory import build_model
from module_a.src.reproducibility import seed_everything
from module_a.src.trainer import prepare_training_data, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded A3 WavLM/CAM++ mini-training. The command refuses an "
            "unbounded run unless --mini or --max-steps is supplied."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mini", action="store_true", help="Use 50 speakers and 50 steps.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-speakers", type=int)
    parser.add_argument("--max-monitor-speakers", type=int)
    parser.add_argument("--speakers-per-batch", type=int)
    parser.add_argument("--utterances-per-speaker", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable CUDA AMP (automatically disabled on CPU).",
    )
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        parser.error(f"{name} must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.mini and args.max_steps is None:
        parser.error("Pass --mini or an explicit --max-steps; full training is not implicit.")
    for name in (
        "epochs",
        "max_steps",
        "max_train_speakers",
        "max_monitor_speakers",
        "speakers_per_batch",
        "utterances_per_speaker",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    config = load_config(dataset_root=args.dataset_root, output_root=args.output_dir)
    seed = config.seed if args.seed is None else args.seed
    max_steps = args.max_steps if args.max_steps is not None else (50 if args.mini else None)
    max_train = args.max_train_speakers
    max_monitor = args.max_monitor_speakers
    if args.mini:
        max_train = 50 if max_train is None else max_train
        max_monitor = 10 if max_monitor is None else max_monitor
    training = replace(
        config.training,
        epochs=config.training.epochs if args.epochs is None else args.epochs,
        max_steps=max_steps,
        mixed_precision=(
            config.training.mixed_precision if args.amp is None else args.amp
        ),
        speakers_per_batch=(
            config.training.speakers_per_batch
            if args.speakers_per_batch is None
            else args.speakers_per_batch
        ),
        utterances_per_speaker=(
            config.training.utterances_per_speaker
            if args.utterances_per_speaker is None
            else args.utterances_per_speaker
        ),
        num_workers=(
            config.training.num_workers if args.num_workers is None else args.num_workers
        ),
        max_train_speakers=max_train,
        max_monitor_speakers=max_monitor,
    )
    config = replace(
        config,
        seed=seed,
        split=replace(config.split, seed=seed),
        training=training,
    )
    seed_everything(seed)
    device = resolve_device(args.device)
    data = prepare_training_data(
        config,
        train_manifest=args.train_manifest,
        dataset_root=args.dataset_root,
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "selected_speaker_count": len(data.selected_speakers),
                "selected_speakers": list(data.selected_speakers),
                "monitor_speaker_count": len(data.monitor_speakers),
                "fit_utterances": len(data.fit_records),
                "monitor_utterances": len(data.monitor_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    model = build_model(
        config,
        num_classes=len(data.speaker_to_index),
        local_files_only=args.local_files_only,
    )
    result = train_model(
        model=model,
        config=config,
        data=data,
        device=device,
        output_dir=Path(args.output_dir),
        resume=args.resume,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
