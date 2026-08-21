"""CLI for deterministic speaker-disjoint manifest preparation."""

from __future__ import annotations

import argparse
import json
import sys

from module_a.src.config import DEFAULT_DATASET_CONFIG, DEFAULT_EXPERIMENT_CONFIG, load_config
from module_a.src.pipeline import prepare_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create validated train/val/test manifests split by speaker."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--dataset-config", default=str(DEFAULT_DATASET_CONFIG), help="Dataset YAML path."
    )
    parser.add_argument(
        "--experiment-config",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help="Experiment YAML path.",
    )
    parser.add_argument("--output-dir", help="Override the runtime output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            args.dataset_config,
            args.experiment_config,
            dataset_root=args.dataset_root,
            output_root=args.output_dir,
        )
        result = prepare_manifests(config)
    except Exception as exc:
        print(f"Manifest preparation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.split_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
