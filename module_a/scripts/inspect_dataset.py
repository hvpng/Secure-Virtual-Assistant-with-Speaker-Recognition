"""CLI for recursive dataset discovery and metadata inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from module_a.src.config import DEFAULT_DATASET_CONFIG, DEFAULT_EXPERIMENT_CONFIG, load_config
from module_a.src.pipeline import inspect_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect audio metadata without loading or training a speaker model."
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
        result = inspect_dataset(config)
    except Exception as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    compact = {
        key: result.summary[key]
        for key in (
            "dataset_name",
            "total_discovered_files",
            "usable_files",
            "corrupt_files",
            "num_speakers",
        )
    }
    compact["speaker_id_source"] = config.dataset.speaker_id_source
    compact["summary_path"] = str(Path(config.output_root) / "dataset_summary.json")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
