"""Single-file YAML configuration for the ECAPA/VoxVietnam pipeline."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


MODULE_A_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODULE_A_ROOT / "configs" / "ecapa_voxvietnam.yaml"


class ConfigError(ValueError):
    """Raised when the canonical Module A configuration is invalid."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping.")
    return dict(value)


def _probability(value: Any, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ConfigError(f"'{name}' must be between 0 and 1.")
    if not allow_zero and result == 0:
        raise ConfigError(f"'{name}' must be greater than 0.")
    return result


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration: {config_path}") from exc
    config = _mapping(document, "root")
    required = (
        "dataset", "audio", "features", "model", "loss", "optimizer",
        "training", "augmentation", "calibration", "evaluation",
    )
    for section in required:
        config[section] = _mapping(config.get(section), section)

    if config["model"].get("architecture") != "ecapa_tdnn":
        raise ConfigError("The only supported architecture is ecapa_tdnn.")
    if int(config["audio"].get("sample_rate", 0)) != 16_000:
        raise ConfigError("Canonical sample rate must be 16000 Hz.")
    if int(config["features"].get("fbank_dim", 0)) != 80:
        raise ConfigError("Canonical fbank dimension must be 80.")
    if int(config["model"].get("embedding_dim", 0)) != 192:
        raise ConfigError("Canonical embedding dimension must be 192.")
    if config["loss"].get("name") != "aam_softmax":
        raise ConfigError("The training loss must be aam_softmax.")
    if config["optimizer"].get("name") != "adamw":
        raise ConfigError("The optimizer must be adamw.")

    train_ratio = _probability(config["dataset"].get("train_ratio"), "dataset.train_ratio")
    validation_ratio = _probability(
        config["dataset"].get("validation_ratio"), "dataset.validation_ratio"
    )
    if not math.isclose(train_ratio + validation_ratio, 1.0):
        raise ConfigError("Train and validation ratios must sum to 1.")
    if int(config.get("seed", -1)) < 0:
        raise ConfigError("'seed' must be a non-negative integer.")
    if int(config["training"].get("batch_size", 0)) != (
        int(config["training"].get("speakers_per_batch", 0))
        * int(config["training"].get("utterances_per_speaker", 0))
    ):
        raise ConfigError(
            "training.batch_size must equal speakers_per_batch * utterances_per_speaker."
        )
    for key in ("sv_target_far", "sid_target_unknown_far"):
        _probability(config["calibration"].get(key), f"calibration.{key}")
    sid_known_ratio = _probability(
        config["calibration"].get("sid_known_ratio"),
        "calibration.sid_known_ratio",
        allow_zero=False,
    )
    if sid_known_ratio >= 1:
        raise ConfigError("calibration.sid_known_ratio must be less than 1.")
    return copy.deepcopy(config)


def save_json(path: str | Path, value: Any) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
