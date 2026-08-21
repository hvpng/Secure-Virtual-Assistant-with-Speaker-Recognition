"""Small, validated YAML configuration layer for Module A A0/A1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


MODULE_A_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = MODULE_A_ROOT / "configs" / "dataset.yaml"
DEFAULT_EXPERIMENT_CONFIG = MODULE_A_ROOT / "configs" / "experiment.yaml"
SUPPORTED_SPEAKER_ID_SOURCES = {"parent_dir", "path_component", "metadata_csv"}


class ConfigurationError(ValueError):
    """Raised when an A0/A1 configuration is missing or inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path | None
    audio_extensions: tuple[str, ...]
    min_utterances_per_speaker: int
    speaker_id_source: str
    speaker_id_path_component: int | None
    speaker_metadata_csv: Path | None
    speaker_metadata_path_column: str
    speaker_metadata_id_column: str


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int


@dataclass(frozen=True)
class InspectionConfig:
    max_files: int | None


@dataclass(frozen=True)
class ModuleAConfig:
    dataset: DatasetConfig
    split: SplitConfig
    audio: AudioConfig
    inspection: InspectionConfig
    output_root: Path

    def with_overrides(
        self,
        *,
        dataset_root: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> "ModuleAConfig":
        dataset = self.dataset
        if dataset_root is not None:
            dataset = replace(dataset, root=Path(dataset_root).expanduser().resolve())
        resolved_output = self.output_root
        if output_root is not None:
            resolved_output = Path(output_root).expanduser().resolve()
        return replace(self, dataset=dataset, output_root=resolved_output)

    def require_dataset_root(self) -> Path:
        if self.dataset.root is None:
            raise ConfigurationError(
                "Dataset root is not configured. Pass --dataset-root or set dataset.root."
            )
        root = self.dataset.root.expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"Dataset root does not exist or is not a directory: {root}")
        return root


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read YAML config: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Config must contain a YAML mapping: {path}")
    return loaded


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Config section '{key}' must be a mapping.")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive integer.")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"'{field}' must be an integer.")
    return value


def _ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive number.")
    return float(value)


def _optional_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("Configured paths must be non-empty strings or null.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(
    dataset_config_path: str | Path = DEFAULT_DATASET_CONFIG,
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ModuleAConfig:
    """Load A0/A1 configuration without requiring the dataset to exist locally."""

    dataset_path = Path(dataset_config_path).expanduser().resolve()
    experiment_path = Path(experiment_config_path).expanduser().resolve()
    dataset_document = _load_yaml(dataset_path)
    experiment_document = _load_yaml(experiment_path)

    dataset_data = _mapping(dataset_document, "dataset")
    split_data = _mapping(dataset_document, "split")
    audio_data = _mapping(dataset_document, "audio")
    inspection_data = _mapping(dataset_document, "inspection")
    output_data = _mapping(experiment_document, "output")

    name = dataset_data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("'dataset.name' must be a non-empty string.")

    raw_extensions = dataset_data.get("audio_extensions")
    if not isinstance(raw_extensions, list) or not raw_extensions:
        raise ConfigurationError("'dataset.audio_extensions' must be a non-empty list.")
    extensions: list[str] = []
    for extension in raw_extensions:
        if not isinstance(extension, str) or not extension.strip():
            raise ConfigurationError("Audio extensions must be non-empty strings.")
        normalized = extension.lower().strip()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        extensions.append(normalized)

    speaker_id_source = dataset_data.get("speaker_id_source", "parent_dir")
    if speaker_id_source not in SUPPORTED_SPEAKER_ID_SOURCES:
        raise ConfigurationError(
            "'dataset.speaker_id_source' must be parent_dir, path_component, or metadata_csv."
        )
    component = dataset_data.get("speaker_id_path_component")
    if component is not None and (isinstance(component, bool) or not isinstance(component, int)):
        raise ConfigurationError("'dataset.speaker_id_path_component' must be an integer or null.")

    train_ratio = _ratio(split_data.get("train_ratio"), "split.train_ratio")
    val_ratio = _ratio(split_data.get("val_ratio"), "split.val_ratio")
    test_ratio = _ratio(split_data.get("test_ratio"), "split.test_ratio")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ConfigurationError("Split ratios must sum to exactly 1.0.")

    max_files = inspection_data.get("max_files")
    if max_files is not None:
        max_files = _positive_int(max_files, "inspection.max_files")

    configured_dataset_root = _optional_path(dataset_data.get("root"), dataset_path.parent)
    metadata_csv = _optional_path(dataset_data.get("speaker_metadata_csv"), dataset_path.parent)
    configured_output = output_data.get("root", "outputs")
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise ConfigurationError("'output.root' must be a non-empty path string.")
    output_path = Path(configured_output).expanduser()
    if not output_path.is_absolute():
        output_path = (MODULE_A_ROOT / output_path).resolve()

    config = ModuleAConfig(
        dataset=DatasetConfig(
            name=name.strip(),
            root=configured_dataset_root,
            audio_extensions=tuple(dict.fromkeys(extensions)),
            min_utterances_per_speaker=_positive_int(
                dataset_data.get("min_utterances_per_speaker"),
                "dataset.min_utterances_per_speaker",
            ),
            speaker_id_source=speaker_id_source,
            speaker_id_path_component=component,
            speaker_metadata_csv=metadata_csv,
            speaker_metadata_path_column=str(
                dataset_data.get("speaker_metadata_path_column", "path")
            ),
            speaker_metadata_id_column=str(
                dataset_data.get("speaker_metadata_id_column", "speaker_id")
            ),
        ),
        split=SplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=_integer(split_data.get("seed"), "split.seed"),
        ),
        audio=AudioConfig(
            target_sample_rate=_positive_int(
                audio_data.get("target_sample_rate"), "audio.target_sample_rate"
            )
        ),
        inspection=InspectionConfig(max_files=max_files),
        output_root=output_path,
    )
    return config.with_overrides(dataset_root=dataset_root, output_root=output_root)
