"""Small, validated YAML configuration layer for Module A A0-A3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    segment_seconds: float


@dataclass(frozen=True)
class InspectionConfig:
    max_files: int | None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    wavlm_model_name: str
    wavlm_frozen: bool
    stage2_enabled: bool
    wavlm_hidden_dimension: int
    adapter_dimension: int
    embedding_dimension: int
    campp_growth_rate: int
    campp_block_layers: tuple[int, ...]
    campp_init_channels: int
    campp_bottleneck_channels: int
    campp_fcm_channels: int
    campp_segment_frames: int


@dataclass(frozen=True)
class LossConfig:
    type: str
    margin: float
    scale: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    max_steps: int | None
    mixed_precision: bool
    monitor_mixed_precision: bool
    learning_rate: float
    weight_decay: float
    speakers_per_batch: int
    utterances_per_speaker: int
    num_workers: int
    gradient_accumulation_steps: int
    log_every_steps: int
    val_every_steps: int | None
    save_every_steps: int | None
    max_train_speakers: int | None
    max_monitor_speakers: int | None
    monitor_holdout_ratio: float
    max_consecutive_amp_overflows: int


@dataclass(frozen=True)
class SchedulerConfig:
    type: str
    warmup_steps: int


@dataclass(frozen=True)
class ModuleAConfig:
    seed: int
    dataset: DatasetConfig
    split: SplitConfig
    audio: AudioConfig
    inspection: InspectionConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    scheduler: SchedulerConfig
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


def config_to_dict(config: ModuleAConfig) -> dict[str, Any]:
    """Return a checkpoint-safe representation without notebook or YAML state."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(asdict(config))


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


def _non_negative_int(value: Any, field: str) -> int:
    value = _integer(value, field)
    if value < 0:
        raise ConfigurationError(f"'{field}' must be a non-negative integer.")
    return value


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive number.")
    return float(value)


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"'{field}' must be a positive number.")
    return float(value)


def _non_negative_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise ConfigurationError(f"'{field}' must be a non-negative number.")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{field}' must be true or false.")
    return value


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _positive_int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"'{field}' must be a non-empty integer list.")
    return tuple(_positive_int(item, field) for item in value)


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
    """Load A0-A3 configuration without requiring a dataset or model download."""

    dataset_path = Path(dataset_config_path).expanduser().resolve()
    experiment_path = Path(experiment_config_path).expanduser().resolve()
    dataset_document = _load_yaml(dataset_path)
    experiment_document = _load_yaml(experiment_path)

    dataset_data = _mapping(dataset_document, "dataset")
    split_data = _mapping(dataset_document, "split")
    audio_data = _mapping(dataset_document, "audio")
    inspection_data = _mapping(dataset_document, "inspection")
    output_data = _mapping(experiment_document, "output")
    model_data = _mapping(experiment_document, "model")
    model_audio_data = _mapping(experiment_document, "audio")
    loss_data = _mapping(experiment_document, "loss")
    training_data = _mapping(experiment_document, "training")
    scheduler_data = _mapping(experiment_document, "scheduler")

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
    seed = _integer(experiment_document.get("seed"), "seed")
    split_seed = _integer(split_data.get("seed"), "split.seed")
    if seed != split_seed:
        raise ConfigurationError("Experiment seed and split.seed must match.")

    dataset_sample_rate = _positive_int(
        audio_data.get("target_sample_rate"), "audio.target_sample_rate"
    )
    model_sample_rate = _positive_int(
        model_audio_data.get("sample_rate"), "experiment.audio.sample_rate"
    )
    if dataset_sample_rate != model_sample_rate:
        raise ConfigurationError("Dataset and model sample rates must match.")

    architecture = _non_empty_string(model_data.get("architecture"), "model.architecture")
    if architecture != "wavlm_base_plus_campp":
        raise ConfigurationError("A2 model.architecture must be wavlm_base_plus_campp.")
    wavlm_frozen = _boolean(model_data.get("wavlm_frozen"), "model.wavlm_frozen")
    stage2_enabled = _boolean(model_data.get("stage2_enabled"), "model.stage2_enabled")
    if not wavlm_frozen or stage2_enabled:
        raise ConfigurationError("A2 requires frozen WavLM and stage2_enabled=false.")
    adapter_dimension = _positive_int(
        model_data.get("adapter_dimension"), "model.adapter_dimension"
    )
    if adapter_dimension != 80:
        raise ConfigurationError("Canonical A2 adapter_dimension must be 80.")
    loss_type = _non_empty_string(loss_data.get("type"), "loss.type")
    if loss_type != "aam_softmax":
        raise ConfigurationError("A2 loss.type must be aam_softmax.")

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
        seed=seed,
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
            seed=split_seed,
        ),
        audio=AudioConfig(
            target_sample_rate=dataset_sample_rate,
            segment_seconds=_positive_number(
                model_audio_data.get("segment_seconds"),
                "experiment.audio.segment_seconds",
            ),
        ),
        inspection=InspectionConfig(max_files=max_files),
        model=ModelConfig(
            architecture=architecture,
            wavlm_model_name=_non_empty_string(
                model_data.get("wavlm_model_name"), "model.wavlm_model_name"
            ),
            wavlm_frozen=wavlm_frozen,
            stage2_enabled=stage2_enabled,
            wavlm_hidden_dimension=_positive_int(
                model_data.get("wavlm_hidden_dimension"),
                "model.wavlm_hidden_dimension",
            ),
            adapter_dimension=adapter_dimension,
            embedding_dimension=_positive_int(
                model_data.get("embedding_dimension"), "model.embedding_dimension"
            ),
            campp_growth_rate=_positive_int(
                model_data.get("campp_growth_rate"), "model.campp_growth_rate"
            ),
            campp_block_layers=_positive_int_tuple(
                model_data.get("campp_block_layers"), "model.campp_block_layers"
            ),
            campp_init_channels=_positive_int(
                model_data.get("campp_init_channels"), "model.campp_init_channels"
            ),
            campp_bottleneck_channels=_positive_int(
                model_data.get("campp_bottleneck_channels"),
                "model.campp_bottleneck_channels",
            ),
            campp_fcm_channels=_positive_int(
                model_data.get("campp_fcm_channels"), "model.campp_fcm_channels"
            ),
            campp_segment_frames=_positive_int(
                model_data.get("campp_segment_frames"), "model.campp_segment_frames"
            ),
        ),
        loss=LossConfig(
            type=loss_type,
            margin=_positive_number(loss_data.get("margin"), "loss.margin"),
            scale=_positive_number(loss_data.get("scale"), "loss.scale"),
        ),
        training=TrainingConfig(
            epochs=_positive_int(training_data.get("epochs"), "training.epochs"),
            max_steps=_optional_positive_int(
                training_data.get("max_steps"), "training.max_steps"
            ),
            mixed_precision=_boolean(
                training_data.get("mixed_precision"), "training.mixed_precision"
            ),
            monitor_mixed_precision=_boolean(
                training_data.get("monitor_mixed_precision", False),
                "training.monitor_mixed_precision",
            ),
            learning_rate=_positive_number(
                training_data.get("learning_rate"), "training.learning_rate"
            ),
            weight_decay=_non_negative_number(
                training_data.get("weight_decay"), "training.weight_decay"
            ),
            speakers_per_batch=_positive_int(
                training_data.get("speakers_per_batch"),
                "training.speakers_per_batch",
            ),
            utterances_per_speaker=_positive_int(
                training_data.get("utterances_per_speaker"),
                "training.utterances_per_speaker",
            ),
            num_workers=_non_negative_int(
                training_data.get("num_workers"), "training.num_workers"
            ),
            gradient_accumulation_steps=_positive_int(
                training_data.get("gradient_accumulation_steps"),
                "training.gradient_accumulation_steps",
            ),
            log_every_steps=_positive_int(
                training_data.get("log_every_steps"), "training.log_every_steps"
            ),
            val_every_steps=_optional_positive_int(
                training_data.get("val_every_steps"), "training.val_every_steps"
            ),
            save_every_steps=_optional_positive_int(
                training_data.get("save_every_steps"), "training.save_every_steps"
            ),
            max_train_speakers=_optional_positive_int(
                training_data.get("max_train_speakers"),
                "training.max_train_speakers",
            ),
            max_monitor_speakers=_optional_positive_int(
                training_data.get("max_monitor_speakers"),
                "training.max_monitor_speakers",
            ),
            monitor_holdout_ratio=_ratio(
                training_data.get("monitor_holdout_ratio"),
                "training.monitor_holdout_ratio",
            ),
            max_consecutive_amp_overflows=_positive_int(
                training_data.get("max_consecutive_amp_overflows", 20),
                "training.max_consecutive_amp_overflows",
            ),
        ),
        scheduler=SchedulerConfig(
            type=_non_empty_string(scheduler_data.get("type"), "scheduler.type"),
            warmup_steps=_non_negative_int(
                scheduler_data.get("warmup_steps"), "scheduler.warmup_steps"
            ),
        ),
        output_root=output_path,
    )
    if not 0 < config.training.monitor_holdout_ratio < 1:
        raise ConfigurationError("training.monitor_holdout_ratio must be between 0 and 1.")
    if config.scheduler.type != "cosine":
        raise ConfigurationError("A3 scheduler.type must be cosine.")
    return config.with_overrides(dataset_root=dataset_root, output_root=output_root)
