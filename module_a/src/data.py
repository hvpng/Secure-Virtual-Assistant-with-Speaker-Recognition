"""VoxVietnam discovery, manifests, audio loading, and balanced sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset, Sampler

from module_a.src.config import save_json


class DataError(RuntimeError):
    """Raised for controlled dataset, manifest, or audio failures."""


@dataclass(frozen=True)
class AudioRecord:
    path: str
    speaker_id: str
    split: str


def resolve_dataset_subset(
    *,
    subset_name: str,
    local_root: str | Path | None,
    hf_repo_id: str,
    hf_token: str | None,
    hf_cache_dir: str | Path | None,
) -> Path:
    """Resolve exactly one dataset subset without materializing another subset."""

    if local_root is not None:
        root = Path(local_root).expanduser().resolve()
        if not root.is_dir():
            raise DataError(f"Dataset path does not exist: {root}")
        if root.name.casefold() == subset_name.casefold():
            return root
        candidate = root / subset_name
        if candidate.is_dir():
            return candidate.resolve()
        raise DataError(
            f"Cannot find '{subset_name}' directly under {root}. Pass either the "
            "dataset root or the subset directory itself."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DataError("huggingface-hub is required for Hugging Face dataset access.") from exc
    cache_root = (
        Path(hf_cache_dir).expanduser().resolve()
        if hf_cache_dir
        else Path.cwd() / ".cache" / "voxvietnam"
    )
    local_dir = cache_root / hf_repo_id.replace("/", "--") / subset_name
    try:
        snapshot = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            token=hf_token,
            local_dir=local_dir,
            allow_patterns=[
                f"{subset_name}/**",
                f"*/{subset_name}/**",
                f"**/{subset_name}/**",
            ],
        )
    except Exception as exc:
        raise DataError(
            f"Cannot materialize {subset_name} from Hugging Face dataset {hf_repo_id}."
        ) from exc
    snapshot_root = Path(snapshot).resolve()
    direct = snapshot_root / subset_name
    if direct.is_dir():
        return direct
    matches = [
        path for path in snapshot_root.rglob(subset_name) if path.is_dir()
    ]
    if len(matches) != 1:
        raise DataError(
            f"Downloaded dataset layout does not expose one unambiguous {subset_name} directory."
        )
    return matches[0].resolve()


def discover_records(
    root: str | Path,
    *,
    split: str,
    audio_extensions: Sequence[str],
    speaker_component_from_end: int = 2,
) -> list[AudioRecord]:
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DataError(f"Dataset subset does not exist: {dataset_root}")
    if speaker_component_from_end < 2:
        raise DataError("speaker_component_from_end must be at least 2.")
    extensions = {value.lower() for value in audio_extensions}
    records: list[AudioRecord] = []
    for audio_path in sorted(dataset_root.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in extensions:
            continue
        relative = audio_path.relative_to(dataset_root)
        if len(relative.parts) < speaker_component_from_end:
            raise DataError(
                f"Cannot derive speaker ID safely from path: {relative.as_posix()}"
            )
        speaker_id = relative.parts[-speaker_component_from_end].strip()
        if not speaker_id or speaker_id in {".", ".."}:
            raise DataError(f"Invalid speaker ID derived from: {relative.as_posix()}")
        records.append(AudioRecord(relative.as_posix(), speaker_id, split))
    if not records:
        raise DataError(f"No supported audio files were found under {dataset_root}.")
    return records


def split_train_validation(
    records: Sequence[AudioRecord], *, seed: int = 42, validation_ratio: float = 0.1
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    speakers = sorted({record.speaker_id for record in records})
    if len(speakers) < 2:
        raise DataError("At least two speakers are required for a speaker-disjoint split.")
    shuffled = speakers.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(speakers) - 1, round(len(speakers) * validation_ratio)))
    validation_speakers = set(shuffled[:validation_count])
    train = [
        AudioRecord(record.path, record.speaker_id, "train")
        for record in records
        if record.speaker_id not in validation_speakers
    ]
    validation = [
        AudioRecord(record.path, record.speaker_id, "validation")
        for record in records
        if record.speaker_id in validation_speakers
    ]
    validate_split(train, validation)
    return sorted(train, key=lambda item: item.path), sorted(
        validation, key=lambda item: item.path
    )


def validate_split(train: Sequence[AudioRecord], validation: Sequence[AudioRecord]) -> None:
    if not train or not validation:
        raise DataError("Train and validation splits must both be non-empty.")
    train_speakers = {record.speaker_id for record in train}
    validation_speakers = {record.speaker_id for record in validation}
    if train_speakers & validation_speakers:
        raise DataError("Train/validation speaker leakage detected.")
    train_paths = {record.path for record in train}
    validation_paths = {record.path for record in validation}
    if len(train_paths) != len(train) or len(validation_paths) != len(validation):
        raise DataError("Duplicate paths detected inside a manifest split.")
    if train_paths & validation_paths:
        raise DataError("Train/validation file leakage detected.")


def write_manifest(path: str | Path, records: Sequence[AudioRecord]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "speaker_id", "split"))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    temporary.replace(output)
    return output


def load_manifest(path: str | Path, expected_split: str | None = None) -> list[AudioRecord]:
    manifest = Path(path).expanduser().resolve()
    try:
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise DataError(f"Cannot read manifest: {manifest}") from exc
    records: list[AudioRecord] = []
    for row in rows:
        try:
            record = AudioRecord(row["path"], row["speaker_id"], row["split"])
        except KeyError as exc:
            raise DataError("Manifest columns must be path,speaker_id,split.") from exc
        if not record.path or not record.speaker_id or not record.split:
            raise DataError("Manifest contains an empty required value.")
        if expected_split is not None and record.split != expected_split:
            raise DataError(f"Manifest contains records outside split '{expected_split}'.")
        records.append(record)
    if not records:
        raise DataError(f"Manifest is empty: {manifest}")
    return records


def records_fingerprint(records: Sequence[AudioRecord]) -> str:
    payload = [asdict(item) for item in sorted(records, key=lambda item: item.path)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_train_validation_manifests(
    subset_root: str | Path,
    output_dir: str | Path,
    *,
    audio_extensions: Sequence[str],
    speaker_component_from_end: int,
    seed: int,
    validation_ratio: float,
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    discovered = discover_records(
        subset_root,
        split="voxvietnam_t",
        audio_extensions=audio_extensions,
        speaker_component_from_end=speaker_component_from_end,
    )
    output = Path(output_dir).expanduser().resolve()
    manifest_dir = output / "manifests"
    metadata_path = manifest_dir / "metadata.json"
    fingerprint = records_fingerprint(discovered)
    expected = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "speaker_component_from_end": speaker_component_from_end,
    }
    train_path = manifest_dir / "train.csv"
    validation_path = manifest_dir / "validation.csv"
    if metadata_path.is_file() and train_path.is_file() and validation_path.is_file():
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == expected:
            train = load_manifest(train_path, "train")
            validation = load_manifest(validation_path, "validation")
            validate_split(train, validation)
            return train, validation
    train, validation = split_train_validation(
        discovered, seed=seed, validation_ratio=validation_ratio
    )
    write_manifest(train_path, train)
    write_manifest(validation_path, validation)
    save_json(metadata_path, expected)
    return train, validation


def build_speaker_to_index(records: Sequence[AudioRecord]) -> dict[str, int]:
    speakers = sorted({record.speaker_id for record in records})
    if not speakers:
        raise DataError("Cannot build a classifier mapping from empty training data.")
    return {speaker: index for index, speaker in enumerate(speakers)}


def load_waveform(path: str | Path, sample_rate: int) -> torch.Tensor:
    audio_path = Path(path).expanduser().resolve()
    try:
        samples, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise DataError(f"Cannot decode audio file: {audio_path}") from exc
    if samples.size == 0 or source_rate <= 0 or not np.isfinite(samples).all():
        raise DataError(f"Audio is empty or non-finite: {audio_path}")
    waveform = torch.from_numpy(samples).transpose(0, 1).mean(dim=0)
    if source_rate != sample_rate:
        waveform = AF.resample(waveform, source_rate, sample_rate)
    waveform = waveform.to(torch.float32).contiguous()
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise DataError(f"Decoded waveform violates the mono float32 contract: {audio_path}")
    return waveform


def repeat_or_crop(
    waveform: torch.Tensor,
    target_samples: int,
    *,
    random_crop: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if waveform.ndim != 1 or waveform.numel() == 0 or target_samples <= 0:
        raise DataError("Segment preparation requires a non-empty rank-1 waveform.")
    length = waveform.numel()
    if length < target_samples:
        waveform = waveform.repeat(math.ceil(target_samples / length))[:target_samples]
    elif length > target_samples:
        if random_crop:
            start = int(
                torch.randint(
                    0, length - target_samples + 1, (1,), generator=generator
                ).item()
            )
        else:
            start = (length - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    result = waveform.to(torch.float32).contiguous()
    if result.shape != (target_samples,) or not torch.isfinite(result).all():
        raise DataError("Segment preparation produced an invalid waveform.")
    return result


class AudioAugmenter:
    """Small optional waveform augmenter with explicit enabled-resource reporting."""

    def __init__(self, config: Mapping[str, Any], sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.speed_enabled = bool(config.get("speed_perturb", False))
        self.noise_files = self._resource_files(config.get("noise_dir"))
        self.reverb_files = self._resource_files(config.get("reverb_dir"))
        self.noise_enabled = bool(config.get("additive_noise", False) and self.noise_files)
        self.reverb_enabled = bool(config.get("reverb", False) and self.reverb_files)

    @staticmethod
    def _resource_files(path: Any) -> list[Path]:
        if not path:
            return []
        root = Path(str(path)).expanduser().resolve()
        if not root.is_dir():
            return []
        return sorted(
            item for item in root.rglob("*") if item.suffix.lower() in {".wav", ".flac"}
        )

    @property
    def enabled(self) -> dict[str, bool]:
        return {
            "speed_perturb": self.speed_enabled,
            "additive_noise": self.noise_enabled,
            "reverb": self.reverb_enabled,
        }

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        output = waveform
        if self.speed_enabled:
            factors = (0.9, 1.0, 1.1)
            factor = factors[int(torch.randint(0, len(factors), (1,)).item())]
            if factor != 1.0:
                size = max(1, round(output.numel() / factor))
                output = F.interpolate(
                    output[None, None, :], size=size, mode="linear", align_corners=False
                )[0, 0]
        if self.noise_enabled and torch.rand(()) < 0.5:
            noise = load_waveform(random.choice(self.noise_files), self.sample_rate)
            noise = repeat_or_crop(noise, output.numel(), random_crop=True)
            snr_db = float(torch.empty(()).uniform_(5.0, 20.0).item())
            signal_rms = output.square().mean().sqrt().clamp_min(1e-6)
            noise_rms = noise.square().mean().sqrt().clamp_min(1e-6)
            output = output + noise * (signal_rms / noise_rms) * (10 ** (-snr_db / 20))
        if self.reverb_enabled and torch.rand(()) < 0.5:
            impulse = load_waveform(random.choice(self.reverb_files), self.sample_rate)
            impulse = impulse[: min(impulse.numel(), self.sample_rate)]
            impulse = impulse / impulse.abs().sum().clamp_min(1e-6)
            output = F.conv1d(
                output[None, None, :], impulse.flip(0)[None, None, :], padding=impulse.numel() - 1
            )[0, 0, : output.numel()]
        if not torch.isfinite(output).all():
            raise DataError("Audio augmentation produced non-finite samples.")
        return output


class VoxVietnamDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[AudioRecord],
        root: str | Path,
        *,
        sample_rate: int,
        segment_seconds: float,
        training: bool,
        speaker_to_index: Mapping[str, int] | None = None,
        augmenter: AudioAugmenter | None = None,
    ) -> None:
        if not records:
            raise DataError("Dataset records are empty.")
        self.records = list(records)
        self.root = Path(root).expanduser().resolve()
        self.sample_rate = sample_rate
        self.target_samples = round(sample_rate * segment_seconds)
        self.training = training
        self.speaker_to_index = dict(speaker_to_index or {})
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio_path = (self.root / Path(record.path)).resolve()
        if self.root not in audio_path.parents:
            raise DataError("Manifest path escapes the dataset root.")
        waveform = load_waveform(audio_path, self.sample_rate)
        if self.training and self.augmenter is not None:
            waveform = self.augmenter(waveform)
        waveform = repeat_or_crop(
            waveform, self.target_samples, random_crop=self.training
        )
        label = self.speaker_to_index.get(record.speaker_id, -1)
        if self.training and label < 0:
            raise DataError(f"Training speaker is absent from classifier mapping: {record.speaker_id}")
        return {
            "waveform": waveform,
            "class_index": label,
            "speaker_id": record.speaker_id,
            "path": record.path,
        }


class SpeakerBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: Sequence[AudioRecord],
        *,
        speakers_per_batch: int,
        utterances_per_speaker: int,
        seed: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.speaker_id].append(index)
        if speakers_per_batch <= 0 or utterances_per_speaker <= 0:
            raise DataError("Balanced sampler dimensions must be positive.")
        if len(grouped) < speakers_per_batch:
            raise DataError("Not enough speakers for speakers_per_batch.")
        self.grouped = {key: tuple(value) for key, value in sorted(grouped.items())}
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        self.seed = seed
        batch_size = speakers_per_batch * utterances_per_speaker
        self.batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(records) / batch_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        speakers = list(self.grouped)
        for _ in range(self.batches_per_epoch):
            selected = rng.sample(speakers, self.speakers_per_batch)
            batch: list[int] = []
            for speaker in selected:
                indices = self.grouped[speaker]
                if len(indices) >= self.utterances_per_speaker:
                    batch.extend(rng.sample(list(indices), self.utterances_per_speaker))
                else:
                    batch.extend(rng.choices(indices, k=self.utterances_per_speaker))
            yield batch


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def hf_token_from_environment(variable_name: str) -> str | None:
    value = os.getenv(variable_name)
    return value.strip() if value and value.strip() else None

