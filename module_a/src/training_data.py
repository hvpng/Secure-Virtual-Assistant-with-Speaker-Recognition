"""A3 manifest datasets, train-only monitor holdout, and balanced sampling."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from module_a.src.audio_batch import (
    collate_fixed_waveforms,
    fit_waveform_to_segment_random,
    load_waveform,
)


class TrainingDataError(ValueError):
    """Raised when A3 data would violate the train-speaker classifier contract."""


@dataclass(frozen=True)
class TrainingRecord:
    path: str
    speaker_id: str
    split: str = "train"


@dataclass(frozen=True)
class TrainMonitorSplit:
    fit: tuple[TrainingRecord, ...]
    monitor: tuple[TrainingRecord, ...]
    monitor_speakers: tuple[str, ...]


def load_manifest(path: str | Path) -> list[TrainingRecord]:
    """Load stable A1 columns without resolving paths outside dataset_root."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise TrainingDataError(f"Manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"path", "speaker_id", "split"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise TrainingDataError(
                    "Manifest must contain path, speaker_id, and split columns."
                )
            records: list[TrainingRecord] = []
            for line_number, row in enumerate(reader, start=2):
                relative_path = (row.get("path") or "").strip()
                speaker_id = (row.get("speaker_id") or "").strip()
                split = (row.get("split") or "").strip()
                candidate = Path(relative_path)
                if not relative_path or candidate.is_absolute() or ".." in candidate.parts:
                    raise TrainingDataError(
                        f"Manifest line {line_number} has an unsafe relative path."
                    )
                if not speaker_id or split not in {"train", "val", "test"}:
                    raise TrainingDataError(
                        f"Manifest line {line_number} has an invalid speaker or split."
                    )
                records.append(
                    TrainingRecord(candidate.as_posix(), speaker_id, split)
                )
    except OSError as exc:
        raise TrainingDataError(f"Cannot read manifest: {manifest_path}") from exc
    if not records:
        raise TrainingDataError("Manifest contains no records.")
    return records


def select_train_records(
    records: Sequence[TrainingRecord],
    *,
    max_speakers: int | None,
    seed: int,
) -> tuple[list[TrainingRecord], tuple[str, ...]]:
    """Select a reproducible seeded subset from sorted A1 train speakers."""

    train_records = [record for record in records if record.split == "train"]
    speakers = sorted({record.speaker_id for record in train_records})
    if not speakers:
        raise TrainingDataError("No train speakers are available for A3.")
    if max_speakers is not None:
        if max_speakers <= 1:
            raise TrainingDataError("max_train_speakers must be at least 2.")
        if max_speakers < len(speakers):
            speakers = sorted(random.Random(seed).sample(speakers, max_speakers))
    selected = set(speakers)
    filtered = sorted(
        (record for record in train_records if record.speaker_id in selected),
        key=lambda record: (record.speaker_id, record.path),
    )
    return filtered, tuple(speakers)


def build_speaker_to_index(
    records: Sequence[TrainingRecord],
) -> dict[str, int]:
    """Build classes only from sorted train speakers; val/test are ignored."""

    speakers = sorted(
        {record.speaker_id for record in records if record.split == "train"}
    )
    if len(speakers) < 2:
        raise TrainingDataError("AAM training requires at least two train speakers.")
    return {speaker_id: index for index, speaker_id in enumerate(speakers)}


def _speaker_seed(seed: int, speaker_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{speaker_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_train_monitor(
    records: Sequence[TrainingRecord],
    *,
    holdout_ratio: float,
    seed: int,
    max_monitor_speakers: int | None = None,
) -> TrainMonitorSplit:
    """Hold out utterances from train speakers without touching A1 val/test.

    Every selected classifier speaker keeps at least one fit utterance. Only
    speakers with at least two utterances can contribute monitor samples.
    """

    if not 0 < holdout_ratio < 1:
        raise TrainingDataError("monitor holdout_ratio must be between 0 and 1.")
    grouped: dict[str, list[TrainingRecord]] = defaultdict(list)
    for record in records:
        if record.split != "train":
            raise TrainingDataError("Train-monitor split accepts train records only.")
        grouped[record.speaker_id].append(record)
    if len(grouped) < 2:
        raise TrainingDataError("Train-monitor split requires at least two speakers.")

    eligible = sorted(speaker for speaker, items in grouped.items() if len(items) >= 2)
    if max_monitor_speakers is not None:
        if max_monitor_speakers <= 0:
            raise TrainingDataError("max_monitor_speakers must be positive.")
        if max_monitor_speakers < len(eligible):
            eligible = sorted(random.Random(seed + 1).sample(eligible, max_monitor_speakers))
    monitor_speakers = set(eligible)
    fit: list[TrainingRecord] = []
    monitor: list[TrainingRecord] = []
    for speaker_id in sorted(grouped):
        items = sorted(grouped[speaker_id], key=lambda record: record.path)
        if speaker_id not in monitor_speakers:
            fit.extend(items)
            continue
        shuffled = items.copy()
        random.Random(_speaker_seed(seed, speaker_id)).shuffle(shuffled)
        monitor_count = min(
            len(items) - 1,
            max(1, math.floor(len(items) * holdout_ratio)),
        )
        monitor_paths = {record.path for record in shuffled[:monitor_count]}
        monitor.extend(record for record in items if record.path in monitor_paths)
        fit.extend(record for record in items if record.path not in monitor_paths)

    fit.sort(key=lambda record: (record.speaker_id, record.path))
    monitor.sort(key=lambda record: (record.speaker_id, record.path))
    if not fit or not monitor:
        raise TrainingDataError(
            "Train-monitor holdout is empty; provide speakers with at least two utterances."
        )
    if {record.path for record in fit} & {record.path for record in monitor}:
        raise TrainingDataError("Train-monitor path leakage detected.")
    if set(grouped) != {record.speaker_id for record in fit}:
        raise TrainingDataError("Every classifier speaker must remain in the fit split.")
    return TrainMonitorSplit(tuple(fit), tuple(monitor), tuple(eligible))


class SpeakerManifestDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[TrainingRecord],
        *,
        dataset_root: str | Path,
        speaker_to_index: Mapping[str, int],
        sample_rate: int,
    ) -> None:
        if not records:
            raise TrainingDataError("Speaker dataset cannot be empty.")
        self.records = tuple(records)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.speaker_to_index = dict(speaker_to_index)
        self.sample_rate = sample_rate
        if not self.dataset_root.is_dir() or sample_rate <= 0:
            raise TrainingDataError("Dataset root and sample rate must be valid.")
        for record in self.records:
            if record.speaker_id not in self.speaker_to_index:
                raise TrainingDataError(
                    f"Speaker is absent from train classifier mapping: {record.speaker_id}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.dataset_root / Path(relative_path)).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise TrainingDataError("Audio path escapes dataset_root.") from exc
        return path

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        resolved = self._resolve(record.path)
        return {
            "waveform": load_waveform(resolved, target_sample_rate=self.sample_rate),
            "class_index": self.speaker_to_index[record.speaker_id],
            "speaker_id": record.speaker_id,
            "path": record.path,
        }


class WaveformBatchCollator:
    def __init__(
        self,
        *,
        sample_rate: int,
        segment_seconds: float,
        training: bool,
    ) -> None:
        if sample_rate <= 0 or segment_seconds <= 0:
            raise TrainingDataError("Collator audio settings must be positive.")
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.training = training

    def __call__(self, items: Sequence[dict[str, object]]) -> dict[str, object]:
        if not items:
            raise TrainingDataError("Cannot collate an empty batch.")
        waveforms = [item["waveform"] for item in items]
        if not all(isinstance(waveform, Tensor) for waveform in waveforms):
            raise TrainingDataError("Dataset waveform values must be tensors.")
        if self.training:
            segment_samples = round(self.sample_rate * self.segment_seconds)
            batch = torch.stack(
                [
                    fit_waveform_to_segment_random(waveform, segment_samples)
                    for waveform in waveforms
                ]
            )
            attention_mask = torch.ones(batch.shape, dtype=torch.long)
        else:
            batch, attention_mask = collate_fixed_waveforms(
                waveforms,
                sample_rate=self.sample_rate,
                segment_seconds=self.segment_seconds,
            )
        return {
            "waveforms": batch,
            "attention_mask": attention_mask,
            "labels": torch.tensor(
                [int(item["class_index"]) for item in items], dtype=torch.long
            ),
            "speaker_ids": [str(item["speaker_id"]) for item in items],
            "paths": [str(item["path"]) for item in items],
        }


class SpeakerBalancedBatchSampler(Sampler[list[int]]):
    """Uniformly choose speakers, then uniformly choose their utterances."""

    def __init__(
        self,
        records: Sequence[TrainingRecord],
        *,
        speakers_per_batch: int,
        utterances_per_speaker: int,
        seed: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        if speakers_per_batch <= 0 or utterances_per_speaker <= 0:
            raise TrainingDataError("Balanced batch dimensions must be positive.")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.speaker_id].append(index)
        self.speakers = tuple(sorted(grouped))
        if len(self.speakers) < speakers_per_batch:
            raise TrainingDataError(
                "speakers_per_batch exceeds the number of fit speakers."
            )
        self.indices_by_speaker = {
            speaker: tuple(indices) for speaker, indices in grouped.items()
        }
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        self.seed = seed
        self.epoch = 0
        default_batches = math.ceil(
            len(records) / (speakers_per_batch * utterances_per_speaker)
        )
        self.batches_per_epoch = batches_per_epoch or default_batches
        if self.batches_per_epoch <= 0:
            raise TrainingDataError("Balanced sampler cannot be empty.")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise TrainingDataError("Sampler epoch must be non-negative.")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        for _ in range(self.batches_per_epoch):
            selected_speakers = rng.sample(self.speakers, self.speakers_per_batch)
            batch: list[int] = []
            for speaker in selected_speakers:
                candidates = self.indices_by_speaker[speaker]
                if len(candidates) >= self.utterances_per_speaker:
                    batch.extend(rng.sample(candidates, self.utterances_per_speaker))
                else:
                    batch.extend(
                        rng.choice(candidates)
                        for _ in range(self.utterances_per_speaker)
                    )
            yield batch


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python/NumPy from the DataLoader-provided torch worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
