"""Deterministic development-only speaker backend.

This module deliberately performs no model download and does not approximate
production accuracy. Fixture names such as ``alice_01.wav`` and
``alice_query.wav`` map to the same deterministic identity embedding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EMBEDDING_DIMENSION = 64


@dataclass(frozen=True)
class FakeSpeakerModel:
    device: str


def load_model(device: str = "auto", **_: object) -> FakeSpeakerModel:
    return FakeSpeakerModel(device=device)


def _fixture_identity(audio_path: str) -> str:
    stem = Path(audio_path).stem.lower()
    identity = stem.split("_", maxsplit=1)[0]
    return identity or "unknown"


def extract_embedding(
    model: FakeSpeakerModel, audio_path: str
) -> np.ndarray:
    del model
    identity = _fixture_identity(audio_path)
    values = bytearray()
    counter = 0
    while len(values) < EMBEDDING_DIMENSION:
        values.extend(hashlib.sha256(f"{identity}:{counter}".encode()).digest())
        counter += 1

    embedding = np.asarray(values[:EMBEDDING_DIMENSION], dtype=np.float32)
    embedding = embedding / 127.5 - 1.0
    return embedding / np.linalg.norm(embedding)
