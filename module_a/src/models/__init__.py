"""Training-side speaker model components for Module A."""

from module_a.src.models.aam_softmax import AAMSoftmax
from module_a.src.models.campp import CAMPlusPlus
from module_a.src.models.speaker_model import SpeakerTrainingModel, WavLMCAMPlusPlus
from module_a.src.models.wavlm_frontend import (
    DeterministicFakeWavLM,
    HuggingFaceWavLMFrontend,
)

__all__ = [
    "AAMSoftmax",
    "CAMPlusPlus",
    "DeterministicFakeWavLM",
    "HuggingFaceWavLMFrontend",
    "SpeakerTrainingModel",
    "WavLMCAMPlusPlus",
]

