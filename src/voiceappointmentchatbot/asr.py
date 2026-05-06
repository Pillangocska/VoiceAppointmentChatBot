"""Speech recognition wrapper around faster-whisper.

Loads a single Whisper model sized to the available device and exposes a
``transcribe`` method that returns the decoded text along with the
language Whisper detected for the utterance.
"""

from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import sys

import numpy as np
from faster_whisper import WhisperModel

from voiceappointmentchatbot.config import Device, WhisperConfig


def _register_cuda_dll_dirs() -> None:
    """Add bundled NVIDIA wheel ``bin`` dirs to the DLL search path.

    ``faster-whisper`` loads ``cublas64_12.dll`` and ``cudnn*64_9.dll``
    dynamically. The ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12``
    wheels ship those DLLs under ``nvidia/<pkg>/bin`` but do not put
    them on the loader path, so on Windows the loader can't find them
    unless we register the directories explicitly. No-op on non-Windows
    or when the wheels are not installed.
    """
    if sys.platform != "win32":
        return
    for package in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))


@dataclass(frozen=True)
class Transcript:
    """Result of an ASR pass over a single utterance.

    Attributes:
        text: Decoded transcript with leading/trailing whitespace stripped.
        language: ISO 639-1 code Whisper detected (``en``, ``hu``, ...).
        language_probability: Confidence in the detected language.
    """

    text: str
    language: str
    language_probability: float


_SUPPORTED_LANGUAGES = ("en", "hu")
_LOCK_CONFIDENCE = 0.9


class WhisperTranscriber:
    """Thin wrapper that lazily loads a faster-whisper model.

    The transcriber starts in auto-detect mode. The first time Whisper
    reports an EN or HU detection at probability ``>= 0.9`` the language
    is *locked*: every subsequent call passes that code as the
    ``language=`` argument to ``model.transcribe`` so a one-off mistaken
    detection (Hungarian utterance tagged ``sk``, etc.) cannot derail
    the rest of the conversation.

    Attributes:
        device: Compute device the underlying model runs on.
        config: Whisper model selection rules.
        locked_language: ISO 639-1 code Whisper is pinned to, or
            ``None`` while still auto-detecting.
    """

    def __init__(self, device: Device, config: WhisperConfig) -> None:
        """Initialise without loading the model yet."""
        self.device = device
        self.config = config
        self.locked_language: Optional[str] = None
        self._model: Optional[WhisperModel] = None

    def _ensure_loaded(self) -> WhisperModel:
        """Load the model on first use and cache it for subsequent calls."""
        if self._model is None:
            if self.device == "cuda":
                _register_cuda_dll_dirs()
            self._model = WhisperModel(
                self.config.model_for(self.device),
                device=self.device,
                compute_type=self.config.compute_type_for(self.device),
            )
        return self._model

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe an utterance and detect its language.

        Once :attr:`locked_language` is set the language argument is
        forwarded to faster-whisper so it skips its own detection step
        and decodes the utterance in the locked language.

        Args:
            audio: Mono float32 PCM samples at 16 kHz.

        Returns:
            Transcript with text and detected language metadata. Empty
            input yields an empty transcript with language ``en``.
        """
        if audio.size == 0:
            return Transcript(text="", language="en", language_probability=0.0)

        model = self._ensure_loaded()
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language=self.locked_language,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        language = info.language
        probability = float(info.language_probability)
        if (
            self.locked_language is None
            and language in _SUPPORTED_LANGUAGES
            and probability >= _LOCK_CONFIDENCE
        ):
            self.locked_language = language
        return Transcript(
            text=text,
            language=language,
            language_probability=probability,
        )
