"""Speech recognition wrapper around faster-whisper.

Loads a single Whisper model sized to the available device and exposes a
``transcribe`` method that returns the decoded text along with the
language Whisper detected for the utterance.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from voiceappointmentchatbot.config import Device, WhisperConfig


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


class WhisperTranscriber:
    """Thin wrapper that lazily loads a faster-whisper model.

    Attributes:
        device: Compute device the underlying model runs on.
        config: Whisper model selection rules.
    """

    def __init__(self, device: Device, config: WhisperConfig) -> None:
        """Initialise without loading the model yet."""
        self.device = device
        self.config = config
        self._model: Optional[WhisperModel] = None

    def _ensure_loaded(self) -> WhisperModel:
        """Load the model on first use and cache it for subsequent calls."""
        if self._model is None:
            self._model = WhisperModel(
                self.config.model_for(self.device),
                device=self.device,
                compute_type=self.config.compute_type_for(self.device),
            )
        return self._model

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe an utterance and detect its language.

        Args:
            audio: Mono float32 PCM samples at 16 kHz.

        Returns:
            Transcript with text and detected language metadata. Empty
            input yields an empty transcript with language ``en``.
        """
        if audio.size == 0:
            return Transcript(text="", language="en", language_probability=0.0)

        model = self._ensure_loaded()
        segments, info = model.transcribe(audio, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )
