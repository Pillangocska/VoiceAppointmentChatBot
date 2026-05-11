"""Speech recognition wrapper around faster-whisper.

Loads a single Whisper model sized to the available device and exposes a
``transcribe`` method that returns the decoded text along with the
language Whisper detected for the utterance.
"""

import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

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
_LOCK_CONFIDENCE = 0.8


class WhisperTranscriber:
    """Thin wrapper that lazily loads a faster-whisper model.

    Every utterance is decoded with Whisper's per-utterance language
    auto-detection so the user can switch between English and Hungarian
    mid-conversation. The last EN or HU detection at probability
    ``>= _LOCK_CONFIDENCE`` is remembered as a *hint* — used only to
    pick the ``initial_prompt`` for short follow-up utterances — and
    never forced onto the decoder.

    A per-language ``initial_prompt`` biases the decoder toward the
    domain vocabulary (pet names, "foglalni", "rendelő", ...) so common
    booking words are less likely to come back as garbled neighbours.

    Attributes:
        device: Compute device the underlying model runs on.
        config: Whisper model selection rules.
        last_language: Last EN/HU detection at the confidence threshold,
            or ``None`` if no confident detection has happened yet. Used
            only as a prompt hint, not to constrain decoding.
        prompts: Mapping from ISO 639-1 code to the ``initial_prompt``
            forwarded to faster-whisper for utterances in that language.
    """

    def __init__(
        self,
        device: Device,
        config: WhisperConfig,
        prompts: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialise without loading the model yet.

        Args:
            device: Target compute device for the Whisper model.
            config: Whisper model selection and decoding parameters.
            prompts: Optional language-keyed ``initial_prompt`` strings
                used to bias decoding toward domain vocabulary.
        """
        self.device = device
        self.config = config
        self.last_language: Optional[str] = None
        self.prompts: dict[str, str] = dict(prompts or {})
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

    def warm_up(self) -> None:
        """Eagerly load the model and run one dummy decode.

        Calling this from startup avoids the multi-second pause that
        otherwise happens on the first user utterance — both because the
        weights download (or cache hit) is paid up front, and because
        the first decode triggers CUDA kernel compilation and
        CTranslate2 graph construction that subsequent calls reuse.
        """
        model = self._ensure_loaded()
        warmup_audio = np.zeros(self.config.warmup_samples, dtype=np.float32)
        segments, _ = model.transcribe(warmup_audio, beam_size=1)
        for _ in segments:
            pass

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe an utterance and detect its language.

        Whisper's auto-detection runs on every call so the user can
        switch between English and Hungarian mid-conversation. When a
        prompt is registered for the most recent confidently-detected
        language it is passed as ``initial_prompt`` to bias decoding
        toward domain vocabulary.

        Args:
            audio: Mono float32 PCM samples at 16 kHz.

        Returns:
            Transcript with text and detected language metadata. Empty
            input yields an empty transcript with language ``en``.
        """
        if audio.size == 0:
            return Transcript(text="", language="en", language_probability=0.0)

        model = self._ensure_loaded()
        initial_prompt = self.prompts.get(self.last_language) if self.last_language else None
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        language = info.language
        probability = float(info.language_probability)
        if language in _SUPPORTED_LANGUAGES and probability >= _LOCK_CONFIDENCE:
            self.last_language = language
        return Transcript(
            text=text,
            language=language,
            language_probability=probability,
        )
