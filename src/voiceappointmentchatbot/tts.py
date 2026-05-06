"""Text-to-speech synthesis backed by Piper voices.

Synthesises an utterance into a float32 PCM stream and plays it through
the default audio output. Voices are loaded lazily and cached per
language so switching between EN and HU during a session is cheap, and
missing voice files are downloaded on first use.
"""

from urllib.request import urlretrieve
from typing import Dict, List
from pathlib import Path

import numpy as np
import sounddevice as sd

from voiceappointmentchatbot.config import PiperConfig, PiperVoiceSpec


_TRAILING_SILENCE_SECONDS = 0.3


class PiperSpeaker:
    """Lazy multi-voice Piper TTS wrapper.

    Attributes:
        config: Piper voice configuration.
    """

    def __init__(self, config: PiperConfig) -> None:
        """Initialise without loading any voices yet."""
        self.config = config
        self._voices: Dict[str, object] = {}

    def _ensure_voice(self, language: str) -> object:
        """Load the voice for ``language``, fetching files when missing.

        Args:
            language: ISO 639-1 language code.

        Returns:
            The cached PiperVoice instance.
        """
        if language in self._voices:
            return self._voices[language]

        from piper.voice import PiperVoice

        spec = self.config.voices[language]
        model_path, config_path = self.config.voice_for(language)
        self._download_if_missing(model_path, spec.model_url)
        self._download_if_missing(config_path, spec.config_url)

        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._voices[language] = voice
        return voice

    @staticmethod
    def _download_if_missing(target: Path, url: str) -> None:
        """Download ``url`` to ``target`` when ``target`` does not exist.

        Args:
            target: Destination path; parent directories are created.
            url: Public URL to fetch the file from.
        """
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[piper] downloading {target.name}")
        urlretrieve(url, target)

    @staticmethod
    def _spec_for(config: PiperConfig, language: str) -> PiperVoiceSpec:
        """Return the voice specification registered for ``language``."""
        return config.voices[language]

    def warm_up(self) -> None:
        """Eagerly load every configured voice.

        Downloads any missing model/config files for the registered
        languages and instantiates each :class:`PiperVoice` so the first
        spoken reply does not stall on disk I/O or model construction.
        """
        for language in self.config.voices:
            self._ensure_voice(language)

    def speak(self, text: str, language: str) -> None:
        """Synthesise ``text`` in ``language`` and play it on the speakers.

        Args:
            text: Utterance to speak. Empty text is a no-op.
            language: ISO 639-1 language code; falls back to ``en`` when
                no voice is registered for the detected language.
        """
        if not text.strip():
            return

        if language not in self.config.voices:
            language = "en"

        voice = self._ensure_voice(language)
        sample_rate, samples = self._synthesize(voice, text)
        padding = np.zeros(int(sample_rate * _TRAILING_SILENCE_SECONDS), dtype=np.float32)
        padded = np.concatenate([samples, padding])
        sd.play(padded, samplerate=sample_rate)
        sd.wait()

    @staticmethod
    def _synthesize(voice: object, text: str) -> tuple[int, np.ndarray]:
        """Render ``text`` through ``voice`` into a NumPy float32 array.

        Concatenates the float audio from every ``AudioChunk`` yielded by
        Piper's streaming ``synthesize`` API. The sample rate is read
        from the first chunk; subsequent chunks share it.

        Args:
            voice: A loaded PiperVoice instance.
            text: Utterance to synthesise.

        Returns:
            Tuple of (sample rate in Hz, mono float32 samples in [-1, 1]).
        """
        chunks: List[np.ndarray] = []
        sample_rate = 0
        for chunk in voice.synthesize(text):  # type: ignore[attr-defined]
            if sample_rate == 0:
                sample_rate = int(chunk.sample_rate)
            chunks.append(np.asarray(chunk.audio_float_array, dtype=np.float32))

        if not chunks:
            return sample_rate or 22_050, np.zeros(0, dtype=np.float32)
        return sample_rate, np.concatenate(chunks)
