"""Unit tests for :mod:`voiceappointmentchatbot.asr`.

The faster-whisper model is replaced with a stub that records every
``transcribe`` call so we can assert on the keyword arguments without
loading the real weights.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from voiceappointmentchatbot import asr as asr_module
from voiceappointmentchatbot.asr import Transcript, WhisperTranscriber
from voiceappointmentchatbot.config import WhisperConfig


class _StubInfo:
    """Minimal stand-in for the ``info`` object returned by faster-whisper."""

    def __init__(self, language: str, probability: float) -> None:
        self.language = language
        self.language_probability = probability


class _StubSegment:
    """Tiny duck-typed segment with the only attribute the wrapper reads."""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubWhisperModel:
    """Records ``transcribe`` invocations and replays scripted segments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: List[Dict[str, Any]] = []
        self.next_language = "hu"
        self.next_probability = 0.95
        self.next_segments = [_StubSegment("Jó reggelt!")]

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Any:
        """Capture call arguments and return the scripted segments."""
        self.calls.append(kwargs)
        info = _StubInfo(self.next_language, self.next_probability)
        return iter(self.next_segments), info


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> _StubWhisperModel:
    """Replace ``WhisperModel`` with the stub for the duration of the test."""
    stub = _StubWhisperModel()

    def _factory(*args: Any, **kwargs: Any) -> _StubWhisperModel:
        stub.init_args = args
        stub.init_kwargs = kwargs
        return stub

    monkeypatch.setattr(asr_module, "WhisperModel", _factory)
    return stub


def test_initial_prompt_is_forwarded_for_locked_language(
    stub_model: _StubWhisperModel,
) -> None:
    """When a prompt exists for the locked language it reaches the decoder."""
    transcriber = WhisperTranscriber(
        device="cpu",
        config=WhisperConfig(),
        prompts={"hu": "rendelő foglalás kutya"},
    )
    transcriber.locked_language = "hu"

    transcriber.transcribe(np.zeros(1600, dtype=np.float32))

    assert stub_model.calls[-1]["initial_prompt"] == "rendelő foglalás kutya"
    assert stub_model.calls[-1]["language"] == "hu"


def test_initial_prompt_is_none_before_language_is_locked(
    stub_model: _StubWhisperModel,
) -> None:
    """Without a locked language we cannot pick a prompt yet — pass ``None``."""
    transcriber = WhisperTranscriber(
        device="cpu",
        config=WhisperConfig(),
        prompts={"hu": "rendelő foglalás", "en": "appointment booking"},
    )

    transcriber.transcribe(np.zeros(1600, dtype=np.float32))

    assert stub_model.calls[-1]["initial_prompt"] is None


def test_transcribe_returns_decoded_text(stub_model: _StubWhisperModel) -> None:
    """The wrapper concatenates segment text and exposes language metadata."""
    transcriber = WhisperTranscriber(device="cpu", config=WhisperConfig())
    stub_model.next_segments = [_StubSegment("Hello"), _StubSegment(" there")]
    stub_model.next_language = "en"
    stub_model.next_probability = 0.97

    result = transcriber.transcribe(np.zeros(1600, dtype=np.float32))

    assert result == Transcript(
        text="Hello there",
        language="en",
        language_probability=pytest.approx(0.97),
    )


def test_empty_audio_short_circuits(stub_model: _StubWhisperModel) -> None:
    """An empty input never reaches the model."""
    transcriber = WhisperTranscriber(device="cpu", config=WhisperConfig())

    result = transcriber.transcribe(np.zeros(0, dtype=np.float32))

    assert result.text == ""
    assert stub_model.calls == []


def test_language_lock_triggers_at_confidence_threshold(
    stub_model: _StubWhisperModel,
) -> None:
    """A high-confidence detection pins subsequent calls to that language."""
    transcriber = WhisperTranscriber(device="cpu", config=WhisperConfig())
    stub_model.next_language = "hu"
    stub_model.next_probability = 0.9

    transcriber.transcribe(np.zeros(1600, dtype=np.float32))

    assert transcriber.locked_language == "hu"

    transcriber.transcribe(np.zeros(1600, dtype=np.float32))
    assert stub_model.calls[-1]["language"] == "hu"
