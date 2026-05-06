"""Unit tests for helpers exposed by :mod:`voiceappointmentchatbot.main`.

These tests exercise pure helpers only — they avoid importing the full
CLI loop's heavyweight side effects (microphone, model loading) by
importing the helper symbol directly.
"""

from voiceappointmentchatbot.booking_writer import SentimentSummary
from voiceappointmentchatbot.main import summarise_sentiments


def test_summarise_sentiments_empty_returns_none() -> None:
    """No samples → no summary; the writer omits the key entirely."""
    assert summarise_sentiments([]) is None


def test_summarise_sentiments_single_sample_echoes_the_value() -> None:
    """A single sample is echoed back with ``samples == 1``."""
    summary = summarise_sentiments([("positive", 0.91)])

    assert summary == SentimentSummary(label="positive", score=0.91, samples=1)


def test_summarise_sentiments_majority_wins_on_label_count() -> None:
    """The most-frequent label wins regardless of confidence spread."""
    samples = [
        ("positive", 0.95),
        ("neutral", 0.60),
        ("neutral", 0.55),
    ]

    summary = summarise_sentiments(samples)

    assert summary is not None
    assert summary.label == "neutral"
    assert summary.samples == 3
    assert abs(summary.score - 0.575) < 1e-6


def test_summarise_sentiments_tie_broken_by_higher_mean_score() -> None:
    """When two labels tie on count the higher mean confidence wins."""
    samples = [
        ("positive", 0.55),
        ("positive", 0.60),
        ("negative", 0.90),
        ("negative", 0.95),
    ]

    summary = summarise_sentiments(samples)

    assert summary is not None
    assert summary.label == "negative"
    assert summary.samples == 4
    assert abs(summary.score - 0.925) < 1e-6
