"""Unit tests for the week-one echo dialogue policy."""

from voiceappointmentchatbot.asr import Transcript
from voiceappointmentchatbot.dialogue import echo_reply
from voiceappointmentchatbot.sentiment import SentimentResult


def test_english_positive_reply_quotes_user() -> None:
    """An English transcript is echoed with a positive acknowledgement."""
    transcript = Transcript(text="hello there", language="en", language_probability=0.99)
    sentiment = SentimentResult(label="positive", score=0.9)

    reply = echo_reply(transcript, sentiment)

    assert "hello there" in reply
    assert reply.startswith("You said:")


def test_hungarian_neutral_reply_uses_hungarian_template() -> None:
    """A Hungarian transcript is echoed with the Hungarian template."""
    transcript = Transcript(text="szia", language="hu", language_probability=0.95)
    sentiment = SentimentResult(label="neutral", score=0.8)

    reply = echo_reply(transcript, sentiment)

    assert reply.startswith("Azt mondtad:")
    assert "szia" in reply


def test_unknown_language_falls_back_to_english() -> None:
    """A language without a template falls back to the English template."""
    transcript = Transcript(text="bonjour", language="fr", language_probability=0.7)
    sentiment = SentimentResult(label="neutral", score=0.5)

    reply = echo_reply(transcript, sentiment)

    assert reply.startswith("You said:")


def test_empty_transcript_yields_silence_message() -> None:
    """An empty transcript produces a polite "did not hear" message."""
    transcript = Transcript(text="", language="en", language_probability=0.0)
    sentiment = SentimentResult(label="neutral", score=0.0)

    reply = echo_reply(transcript, sentiment)

    assert reply == "I did not hear anything."
