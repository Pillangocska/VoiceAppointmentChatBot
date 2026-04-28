"""Dialogue policy used during the first project week.

Implements a deliberately trivial echo policy that repeats the user's
utterance back together with a short sentiment acknowledgement, in the
language Whisper detected. This component will be replaced by a
slot-filling state machine in week two.
"""

from voiceappointmentchatbot.asr import Transcript
from voiceappointmentchatbot.sentiment import SentimentResult

_ACKNOWLEDGEMENT = {
    "en": {
        "positive": "You sound positive.",
        "neutral": "Got it.",
        "negative": "You sound a bit down.",
    },
    "hu": {
        "positive": "Vidámnak hangzol.",
        "neutral": "értem.",
        "negative": "Kicsit lehangoltnak hangzol.",
    },
}


def echo_reply(transcript: Transcript, sentiment: SentimentResult) -> str:
    """Build a short reply that echoes the transcript and notes sentiment.

    Args:
        transcript: ASR output for the latest utterance.
        sentiment: Sentiment result for the same utterance.

    Returns:
        A reply string in the language Whisper detected, falling back to
        English when the language is not yet supported.
    """
    language = transcript.language if transcript.language in _ACKNOWLEDGEMENT else "en"
    ack = _ACKNOWLEDGEMENT[language].get(sentiment.label, _ACKNOWLEDGEMENT[language]["neutral"])

    if not transcript.text:
        return "I did not hear anything." if language == "en" else "Nem hallottam semmit."

    quoted = transcript.text.rstrip(" .!?,;:")
    if language == "hu":
        return f"Azt mondtad: {quoted}. {ack}"
    return f"You said: {quoted}. {ack}"
