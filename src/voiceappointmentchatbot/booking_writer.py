"""Persist confirmed appointments to a JSON file under ``output/``.

The dialogue manager calls :func:`write_appointment` once the model
issues ``confirm_appointment`` against a complete booking state. The
file format is a flat JSON object keyed by stable field names so the
contents can be skimmed or diffed without per-domain knowledge:

    {
        "domain": "vet",
        "created_at": "2026-05-06T14:32:11+02:00",
        "language": "hu",
        "customer": {"name": "Anna Kovács", "phone": "+3615550199"},
        "slots": {"pet_name": "Bodri", "species": "dog", ...},
        "sentiment": {"label": "neutral", "score": 0.78, "samples": 4},
        "transcript": [{"role": "user", "text": "...", "language": "hu"}, ...]
    }

``customer.name`` and ``customer.phone`` are promoted from the
``customer_name`` and ``phone`` slots because every domain in this
project carries those two. Everything else is dumped verbatim under
``slots`` so adding a new domain only needs a new YAML file, not a
change to this module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import re

from voiceappointmentchatbot.booking import BookingState


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_PROMOTED_SLOTS = ("customer_name", "phone")


@dataclass(frozen=True)
class TranscriptTurn:
    """One user-or-assistant turn captured for the appointment record.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        text: Spoken or recognised text for the turn.
        language: ISO 639-1 code for the turn (``en`` or ``hu``).
    """

    role: str
    text: str
    language: str


@dataclass(frozen=True)
class SentimentSummary:
    """Aggregate sentiment across the user's turns.

    Attributes:
        label: Dominant label (``positive`` / ``neutral`` / ``negative``).
        score: Mean confidence of the dominant label, in [0, 1].
        samples: Number of user turns that contributed to the summary.
    """

    label: str
    score: float
    samples: int


@dataclass(frozen=True)
class AppointmentRecord:
    """Everything the JSON writer needs to serialise one booking.

    Attributes:
        state: Final booking state with every slot confirmed.
        language: Language code of the most recent user turn — used as
            the appointment's session language.
        sentiment: Optional aggregate sentiment summary.
        transcript: Optional list of recorded turns. Empty when
            transcript capture is disabled.
        created_at: Timezone-aware timestamp; defaults to ``datetime.now``
            at the system local timezone when omitted.
    """

    state: BookingState
    language: str
    sentiment: Optional[SentimentSummary] = None
    transcript: Sequence[TranscriptTurn] = field(default_factory=tuple)
    created_at: Optional[datetime] = None


def serialise_record(record: AppointmentRecord) -> Dict[str, Any]:
    """Convert an :class:`AppointmentRecord` into a plain dict.

    Args:
        record: Source record.

    Returns:
        Dictionary ready for ``json.dumps``.
    """
    state = record.state
    created_at = record.created_at or datetime.now().astimezone()

    customer = {
        "name": state.slots.get("customer_name", ""),
        "phone": state.slots.get("phone", ""),
    }
    other_slots: Dict[str, str] = {
        name: value
        for name, value in state.slots.items()
        if name not in _PROMOTED_SLOTS
    }

    payload: Dict[str, Any] = {
        "domain": state.domain.name,
        "created_at": created_at.isoformat(timespec="seconds"),
        "language": record.language,
        "customer": customer,
        "slots": other_slots,
    }
    if record.sentiment is not None:
        payload["sentiment"] = {
            "label": record.sentiment.label,
            "score": round(record.sentiment.score, 3),
            "samples": record.sentiment.samples,
        }
    if record.transcript:
        payload["transcript"] = [
            {"role": turn.role, "text": turn.text, "language": turn.language}
            for turn in record.transcript
        ]
    return payload


def write_appointment(
    record: AppointmentRecord,
    output_dir: Path,
) -> Path:
    """Write ``record`` as JSON under ``output_dir`` and return the path.

    The filename embeds the domain name and a UTC timestamp so two
    bookings made within the same second never collide. The output
    directory is created if missing.

    Args:
        record: Booking to persist.
        output_dir: Destination directory.

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = serialise_record(record)
    filename = _build_filename(record.state, record.created_at)
    path = output_dir / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _build_filename(state: BookingState, created_at: Optional[datetime]) -> str:
    """Compose a deterministic filename for one appointment."""
    timestamp = (created_at or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S")
    safe_domain = _FILENAME_SAFE.sub("_", state.domain.name) or "domain"
    return f"appointment_{safe_domain}_{timestamp}.json"


def appointment_record_from(
    state: BookingState,
    language: str,
    sentiment: Optional[SentimentSummary] = None,
    transcript: Optional[Sequence[TranscriptTurn]] = None,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
) -> AppointmentRecord:
    """Build a record from a booking state plus optional context.

    ``history`` is a convenience alternative to ``transcript`` for
    callers that already keep the dialogue manager's history list. When
    supplied, only ``user`` and ``assistant`` text blocks are
    included; tool-use and tool-result entries are dropped.

    Args:
        state: Final booking state.
        language: Latest user-turn language code.
        sentiment: Optional aggregate sentiment.
        transcript: Pre-built transcript turns. Wins over ``history``
            when both are supplied.
        history: Raw dialogue-manager history; converted to transcript
            turns when ``transcript`` is ``None``.

    Returns:
        Fully populated :class:`AppointmentRecord`.
    """
    if transcript is not None:
        turns = tuple(transcript)
    elif history is not None:
        turns = tuple(_history_to_transcript(history, language))
    else:
        turns = ()
    return AppointmentRecord(
        state=state,
        language=language,
        sentiment=sentiment,
        transcript=turns,
    )


def _history_to_transcript(
    history: Sequence[Mapping[str, Any]],
    language: str,
) -> List[TranscriptTurn]:
    """Filter the dialogue history down to spoken turns only."""
    turns: List[TranscriptTurn] = []
    for entry in history:
        role = entry.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(entry.get("content"))
        if not text:
            continue
        turns.append(TranscriptTurn(role=role, text=text, language=language))
    return turns


def _extract_text(content: Any) -> str:
    """Pull plain-text segments out of an Anthropic-style message body."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return " ".join(part.strip() for part in parts if part.strip())
    return ""
