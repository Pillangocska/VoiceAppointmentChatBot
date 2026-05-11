"""Unit tests for the appointment JSON writer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.booking_writer import (
    AppointmentRecord,
    SentimentSummary,
    TranscriptTurn,
    appointment_record_from,
    serialise_record,
    write_appointment,
)
from voiceappointmentchatbot.domains import load_domain


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"
FIXED_TIMESTAMP = datetime(2026, 5, 6, 14, 32, 11, tzinfo=timezone.utc)
TIME_ANCHOR = datetime(2026, 5, 6, 14, 32, 11, tzinfo=timezone(timedelta(hours=2)))


def _complete_vet_state() -> BookingState:
    """Return a vet state with every slot filled and confirmed."""
    state = BookingState(domain=load_domain("vet", DOMAINS_DIR))
    state.set_time_anchor(TIME_ANCHOR)
    state.set_slot("customer_name", "Anna Kovács")
    state.set_slot("phone", "+36 1 555 0199")
    state.confirm_phone()
    state.set_slot("pet_name", "Bodri")
    state.set_slot("species", "dog")
    state.set_slot("complaint", "limping for two days")
    state.set_slot("time", "Friday at 10:00", iso="2026-05-08T10:00+02:00")
    return state


def _complete_hairdresser_state() -> BookingState:
    """Return a hairdresser state with every slot filled and confirmed."""
    state = BookingState(domain=load_domain("hairdresser", DOMAINS_DIR))
    state.set_time_anchor(TIME_ANCHOR)
    state.set_slot("customer_name", "Réka Nagy")
    state.set_slot("phone", "+36 30 123 4567")
    state.confirm_phone()
    state.set_slot("service", "balayage")
    state.set_slot(
        "time", "next Saturday at 14:00", iso="2026-05-16T14:00+02:00"
    )
    return state


def test_serialise_promotes_customer_name_and_phone() -> None:
    """``customer_name`` and ``phone`` move to a top-level customer block."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="hu",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["customer"] == {
        "name": "Anna Kovács",
        "phone": "+3615550199",
    }
    assert "customer_name" not in payload["slots"]
    assert "phone" not in payload["slots"]


def test_serialise_keeps_other_slots_under_slots_key() -> None:
    """Domain-specific slots are dumped verbatim under ``slots``."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["slots"] == {
        "pet_name": "Bodri",
        "species": "dog",
        "complaint": "limping for two days",
        "time": "Friday at 10:00",
        "time_iso": "2026-05-08T10:00+02:00",
    }


def test_serialise_records_domain_and_timestamp() -> None:
    """The payload echoes the domain name and an ISO 8601 timestamp."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["domain"] == "vet"
    assert payload["created_at"] == "2026-05-06T14:32:11+00:00"
    assert payload["language"] == "en"


def test_serialise_includes_sentiment_when_present() -> None:
    """A sentiment summary is rendered with rounded score and sample count."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        sentiment=SentimentSummary(label="neutral", score=0.7843, samples=4),
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["sentiment"] == {
        "label": "neutral",
        "score": 0.784,
        "samples": 4,
    }


def test_serialise_omits_sentiment_when_absent() -> None:
    """Without a sentiment summary, the key is left out entirely."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert "sentiment" not in payload


def test_serialise_includes_transcript_when_supplied() -> None:
    """Transcript turns survive serialisation in order."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        transcript=(
            TranscriptTurn(role="user", text="Hi, I need an appointment.", language="en"),
            TranscriptTurn(role="assistant", text="Of course, what's your name?", language="en"),
        ),
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["transcript"] == [
        {"role": "user", "text": "Hi, I need an appointment.", "language": "en"},
        {"role": "assistant", "text": "Of course, what's your name?", "language": "en"},
    ]


def test_serialise_omits_transcript_when_empty() -> None:
    """Empty transcripts are not written out as an empty list."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert "transcript" not in payload


def test_hairdresser_record_only_promotes_two_fields() -> None:
    """A four-slot domain still produces a clean customer/slots split."""
    record = AppointmentRecord(
        state=_complete_hairdresser_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["customer"] == {
        "name": "Réka Nagy",
        "phone": "+36301234567",
    }
    assert payload["slots"] == {
        "service": "balayage",
        "time": "next Saturday at 14:00",
        "time_iso": "2026-05-16T14:00+02:00",
    }


def test_write_appointment_creates_file_and_returns_path(tmp_path: Path) -> None:
    """The writer creates the output directory and returns the file path."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=FIXED_TIMESTAMP,
    )

    path = write_appointment(record, tmp_path / "deeper" / "output")

    assert path.exists()
    assert path.parent.name == "output"
    assert path.name.startswith("appointment_vet_20260506T143211")
    assert path.suffix == ".json"


def test_write_appointment_emits_valid_utf8_json(tmp_path: Path) -> None:
    """The written file parses back as JSON with non-ASCII intact."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="hu",
        created_at=FIXED_TIMESTAMP,
    )

    path = write_appointment(record, tmp_path)
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert parsed["customer"]["name"] == "Anna Kovács"
    assert parsed["domain"] == "vet"


def test_appointment_record_from_history_filters_to_spoken_turns() -> None:
    """``appointment_record_from`` discards tool-use and tool-result entries."""
    state = _complete_vet_state()
    history = [
        {"role": "user", "content": "Hello, I'd like an appointment."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Of course, what's your name?"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "update_slot",
                    "input": {"name": "customer_name", "value": "Anna"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "stored",
                    "is_error": False,
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Thanks, Anna."}]},
    ]

    record = appointment_record_from(
        state=state, language="en", history=history
    )

    assert [(turn.role, turn.text) for turn in record.transcript] == [
        ("user", "Hello, I'd like an appointment."),
        ("assistant", "Of course, what's your name?"),
        ("assistant", "Thanks, Anna."),
    ]


def test_appointment_record_from_prefers_explicit_transcript() -> None:
    """An explicit ``transcript`` argument wins over ``history``."""
    state = _complete_vet_state()
    explicit = (TranscriptTurn(role="user", text="hi", language="en"),)

    record = appointment_record_from(
        state=state,
        language="en",
        transcript=explicit,
        history=[{"role": "user", "content": "ignored"}],
    )

    assert record.transcript == explicit


def test_filename_is_unique_across_two_writes(tmp_path: Path) -> None:
    """Two records at different timestamps produce different filenames."""
    earlier = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=datetime(2026, 5, 6, 14, 32, 10, tzinfo=timezone.utc),
    )
    later = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        created_at=datetime(2026, 5, 6, 14, 32, 11, tzinfo=timezone.utc),
    )

    first = write_appointment(earlier, tmp_path)
    second = write_appointment(later, tmp_path)

    assert first != second
    assert first.exists() and second.exists()


def test_serialise_round_trips_score_to_three_decimals() -> None:
    """Sentiment scores are rounded for human readability."""
    record = AppointmentRecord(
        state=_complete_vet_state(),
        language="en",
        sentiment=SentimentSummary(label="positive", score=0.123456, samples=2),
        created_at=FIXED_TIMESTAMP,
    )

    payload = serialise_record(record)

    assert payload["sentiment"]["score"] == pytest.approx(0.123)
