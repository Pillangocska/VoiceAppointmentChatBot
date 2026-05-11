"""Unit tests for the booking state and slot validation."""

from datetime import datetime
from pathlib import Path

import pytest

from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.domains import load_domain


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


@pytest.fixture
def vet_state() -> BookingState:
    """Fresh booking state for the vet domain."""
    return BookingState(domain=load_domain("vet", DOMAINS_DIR))


@pytest.fixture
def hairdresser_state() -> BookingState:
    """Fresh booking state for the hairdresser domain."""
    return BookingState(domain=load_domain("hairdresser", DOMAINS_DIR))


def test_new_state_is_empty_and_incomplete(vet_state: BookingState) -> None:
    """A fresh state has no slots filled and is not complete."""
    assert vet_state.slots == {}
    assert not vet_state.is_complete()
    assert {spec.name for spec in vet_state.missing_slots()} == set(vet_state.domain.slot_names)


def test_setting_text_slot_marks_it_confirmed(vet_state: BookingState) -> None:
    """Non-phone slots are auto-confirmed once set."""
    vet_state.set_slot("customer_name", "  Anna Kovács  ")

    assert vet_state.slots["customer_name"] == "Anna Kovács"
    assert "customer_name" in vet_state.confirmed_slots


def test_phone_slot_normalises_and_awaits_confirmation(vet_state: BookingState) -> None:
    """Phone slots strip formatting and require explicit confirmation."""
    vet_state.set_slot("phone", "+36 (1) 555-0199")

    assert vet_state.slots["phone"] == "+3615550199"
    assert vet_state.pending_phone_confirmation is True
    assert "phone" not in vet_state.confirmed_slots
    assert any(spec.name == "phone" for spec in vet_state.missing_slots())


def test_phone_confirmation_marks_slot_confirmed(vet_state: BookingState) -> None:
    """Calling ``confirm_phone`` clears the pending flag and confirms the slot."""
    vet_state.set_slot("phone", "+36 1 555 0199")

    vet_state.confirm_phone()

    assert "phone" in vet_state.confirmed_slots
    assert vet_state.pending_phone_confirmation is False


def test_phone_rejection_clears_value(vet_state: BookingState) -> None:
    """Calling ``reject_phone`` removes the stored value."""
    vet_state.set_slot("phone", "+36 1 555 0199")

    vet_state.reject_phone()

    assert "phone" not in vet_state.slots
    assert vet_state.pending_phone_confirmation is False


def test_too_short_phone_is_rejected(vet_state: BookingState) -> None:
    """Phone numbers with fewer than seven digits are rejected."""
    with pytest.raises(ValueError, match="at least"):
        vet_state.set_slot("phone", "12345")


def test_unknown_slot_raises_key_error(vet_state: BookingState) -> None:
    """Setting a slot the domain does not declare raises ``KeyError``."""
    with pytest.raises(KeyError):
        vet_state.set_slot("nonexistent", "value")


def test_empty_slot_value_is_rejected(vet_state: BookingState) -> None:
    """Whitespace-only slot values raise ``ValueError``."""
    with pytest.raises(ValueError, match="cannot be empty"):
        vet_state.set_slot("customer_name", "   ")


def test_full_vet_booking_is_complete(vet_state: BookingState) -> None:
    """A vet booking with every slot filled and confirmed is complete."""
    vet_state.set_slot("customer_name", "Anna Kovács")
    vet_state.set_slot("phone", "+36 1 555 0199")
    vet_state.confirm_phone()
    vet_state.set_slot("pet_name", "Bodri")
    vet_state.set_slot("species", "dog")
    vet_state.set_slot("complaint", "limping for two days")
    vet_state.set_slot("time", "Friday at 10:00", iso="2026-05-15T10:00+02:00")

    assert vet_state.is_complete()
    assert vet_state.missing_slots() == ()


def test_hairdresser_booking_completes_with_four_slots(
    hairdresser_state: BookingState,
) -> None:
    """The hairdresser domain only needs four confirmed slots."""
    hairdresser_state.set_slot("customer_name", "Réka")
    hairdresser_state.set_slot("phone", "+36 30 123 4567")
    hairdresser_state.confirm_phone()
    hairdresser_state.set_slot("service", "balayage")
    hairdresser_state.set_slot(
        "time", "next Saturday at 14:00", iso="2026-05-16T14:00+02:00"
    )

    assert hairdresser_state.is_complete()


def test_phone_readback_in_english(vet_state: BookingState) -> None:
    """English readback expands each digit to its word form."""
    vet_state.set_slot("phone", "+36 1 555 0199")

    readback = vet_state.phone_readback("en")

    assert readback is not None
    assert readback.startswith("plus three six")
    assert "zero" in readback


def test_phone_readback_in_hungarian(vet_state: BookingState) -> None:
    """Hungarian readback uses Hungarian digit words."""
    vet_state.set_slot("phone", "+36 1 555 0199")

    readback = vet_state.phone_readback("hu")

    assert readback is not None
    assert readback.startswith("plusz három hat")
    assert "kilenc" in readback


def test_phone_readback_returns_none_when_unset(vet_state: BookingState) -> None:
    """Without a stored phone the readback is ``None``."""
    assert vet_state.phone_readback("en") is None


def test_confirm_phone_without_value_raises(vet_state: BookingState) -> None:
    """Confirming when no phone is stored raises ``RuntimeError``."""
    with pytest.raises(RuntimeError):
        vet_state.confirm_phone()


def test_confirm_phone_when_already_confirmed_raises(vet_state: BookingState) -> None:
    """Double-confirming the phone slot raises ``RuntimeError``."""
    vet_state.set_slot("phone", "+36 1 555 0199")
    vet_state.confirm_phone()

    with pytest.raises(RuntimeError):
        vet_state.confirm_phone()


def test_datetime_slot_stores_raw_value_and_iso(vet_state: BookingState) -> None:
    """The raw phrase goes under ``slots`` and the iso under ``normalised``."""
    vet_state.set_slot(
        "time", "holnap tizenhárom órakor", iso="2026-05-12T13:00+02:00"
    )

    assert vet_state.slots["time"] == "holnap tizenhárom órakor"
    assert vet_state.normalised["time"] == "2026-05-12T13:00+02:00"


def test_datetime_slot_requires_iso(vet_state: BookingState) -> None:
    """Omitting ``iso`` for a datetime slot is an error the model must fix."""
    with pytest.raises(ValueError, match="datetime slot"):
        vet_state.set_slot("time", "holnap 12:30")


def test_datetime_slot_rejects_malformed_iso(vet_state: BookingState) -> None:
    """A malformed ISO timestamp surfaces a clear ``ValueError``."""
    with pytest.raises(ValueError, match="invalid ISO timestamp"):
        vet_state.set_slot("time", "holnap 12:30", iso="tomorrow at 1pm")


def test_datetime_slot_rejects_naive_iso(vet_state: BookingState) -> None:
    """An ISO timestamp without a timezone offset is rejected."""
    with pytest.raises(ValueError, match="timezone offset"):
        vet_state.set_slot("time", "holnap 12:30", iso="2026-05-12T13:00")


def test_datetime_iso_is_canonicalised(vet_state: BookingState) -> None:
    """Stored iso strings are reformatted to minute precision."""
    vet_state.set_slot(
        "time", "tomorrow 1 pm", iso="2026-05-12T13:00:00+02:00"
    )

    assert vet_state.normalised["time"] == "2026-05-12T13:00+02:00"


def test_set_time_anchor_rejects_naive_datetime(vet_state: BookingState) -> None:
    """The anchor must carry timezone information to avoid drift."""
    naive = datetime(2026, 5, 11, 9, 0)

    with pytest.raises(ValueError):
        vet_state.set_time_anchor(naive)
