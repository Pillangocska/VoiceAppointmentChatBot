"""In-progress appointment state and slot validation.

The dialogue manager keeps a :class:`BookingState` per session. The LLM
fills slots through tool calls; this module owns the rules for what
counts as a valid value, when an appointment is complete, and how to
read a phone number back to the user one digit at a time so Whisper
mishears can be caught before the booking is written to disk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

from voiceappointmentchatbot.domains import Domain, SlotSpec


_PHONE_ALLOWED = re.compile(r"[\d+]")
_PHONE_DIGITS_ONLY = re.compile(r"\D")
_MIN_PHONE_DIGITS = 7

_DIGIT_WORDS_EN: Dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "+": "plus",
}

_DIGIT_WORDS_HU: Dict[str, str] = {
    "0": "nulla",
    "1": "egy",
    "2": "kettő",
    "3": "három",
    "4": "négy",
    "5": "öt",
    "6": "hat",
    "7": "hét",
    "8": "nyolc",
    "9": "kilenc",
    "+": "plusz",
}


@dataclass
class BookingState:
    """Mutable per-session booking record.

    Slot values are stored as raw strings as captured from the user. The
    ``confirmed_slots`` set tracks which slots have been verified — for
    most slots that is implicit when the value is set, but for the
    ``phone`` slot we wait until the user explicitly confirms the
    digit-by-digit readback.

    Attributes:
        domain: Domain whose slot list governs this booking.
        slots: Mapping from slot name to the raw string value.
        confirmed_slots: Names of slots the user has confirmed. Untyped
            slots are auto-confirmed; phone slots are confirmed
            explicitly via :meth:`confirm_phone`.
        pending_phone_confirmation: ``True`` when a phone value is set
            but still awaiting digit-readback confirmation.
    """

    domain: Domain
    slots: Dict[str, str] = field(default_factory=dict)
    confirmed_slots: set[str] = field(default_factory=set)
    pending_phone_confirmation: bool = False

    def set_slot(self, name: str, value: str) -> None:
        """Store ``value`` for the slot ``name`` after light validation.

        Phone values are normalised (whitespace, dashes, parentheses are
        stripped) and the slot is marked as needing confirmation. Other
        slots are auto-confirmed once set.

        Args:
            name: Slot identifier declared by the active domain.
            value: Raw value as understood by the LLM.

        Raises:
            KeyError: If the slot is not declared by the domain.
            ValueError: If the value fails validation for the slot type.
        """
        spec = self.domain.slot(name)  # raises KeyError on unknown slot
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"slot {name!r} cannot be empty")

        if spec.type == "phone":
            normalised = _normalise_phone(cleaned)
            if not _is_valid_phone(normalised):
                raise ValueError(
                    f"phone number {value!r} does not contain at least "
                    f"{_MIN_PHONE_DIGITS} digits"
                )
            self.slots[name] = normalised
            self.confirmed_slots.discard(name)
            self.pending_phone_confirmation = True
            return

        self.slots[name] = cleaned
        self.confirmed_slots.add(name)

    def confirm_phone(self) -> None:
        """Mark the stored phone number as confirmed by the user.

        Raises:
            RuntimeError: If no phone value is stored or none is pending
                confirmation.
        """
        phone_slot = self._phone_slot()
        if phone_slot is None or phone_slot.name not in self.slots:
            raise RuntimeError("no phone value to confirm")
        if not self.pending_phone_confirmation:
            raise RuntimeError("phone is not awaiting confirmation")
        self.confirmed_slots.add(phone_slot.name)
        self.pending_phone_confirmation = False

    def reject_phone(self) -> None:
        """Discard the stored phone number after the user rejected the readback."""
        phone_slot = self._phone_slot()
        if phone_slot is None:
            return
        self.slots.pop(phone_slot.name, None)
        self.confirmed_slots.discard(phone_slot.name)
        self.pending_phone_confirmation = False

    def missing_slots(self) -> Tuple[SlotSpec, ...]:
        """Return slot specs that have not yet been filled and confirmed.

        A slot counts as missing if it has no stored value or if it is
        still awaiting confirmation. The result preserves the declaration
        order from the domain so the manager can ask for the next slot
        deterministically.
        """
        missing: List[SlotSpec] = []
        for spec in self.domain.slots:
            if spec.name not in self.slots or spec.name not in self.confirmed_slots:
                missing.append(spec)
        return tuple(missing)

    def is_complete(self) -> bool:
        """Whether every domain slot has a confirmed value."""
        return not self.missing_slots()

    def phone_readback(self, language: str) -> Optional[str]:
        """Return a spoken-style readback of the stored phone number.

        Each character is rendered as a digit word in ``language`` so
        that Whisper-introduced ambiguities ("nine" / "five",
        "kettő" / "hét") are easier for the user to catch.

        Args:
            language: ISO 639-1 language code; falls back to English.

        Returns:
            The spaced word sequence, or ``None`` if no phone is stored.
        """
        phone_slot = self._phone_slot()
        if phone_slot is None:
            return None
        value = self.slots.get(phone_slot.name)
        if not value:
            return None
        words = _DIGIT_WORDS_HU if language == "hu" else _DIGIT_WORDS_EN
        rendered = [words.get(char, char) for char in value]
        return " ".join(rendered)

    def _phone_slot(self) -> Optional[SlotSpec]:
        """Return the first phone-typed slot in the domain, if any."""
        for spec in self.domain.slots:
            if spec.type == "phone":
                return spec
        return None


def _normalise_phone(raw: str) -> str:
    """Strip formatting characters from a phone number.

    Keeps digits and a single leading ``+``. Spaces, dashes, dots and
    parentheses are removed. Internal ``+`` characters are dropped.
    """
    kept = "".join(_PHONE_ALLOWED.findall(raw))
    if not kept:
        return ""
    if kept.startswith("+"):
        return "+" + kept[1:].replace("+", "")
    return kept.replace("+", "")


def _is_valid_phone(value: str) -> bool:
    """Whether ``value`` contains at least :data:`_MIN_PHONE_DIGITS` digits."""
    digits = _PHONE_DIGITS_ONLY.sub("", value)
    return len(digits) >= _MIN_PHONE_DIGITS
