"""Unit tests for the slot-filling dialogue manager.

The tests drive the manager with a scripted fake LLM client so the tool
loop runs end-to-end without hitting the network. Each scripted turn is
either a plain assistant message or a sequence of tool calls; the
manager must dispatch them, append the tool results, and call back into
the client until the script yields a text turn.
"""

from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
import pytest

from voiceappointmentchatbot.asr import Transcript
from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.dialogue import DialogueManager, DialogueResult
from voiceappointmentchatbot.domains import load_domain
from voiceappointmentchatbot.llm import (
    AssistantTurn,
    TOOL_ASK_KB,
    TOOL_CONFIRM_APPOINTMENT,
    TOOL_CONFIRM_PHONE,
    TOOL_UPDATE_SLOT,
    ToolCall,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


class _ScriptedClient:
    """Replays a queue of pre-built :class:`AssistantTurn` objects.

    Each call to :meth:`respond` records the booking state snapshot and
    pops the next scripted turn. Tests assert against ``snapshots`` and
    ``calls`` to verify the manager passed the right history.
    """

    def __init__(self, turns: Sequence[AssistantTurn]) -> None:
        """Store the scripted turns in a FIFO."""
        self._turns: deque[AssistantTurn] = deque(turns)
        self.calls: list[list[dict[str, Any]]] = []
        self.snapshots: list[BookingState] = []

    def respond(
        self,
        history: list[dict[str, Any]],
        booking_state: BookingState,
    ) -> AssistantTurn:
        """Record the call and return the next scripted turn."""
        if not self._turns:
            raise AssertionError("scripted client ran out of turns")
        self.calls.append([dict(item) for item in history])
        self.snapshots.append(booking_state)
        return self._turns.popleft()


def _text_turn(text: str) -> AssistantTurn:
    """Build a scripted turn that just speaks to the user."""
    return AssistantTurn(
        text=text,
        tool_calls=(),
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
    )


def _tool_turn(*calls: ToolCall, text: str = "") -> AssistantTurn:
    """Build a scripted turn that issues one or more tool calls."""
    raw: list[dict[str, Any]] = []
    if text:
        raw.append({"type": "text", "text": text})
    for call in calls:
        raw.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    return AssistantTurn(
        text=text,
        tool_calls=tuple(calls),
        stop_reason="tool_use",
        raw_content=raw,
    )


class _ManagerFactory(Protocol):
    """Callable that builds a manager + scripted client for a test."""

    def __call__(
        self,
        turns: Sequence[AssistantTurn],
        *,
        knowledge_lookup: Callable[[str], str] | None = None,
        on_appointment_confirmed: Callable[[BookingState], None] | None = None,
    ) -> tuple[DialogueManager, _ScriptedClient]: ...


@pytest.fixture
def vet_manager_factory() -> _ManagerFactory:
    """Factory returning a manager bound to a scripted client."""

    def _build(
        turns: Sequence[AssistantTurn],
        *,
        knowledge_lookup: Callable[[str], str] | None = None,
        on_appointment_confirmed: Callable[[BookingState], None] | None = None,
    ) -> tuple[DialogueManager, _ScriptedClient]:
        domain = load_domain("vet", DOMAINS_DIR)
        client = _ScriptedClient(turns)
        manager = DialogueManager(
            domain=domain,
            client=client,
            knowledge_lookup=knowledge_lookup,
            on_appointment_confirmed=on_appointment_confirmed,
        )
        return manager, client

    return _build


def test_empty_transcript_returns_silence_message_in_english(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """An empty English transcript short-circuits without calling the LLM."""
    manager, client = vet_manager_factory(turns=[])

    result = manager.handle_user_turn(
        Transcript(text="", language="en", language_probability=0.0)
    )

    assert result == DialogueResult(
        reply="I did not hear anything. Could you repeat that, please?",
        language="en",
        booking_complete=False,
    )
    assert client.calls == []


def test_empty_transcript_returns_silence_message_in_hungarian(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """The silence message is localised when the user speaks Hungarian."""
    manager, _ = vet_manager_factory(turns=[])

    result = manager.handle_user_turn(
        Transcript(text="", language="hu", language_probability=0.9)
    )

    assert result.language == "hu"
    assert "Nem hallottam" in result.reply


def test_plain_text_turn_is_returned_verbatim(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """A turn with no tool calls becomes the user-facing reply."""
    manager, _ = vet_manager_factory(turns=[_text_turn("Hello, what is your name?")])

    result = manager.handle_user_turn(
        Transcript(text="Hi", language="en", language_probability=0.99)
    )

    assert result.reply == "Hello, what is your name?"
    assert result.language == "en"
    assert not result.booking_complete


def test_update_slot_tool_writes_to_booking_state(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """An ``update_slot`` tool call mutates the manager's booking state."""
    turns = [
        _tool_turn(
            ToolCall(
                id="t1",
                name=TOOL_UPDATE_SLOT,
                arguments={"name": "customer_name", "value": "Anna Kovács"},
            )
        ),
        _text_turn("Thanks, Anna. What's the best phone number for you?"),
    ]
    manager, client = vet_manager_factory(turns=turns)

    result = manager.handle_user_turn(
        Transcript(
            text="My name is Anna Kovács.",
            language="en",
            language_probability=0.99,
        )
    )

    assert manager.state.slots["customer_name"] == "Anna Kovács"
    assert "customer_name" in manager.state.confirmed_slots
    assert result.reply.startswith("Thanks, Anna")
    assert len(client.calls) == 2  # initial call + after-tool call


def test_phone_update_then_confirmation_flow(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Phone slots stay pending until ``confirm_phone(accepted=True)`` runs."""
    turns = [
        _tool_turn(
            ToolCall(
                id="t1",
                name=TOOL_UPDATE_SLOT,
                arguments={"name": "phone", "value": "+36 1 555 0199"},
            )
        ),
        _text_turn("Just to check, that was plus three six, one, five five five..."),
    ]
    manager, _ = vet_manager_factory(turns=turns)

    manager.handle_user_turn(
        Transcript(
            text="My number is plus 36 1 555 0199",
            language="en",
            language_probability=0.99,
        )
    )

    assert manager.state.pending_phone_confirmation is True
    assert "phone" not in manager.state.confirmed_slots

    follow_up = [
        _tool_turn(
            ToolCall(id="t2", name=TOOL_CONFIRM_PHONE, arguments={"accepted": True})
        ),
        _text_turn("Great. What is your pet's name?"),
    ]
    manager.client = _ScriptedClient(follow_up)

    manager.handle_user_turn(
        Transcript(text="Yes, that's right.", language="en", language_probability=0.99)
    )

    assert manager.state.pending_phone_confirmation is False
    assert "phone" in manager.state.confirmed_slots


def test_confirm_phone_with_rejection_clears_value(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Rejecting the readback discards the stored phone number."""
    turns = [
        _tool_turn(
            ToolCall(
                id="t1",
                name=TOOL_UPDATE_SLOT,
                arguments={"name": "phone", "value": "+36 1 555 0199"},
            )
        ),
        _text_turn("Reading back: plus three six..."),
    ]
    manager, _ = vet_manager_factory(turns=turns)
    manager.handle_user_turn(
        Transcript(
            text="my number is...",
            language="en",
            language_probability=0.99,
        )
    )

    follow_up = [
        _tool_turn(
            ToolCall(id="t2", name=TOOL_CONFIRM_PHONE, arguments={"accepted": False})
        ),
        _text_turn("OK, could you say it again?"),
    ]
    manager.client = _ScriptedClient(follow_up)
    manager.handle_user_turn(
        Transcript(text="No, that's wrong.", language="en", language_probability=0.99)
    )

    assert "phone" not in manager.state.slots
    assert manager.state.pending_phone_confirmation is False


def test_ask_kb_uses_supplied_lookup(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Knowledge-base lookups are routed through the supplied callback."""
    captured: list[str] = []

    def lookup(question: str) -> str:
        captured.append(question)
        return "We are open Monday to Friday from 8 to 19."

    turns = [
        _tool_turn(
            ToolCall(
                id="t1",
                name=TOOL_ASK_KB,
                arguments={"question": "When are you open?"},
            )
        ),
        _text_turn("We are open weekdays from 8 to 19."),
    ]
    manager, _ = vet_manager_factory(turns=turns, knowledge_lookup=lookup)

    result = manager.handle_user_turn(
        Transcript(
            text="What are your opening hours?",
            language="en",
            language_probability=0.99,
        )
    )

    assert captured == ["When are you open?"]
    assert result.reply.startswith("We are open weekdays")


def test_ask_kb_returns_stub_when_no_lookup_configured(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Without RAG wired in, the tool returns an unavailability stub."""
    turns = [
        _tool_turn(
            ToolCall(id="t1", name=TOOL_ASK_KB, arguments={"question": "Prices?"})
        ),
        _text_turn("Sorry, I do not have pricing details to hand."),
    ]
    manager, client = vet_manager_factory(turns=turns)

    manager.handle_user_turn(
        Transcript(text="how much?", language="en", language_probability=0.99)
    )

    last_history = client.calls[-1]
    tool_result = last_history[-1]
    assert tool_result["role"] == "user"
    assert "knowledge base not available" in tool_result["content"][0]["content"]


def test_confirm_appointment_invokes_sink_when_state_complete(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Sink is called once when the model finalises a complete booking."""
    sinks: list[BookingState] = []

    def sink(state: BookingState) -> None:
        sinks.append(state)

    turns = [
        _tool_turn(
            ToolCall(id="t1", name=TOOL_CONFIRM_APPOINTMENT, arguments={})
        ),
        _text_turn("All booked, see you Friday!"),
    ]
    manager, _ = vet_manager_factory(turns=turns, on_appointment_confirmed=sink)

    manager.state.set_slot("customer_name", "Anna Kovács")
    manager.state.set_slot("phone", "+36 1 555 0199")
    manager.state.confirm_phone()
    manager.state.set_slot("pet_name", "Bodri")
    manager.state.set_slot("species", "dog")
    manager.state.set_slot("complaint", "limping")
    manager.state.set_slot("time", "Friday at 10:00")

    result = manager.handle_user_turn(
        Transcript(text="yes please book it", language="en", language_probability=0.99)
    )

    assert len(sinks) == 1
    assert result.booking_complete is True
    assert result.reply.startswith("All booked")


def test_confirm_appointment_with_missing_slots_is_rejected(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """An incomplete state must not trigger the appointment sink."""
    sinks: list[BookingState] = []

    turns = [
        _tool_turn(
            ToolCall(id="t1", name=TOOL_CONFIRM_APPOINTMENT, arguments={})
        ),
        _text_turn("Actually, I still need the pet's name."),
    ]
    manager, _ = vet_manager_factory(
        turns=turns, on_appointment_confirmed=lambda s: sinks.append(s)
    )

    result = manager.handle_user_turn(
        Transcript(text="book it", language="en", language_probability=0.99)
    )

    assert sinks == []
    assert result.booking_complete is False


def test_unknown_language_falls_back_to_english_reply(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """A non-EN/HU detection still produces an English reply."""
    manager, _ = vet_manager_factory(turns=[_text_turn("Hello there.")])

    result = manager.handle_user_turn(
        Transcript(text="bonjour", language="fr", language_probability=0.7)
    )

    assert result.language == "en"
    assert result.reply == "Hello there."


def test_tool_use_turn_text_is_used_when_followup_is_empty(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Text emitted alongside a tool call is spoken if the next turn is empty."""
    turns = [
        _tool_turn(
            ToolCall(id="t1", name=TOOL_CONFIRM_PHONE, arguments={"accepted": True}),
            text="Great, your phone is confirmed. What is your pet's name?",
        ),
        _text_turn(""),
    ]
    manager, _ = vet_manager_factory(turns=turns)
    manager.state.set_slot("phone", "+36 1 555 0199")

    result = manager.handle_user_turn(
        Transcript(text="Yes, that's right.", language="en", language_probability=0.99)
    )

    assert result.reply == "Great, your phone is confirmed. What is your pet's name?"
    assert "Sorry" not in result.reply


def test_markdown_bullets_and_emphasis_are_stripped_for_speech(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """Markdown bullets and bold/italic markers are removed before TTS."""
    reply = (
        "Here are our services:\n"
        "- **Vaccinations** for dogs\n"
        "- *Dental* cleaning\n"
        "1. Annual checkups\n"
        "Use `Bodri` as the pet name."
    )
    manager, _ = vet_manager_factory(turns=[_text_turn(reply)])

    result = manager.handle_user_turn(
        Transcript(text="What do you offer?", language="en", language_probability=0.99)
    )

    assert "**" not in result.reply
    assert "- " not in result.reply
    assert "`" not in result.reply
    assert "Vaccinations for dogs." in result.reply
    assert "Dental cleaning." in result.reply
    assert "Annual checkups." in result.reply
    assert "Use Bodri as the pet name." in result.reply


def test_tool_loop_iteration_cap_emits_fallback_message(
    vet_manager_factory: _ManagerFactory,
) -> None:
    """If the model never stops calling tools the manager bails out."""
    bad_turn = _tool_turn(
        ToolCall(
            id="loop",
            name=TOOL_UPDATE_SLOT,
            arguments={"name": "customer_name", "value": "Anna"},
        )
    )
    turns = [bad_turn for _ in range(20)]
    manager, _ = vet_manager_factory(turns=turns)

    result = manager.handle_user_turn(
        Transcript(text="hello", language="en", language_probability=0.99)
    )

    assert "Sorry" in result.reply
    assert result.language == "en"
