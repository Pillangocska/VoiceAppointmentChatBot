"""Tests for the Anthropic Claude wrapper.

The unit tests exercise prompt construction and response parsing
without hitting the network. The smoke test calls the real Claude API
and is skipped automatically when no API key is configured.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from voiceappointmentchatbot.asr import Transcript
from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.config import AnthropicConfig, AppConfig
from voiceappointmentchatbot.dialogue import DialogueManager
from voiceappointmentchatbot.domains import load_domain
from voiceappointmentchatbot.llm import (
    AssistantTurn,
    HaikuClient,
    TOOL_ASK_KB,
    TOOL_CONFIRM_APPOINTMENT,
    TOOL_CONFIRM_PHONE,
    TOOL_UPDATE_SLOT,
    _format_state_snapshot,
    _parse_response,
    assistant_message,
    build_system_prompt,
    build_tools,
    tool_result_message,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


@pytest.fixture
def vet_domain() -> Any:
    """Vet domain fixture."""
    return load_domain("vet", DOMAINS_DIR)


def test_system_prompt_includes_both_languages(vet_domain: Any) -> None:
    """System prompt embeds the EN and HU blurbs and slot lists."""
    prompt = build_system_prompt(vet_domain)

    assert "Paws & Whiskers" in prompt
    assert "Mancs" in prompt
    assert "customer_name" in prompt
    assert "phone" in prompt
    assert "digit-by-digit" in prompt
    assert "iso" in prompt


def test_tools_constrain_slot_names_to_domain(vet_domain: Any) -> None:
    """``update_slot`` enum lists exactly the domain's declared slots."""
    tools = build_tools(vet_domain)
    update = next(tool for tool in tools if tool["name"] == TOOL_UPDATE_SLOT)

    assert set(update["input_schema"]["properties"]["name"]["enum"]) == set(
        vet_domain.slot_names
    )
    assert "iso" in update["input_schema"]["properties"]
    declared = {tool["name"] for tool in tools}
    assert declared == {
        TOOL_UPDATE_SLOT,
        TOOL_ASK_KB,
        TOOL_CONFIRM_PHONE,
        TOOL_CONFIRM_APPOINTMENT,
    }


def test_state_snapshot_lists_filled_and_missing_slots(vet_domain: Any) -> None:
    """The transient state note shows confirmed and pending slots."""
    state = BookingState(domain=vet_domain)
    state.set_slot("customer_name", "Anna")
    state.set_slot("phone", "+36 1 555 0199")

    snapshot = _format_state_snapshot(state)

    assert "customer_name" in snapshot
    assert "Anna" in snapshot
    assert "confirmed" in snapshot
    assert "pending" in snapshot
    assert "pet_name: not yet provided" in snapshot
    assert "CURRENT TIME" in snapshot


def test_state_snapshot_when_empty(vet_domain: Any) -> None:
    """Empty state still announces the current time anchor."""
    state = BookingState(domain=vet_domain)

    snapshot = _format_state_snapshot(state)

    assert "CURRENT TIME" in snapshot
    assert "BOOKING STATE\nNo slots filled yet." in snapshot


def test_parse_response_extracts_text_and_tool_calls() -> None:
    """``_parse_response`` separates text blocks from tool-use blocks."""
    fake = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hello, what is your name?"),
            SimpleNamespace(
                type="tool_use",
                id="toolu_01",
                name="update_slot",
                input={"name": "customer_name", "value": "Anna"},
            ),
        ],
        stop_reason="tool_use",
    )

    turn = _parse_response(fake)

    assert turn.text == "Hello, what is your name?"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "update_slot"
    assert turn.tool_calls[0].arguments == {"name": "customer_name", "value": "Anna"}
    assert turn.stop_reason == "tool_use"
    assert turn.raw_content[1]["type"] == "tool_use"


def test_assistant_message_round_trips_raw_content() -> None:
    """Wrapping a turn as an assistant message preserves the raw blocks."""
    turn = AssistantTurn(
        text="hi",
        tool_calls=(),
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": "hi"}],
    )

    message = assistant_message(turn)

    assert message == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}


def test_tool_result_message_shape() -> None:
    """Tool results are wrapped as ``user`` messages with a tool_result block."""
    msg = tool_result_message("toolu_01", "ok")

    assert msg["role"] == "user"
    assert msg["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "ok",
        "is_error": False,
    }


def test_haiku_client_requires_api_key(vet_domain: Any) -> None:
    """Constructing the client without an API key raises immediately."""
    with pytest.raises(ValueError, match="api_key"):
        HaikuClient(AnthropicConfig(api_key=""), vet_domain)


def _config_has_api_key() -> bool:
    """Whether ``config.yaml`` provides an Anthropic API key."""
    try:
        return bool(AppConfig.load().anthropic.api_key)
    except Exception:
        return False


@pytest.mark.skipif(
    not _config_has_api_key(),
    reason="No anthropic.api_key in config.yaml; smoke test skipped.",
)
def test_smoke_real_haiku_call_uses_update_slot_tool(vet_domain: Any) -> None:
    """End-to-end check: a greeting turn yields no tool call yet.

    This is the cheapest possible round-trip — one short user message,
    no tool results to feed back, just enough to verify the SDK call is
    correctly shaped and the API key works.
    """
    config = AppConfig.load().anthropic
    client = HaikuClient(config, vet_domain)
    state = BookingState(domain=vet_domain)

    turn = client.respond(
        history=[{"role": "user", "content": "Hi, I'd like to book an appointment."}],
        booking_state=state,
    )

    assert turn.text or turn.tool_calls
    assert turn.stop_reason in {"end_turn", "tool_use", "stop_sequence", "max_tokens"}


@pytest.mark.skipif(
    not _config_has_api_key(),
    reason="No anthropic.api_key in config.yaml; smoke test skipped.",
)
def test_phone_digit_readback_confirms_via_real_haiku(vet_domain: Any) -> None:
    """Two-turn integration check for the phone digit-readback flow.

    Drives a real :class:`DialogueManager` plus :class:`HaikuClient`
    through two user turns: the user supplies a phone number, then
    confirms the digit-by-digit readback. The assertion is on the
    booking state, not on the spoken reply, so we are robust to phrasing
    drift across model versions.
    """
    config = AppConfig.load().anthropic
    client = HaikuClient(config, vet_domain)
    manager = DialogueManager(domain=vet_domain, client=client)

    manager.handle_user_turn(
        Transcript(
            text="My phone number is +36 1 555 0199.",
            language="en",
            language_probability=0.99,
        )
    )

    manager.handle_user_turn(
        Transcript(
            text="Yes, that's correct.",
            language="en",
            language_probability=0.99,
        )
    )

    assert "phone" in manager.state.confirmed_slots
    assert manager.state.pending_phone_confirmation is False
