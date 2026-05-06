"""Slot-filling dialogue manager driven by Claude Haiku.

The manager is the heart of the week-2 pipeline. It owns the booking
state and the conversation history, calls the LLM each turn, executes
the tool calls the model issues against local code (booking state
updates, knowledge-base lookups, phone confirmation, appointment
finalisation), and feeds the results back until the model replies with
plain text the user can hear.

The bot's spoken language is always the language of the latest user
turn so Whisper-detected EN/HU switches propagate end-to-end through
the rest of the pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
import re

from voiceappointmentchatbot.asr import Transcript
from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.domains import Domain
from voiceappointmentchatbot.llm import (
    AssistantTurn,
    TOOL_ASK_KB,
    TOOL_CONFIRM_APPOINTMENT,
    TOOL_CONFIRM_PHONE,
    TOOL_UPDATE_SLOT,
    ToolCall,
    assistant_message,
    tool_result_message,
)


_MAX_TOOL_ITERATIONS = 16

_BULLET_LINE_RE = re.compile(r"^[ \t]*(?:[-*•‣◦]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}\s+")
_BOLD_ITALIC_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_INLINE_CODE_RE = re.compile(r"`+([^`]+)`+")
_WHITESPACE_RE = re.compile(r"[ \t]+")


class _LLMClient(Protocol):
    """Subset of :class:`llm.HaikuClient` the manager depends on."""

    def respond(
        self,
        history: list,
        booking_state: BookingState,
    ) -> AssistantTurn:
        ...


KnowledgeLookup = Callable[[str], str]
"""Callable that takes a question and returns retrieved KB context."""

AppointmentSink = Callable[[BookingState], None]
"""Callable invoked when the model issues ``confirm_appointment``."""


@dataclass
class DialogueResult:
    """Outcome of processing one user utterance.

    Attributes:
        reply: Text the bot will speak.
        language: ISO 639-1 code the reply should be voiced in.
        booking_complete: ``True`` when the model has just finalised the
            appointment in this turn. Main loop uses this to stop after
            the closing utterance is spoken.
    """

    reply: str
    language: str
    booking_complete: bool = False


@dataclass
class DialogueManager:
    """Runs the slot-filling conversation for a single booking session.

    Attributes:
        domain: Active business domain (slots, blurb, KB path).
        client: LLM wrapper used to generate replies.
        knowledge_lookup: Optional RAG callback. When ``None`` the
            ``ask_kb`` tool returns a placeholder message so the rest of
            the pipeline still works without retrieval wired in.
        on_appointment_confirmed: Optional sink invoked once when the
            model calls ``confirm_appointment`` with a complete state.
    """

    domain: Domain
    client: _LLMClient
    knowledge_lookup: Optional[KnowledgeLookup] = None
    on_appointment_confirmed: Optional[AppointmentSink] = None
    state: BookingState = field(init=False)
    history: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _last_user_language: str = field(init=False, default="en")
    _last_finalised: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialise the booking state from the provided domain."""
        self.state = BookingState(domain=self.domain)

    def handle_user_turn(self, transcript: Transcript) -> DialogueResult:
        """Process one user utterance and return the bot's spoken reply.

        Args:
            transcript: Whisper transcript for the user's latest utterance.

        Returns:
            :class:`DialogueResult` carrying the reply text, the language
            the reply should be voiced in, and a flag set on the turn the
            model finalises the appointment.
        """
        self._last_user_language = self._reply_language(transcript.language)
        if not transcript.text.strip():
            return DialogueResult(
                reply=self._silence_message(self._last_user_language),
                language=self._last_user_language,
                booking_complete=False,
            )

        self._last_finalised = False
        self.history.append({"role": "user", "content": transcript.text})

        buffered_text: str = ""
        for iteration in range(_MAX_TOOL_ITERATIONS):
            turn = self.client.respond(history=self.history, booking_state=self.state)
            self.history.append(assistant_message(turn))
            tool_names = [call.name for call in turn.tool_calls]
            print(
                f"[dialogue] iter={iteration + 1}/{_MAX_TOOL_ITERATIONS} "
                f"stop={turn.stop_reason} tools={tool_names} "
                f"text_len={len(turn.text)}"
            )

            if turn.text:
                buffered_text = turn.text

            if not turn.tool_calls:
                if turn.text:
                    reply = turn.text
                elif buffered_text:
                    print(
                        "[dialogue] empty text on final turn; "
                        "using buffered text from tool-use turn"
                    )
                    reply = buffered_text
                else:
                    print(
                        "[dialogue] empty text and no tool calls; "
                        "using fallback message"
                    )
                    reply = self._fallback_message(self._last_user_language)
                return DialogueResult(
                    reply=_strip_markdown_for_speech(reply),
                    language=self._last_user_language,
                    booking_complete=self._last_finalised,
                )

            for call in turn.tool_calls:
                self._execute_tool(call)

        print(
            f"[dialogue] tool loop exceeded {_MAX_TOOL_ITERATIONS} "
            f"iterations; using fallback message"
        )
        fallback = buffered_text or self._fallback_message(self._last_user_language)
        return DialogueResult(
            reply=_strip_markdown_for_speech(fallback),
            language=self._last_user_language,
            booking_complete=False,
        )

    def _execute_tool(self, call: ToolCall) -> None:
        """Run a tool call locally and append its result to history."""
        try:
            content = self._dispatch(call)
            error = False
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            content = f"error: {exc}"
            error = True
        self.history.append(tool_result_message(call.id, content, is_error=error))

    def _dispatch(self, call: ToolCall) -> str:
        """Dispatch a single tool call and return its serialised result."""
        if call.name == TOOL_UPDATE_SLOT:
            return self._tool_update_slot(call.arguments)
        if call.name == TOOL_ASK_KB:
            return self._tool_ask_kb(call.arguments)
        if call.name == TOOL_CONFIRM_PHONE:
            return self._tool_confirm_phone(call.arguments)
        if call.name == TOOL_CONFIRM_APPOINTMENT:
            return self._tool_confirm_appointment()
        raise ValueError(f"unknown tool: {call.name!r}")

    def _tool_update_slot(self, arguments: Dict[str, Any]) -> str:
        """Apply an ``update_slot`` call to the booking state."""
        name = arguments.get("name")
        value = arguments.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("update_slot requires 'name' and 'value' strings")
        self.state.set_slot(name, value)
        spec = self.state.domain.slot(name)
        stored = self.state.slots[name]
        if spec.type == "phone":
            readback = self.state.phone_readback(self._last_user_language) or ""
            return (
                f"stored {name}={stored!r}; readback ({self._last_user_language}): "
                f"{readback}; awaiting confirm_phone after the user responds."
            )
        return f"stored {name}={stored!r}"

    def _tool_ask_kb(self, arguments: Dict[str, Any]) -> str:
        """Run a knowledge-base lookup; falls back to a stub when disabled."""
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("ask_kb requires a non-empty 'question' string")
        if self.knowledge_lookup is None:
            return (
                "knowledge base not available in this build; answer from "
                "general knowledge or apologise that you cannot answer."
            )
        return self.knowledge_lookup(question)

    def _tool_confirm_phone(self, arguments: Dict[str, Any]) -> str:
        """Apply the user's response to the phone digit readback."""
        accepted = arguments.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("confirm_phone requires a boolean 'accepted'")
        if accepted:
            self.state.confirm_phone()
            return "phone confirmed"
        self.state.reject_phone()
        return "phone rejected; ask the user to repeat the number"

    def _tool_confirm_appointment(self) -> str:
        """Finalise the booking when the state is complete."""
        if not self.state.is_complete():
            missing = ", ".join(spec.name for spec in self.state.missing_slots())
            return f"cannot confirm: missing or unconfirmed slots: {missing}"
        if self.on_appointment_confirmed is not None:
            self.on_appointment_confirmed(self.state)
        self._last_finalised = True
        return "appointment confirmed and saved"

    def _reply_language(self, detected: str) -> str:
        """Pick the reply language from the latest user-turn detection."""
        return detected if detected in ("en", "hu") else "en"

    @staticmethod
    def _silence_message(language: str) -> str:
        """Spoken reply for an empty utterance."""
        if language == "hu":
            return "Nem hallottam semmit. Megismételnéd, kérlek?"
        return "I did not hear anything. Could you repeat that, please?"

    @staticmethod
    def _fallback_message(language: str) -> str:
        """Spoken reply when the tool loop exceeds the iteration cap."""
        if language == "hu":
            return "Bocsánat, valami elakadt. Kezdjük újra a foglalást?"
        return "Sorry, something got stuck on my end. Shall we start the booking over?"


def _strip_markdown_for_speech(text: str) -> str:
    """Flatten markdown formatting into a clean spoken string.

    Removes bullets, numbered-list markers, headings, emphasis markers
    (``**bold**``, ``*italic*``, ``_underline_``), and inline code
    backticks. Each list item is terminated with a period so consecutive
    items do not slur together when synthesised.

    Args:
        text: Raw assistant reply, potentially containing markdown.

    Returns:
        A speech-friendly string with no markdown syntax characters.
    """
    if not text:
        return text

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = _HEADING_RE.sub("", raw_line)
        is_bullet = bool(_BULLET_LINE_RE.match(line))
        line = _BULLET_LINE_RE.sub("", line)
        line = _INLINE_CODE_RE.sub(r"\1", line)
        line = _BOLD_ITALIC_RE.sub(r"\2", line)
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            continue
        if is_bullet and line[-1] not in ".!?…":
            line = f"{line}."
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)
