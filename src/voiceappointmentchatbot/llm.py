"""Anthropic Claude wrapper used by the dialogue manager.

Encapsulates everything that talks to the Anthropic API:

* Building a bilingual system prompt from the active domain.
* Declaring the booking tools (``update_slot``, ``ask_kb``,
  ``confirm_phone``, ``confirm_appointment``) so the model can drive
  the booking state machine instead of free-texting JSON.
* Calling ``messages.create`` with prompt caching on the system block
  so we only pay full-price tokens for it on the first turn of each
  session.

The dialogue manager calls :meth:`HaikuClient.respond` once per user
turn and receives a structured :class:`AssistantTurn` it can apply to
the booking state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.config import AnthropicConfig
from voiceappointmentchatbot.domains import Domain


TOOL_UPDATE_SLOT = "update_slot"
TOOL_ASK_KB = "ask_kb"
TOOL_CONFIRM_PHONE = "confirm_phone"
TOOL_CONFIRM_APPOINTMENT = "confirm_appointment"


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation issued by the model in a single turn.

    Attributes:
        id: Anthropic-assigned tool-use identifier; echoed back when the
            manager submits the tool result.
        name: Tool name (one of the ``TOOL_*`` constants).
        arguments: Decoded arguments as a plain dict.
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """Result of a single ``messages.create`` call.

    Attributes:
        text: Concatenated ``text`` blocks from the assistant message,
            ready to be spoken or printed.
        tool_calls: Tool invocations the manager must execute and feed
            back to the model on the next call.
        stop_reason: Raw stop reason from the SDK, useful for logging.
        raw_content: Full ``content`` list from the assistant message,
            preserved verbatim so the manager can pass it back unmodified
            on the follow-up call (Anthropic requires this round-trip).
    """

    text: str
    tool_calls: Tuple[ToolCall, ...]
    stop_reason: Optional[str]
    raw_content: List[Dict[str, Any]] = field(default_factory=list)


def build_system_prompt(domain: Domain) -> str:
    """Compose the bilingual system prompt for ``domain``.

    The single block covers both languages so we can rely on the model
    to follow the user's language turn by turn rather than swapping
    prompts. The prompt also lists the slot names so the model knows
    which arguments to pass to ``update_slot``.

    Args:
        domain: Active business domain.

    Returns:
        The system prompt as a single string.
    """
    slot_lines_en = []
    slot_lines_hu = []
    for spec in domain.slots:
        if spec.type == "phone":
            suffix = " (must be confirmed digit-by-digit)"
            suffix_hu = " (számjegyenként meg kell erősíteni)"
        elif spec.type == "datetime":
            suffix = " (include both a date and a clock time)"
            suffix_hu = " (dátumot és pontos időt is meg kell adni)"
        else:
            suffix = ""
            suffix_hu = ""
        slot_lines_en.append(f"  - {spec.name}: {spec.prompt_for('en')}{suffix}")
        slot_lines_hu.append(f"  - {spec.name}: {spec.prompt_for('hu')}{suffix_hu}")
    slots_block_en = "\n".join(slot_lines_en)
    slots_block_hu = "\n".join(slot_lines_hu)

    return (
        "ROLE\n"
        f"{domain.blurb('en').strip()}\n\n"
        f"{domain.blurb('hu').strip()}\n\n"
        "LANGUAGE\n"
        "Always reply in the same language the user just used. The user "
        "may switch between English and Hungarian mid-conversation; "
        "follow them. Keep replies short — one or two sentences is "
        "ideal because they are spoken aloud.\n\n"
        "BOOKING SLOTS\n"
        "You must collect the following slots before the booking is "
        "complete. Call the `update_slot` tool exactly once per slot "
        "value the user provides. Do not invent or assume values. Ask "
        "for one slot at a time when slots are still missing.\n\n"
        "PARALLEL TOOL CALLS\n"
        "If the user provides multiple slot values in a single utterance "
        "(for example, name, pet name, date, and time all at once), emit "
        "all the corresponding `update_slot` calls in the *same* response, "
        "as parallel tool_use blocks. Do not spread them across multiple "
        "turns. After the tool results come back, produce your spoken "
        "reply that asks for the next missing slot.\n\n"
        f"English descriptions:\n{slots_block_en}\n\n"
        f"Hungarian descriptions:\n{slots_block_hu}\n\n"
        "TOOLS\n"
        "- `update_slot(name, value, iso?)`: Store a slot value. `name` "
        "must be one of the slot identifiers above; `value` is the raw "
        "string the user said, in the original language. For datetime "
        "slots you must ALSO pass `iso`: the absolute timestamp in "
        "`YYYY-MM-DDTHH:MM` form with a `+HH:MM` / `-HH:MM` offset, "
        "resolved against the CURRENT TIME block. Resolve relative "
        "phrases like 'holnap', 'tomorrow', 'next Friday' yourself "
        "using that anchor. Omit `iso` for non-datetime slots.\n"
        "- `ask_kb(question)`: Search the practice's knowledge base "
        "(prices, hours, services) when the user asks a factual "
        "question you do not already know. Use the answer to ground "
        "your reply; cite numbers verbatim.\n"
        "- `confirm_phone(accepted)`: Call **only** after the user "
        "responded to a digit-by-digit readback of their phone number. "
        "`accepted` is `true` if they confirmed the number is correct, "
        "`false` if they want to re-enter it.\n"
        "- `confirm_appointment()`: Call when every slot is filled and "
        "you have summarised the booking back to the user and they "
        "agreed. This finalises the appointment and writes it to disk.\n\n"
        "PHONE NUMBERS\n"
        "When the user gives a phone number, store it with `update_slot` "
        "and then in your spoken reply read it back digit by digit "
        "(e.g. 'plus three six, one, five five five, ...') and ask "
        "whether it is correct. Wait for their answer, then call "
        "`confirm_phone` with `accepted` set accordingly.\n\n"
        "STYLE\n"
        "Friendly, professional, concise. Do not list every slot at "
        "once — collect them naturally, one question per turn. Never "
        "output JSON to the user; the tools handle structure."
    )


def build_tools(domain: Domain) -> List[Dict[str, Any]]:
    """Return the JSON-Schema tool declarations for the given domain.

    The ``update_slot`` schema constrains ``name`` to the domain's
    declared slot identifiers so the model cannot invent fields.

    Args:
        domain: Active business domain.

    Returns:
        List of tool dictionaries in the format Anthropic expects.
    """
    slot_names = list(domain.slot_names)
    return [
        {
            "name": TOOL_UPDATE_SLOT,
            "description": (
                "Store one slot value extracted from the user's last "
                "message. Call once per slot. Do not call with values "
                "the user has not actually given. For datetime slots "
                "also pass `iso` with the absolute timestamp resolved "
                "against CURRENT TIME."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": slot_names,
                        "description": "Identifier of the slot to update.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Raw user-provided value for the slot.",
                    },
                    "iso": {
                        "type": "string",
                        "description": (
                            "Absolute timestamp for datetime slots in "
                            "`YYYY-MM-DDTHH:MM+HH:MM` form (e.g. "
                            "`2026-05-12T13:00+02:00`). Required when "
                            "the slot is a datetime slot; omit otherwise."
                        ),
                    },
                },
                "required": ["name", "value"],
            },
        },
        {
            "name": TOOL_ASK_KB,
            "description": (
                "Ask the practice's knowledge base a factual question "
                "(opening hours, prices, services, policies). Returns "
                "short retrieved passages."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's factual question, paraphrased if helpful.",
                    },
                },
                "required": ["question"],
            },
        },
        {
            "name": TOOL_CONFIRM_PHONE,
            "description": (
                "Confirm or reject the previously stored phone number "
                "after the digit-by-digit readback."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "accepted": {
                        "type": "boolean",
                        "description": "True if the user confirmed the readback.",
                    },
                },
                "required": ["accepted"],
            },
        },
        {
            "name": TOOL_CONFIRM_APPOINTMENT,
            "description": (
                "Finalise the appointment once every slot is filled and "
                "the user has agreed to the summary. Writes the booking "
                "to disk."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


class HaikuClient:
    """Thin wrapper around ``anthropic.Anthropic`` for the dialogue loop.

    The client is configured once per session and reused across turns.
    Prompt caching is enabled on the system prompt so its tokens are
    billed at the cheaper cache-read rate after the first turn.

    Attributes:
        config: Anthropic API settings.
        domain: Active business domain.
    """

    def __init__(self, config: AnthropicConfig, domain: Domain) -> None:
        """Initialise the client; the SDK is loaded lazily on first call."""
        if not config.api_key:
            raise ValueError(
                "AnthropicConfig.api_key is empty; set anthropic.api_key in config.yaml"
            )
        self.config = config
        self.domain = domain
        self._client: Optional[Any] = None
        self._system_prompt = build_system_prompt(domain)
        self._tools = build_tools(domain)

    def _ensure_client(self) -> Any:
        """Instantiate the SDK client on first use."""
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def respond(
        self,
        history: Sequence[Mapping[str, Any]],
        booking_state: BookingState,
    ) -> AssistantTurn:
        """Send the conversation history to Claude and return one turn.

        Args:
            history: Prior messages in Anthropic format. Each entry must
                have ``role`` (``user`` or ``assistant``) and ``content``
                (string or list of content blocks). Tool-result messages
                are sent as ``user`` role with a ``tool_result`` block.
            booking_state: Current booking state. Its slot snapshot is
                appended to the system prompt as a transient note so the
                model knows what is already filled without us re-sending
                full history each turn.

        Returns:
            Parsed :class:`AssistantTurn`.
        """
        client = self._ensure_client()
        system_blocks = [
            {
                "type": "text",
                "text": self._system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": _format_state_snapshot(booking_state),
            },
        ]

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_blocks,
            tools=self._tools,
            messages=list(history),
        )
        return _parse_response(response)


def _format_state_snapshot(state: BookingState) -> str:
    """Render the current booking state and time anchor as a per-turn note."""
    now_iso = state.time_anchor.isoformat(timespec="minutes")
    weekday = state.time_anchor.strftime("%A")
    header = (
        "CURRENT TIME\n"
        f"{now_iso} ({weekday}). Use this when resolving relative dates "
        f"like 'holnap' or 'tomorrow' for the `iso` argument of "
        f"`update_slot`."
    )
    if not state.slots:
        body = "BOOKING STATE\nNo slots filled yet."
    else:
        lines = ["BOOKING STATE"]
        for spec in state.domain.slots:
            if spec.name in state.slots:
                confirmed = (
                    "confirmed" if spec.name in state.confirmed_slots else "pending"
                )
                lines.append(
                    f"- {spec.name} = {state.slots[spec.name]!r} ({confirmed})"
                )
            else:
                lines.append(f"- {spec.name}: not yet provided")
        if state.pending_phone_confirmation:
            lines.append("Phone is awaiting digit-readback confirmation.")
        body = "\n".join(lines)
    return f"{header}\n\n{body}"


def _parse_response(response: Any) -> AssistantTurn:
    """Convert an Anthropic ``Message`` into an :class:`AssistantTurn`."""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    raw_content: List[Dict[str, Any]] = []

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
            raw_content.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            arguments = dict(block.input) if isinstance(block.input, Mapping) else {}
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=arguments)
            )
            raw_content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": arguments,
                }
            )

    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        stop_reason=getattr(response, "stop_reason", None),
        raw_content=raw_content,
    )


def tool_result_message(tool_use_id: str, content: str, is_error: bool = False) -> Dict[str, Any]:
    """Build the ``user`` message that returns a tool result to the model.

    Args:
        tool_use_id: ``id`` from the originating :class:`ToolCall`.
        content: Stringified result the model should see.
        is_error: Whether the tool execution failed.

    Returns:
        A message dictionary suitable for appending to history.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            }
        ],
    }


def assistant_message(turn: AssistantTurn) -> Dict[str, Any]:
    """Wrap an :class:`AssistantTurn` as an assistant history message."""
    return {"role": "assistant", "content": list(turn.raw_content)}
