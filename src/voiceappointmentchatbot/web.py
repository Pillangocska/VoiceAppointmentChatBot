"""FastAPI WebSocket server exposing the booking pipeline to a browser.

A single ``/ws`` WebSocket carries one booking session: the browser sends
push-to-talk audio blobs (any container ffmpeg can decode — WebM/Opus or
WAV typically), and the server runs the existing Whisper / sentiment /
:class:`DialogueManager` / Piper pipeline and streams back transcripts,
the bot's reply, the live booking state, and the synthesised audio as
base64-encoded WAV.

The HTTP surface also serves the built React SPA from
``frontend/dist`` when present, so a single ``vetbot-web`` command
launches the whole app.
"""

from base64 import b64decode, b64encode
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import asyncio
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from voiceappointmentchatbot.asr import Transcript, WhisperTranscriber
from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.booking_writer import (
    SentimentSummary,
    TranscriptTurn,
    appointment_record_from,
    serialise_record,
    write_appointment,
)
from voiceappointmentchatbot.config import AppConfig
from voiceappointmentchatbot.dialogue import DialogueManager, DialogueResult
from voiceappointmentchatbot.domains import Domain, load_domain
from voiceappointmentchatbot.knowledge import KnowledgeBase
from voiceappointmentchatbot.llm import HaikuClient
from voiceappointmentchatbot.sentiment import TextSentimentAnalyzer
from voiceappointmentchatbot.tts import PiperSpeaker


_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_TARGET_SAMPLE_RATE = 16_000


class _Pipeline:
    """Process-wide singletons for the heavy pipeline components.

    Whisper, Piper and the sentiment classifier are expensive to load and
    safe to share across sessions; only the per-session state (booking,
    dialogue history, sentiment samples, transcript turns) is rebuilt on
    each WebSocket connection.

    Attributes:
        config: Application configuration.
        domain: Currently active domain (vet only for now).
    """

    def __init__(self) -> None:
        """Initialise pipeline without loading any models yet."""
        self.config: AppConfig = AppConfig.load()
        self.domain: Domain = load_domain(
            self.config.domain, self.config.domains_dir
        )
        self.transcriber = WhisperTranscriber(self.config.device, self.config.whisper)
        self.sentiment = TextSentimentAnalyzer(
            self.config.device, self.config.sentiment
        )
        self.speaker = PiperSpeaker(self.config.piper)
        self.llm = HaikuClient(self.config.anthropic, self.domain)
        self.knowledge_base = KnowledgeBase(
            self.domain.knowledge_base_path, self.config.knowledge
        )

    def warm_up(self) -> None:
        """Eagerly load every model so the first user turn is snappy."""
        print("[web] warming up whisper")
        self.transcriber.warm_up()
        print("[web] warming up sentiment")
        self.sentiment.warm_up()
        print("[web] warming up knowledge base")
        self.knowledge_base.warm_up()
        print("[web] warming up piper voices")
        self.speaker.warm_up()


def _domain_payload(domain: Domain) -> Dict[str, Any]:
    """Serialise the active domain for the frontend slot panel."""
    return {
        "name": domain.name,
        "display_name": {
            "en": domain.display_name("en"),
            "hu": domain.display_name("hu"),
        },
        "slots": [
            {
                "name": spec.name,
                "type": spec.type,
                "prompt": {
                    "en": spec.prompt_for("en"),
                    "hu": spec.prompt_for("hu"),
                },
            }
            for spec in domain.slots
        ],
    }


def _booking_payload(state: BookingState) -> Dict[str, Any]:
    """Serialise the live booking state for the frontend."""
    return {
        "slots": dict(state.slots),
        "confirmed": sorted(state.confirmed_slots),
        "pending_phone_confirmation": state.pending_phone_confirmation,
        "missing": [spec.name for spec in state.missing_slots()],
        "is_complete": state.is_complete(),
    }


def _decode_audio_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode an arbitrary browser audio blob to mono float32 16 kHz PCM.

    Browsers emit WebM/Opus from ``MediaRecorder`` by default, which is
    not something Python can decode in the stdlib. We shell out to ffmpeg
    once per utterance to get back canonical WAV PCM, then read the
    result with :mod:`wave`. ffmpeg is required at runtime — the error
    message points users at the install step.

    Args:
        audio_bytes: Raw container bytes posted by the browser.

    Returns:
        One-dimensional float32 NumPy array in ``[-1, 1]`` at 16 kHz.

    Raises:
        RuntimeError: If ffmpeg is not on PATH or fails to decode.
    """
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to decode browser audio but was not found "
            "on PATH; install it (e.g. `winget install ffmpeg`) and retry."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "in.bin"
        out_path = Path(tmp_dir) / "out.wav"
        in_path.write_bytes(audio_bytes)
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(in_path),
                "-ac", "1",
                "-ar", str(_TARGET_SAMPLE_RATE),
                "-f", "wav",
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode audio: {result.stderr.decode(errors='replace')}"
            )
        with wave.open(str(out_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise RuntimeError(
            f"unexpected sample width from ffmpeg: {sample_width}"
        )
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def _synthesize_wav(speaker: PiperSpeaker, text: str, language: str) -> Optional[str]:
    """Render ``text`` with Piper and return base64 WAV for the browser.

    Reuses :meth:`PiperSpeaker._ensure_voice` and ``_synthesize`` to get
    the float samples rather than playing them through the local
    speakers; the browser does the playback instead.

    Args:
        speaker: Pipeline-shared Piper wrapper.
        text: Utterance to speak.
        language: ISO 639-1 language code; falls back to ``en``.

    Returns:
        Base64-encoded WAV string, or ``None`` if ``text`` is empty.
    """
    if not text.strip():
        return None
    if language not in speaker.config.voices:
        language = "en"
    voice = speaker._ensure_voice(language)  # noqa: SLF001 - reuse internal
    sample_rate, samples = PiperSpeaker._synthesize(voice, text)  # noqa: SLF001
    if samples.size == 0:
        return None
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return b64encode(buffer.getvalue()).decode("ascii")


def _summarise_sentiments(
    samples: List[Tuple[str, float]],
) -> Optional[SentimentSummary]:
    """Aggregate per-turn sentiment readings (same rule as the CLI)."""
    if not samples:
        return None
    counts: Counter[str] = Counter()
    buckets: Dict[str, List[float]] = defaultdict(list)
    for label, score in samples:
        counts[label] += 1
        buckets[label].append(score)
    top = max(counts.values())
    contenders = [label for label, count in counts.items() if count == top]
    if len(contenders) == 1:
        winner = contenders[0]
    else:
        winner = max(
            contenders,
            key=lambda label: sum(buckets[label]) / len(buckets[label]),
        )
    winning = buckets[winner]
    return SentimentSummary(
        label=winner,
        score=sum(winning) / len(winning),
        samples=len(samples),
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up the pipeline once before serving traffic."""
    pipeline: _Pipeline = app.state.pipeline
    await asyncio.to_thread(pipeline.warm_up)
    print("[web] pipeline ready")
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app with the pipeline attached to ``app.state``.

    Returns:
        Configured :class:`FastAPI` ready for uvicorn.
    """
    pipeline = _Pipeline()
    app = FastAPI(lifespan=_lifespan)
    app.state.pipeline = pipeline

    @app.get("/api/domain")
    async def get_domain() -> JSONResponse:
        """Return the active domain's slot list and display names."""
        return JSONResponse(_domain_payload(pipeline.domain))

    @app.websocket("/ws")
    async def booking_socket(websocket: WebSocket) -> None:
        """Run one booking session over a single WebSocket connection."""
        await websocket.accept()
        await _run_session(websocket, pipeline)

    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIST), html=True),
            name="spa",
        )
    else:
        @app.get("/")
        async def missing_frontend() -> JSONResponse:
            """Friendly hint when the SPA bundle is not built yet."""
            return JSONResponse(
                {
                    "error": "frontend bundle not found",
                    "expected_at": str(_FRONTEND_DIST),
                    "hint": "run `npm install && npm run build` inside `frontend/`",
                },
                status_code=503,
            )

    return app


async def _run_session(websocket: WebSocket, pipeline: _Pipeline) -> None:
    """Handle one booking session over ``websocket`` until it closes.

    Each user utterance arrives as a single JSON text frame carrying
    base64-encoded audio. The server emits multiple JSON events back:
    ``transcript``, ``sentiment``, ``reply`` (with audio + booking
    state), and finally ``saved`` when the LLM finalises the booking.
    """
    domain = pipeline.domain
    state = BookingState(domain=domain)
    sentiment_samples: List[Tuple[str, float]] = []
    transcript_turns: List[TranscriptTurn] = []
    last_language = {"value": "en"}
    saved_path: Dict[str, Optional[Path]] = {"value": None}

    def on_confirmed(final_state: BookingState) -> None:
        record = appointment_record_from(
            state=final_state,
            language=last_language["value"],
            sentiment=_summarise_sentiments(sentiment_samples),
            transcript=transcript_turns,
        )
        path = write_appointment(record, pipeline.config.output_dir)
        saved_path["value"] = path
        print(f"[web] appointment saved to {path}")

    manager = DialogueManager(
        domain=domain,
        client=pipeline.llm,
        knowledge_lookup=pipeline.knowledge_base.query,
        on_appointment_confirmed=on_confirmed,
    )

    await websocket.send_json(
        {
            "type": "session_ready",
            "domain": _domain_payload(domain),
            "booking": _booking_payload(state),
        }
    )

    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind != "utterance":
                await websocket.send_json(
                    {"type": "error", "message": f"unknown message type: {kind!r}"}
                )
                continue

            audio_b64 = message.get("audio_base64", "")
            try:
                audio = await asyncio.to_thread(
                    _decode_audio_to_pcm, b64decode(audio_b64)
                )
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "message": f"audio decode failed: {exc}"}
                )
                continue

            transcript = await asyncio.to_thread(
                pipeline.transcriber.transcribe, audio
            )
            await websocket.send_json(
                {
                    "type": "transcript",
                    "text": transcript.text,
                    "language": transcript.language,
                    "language_probability": transcript.language_probability,
                }
            )

            mood = await asyncio.to_thread(
                pipeline.sentiment.analyze, transcript.text
            )
            if transcript.text.strip():
                sentiment_samples.append((mood.label, mood.score))
                transcript_turns.append(
                    TranscriptTurn(
                        role="user",
                        text=transcript.text,
                        language=transcript.language,
                    )
                )
            await websocket.send_json(
                {
                    "type": "sentiment",
                    "label": mood.label,
                    "score": mood.score,
                }
            )

            result: DialogueResult = await asyncio.to_thread(
                manager.handle_user_turn, transcript
            )
            last_language["value"] = result.language
            transcript_turns.append(
                TranscriptTurn(
                    role="assistant",
                    text=result.reply,
                    language=result.language,
                )
            )

            audio_payload = await asyncio.to_thread(
                _synthesize_wav, pipeline.speaker, result.reply, result.language
            )
            await websocket.send_json(
                {
                    "type": "reply",
                    "text": result.reply,
                    "language": result.language,
                    "audio_base64": audio_payload,
                    "booking": _booking_payload(manager.state),
                    "booking_complete": result.booking_complete,
                }
            )

            if result.booking_complete:
                summary = _summarise_sentiments(sentiment_samples)
                record = appointment_record_from(
                    state=manager.state,
                    language=last_language["value"],
                    sentiment=summary,
                    transcript=transcript_turns,
                )
                await websocket.send_json(
                    {
                        "type": "saved",
                        "path": (
                            str(saved_path["value"]) if saved_path["value"] else None
                        ),
                        "payload": serialise_record(record),
                    }
                )
                break
    except WebSocketDisconnect:
        print("[web] client disconnected")
    except Exception as exc:  # noqa: BLE001
        print(f"[web] session error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    """Run the web server with uvicorn on ``127.0.0.1:8000``."""
    import uvicorn

    uvicorn.run(
        "voiceappointmentchatbot.web:create_app",
        host="127.0.0.1",
        port=8000,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
