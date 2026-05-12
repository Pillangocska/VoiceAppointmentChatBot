# VoiceAppointmentChatBot

A bilingual (English / Hungarian) voice chatbot that books appointments
with a veterinary practice. The user speaks into the microphone, the bot transcribes the utterance, detects the language and sentiment, and responds with synthesised speech. The dialogue asks for the most important information to be confirmed. User identification (name, email, phone number or other method), time, treatment or service chosen, other extra information requested (e.g. emotional state). Final appointment details are written to a JSON file.

## Setup and run

Requires Python 3.12, [`uv`](https://docs.astral.sh/uv/), Node 18+ and `ffmpeg` on PATH (the server shells out to ffmpeg once per utterance to decode the browser's WebM/Opus audio into 16 kHz PCM for Whisper).

```bash
# CPU-only install
uv sync
# NVIDIA GPU users (~1.2 GB extra wheels)
uv sync --extra cuda
# then edit huggingface.token
cp config.yaml.example config.yaml
# build frontend
cd frontend
pnpm install
pnpm run build
# start the app
cd ..
# serves SPA + WebSocket on http://127.0.0.1:8000
uv run vetbot-web
```

GPU acceleration on NVIDIA cards is opt-in via the `cuda` extra, which
installs the cuBLAS, cuDNN, and NVRTC runtime wheels needed by
CTranslate2. No separate CUDA Toolkit install is required; the wheels
ship the DLLs and `voiceappointmentchatbot.gpu_runtime` registers them
on the Windows DLL search path (and prepends them to `PATH`) before
CTranslate2 loads.

## Development
```bash
# unit tests
uv run --extra dev pytest
# lint
uv run --extra dev ruff check .
# type-check
uv run --extra dev mypy src
```

## Architecture

The system is a classic ASR -> NLU -> TTS cascade wrapped behind a
single FastAPI process. The browser captures push-to-talk audio and
exchanges JSON frames over one WebSocket per session; the server runs
the heavy models as process-wide singletons and rebuilds only the
per-session booking state on each connection.

```mermaid
flowchart LR
    classDef client fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef edge fill:#fff8e1,stroke:#f9a825,color:#5d4037
    classDef pipeline fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef external fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef store fill:#eceff1,stroke:#455a64,color:#263238

    subgraph Client["Browser (React + Vite SPA)"]
        direction TB
        mic([Microphone]):::client
        rec[MediaRecorder<br/>WebM/Opus capture]:::client
        ui[Slot panel · transcript ·<br/>audio playback]:::client
        mic --> rec --> ui
    end

    subgraph Server["FastAPI process (uvicorn, 127.0.0.1:8000)"]
        direction TB
        ws[/"WebSocket /ws<br/>per-session JSON frames"/]:::edge
        ffmpeg["ffmpeg decode<br/>-> 16 kHz mono PCM"]:::edge

        subgraph Pipeline["Per-utterance pipeline (shared singletons)"]
            direction LR
            asr["WhisperTranscriber<br/>faster-whisper · large-v3 / small"]:::pipeline
            sent["TextSentimentAnalyzer<br/>XLM-RoBERTa (EN + HU)"]:::pipeline
            dm["DialogueManager<br/>slot-filling · tool dispatch"]:::pipeline
            tts["PiperSpeaker<br/>en_US-lessac · hu_HU-anna"]:::pipeline
            asr --> sent --> dm --> tts
        end

        kb[("KnowledgeBase<br/>RAG over domains/*.md")]:::store
        booking["BookingState<br/>(per session)"]:::pipeline
    end

    subgraph Cloud["External"]
        direction TB
        haiku["Anthropic API<br/>Claude Haiku 4.5<br/>tool use + prompt cache"]:::external
    end

    out[("output/appointment_&lt;domain&gt;_&lt;ts&gt;.json")]:::store

    rec -- "base64 audio" --> ws
    ws --> ffmpeg --> asr
    tts -- "base64 WAV + reply + state" --> ws
    ws --> ui

    dm <--> haiku
    kb --> dm
    dm <--> booking
    booking -- "on confirm" --> out
```

### Sequence: one user turn end-to-end

A push-to-talk press in the browser produces one `utterance` frame on
the WebSocket. The server emits four events back — `transcript`,
`sentiment`, `reply`, and (on the final turn) `saved` — interleaved
with the heavy work running off the event loop via
`asyncio.to_thread`.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React SPA
    participant WS as FastAPI /ws
    participant FF as ffmpeg
    participant ASR as WhisperTranscriber
    participant SEN as SentimentAnalyzer
    participant DM as DialogueManager
    participant TTS as PiperSpeaker

    User->>UI: hold mic / Enter, speak, release
    UI->>UI: MediaRecorder -> WebM/Opus blob -> base64
    UI->>WS: {type: "utterance", audio_base64}
    WS->>FF: decode container
    FF-->>WS: 16 kHz mono float32 PCM
    WS->>ASR: transcribe(pcm)
    ASR-->>WS: Transcript{text, language}
    WS-->>UI: {type: "transcript", text, language}
    WS->>SEN: analyze(text)
    SEN-->>WS: SentimentResult{label, score}
    WS-->>UI: {type: "sentiment", label, score}
    WS->>DM: handle_user_turn(transcript)
    Note over DM: tool-use loop (see next diagram)
    DM-->>WS: DialogueResult{reply, language, booking_complete}
    WS->>TTS: synthesize(reply, language)
    TTS-->>WS: WAV bytes -> base64
    WS-->>UI: {type: "reply", text, audio_base64, booking, booking_complete}
    UI-->>User: speak audio + update slot panel
```

### Sequence: dialogue manager tool-use loop

`DialogueManager.handle_user_turn` runs an inner loop against
Claude Haiku. Each call returns either plain text (terminal) or one or
more tool calls; the manager dispatches every call locally, appends
its result to the history, and re-invokes the model up to
`_MAX_TOOL_ITERATIONS` (16) times. `update_slot`, `ask_kb`,
`confirm_phone`, and `confirm_appointment` all flow through this loop.

```mermaid
sequenceDiagram
    autonumber
    participant DM as DialogueManager
    participant LLM as HaikuClient
    participant API as Anthropic API
    participant BS as BookingState
    participant KB as KnowledgeBase

    DM->>DM: history.append(user transcript)
    loop up to 16 iterations
        DM->>LLM: respond(history, booking_state)
        LLM->>API: messages.create (cached system prompt + tools)
        API-->>LLM: AssistantTurn{text, tool_calls, stop_reason}
        LLM-->>DM: AssistantTurn
        DM->>DM: history.append(assistant message)
        alt no tool_calls
            DM-->>DM: strip markdown, return DialogueResult
        else has tool_calls
            loop for each ToolCall
                alt update_slot
                    DM->>BS: set_slot(name, value, iso?)
                    BS-->>DM: stored value (+ phone readback if phone)
                else ask_kb
                    DM->>KB: query(question)
                    KB-->>DM: retrieved markdown chunks
                else confirm_phone
                    DM->>BS: confirm_phone() / reject_phone()
                    BS-->>DM: ack
                else confirm_appointment
                    DM->>BS: is_complete()?
                    alt complete
                        DM->>DM: on_appointment_confirmed(state)
                        Note over DM: sets _last_finalised = True
                    else missing slots
                        BS-->>DM: list of missing slots
                    end
                end
                DM->>DM: history.append(tool_result)
            end
        end
    end
```

### Sequence: appointment finalisation

When the model is satisfied that every slot is filled and confirmed it
issues `confirm_appointment`. The session sink writes the JSON file
under `output/`, the next assistant turn carries the closing utterance,
and the server emits one final `saved` event before closing the
WebSocket.

```mermaid
sequenceDiagram
    autonumber
    participant DM as DialogueManager
    participant BS as BookingState
    participant Sink as on_appointment_confirmed
    participant Writer as booking_writer
    participant FS as output/*.json
    participant WS as FastAPI /ws
    participant UI as React SPA

    DM->>BS: is_complete()
    BS-->>DM: True
    DM->>Sink: on_appointment_confirmed(state)
    Sink->>Writer: appointment_record_from(state, language, sentiment, transcript)
    Writer-->>Sink: AppointmentRecord
    Sink->>FS: write_appointment(record, output_dir)
    FS-->>Sink: Path to appointment_DOMAIN_TS.json
    Note over DM: _last_finalised = True, reply returned this turn
    DM-->>WS: DialogueResult booking_complete=True
    WS-->>UI: reply event
    WS-->>UI: saved event with path and payload
    WS->>WS: break session loop
```

## Decisions

The table below records every architectural decision made during the project.

| Area | Decision | Rationale |
| --- | --- | --- |
| Pipeline shape | Classic cascade (ASR -> NLU -> TTS) | Each stage is debuggable and screenshot-friendly for the lab report; matches the assignment hints. |
| Languages | English and Hungarian from day one | Course requirement and likely competition criterion; covered by a single ASR model. |
| ASR engine | `faster-whisper` (CTranslate2) | Same accuracy as `openai-whisper` with ~4x throughput and lower VRAM, on both GPU and CPU. |
| Whisper model | `large-v3` on CUDA, `small` on CPU | `large-v3` gives the best Hungarian accuracy and fits in 16 GB VRAM; `small` keeps the CPU fallback usable. |
| Whisper compute type | `float16` on CUDA, `int8` on CPU | Default fast path on each device. |
| Language detection | Whisper auto-detect per utterance | One model handles EN and HU; downstream code keys off `transcript.language`. |
| Extra signal | Text-based sentiment (positive / neutral / negative) | Simpler and more reliable than audio emotion across EN+HU; easy to write up. |
| Sentiment model | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual XLM-RoBERTa, trained on three-class sentiment, supports HU. |
| TTS engine | Piper, voices `en_US-lessac-medium` and `hu_HU-anna-medium` | Local, fast, decent native Hungarian voice; no API dependency. |
| Dialogue (week 1) | Echo policy with sentiment acknowledgement | Assignment only requires a simple response in week one; slot-filling state machine planned for week 2. |
| Runtime | CLI with hold-to-record (Enter to start, Enter to stop); web client (`vetbot-web`) added in week 3 as the primary entry point | Hold-to-record gives robust, unambiguous endpointing for a noisy microphone; the web client became primary for the lab demo while the CLI stays for headless testing and as the no-browser fallback. |
| Domain | Veterinary appointments | Chosen by the author. Slot vocabulary will reflect pet name, species, complaint, etc. |
| Device handling | Auto-detect CUDA via ctranslate2, fall back to CPU | faster-whisper drives the GPU through CTranslate2, so we do not need a CUDA-enabled PyTorch wheel in week 1. Avoids the Blackwell / cu128 wheel issue on the RTX 5070 Ti. |
| CUDA runtime | NVIDIA cuBLAS + cuDNN + NVRTC pip wheels via the `cuda` extra | Avoids forcing users to install the ~3 GB system CUDA Toolkit; wheels are vendored into `site-packages/nvidia/<lib>/bin/` and exposed to CTranslate2 by `gpu_runtime.py` patching `PATH` and the DLL search path on import. |
| Local config | `config.yaml` (gitignored) overlaying defaults, `config.yaml.example` committed | Keeps the Hugging Face token and other personal overrides out of version control while letting users tune model choices without editing code. |
| Python version | 3.12 | NeMo / PyTorch / faster-whisper wheels are stable on 3.12; 3.14 is too new in early 2026. |
| Project layout | `src/` layout, package name `voiceappointmentchatbot` | Standard modern Python packaging; prevents accidental imports of the working tree. |
| Build backend | Hatchling via `pyproject.toml` | Default for modern Python projects, already what `uv` ships with. |
| Output schema | Single JSON object per appointment under `output/` | Required deliverable; one file per booking keeps inspection trivial. (Slot-filling lands in week 2.) |
| TTS audio path | Concatenate `AudioChunk.audio_float_array` from Piper's streaming `synthesize`, append 300 ms of trailing silence | Piper 1.4+ no longer writes WAVs; assembling float samples directly avoids a WAV roundtrip. The trailing silence prevents Windows audio drivers from clipping the last word when `sd.wait()` returns before the buffer fully drains. |
| Tokenizer dependency | Pin `sentencepiece` explicitly | XLM-RoBERTa's tokenizer is a SentencePiece BPE model; without it `transformers` silently falls back to TikToken and crashes. |
| LLM (week 2) | Claude Haiku 4.5 | Fast, cheap, strong at tool use, multilingual out of the box; the booking flow needs reliable structured output more than long-context reasoning. |
| Tool use over freeform JSON | Slot updates flow through declared tools | Asking the model to "respond as JSON" is unreliable; explicit tools with JSON-Schema arguments mean Haiku cannot invent fields and the local code can validate every call. |
| Prompt caching | `cache_control: ephemeral` on the system block | The bilingual system prompt is large and identical across turns; cache reads turn it into pennies after the first turn of a session. |
| Bilingual single system prompt | One block with EN and HU sections | Simpler than swapping system prompts on every detected language change, and lets the model follow mid-conversation EN/HU switches without losing booking state. |
| RAG implementation | Inlined sentence-transformers + numpy cosine | A real vector database is overkill at the few-dozen-chunk scale of these knowledge bases; the inlined version is one file with no extra services. |
| Markdown chunking | Split tables row-by-row | The first version split on headings and paragraphs only, and the real-embedder integration test failed: a "how much is dental cleaning?" query retrieved the surrounding paragraph instead of the price-table row. Splitting tables row-by-row fixes that and is the most interesting decision in week 2. |
| Domain registry | YAML + sibling Markdown pair under `domains/` | Adding a new business domain is two files (slots + KB) with no Python changes; keeps the project demo-friendly and extensible without new abstractions. |
| Output JSON shape | Promote `customer.name` and `customer.phone`, leave the rest flat under `slots` | Every domain carries those two fields and graders skim the file by eye; keeping the rest flat avoids per-domain nesting churn. |
| Phone digit-readback | Render stored phone as EN/HU digit words and require `confirm_phone` | Whisper mishears digits often (especially in HU: "kettő" / "hét"); reading back digit-by-digit before persisting is cheap insurance against a saved booking we can't actually call back. |
| Sentiment aggregation | Majority label across all scored user turns, ties broken by mean confidence | Single-utterance sentiment is noisy — one short "yes" turn often flips the label; averaging across the whole booking is more representative of the customer's actual mood. |
| Web frontend | Vite + React 18 + TypeScript SPA built to `frontend/dist`, mounted by FastAPI's `StaticFiles` | A single `vetbot-web` command serves both UI and API; React+TS keeps the slot panel and transcript state manageable without pulling in a heavier framework. |
| Web server | FastAPI + uvicorn on `127.0.0.1:8000`, single `factory=True` app | FastAPI gives the WebSocket route, a lifespan hook for warming up models, and `StaticFiles` mounting in one stack; bound to loopback because this is a single-user lab demo, not a deployed service. |
| Browser ↔ server transport | One `/ws` WebSocket per booking session, JSON frames with base64 audio | Natural fit for a stateful per-session bidirectional channel — the server emits `transcript`, `sentiment`, `reply`, and `saved` as separate events per turn. Not benchmarked against SSE / chunked HTTP; WebSocket was the default and stuck. |
| Browser audio capture | `MediaRecorder` with the browser's native mime type (WebM/Opus on Chrome, WAV on Safari), decoded server-side with ffmpeg | Re-encoding in the browser is fragile across vendors; letting `MediaRecorder` pick its native format and shelling out to ffmpeg once per utterance to get 16 kHz mono PCM is robust and reuses the install step we already require for the CLI. |
| TTS delivery to the browser | Synthesise Piper output server-side, return base64 WAV per `reply` event, play through a hidden `<audio>` element | Avoids shipping a Piper port (or its voice files) to the browser; WAV is universally playable so the client needs no decoder, and reusing `PiperSpeaker._synthesize` keeps the CLI and web paths producing identical audio. |
| Push-to-talk UX (web) | Hold the on-screen mic button or the Enter key; release to send | Mirrors the CLI's hold-to-record semantics so the original endpointing rationale carries over unchanged, and gives keyboard users parity with the pointer affordance without inventing a separate "stop" gesture. |
| Pipeline lifetime | Whisper / Piper / sentiment / KB loaded once at FastAPI startup via the lifespan hook; only `BookingState`, dialogue history, sentiment samples and transcript turns rebuilt per WebSocket | Model loads dominate startup cost and the heavy components are stateless across sessions; per-connection state stays small, so warm-loading once and rebuilding the session struct on `accept()` is both cheap and keeps concurrent sessions isolated. |

## Project layout

```
.
├── src/voiceappointmentchatbot/
│   ├── __init__.py
│   ├── config.py             # AppConfig, YAML loader, device detection, model selection
│   ├── gpu_runtime.py        # Registers vendored NVIDIA DLLs on the Windows search path
│   ├── audio_io.py           # HoldToRecord microphone capture
│   ├── asr.py                # WhisperTranscriber + Transcript dataclass
│   ├── sentiment.py          # TextSentimentAnalyzer + SentimentResult
│   ├── tts.py                # PiperSpeaker (lazy multi-voice, auto-download, silence pad)
│   ├── domains.py            # Domain + SlotSpec loader for domains/*.yaml
│   ├── booking.py            # BookingState, slot validation, phone digit readback
│   ├── booking_writer.py     # AppointmentRecord, SentimentSummary, JSON writer
│   ├── knowledge.py          # KnowledgeBase: markdown chunking + sentence-transformer RAG
│   ├── llm.py                # HaikuClient, system prompt + tool declarations
│   ├── dialogue.py           # Slot-filling DialogueManager + tool dispatch loop
│   ├── main.py               # CLI entry point: `vetbot`
│   └── web.py                # FastAPI app + WebSocket server: `vetbot-web`
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── pnpm-workspace.yaml
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx           # Main React component (record / transcript / status UI)
│       ├── App.css
│       ├── api.ts            # WebSocket client wrapper
│       ├── recorder.ts       # MediaRecorder helper (WebM/Opus capture)
│       ├── types.ts          # Shared message types between FE and BE
│       ├── main.tsx          # React entry point
│       └── index.css
├── domains/
│   ├── vet.yaml / vet.md
│   └── hairdresser.yaml / hairdresser.md
├── tests/
│   ├── __init__.py
│   ├── test_asr.py
│   ├── test_dialogue.py
│   ├── test_domains.py
│   ├── test_booking.py
│   ├── test_booking_writer.py
│   ├── test_knowledge.py
│   ├── test_llm.py
│   └── test_main.py
├── models/piper/        # Piper voice files (gitignored, auto-populated on first run)
├── output/              # Appointment JSON files (gitignored)
├── config.yaml.example  # template (committed)
├── config.yaml          # local overrides + HF token (gitignored)
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```
