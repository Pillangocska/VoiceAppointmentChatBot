# VoiceAppointmentChatBot

A bilingual (English / Hungarian) voice chatbot that books appointments
with a veterinary practice. Built as a four-week university project.

The user speaks into the microphone, the bot transcribes the utterance,
detects the language and sentiment, and responds with synthesised speech.
Final appointment details are written to a JSON file.

## Architecture

The pipeline is a classic cascade so each stage is observable, testable,
and swappable. Audio flows left to right; the JSON output is produced by
the dialogue manager once all required slots are filled.

```
mic -> audio_io -> asr (Whisper) -> sentiment -> dialogue -> tts (Piper) -> speakers
                                                       |
                                                       +-> appointment.json
```

## Decisions

The table below records every architectural decision made during week-one
brainstorming and why it was chosen. Each row is durable: when a decision
changes in a later week, update it here together with the reason.

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
| Runtime | CLI with hold-to-record (Enter to start, Enter to stop) | Robust and unambiguous endpointing for a noisy microphone in week one; web/custom frontend planned later. |
| Domain | Veterinary appointments | Chosen by the author. Slot vocabulary will reflect pet name, species, complaint, etc. |
| Device handling | Auto-detect CUDA via ctranslate2, fall back to CPU | faster-whisper drives the GPU through CTranslate2, so we do not need a CUDA-enabled PyTorch wheel in week 1. Avoids the Blackwell / cu128 wheel issue on the RTX 5070 Ti. |
| CUDA runtime | NVIDIA cuBLAS + cuDNN + NVRTC pip wheels via the `cuda` extra | Avoids forcing users to install the ~3 GB system CUDA Toolkit; wheels are vendored into `site-packages/nvidia/<lib>/bin/` and exposed to CTranslate2 by `gpu_runtime.py` patching `PATH` and the DLL search path on import. |
| Local config | `config.yaml` (gitignored) overlaying defaults, `config.yaml.example` committed | Keeps the Hugging Face token and other personal overrides out of version control while letting users tune model choices without editing code. |
| Python version | 3.12 | NeMo / PyTorch / faster-whisper wheels are stable on 3.12; 3.14 is too new in early 2026. |
| Project layout | `src/` layout, package name `voiceappointmentchatbot` | Standard modern Python packaging; prevents accidental imports of the working tree. |
| Build backend | Hatchling via `pyproject.toml` | Default for modern Python projects, already what `uv` ships with. |
| Output schema | Single JSON object per appointment under `output/` | Required deliverable; one file per booking keeps inspection trivial. |

## Project layout

```
.
├── src/voiceappointmentchatbot/
│   ├── __init__.py
│   ├── config.py        # AppConfig, device detection, model selection
│   ├── audio_io.py      # HoldToRecord microphone capture
│   ├── asr.py           # WhisperTranscriber + Transcript dataclass
│   ├── sentiment.py     # TextSentimentAnalyzer + SentimentResult
│   ├── tts.py           # PiperSpeaker (lazy multi-voice)
│   ├── dialogue.py      # Week-1 echo policy
│   └── main.py          # CLI entry point: `vetbot`
├── scripts/
│   └── download_piper_voices.py
├── tests/
│   └── test_dialogue.py
├── models/piper/        # Piper voice files (gitignored, populated by script)
├── output/              # Appointment JSON files (gitignored)
├── docs/                # Lab report assets and screenshots
├── config.yaml.example  # template (committed)
├── config.yaml          # local overrides + HF token (gitignored)
├── pyproject.toml
└── README.md
```

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                  # CPU-only install
uv sync --extra cuda                     # NVIDIA GPU users (~1.2 GB extra wheels)
uv run python scripts/download_piper_voices.py
cp config.yaml.example config.yaml   # then edit huggingface.token
uv run python scripts/prefetch_models.py   # one-time, shows progress
```

The prefetch step is optional but recommended: the first `vetbot` run
otherwise downloads the Whisper model (~1.5 GB for `large-v3`) without
visible progress while the dialogue loop appears to hang.

GPU acceleration on NVIDIA cards is opt-in via the `cuda` extra, which
installs the cuBLAS, cuDNN, and NVRTC runtime wheels needed by
CTranslate2. No separate CUDA Toolkit install is required; the wheels
ship the DLLs and `voiceappointmentchatbot.gpu_runtime` registers them
on the Windows DLL search path before CTranslate2 loads.

Windows users: enabling Developer Mode (Settings -> Privacy & Security
-> For Developers) lets the Hugging Face cache use symlinks, which roughly
halves the disk space used by downloaded models.

## Running

```bash
uv run vetbot
```

Press Enter to start recording, speak, press Enter again to stop. The
bot transcribes, classifies sentiment, prints both, and reads back the
echo reply through the speakers. Ctrl+C to exit.

## Development

```bash
uv run pytest         # unit tests
uv run ruff check .   # lint
uv run mypy src       # type-check
```

## Roadmap

- **Week 1 (current):** ASR + sentiment + echo + TTS, CLI loop.
- **Week 2:** Slot-filling dialogue manager, JSON appointment output, web frontend.
- **Week 3:** LLM-driven dialogue (optional upgrade), richer prompts, error recovery.
- **Week 4:** Polish, lab report, demo recording.
