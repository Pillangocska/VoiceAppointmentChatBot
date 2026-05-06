"""Command-line entry point for the week-two booking chatbot.

Wires together microphone capture, Whisper transcription, sentiment
analysis, the slot-filling :class:`DialogueManager` (driven by Claude
Haiku), and Piper synthesis into a single push-to-talk loop. The bot
exits cleanly when the user sends EOF, hits Ctrl+C, or the dialogue
manager finalises a complete appointment.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from voiceappointmentchatbot.asr import WhisperTranscriber
from voiceappointmentchatbot.audio_io import HoldToRecord
from voiceappointmentchatbot.booking import BookingState
from voiceappointmentchatbot.booking_writer import (
    SentimentSummary,
    TranscriptTurn,
    appointment_record_from,
    write_appointment,
)
from voiceappointmentchatbot.config import AppConfig
from voiceappointmentchatbot.dialogue import DialogueManager
from voiceappointmentchatbot.domains import load_domain
from voiceappointmentchatbot.knowledge import KnowledgeBase
from voiceappointmentchatbot.llm import HaikuClient
from voiceappointmentchatbot.sentiment import TextSentimentAnalyzer
from voiceappointmentchatbot.tts import PiperSpeaker


def summarise_sentiments(
    samples: List[Tuple[str, float]],
) -> Optional[SentimentSummary]:
    """Aggregate per-turn sentiment readings into a single summary.

    The dominant label is decided by majority vote across ``samples``.
    Ties are broken by mean confidence: the label whose contributing
    scores have the higher mean wins. The reported ``score`` is the mean
    confidence of the *winning* label only, and ``samples`` is the total
    number of user turns scored (not just the winners).

    Args:
        samples: Sequence of ``(label, score)`` pairs, one per user turn
            that was passed through the sentiment analyser.

    Returns:
        :class:`SentimentSummary` describing the dominant sentiment, or
        ``None`` when ``samples`` is empty.
    """
    if not samples:
        return None

    counts: Counter[str] = Counter()
    score_buckets: Dict[str, List[float]] = defaultdict(list)
    for label, score in samples:
        counts[label] += 1
        score_buckets[label].append(score)

    top_count = max(counts.values())
    contenders = [label for label, count in counts.items() if count == top_count]
    if len(contenders) == 1:
        winner = contenders[0]
    else:
        winner = max(
            contenders,
            key=lambda label: sum(score_buckets[label]) / len(score_buckets[label]),
        )
    winning_scores = score_buckets[winner]
    mean_score = sum(winning_scores) / len(winning_scores)
    return SentimentSummary(
        label=winner,
        score=mean_score,
        samples=len(samples),
    )


def main() -> None:
    """Run the push-to-talk booking chatbot until completion or exit."""
    config = AppConfig.load()
    domain = load_domain(config.domain, config.domains_dir)
    print(f"[init] device={config.device}")
    print(f"[init] whisper={config.whisper.model_for(config.device)}")
    print(f"[init] domain={domain.name}")
    print(f"[init] model={config.anthropic.model}")

    recorder = HoldToRecord(config.audio)
    transcriber = WhisperTranscriber(config.device, config.whisper)
    sentiment = TextSentimentAnalyzer(config.device, config.sentiment)
    speaker = PiperSpeaker(config.piper)
    llm = HaikuClient(config.anthropic, domain)
    knowledge_base = KnowledgeBase(domain.knowledge_base_path, config.knowledge)

    last_language_holder: dict[str, str] = {"value": "en"}
    sentiment_samples: List[Tuple[str, float]] = []
    transcript_turns: List[TranscriptTurn] = []

    def on_confirmed(state: BookingState) -> None:
        record = appointment_record_from(
            state=state,
            language=last_language_holder["value"],
            sentiment=summarise_sentiments(sentiment_samples),
            transcript=transcript_turns,
        )
        path = write_appointment(record, config.output_dir)
        print(f"\n[booking] appointment saved to {path}")

    manager = DialogueManager(
        domain=domain,
        client=llm,
        knowledge_lookup=knowledge_base.query,
        on_appointment_confirmed=on_confirmed,
    )

    print("Booking chatbot ready. Ctrl+C to exit.\n")
    try:
        while True:
            audio = recorder.record()
            transcript = transcriber.transcribe(audio)
            print(
                f"[asr] lang={transcript.language} "
                f"({transcript.language_probability:.2f}): {transcript.text!r}"
            )

            mood = sentiment.analyze(transcript.text)
            print(f"[sentiment] {mood.label} ({mood.score:.2f})")
            if transcript.text.strip():
                sentiment_samples.append((mood.label, mood.score))
                transcript_turns.append(
                    TranscriptTurn(
                        role="user",
                        text=transcript.text,
                        language=transcript.language,
                    )
                )

            result = manager.handle_user_turn(transcript)
            last_language_holder["value"] = result.language
            print(f"[bot] ({result.language}) {result.reply}\n")
            speaker.speak(result.reply, result.language)
            transcript_turns.append(
                TranscriptTurn(
                    role="assistant",
                    text=result.reply,
                    language=result.language,
                )
            )

            if result.booking_complete:
                print("Booking complete. Goodbye.")
                break
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
