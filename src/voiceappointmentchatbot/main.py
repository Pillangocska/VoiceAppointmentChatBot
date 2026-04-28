"""Command-line entry point for the week-one voice chatbot demo.

Wires together microphone capture, Whisper transcription, sentiment
analysis, the echo dialogue policy, and Piper synthesis into a single
push-to-talk loop. Exits cleanly when the user sends EOF or KeyboardInterrupt.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from voiceappointmentchatbot.asr import WhisperTranscriber
from voiceappointmentchatbot.audio_io import HoldToRecord
from voiceappointmentchatbot.config import AppConfig
from voiceappointmentchatbot.dialogue import echo_reply
from voiceappointmentchatbot.sentiment import TextSentimentAnalyzer
from voiceappointmentchatbot.tts import PiperSpeaker


def main() -> None:
    """Run the push-to-talk chatbot loop until the user exits."""
    config = AppConfig.load()
    print(f"[init] device={config.device}")
    print(f"[init] whisper={config.whisper.model_for(config.device)}")

    recorder = HoldToRecord(config.audio)
    transcriber = WhisperTranscriber(config.device, config.whisper)
    sentiment = TextSentimentAnalyzer(config.device, config.sentiment)
    speaker = PiperSpeaker(config.piper)

    print("Vet appointment chatbot ready. Ctrl+C to exit.\n")
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

            reply = echo_reply(transcript, mood)
            print(f"[bot] {reply}\n")
            speaker.speak(reply, transcript.language)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
