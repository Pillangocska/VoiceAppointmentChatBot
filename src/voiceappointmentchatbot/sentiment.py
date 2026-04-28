"""Multilingual text-based sentiment classification.

Wraps a Hugging Face pipeline that scores text on a three-class
positive/neutral/negative scale and supports both English and Hungarian.
The classifier loads lazily so startup stays fast when sentiment is not
exercised in a given run.
"""

from dataclasses import dataclass
from typing import Optional

from voiceappointmentchatbot.config import Device, SentimentConfig


@dataclass(frozen=True)
class SentimentResult:
    """Outcome of a sentiment classification pass.

    Attributes:
        label: One of ``positive``, ``neutral``, ``negative``.
        score: Model confidence in the predicted label, in [0, 1].
    """

    label: str
    score: float


class TextSentimentAnalyzer:
    """Lazy wrapper around a multilingual XLM-RoBERTa sentiment model.

    Attributes:
        device: Compute device the underlying pipeline runs on.
        config: Sentiment model configuration.
    """

    def __init__(self, device: Device, config: SentimentConfig) -> None:
        """Initialise without loading the model yet."""
        self.device = device
        self.config = config
        self._pipeline: Optional[object] = None

    def _ensure_loaded(self) -> object:
        """Load the Hugging Face pipeline on first use."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1,
            )
        return self._pipeline

    def analyze(self, text: str) -> SentimentResult:
        """Classify the sentiment of ``text``.

        Args:
            text: Utterance transcript in any language supported by the
                underlying model.

        Returns:
            Sentiment result with normalised label and confidence. Empty
            text yields a neutral result with score ``0.0``.
        """
        if not text.strip():
            return SentimentResult(label="neutral", score=0.0)

        clf = self._ensure_loaded()
        prediction = clf(text)[0]  # type: ignore[operator,index]
        return SentimentResult(
            label=str(prediction["label"]).lower(),
            score=float(prediction["score"]),
        )
