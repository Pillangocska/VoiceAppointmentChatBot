"""Unit tests for the markdown chunker and the retrieval ranker."""

from pathlib import Path
from typing import Sequence
import os

import numpy as np
import pytest

from voiceappointmentchatbot.config import KnowledgeBaseConfig
from voiceappointmentchatbot.knowledge import (
    Chunk,
    KnowledgeBase,
    chunk_markdown,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


def test_chunk_markdown_groups_paragraphs_under_headings() -> None:
    """Each chunk records the most recent heading from above it."""
    content = (
        "# Title\n\n"
        "First paragraph that should be long enough to stand alone as a chunk.\n\n"
        "## Section A\n\n"
        "Section A first paragraph, again sufficiently long to count.\n\n"
        "Section A second paragraph, also long enough to be its own chunk.\n\n"
        "## Section B\n\n"
        "Final paragraph in section B, definitely long enough on its own.\n"
    )

    chunks = chunk_markdown(content, min_chars=20)

    headings = [chunk.heading for chunk in chunks]
    assert headings == ["Title", "Section A", "Section A", "Section B"]
    assert "First paragraph" in chunks[0].text
    assert "first paragraph" in chunks[1].text
    assert "second paragraph" in chunks[2].text
    assert "Final paragraph" in chunks[3].text


def test_chunk_markdown_merges_short_paragraphs() -> None:
    """Paragraphs below ``min_chars`` are joined with the next one."""
    content = "# Heading\n\nshort\n\nshort\n\nThis paragraph is long enough to flush."

    chunks = chunk_markdown(content, min_chars=40)

    assert len(chunks) == 1
    assert "short" in chunks[0].text
    assert "long enough" in chunks[0].text


def test_chunk_markdown_handles_empty_input() -> None:
    """An empty document produces no chunks."""
    assert chunk_markdown("", min_chars=10) == []


def test_chunk_markdown_expands_tables_into_one_chunk_per_row() -> None:
    """Each data row in a markdown table becomes its own chunk with the header."""
    content = (
        "## Prices\n\n"
        "| Service | Price |\n"
        "| --- | --- |\n"
        "| Cut | 7000 |\n"
        "| Colour | 22000 |\n"
        "| Balayage | 45000 |\n"
    )

    chunks = chunk_markdown(content, min_chars=80)

    assert len(chunks) == 3
    assert all(chunk.heading == "Prices" for chunk in chunks)
    assert "| Cut | 7000 |" in chunks[0].text
    assert "| Service | Price |" in chunks[0].text  # header is preserved
    assert "Colour" in chunks[1].text
    assert "Balayage" in chunks[2].text


def test_chunk_markdown_on_real_vet_document() -> None:
    """The bundled vet KB chunks into more than one section."""
    content = (DOMAINS_DIR / "vet.md").read_text(encoding="utf-8")

    chunks = chunk_markdown(content, min_chars=80)

    headings = {chunk.heading for chunk in chunks}
    assert "Opening hours" in headings
    assert "Services and indicative prices" in headings
    assert "Booking policy" in headings
    assert all(chunk.text.strip() for chunk in chunks)


class _FakeEmbedder:
    """Deterministic stand-in for ``SentenceTransformer`` used in tests.

    Encodes each text into a unit vector whose entries reflect the
    presence of a small set of probe keywords. This makes ranking
    predictable without downloading the real model.
    """

    KEYWORDS: tuple[str, ...] = (
        "opening",
        "hours",
        "price",
        "vaccination",
        "phone",
    )

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return one row per input text, normalised when requested."""
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [1.0 if keyword in lowered else 0.0 for keyword in self.KEYWORDS],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            if normalize_embeddings and norm > 0:
                vector = vector / norm
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


def _kb_with_fake_model(
    path: Path, config: KnowledgeBaseConfig | None = None
) -> KnowledgeBase:
    """Build a :class:`KnowledgeBase` that bypasses the real embedder."""
    kb = KnowledgeBase(
        path=path,
        config=config or KnowledgeBaseConfig(top_k=2, chunk_min_chars=40),
    )
    kb._model = _FakeEmbedder()  # type: ignore[assignment]
    return kb


def test_query_returns_heading_and_text(tmp_path: Path) -> None:
    """The retrieved string includes the heading prefix for each chunk."""
    doc = tmp_path / "kb.md"
    doc.write_text(
        "# Hours\n\n"
        "Our opening hours are nine to five every weekday.\n\n"
        "# Prices\n\n"
        "A vaccination price is around ten thousand forints.\n\n"
        "# Phones\n\n"
        "Reach the office on the listed phone number any time.\n",
        encoding="utf-8",
    )
    kb = _kb_with_fake_model(doc)

    result = kb.query("what are your opening hours?", top_k=1)

    assert "Hours" in result
    assert "opening hours" in result.lower()


def test_query_ranks_keyword_match_above_unrelated(tmp_path: Path) -> None:
    """A vaccination question prefers the chunk containing 'vaccination'."""
    doc = tmp_path / "kb.md"
    doc.write_text(
        "# Hours\n\n"
        "Our opening hours are listed on the front door each day.\n\n"
        "# Prices\n\n"
        "A vaccination price is around ten thousand forints.\n\n"
        "# Phones\n\n"
        "Reach the office on the listed phone number any time.\n",
        encoding="utf-8",
    )
    kb = _kb_with_fake_model(doc)

    result = kb.query("how much is a vaccination?", top_k=1)

    assert "vaccination" in result.lower()
    assert "phone" not in result.lower()


def test_query_with_empty_question_returns_empty_string(tmp_path: Path) -> None:
    """Whitespace-only questions short-circuit without a model call."""
    doc = tmp_path / "kb.md"
    doc.write_text("# x\n\nbody body body body body body\n", encoding="utf-8")
    kb = _kb_with_fake_model(doc)

    assert kb.query("   ") == ""


def test_query_caches_embeddings_across_calls(tmp_path: Path) -> None:
    """The corpus is embedded exactly once even across multiple queries."""
    doc = tmp_path / "kb.md"
    doc.write_text(
        "# Hours\n\n"
        "Our opening hours are listed on the front door each day.\n",
        encoding="utf-8",
    )
    kb = _kb_with_fake_model(doc)

    counter = {"calls": 0}
    real_encode = kb._model.encode  # type: ignore[union-attr]

    def counting_encode(*args: object, **kwargs: object) -> np.ndarray:
        counter["calls"] += 1
        return real_encode(*args, **kwargs)  # type: ignore[arg-type]

    kb._model.encode = counting_encode  # type: ignore[assignment]

    kb.query("hours?")
    kb.query("opening?")

    # 1 corpus encode + 2 query encodes = 3 calls; key is the corpus
    # only goes through encode once.
    assert counter["calls"] == 3


@pytest.mark.skipif(
    os.environ.get("VABCB_RUN_EMBEDDING_TESTS") != "1",
    reason="Real-embedder integration test; set VABCB_RUN_EMBEDDING_TESTS=1 to enable.",
)
def test_real_embedder_retrieves_vet_pricing_section() -> None:
    """Sanity check against the real Sentence-Transformers model.

    Skipped by default because it downloads ~470 MB on first run.
    """
    config = KnowledgeBaseConfig(top_k=2, chunk_min_chars=80)
    kb = KnowledgeBase(path=DOMAINS_DIR / "vet.md", config=config)

    answer = kb.query("how much does a dental cleaning cost?")

    assert "Dental" in answer or "dental" in answer
    assert "HUF" in answer
