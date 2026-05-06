"""Retrieval-augmented knowledge base for the booking chatbot.

Each business domain ships a single markdown file under ``domains/``
that documents prices, opening hours, services, and policies. When the
LLM calls the ``ask_kb`` tool we want to ground its answer in those
documents rather than on whatever the model remembers from training.

The implementation is deliberately tiny: load the file, split into
paragraph-sized chunks, embed them with a multilingual
Sentence-Transformers model, and rank chunks by cosine similarity to
the incoming question. There is no vector database — the corpus is one
file per domain, so a NumPy array in memory is faster, has no extra
dependency, and keeps the test surface small.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import re

import numpy as np

from voiceappointmentchatbot.config import KnowledgeBaseConfig


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit from the knowledge base.

    Attributes:
        text: Raw paragraph text as it appears in the markdown file.
        heading: Most recent ``#`` / ``##`` heading above this chunk,
            empty string when none precedes it. Used so the LLM sees the
            section context alongside the chunk content.
    """

    text: str
    heading: str


def chunk_markdown(content: str, min_chars: int = 80) -> List[Chunk]:
    """Split a markdown document into paragraph-sized retrievable chunks.

    Headings are tracked but not returned as their own chunks; instead
    each chunk records the most recent heading so the model sees the
    section context. Short paragraphs are merged with the following
    paragraph so we do not return single-sentence fragments.

    Markdown tables are expanded into one chunk per data row so a
    question about a single line item (e.g. a specific price) can match
    that row directly rather than the whole table block. The header row
    is prepended to each chunk so the model sees the column context.

    Args:
        content: Full markdown text.
        min_chars: Paragraphs shorter than this are merged forward.

    Returns:
        List of :class:`Chunk` objects in document order.
    """
    chunks: List[Chunk] = []
    current_heading = ""
    pending: List[str] = []

    def flush() -> None:
        if not pending:
            return
        text = "\n\n".join(pending).strip()
        if text:
            chunks.append(Chunk(text=text, heading=current_heading))
        pending.clear()

    for raw_paragraph in _PARAGRAPH_SPLIT.split(content):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith("#"):
            flush()
            current_heading = paragraph.lstrip("# ").strip()
            continue
        if _is_markdown_table(paragraph):
            flush()
            for row_text in _expand_table_rows(paragraph):
                chunks.append(Chunk(text=row_text, heading=current_heading))
            continue
        pending.append(paragraph)
        joined_length = sum(len(item) for item in pending)
        if joined_length >= min_chars:
            flush()

    flush()
    return chunks


def _is_markdown_table(paragraph: str) -> bool:
    """Whether ``paragraph`` is a pipe-delimited markdown table block."""
    lines = paragraph.splitlines()
    if len(lines) < 3:
        return False
    if not lines[0].lstrip().startswith("|"):
        return False
    separator = lines[1].strip()
    return bool(re.match(r"^\|[\s|:\-]+\|$", separator))


def _expand_table_rows(paragraph: str) -> List[str]:
    """Yield one chunk per data row, prefixed with the header row."""
    lines = [line for line in paragraph.splitlines() if line.strip()]
    header_line = lines[0]
    data_lines = lines[2:]  # skip the separator
    return [f"{header_line}\n{row}" for row in data_lines]


class KnowledgeBase:
    """Lazy multilingual retrieval over a single domain markdown file.

    The embedding model and the corpus are loaded the first time
    :meth:`query` is called so startup stays fast when no factual
    questions arise during a session. Subsequent calls reuse the cached
    embeddings.

    Attributes:
        path: Path to the markdown source file.
        config: Embedding model and retrieval parameters.
    """

    def __init__(self, path: Path, config: KnowledgeBaseConfig) -> None:
        """Initialise without touching disk or loading the model."""
        self.path = path
        self.config = config
        self._chunks: Optional[List[Chunk]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._model: Optional[object] = None

    def _load_chunks(self) -> List[Chunk]:
        """Read and chunk the markdown file on first use."""
        if self._chunks is None:
            text = self.path.read_text(encoding="utf-8")
            self._chunks = chunk_markdown(text, min_chars=self.config.chunk_min_chars)
        return self._chunks

    def _ensure_embeddings(self) -> tuple[List[Chunk], np.ndarray]:
        """Embed every chunk on first use and cache the result."""
        chunks = self._load_chunks()
        if self._embeddings is None:
            from sentence_transformers import SentenceTransformer

            if self._model is None:
                self._model = SentenceTransformer(self.config.embedding_model)
            corpus = [_chunk_for_embedding(chunk) for chunk in chunks]
            vectors = self._model.encode(  # type: ignore[attr-defined]
                corpus,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._embeddings = np.asarray(vectors, dtype=np.float32)
        assert self._embeddings is not None
        return chunks, self._embeddings

    def warm_up(self) -> None:
        """Eagerly load the embedder and embed the corpus once.

        Reads the markdown file, builds chunk embeddings, and caches the
        Sentence-Transformers model so the first ``ask_kb`` tool call
        does not pay any of those costs mid-conversation.
        """
        self._ensure_embeddings()

    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve the best matching chunks for ``question``.

        Args:
            question: User question, in any language the embedding model
                supports.
            top_k: Override for the configured number of chunks. Defaults
                to :attr:`KnowledgeBaseConfig.top_k`.

        Returns:
            A single string with the top chunks separated by blank lines
            and prefixed with their section heading, ready to feed back
            to the LLM as the ``ask_kb`` tool result. Empty string when
            the corpus is empty.
        """
        if not question.strip():
            return ""
        chunks, embeddings = self._ensure_embeddings()
        if not chunks:
            return ""
        k = top_k if top_k is not None else self.config.top_k
        ranked = _rank(question, embeddings, self._model, k=k)  # type: ignore[arg-type]
        selected = [chunks[index] for index in ranked]
        return _format_chunks(selected)


def _chunk_for_embedding(chunk: Chunk) -> str:
    """Embed the heading together with the chunk for richer context."""
    if chunk.heading:
        return f"{chunk.heading}\n{chunk.text}"
    return chunk.text


def _rank(
    question: str,
    embeddings: np.ndarray,
    model: object,
    k: int,
) -> Sequence[int]:
    """Return the indices of the top ``k`` chunks for ``question``."""
    query_vector = model.encode(  # type: ignore[attr-defined]
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_array = np.asarray(query_vector, dtype=np.float32)[0]
    scores = embeddings @ query_array
    if k >= len(scores):
        return list(np.argsort(-scores))
    top = np.argpartition(-scores, k)[:k]
    return list(top[np.argsort(-scores[top])])


def _format_chunks(chunks: Sequence[Chunk]) -> str:
    """Render selected chunks as a single newline-separated string."""
    parts: List[str] = []
    for chunk in chunks:
        if chunk.heading:
            parts.append(f"## {chunk.heading}\n{chunk.text}")
        else:
            parts.append(chunk.text)
    return "\n\n".join(parts)
