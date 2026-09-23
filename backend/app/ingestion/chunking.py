"""Token-budgeted chunking measured with the embedding model's own tokenizer.

Chunk size is counted in the tokens the embedder will actually see, not
characters or a generic BPE. Getting this wrong is silent: the model truncates
past its window and the tail of every oversized chunk is simply not embedded,
which is invisible until recall is mysteriously bad.
"""

import re
import threading
from dataclasses import dataclass

from app.config import get_settings
from app.ingestion.loaders import Page, find_section

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Fragments this small are page furniture — a stray header, a figure number,
# the tail end of a section. They cannot answer anything, but they do occupy an
# index slot and can outrank real passages on short queries.
MIN_CHUNK_TOKENS = 16

_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained(
                    get_settings().local_embedding_model
                )
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def model_token_budget() -> int:
    """Usable content tokens, leaving room for [CLS] and [SEP]."""
    tok = get_tokenizer()
    limit = getattr(tok, "model_max_length", 512)
    if limit > 100_000:  # some tokenizers report a sentinel instead of a real limit
        limit = 512
    return limit - 2


@dataclass
class TextChunk:
    content: str
    token_count: int
    page_start: int
    page_end: int
    section: str | None


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int


def _split_units(pages: list[Page], max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for page in pages:
        for para in (p.strip() for p in page.text.split("\n\n")):
            if not para:
                continue
            tokens = count_tokens(para)
            if tokens <= max_tokens:
                units.append(_Unit(para, tokens, page.number))
                continue
            # Paragraph alone busts the window — break on sentences, then
            # fall back to a hard token slice for pathological input.
            for sentence in _SENTENCE_RE.split(para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_tokens = count_tokens(sentence)
                if s_tokens <= max_tokens:
                    units.append(_Unit(sentence, s_tokens, page.number))
                else:
                    units.extend(_hard_split(sentence, max_tokens, page.number))
    return units


def _hard_split(text: str, max_tokens: int, page: int) -> list[_Unit]:
    tok = get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    out = []
    for i in range(0, len(ids), max_tokens):
        window = ids[i : i + max_tokens]
        out.append(_Unit(tok.decode(window), len(window), page))
    return out


def chunk_pages(
    pages: list[Page],
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    settings = get_settings()
    budget = min(chunk_size or settings.chunk_size_tokens, model_token_budget())
    overlap = overlap if overlap is not None else settings.chunk_overlap_tokens
    overlap = min(overlap, budget // 2)

    # Units are capped at budget-overlap, not budget, so that a carried tail
    # plus the next unit still fits. Splitting at the full budget lets a chunk
    # overshoot by up to `overlap`, which silently truncates at embed time.
    units = _split_units(pages, max(budget - overlap, budget // 2))
    if not units:
        return []

    chunks: list[TextChunk] = []
    window: list[_Unit] = []
    total = 0

    def flush() -> None:
        if not window:
            return
        text = "\n\n".join(u.text for u in window)
        chunks.append(
            TextChunk(
                content=text,
                token_count=total,
                page_start=window[0].page,
                page_end=window[-1].page,
                section=find_section(text),
            )
        )

    for unit in units:
        if window and total + unit.tokens > budget:
            flush()
            # Carry the tail of the previous chunk forward so a fact spanning a
            # boundary stays retrievable from either side.
            carried: list[_Unit] = []
            carried_tokens = 0
            for prev in reversed(window):
                if carried_tokens + prev.tokens > overlap:
                    break
                carried.insert(0, prev)
                carried_tokens += prev.tokens
            # Drop the carry outright if it would push this chunk over budget.
            # Overlap is an optimisation; staying inside the model window is not.
            if carried_tokens + unit.tokens > budget:
                carried, carried_tokens = [], 0
            window = carried
            total = carried_tokens

        window.append(unit)
        total += unit.tokens

    flush()

    kept = [c for c in chunks if c.token_count >= MIN_CHUNK_TOKENS]
    # A genuinely tiny document is still worth one chunk; only drop fragments
    # when there is something substantial to keep.
    return kept or chunks[:1]
