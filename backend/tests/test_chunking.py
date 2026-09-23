"""Chunk packing logic, tested against a whitespace tokenizer.

The real tokenizer is swapped out so these run offline and fast. What is being
tested is the packing and overlap logic, which is independent of which
tokenizer produces the counts.
"""

import pytest

from app.ingestion import chunking
from app.ingestion.loaders import Page


class WhitespaceTokenizer:
    model_max_length = 1000

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(chunking, "_tokenizer", WhitespaceTokenizer())


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_input_is_one_chunk():
    chunks = chunking.chunk_pages([Page(1, words(10))], chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].page_start == chunks[0].page_end == 1


def test_no_chunk_exceeds_the_budget():
    pages = [Page(1, "\n\n".join(words(30, f"p{p}") for p in range(10)))]
    chunks = chunking.chunk_pages(pages, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_overlap_carries_content_forward():
    pages = [Page(1, "\n\n".join(words(40, f"p{p}") for p in range(6)))]
    no_overlap = chunking.chunk_pages(pages, chunk_size=120, overlap=0)
    with_overlap = chunking.chunk_pages(pages, chunk_size=120, overlap=40)
    # Carrying a tail forward means more chunks cover the same text.
    assert len(with_overlap) >= len(no_overlap)


def test_budget_is_clamped_to_the_model_window(monkeypatch):
    monkeypatch.setattr(chunking, "_tokenizer", WhitespaceTokenizer())
    pages = [Page(1, words(5000))]
    # Asking for more than the model can accept must not produce oversized chunks.
    chunks = chunking.chunk_pages(pages, chunk_size=99999, overlap=0)
    assert all(c.token_count <= chunking.model_token_budget() for c in chunks)


def test_oversized_paragraph_is_split_not_dropped():
    pages = [Page(1, words(500))]
    chunks = chunking.chunk_pages(pages, chunk_size=50, overlap=0)
    assert len(chunks) >= 10
    recovered = sum(c.content.count("w") for c in chunks)
    assert recovered >= 500


def test_page_range_spans_source_pages():
    pages = [Page(1, words(30, "a")), Page(2, words(30, "b")), Page(3, words(30, "c"))]
    chunks = chunking.chunk_pages(pages, chunk_size=200, overlap=0)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 3


def test_overlap_carry_cannot_push_a_chunk_over_budget():
    # Regression: the carried tail was added to the window and the next unit
    # appended without re-checking, so chunks overshot by up to `overlap`.
    # Observed in the real corpus as 544-token chunks against a 480 budget.
    pages = [Page(1, "\n\n".join(words(200, f"p{p}") for p in range(12)))]
    for overlap in (0, 16, 64, 128):
        chunks = chunking.chunk_pages(pages, chunk_size=480, overlap=overlap)
        assert chunks, f"overlap={overlap} produced nothing"
        worst = max(c.token_count for c in chunks)
        assert worst <= 480, f"overlap={overlap} produced a {worst}-token chunk"


def test_budget_holds_for_one_oversized_paragraph():
    pages = [Page(1, words(3000))]
    chunks = chunking.chunk_pages(pages, chunk_size=480, overlap=64)
    assert max(c.token_count for c in chunks) <= 480


def test_tiny_trailing_fragments_are_dropped():
    pages = [Page(1, words(100, "a") + "\n\n" + words(3, "tail"))]
    chunks = chunking.chunk_pages(pages, chunk_size=50, overlap=0)
    assert all(c.token_count >= chunking.MIN_CHUNK_TOKENS for c in chunks)


def test_a_tiny_document_still_yields_one_chunk():
    chunks = chunking.chunk_pages([Page(1, "short")], chunk_size=50, overlap=0)
    assert len(chunks) == 1


def test_empty_input_produces_nothing():
    assert chunking.chunk_pages([]) == []
    assert chunking.chunk_pages([Page(1, "")]) == []
