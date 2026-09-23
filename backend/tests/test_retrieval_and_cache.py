import pytest

from app.cache.keys import exact_key, normalise_query, query_digest
from app.retrieval.base import RetrievedChunk
from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from app.retrieval.sparse import BM25Index, _Doc, tokenize


def chunk(cid, rank, retriever="dense", score=1.0):
    import uuid

    return RetrievedChunk(
        chunk_id=uuid.UUID(int=cid),
        document_id=uuid.UUID(int=1000 + cid),
        content=f"chunk {cid}",
        score=score,
        rank=rank,
        source_path=f"/tmp/{cid}.pdf",
        sources={retriever: rank},
    )


class TestFusion:
    def test_chunk_found_by_both_outranks_one_found_by_either(self):
        dense = [chunk(1, 1), chunk(2, 2)]
        sparse = [chunk(3, 1, "sparse"), chunk(1, 2, "sparse")]
        fused = reciprocal_rank_fusion([dense, sparse])
        # Chunk 1 is ranked by both lists, so it should win despite not being
        # first in the sparse list.
        assert str(fused[0].chunk_id) == str(chunk(1, 1).chunk_id)

    def test_provenance_from_all_retrievers_is_kept(self):
        fused = reciprocal_rank_fusion([[chunk(1, 1)], [chunk(1, 3, "sparse")]])
        assert fused[0].sources == {"dense": 1, "sparse": 3}

    def test_ranks_are_contiguous_from_one(self):
        fused = reciprocal_rank_fusion([[chunk(i, i) for i in range(1, 6)]])
        assert [c.rank for c in fused] == [1, 2, 3, 4, 5]

    def test_score_matches_rrf_formula(self):
        fused = reciprocal_rank_fusion([[chunk(1, 1)]])
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))

    def test_empty_lists_are_safe(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestBM25:
    @pytest.fixture
    def index(self):
        docs = [
            _Doc("a", "d1", "retrieval augmented generation for question answering", 6,
                 "/a", "A", None, 1, 1, None),
            _Doc("b", "d2", "dense passage retrieval using bi-encoders", 5,
                 "/b", "B", None, 1, 1, None),
            _Doc("c", "d3", "cooking pasta with tomato sauce", 4,
                 "/c", "C", None, 1, 1, None),
        ]
        for d in docs:
            d.length = len(tokenize(d.content))
        return BM25Index(docs, "v1")

    def test_returns_only_matching_documents(self, index):
        results = index.search("retrieval", 10)
        assert len(results) == 2
        assert all("retrieval" in d.content for d, _ in results)

    def test_unmatched_query_returns_nothing(self, index):
        assert index.search("quantum chromodynamics", 10) == []

    def test_rarer_term_scores_higher(self, index):
        # "pasta" appears in one doc, "retrieval" in two, so pasta has higher idf.
        pasta = index.search("pasta", 1)[0][1]
        retrieval = index.search("retrieval", 1)[0][1]
        assert pasta > retrieval

    def test_stopwords_removed_but_negations_kept(self):
        tokens = tokenize("the model is not accurate and it was slow")
        assert "the" not in tokens and "is" not in tokens
        assert "not" in tokens  # negation carries meaning in technical text


class TestCacheKeys:
    def test_case_and_whitespace_normalised(self):
        assert normalise_query("  What IS   RAG?  ") == "what is rag"

    def test_trailing_punctuation_stripped(self):
        assert normalise_query("what is rag?") == normalise_query("what is rag")

    def test_internal_punctuation_preserved(self):
        # f(x) and fx are different questions; merging them would be wrong.
        assert normalise_query("what is f(x)") != normalise_query("what is fx")

    def test_different_queries_get_different_digests(self):
        assert query_digest("what is rag") != query_digest("what is bm25")

    def test_corpus_version_changes_the_key(self):
        """The core Phase 3 guarantee: re-indexing retires every entry."""
        a = exact_key("what is rag", "corpus_aaaaaaaa")
        b = exact_key("what is rag", "corpus_bbbbbbbb")
        assert a != b

    def test_same_query_and_version_is_stable(self):
        assert exact_key("what is rag", "v1") == exact_key("  What is RAG? ", "v1")
