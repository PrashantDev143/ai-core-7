from app.retrieval.base import RetrievedChunk

# The constant from the original RRF paper. It damps the top ranks so a single
# retriever cannot dominate on its #1 alone: at k=60 the gap between rank 1 and
# rank 2 is small, while the gap between rank 1 and rank 50 still matters.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], *, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Merge ranked lists by rank position, not by score.

    Dense cosine similarity and BM25 scores live on different, unnormalised
    scales that also shift per query, so adding or averaging them is
    meaningless. RRF only reads the ordinal position, which is why it needs no
    tuning per retriever and no score calibration.
    """
    scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for item in ranked:
            key = str(item.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + item.rank)

            if key in merged:
                # Keep provenance from every retriever that found this chunk.
                merged[key].sources.update(item.sources)
            else:
                merged[key] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for rank, (key, score) in enumerate(ordered, start=1):
        chunk = merged[key]
        chunk.score = score
        chunk.rank = rank
        out.append(chunk)
    return out
