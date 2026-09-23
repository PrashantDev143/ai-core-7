import hashlib
import re
import unicodedata

NAMESPACE = "aicore:cache"

_WS_RE = re.compile(r"\s+")
# Only trailing punctuation is stripped. Removing it everywhere would merge
# genuinely different queries — "what is RAG" and "what is R.A.G." are fine to
# merge, but "f(x)" and "fx" are not.
_TRAILING_PUNCT_RE = re.compile(r"[\s?!.,;:]+$")


def normalise_query(query: str) -> str:
    """Canonical form for exact-match lookup.

    Conservative on purpose. Layer 1 must never return an answer to a
    *different* question, so normalisation only collapses differences that
    cannot change meaning: unicode form, case, whitespace, trailing
    punctuation. Anything more aggressive belongs in layer 2, where a
    similarity threshold makes the risk explicit and tunable.
    """
    text = unicodedata.normalize("NFKC", query)
    text = _WS_RE.sub(" ", text).strip().lower()
    return _TRAILING_PUNCT_RE.sub("", text)


def query_digest(query: str) -> str:
    return hashlib.sha256(normalise_query(query).encode("utf-8")).hexdigest()[:32]


def exact_key(query: str, corpus_version: str) -> str:
    """Exact-match key, scoped to the corpus it was answered against.

    The corpus version is in the KEY, not in the value. That means a re-index
    cannot serve a stale answer even by accident: the new version produces a
    different key, so old entries become unreachable and expire on their TTL.
    Storing the version in the value and comparing after the read would work
    too, but only as long as every read site remembers to check.
    """
    return f"{NAMESPACE}:exact:{corpus_version[:16]}:{query_digest(query)}"


def semantic_index_key(corpus_version: str) -> str:
    return f"{NAMESPACE}:semidx:{corpus_version[:16]}"


def semantic_entry_key(corpus_version: str, digest: str) -> str:
    return f"{NAMESPACE}:sem:{corpus_version[:16]}:{digest}"


def metrics_key(name: str) -> str:
    return f"{NAMESPACE}:metrics:{name}"
