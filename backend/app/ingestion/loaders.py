import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Version of the whole text-preparation pipeline: extraction here plus the
# chunking in chunking.py. Bump it whenever either changes the stored text, and
# documents whose bytes are unchanged get re-processed on the next ingest.
#   2: strip NUL/control bytes; fix chunk overlap overflow
PARSER_VERSION = 2

# Everything from here on is citations — high token cost, near-zero answer
# value, and it pollutes retrieval with title fragments that match many queries.
_REFERENCES_RE = re.compile(
    r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE
)

# Everything in C0 except tab and newline, which carry structure we use.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SECTION_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*)\s+[A-Z][^\n]{2,80}|(?:Abstract|Introduction|Conclusion|"
    r"Related Work|Method(?:s|ology)?|Experiments?|Results?|Discussion|Appendix))\s*$",
    re.MULTILINE,
)


@dataclass
class Page:
    number: int
    text: str


@dataclass
class ParsedDocument:
    source_path: str
    source_type: str
    pages: list[Page]
    file_hash: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    arxiv_id: str | None = None
    document_date: date | None = None
    version: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.full_text.encode("utf-8")).hexdigest()

    @property
    def page_count(self) -> int:
        return len(self.pages)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _blocks_in_reading_order(page) -> str:
    """Read a two-column page as two columns rather than zig-zagging across it.

    PyMuPDF's built-in sort orders blocks roughly top-to-bottom, which on a
    two-column arXiv paper interleaves the columns and produces sentences that
    jump mid-clause. That wrecks both chunk coherence and embedding quality.
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return ""

    width = page.rect.width
    midpoint = width / 2

    left = [b for b in blocks if b[2] < midpoint * 1.05]
    right = [b for b in blocks if b[0] > midpoint * 0.95]

    # Treat as two-column only when both sides carry real weight; a single-column
    # page with a stray figure caption must not be split.
    spanning = len(blocks) - len(left) - len(right)
    if len(left) >= 3 and len(right) >= 3 and spanning <= len(blocks) * 0.3:
        ordered = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    else:
        ordered = sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

    return "\n".join(b[4].strip() for b in ordered)


def _clean(text: str) -> str:
    # Embedded fonts and broken encodings leave NUL and other C0 control bytes
    # in extracted text. Postgres rejects NUL in text columns outright, so this
    # is a hard failure on roughly one arXiv PDF in eight, not a cosmetic issue.
    text = _CONTROL_CHARS_RE.sub("", text)
    # Rejoin words hyphenated across a line break, which PDFs do constantly and
    # which otherwise produces tokens like "repre-" / "sentation".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_references(pages: list[Page]) -> list[Page]:
    for i, page in enumerate(pages):
        match = _REFERENCES_RE.search(page.text)
        if match and i >= len(pages) // 2:
            trimmed = page.text[: match.start()].strip()
            kept = pages[:i]
            if trimmed:
                kept.append(Page(number=page.number, text=trimmed))
            return kept
    return pages


def load_pdf(path: Path, sidecar: dict | None = None) -> ParsedDocument:
    import pymupdf

    doc = pymupdf.open(path)
    try:
        pages = [
            Page(number=i + 1, text=_clean(_blocks_in_reading_order(page)))
            for i, page in enumerate(doc)
        ]
        pdf_meta = doc.metadata or {}
    finally:
        doc.close()

    pages = [p for p in pages if p.text]
    pages = _strip_references(pages)

    meta = sidecar or {}
    return ParsedDocument(
        source_path=str(path),
        source_type="pdf",
        pages=pages,
        file_hash=file_digest(path),
        title=meta.get("title") or pdf_meta.get("title") or path.stem,
        authors=meta.get("authors", []),
        arxiv_id=meta.get("arxiv_id"),
        document_date=_parse_date(meta.get("published")),
        version=meta.get("version"),
        metadata={k: v for k, v in meta.items() if k not in {"title", "authors"}},
    )


def load_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = None
    first = text.lstrip().split("\n", 1)[0]
    if first.startswith("# "):
        title = first[2:].strip()

    return ParsedDocument(
        source_path=str(path),
        source_type="markdown",
        pages=[Page(number=1, text=_clean(text))],
        file_hash=file_digest(path),
        title=title or path.stem,
    )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def load_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        sidecar_path = path.with_suffix(".json")
        sidecar = (
            json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar_path.exists()
            else None
        )
        return load_pdf(path, sidecar)
    if suffix in {".md", ".markdown"}:
        return load_markdown(path)
    raise ValueError(f"unsupported file type: {path.suffix}")


def find_section(text: str) -> str | None:
    match = _SECTION_RE.search(text)
    return match.group(1).strip() if match else None
