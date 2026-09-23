"""Download the seed corpus from arXiv.

Papers are selected by querying the arXiv API for topics, not by a hardcoded
list of IDs, and the resulting IDs+versions are pinned into manifest.json. The
manifest is committed and the PDFs are not, so the corpus is reproducible
without putting a few hundred MB in git.

arXiv asks API clients to leave ~3s between requests. That is respected here.
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

API_DELAY = 3.0
PDF_DELAY = 1.5

# Chosen to span retrieval, generation, agents, evaluation and efficiency, so
# the corpus supports both narrow factual questions and multi-document
# synthesis. Each maps to a distinct cluster of the Phase 2 eval set.
QUERIES = [
    'abs:"retrieval augmented generation"',
    'abs:"dense passage retrieval" OR abs:"dense retrieval"',
    'abs:"large language model" AND abs:"hallucination"',
    'abs:"chain of thought" AND abs:"reasoning"',
    'abs:"language model" AND abs:"agent" AND abs:"tool use"',
    'abs:"instruction tuning" OR abs:"parameter efficient fine-tuning"',
    'abs:"vector search" OR abs:"approximate nearest neighbor"',
    'abs:"evaluation" AND abs:"large language model" AND abs:"benchmark"',
    'abs:"prompt injection" OR abs:"jailbreak" AND abs:"language model"',
    'abs:"knowledge distillation" AND abs:"language model"',
]


@dataclass
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    summary: str
    pdf_url: str

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id.replace('/', '_')}{self.version}.pdf"


def _text(entry, path: str) -> str:
    node = entry.find(path, ATOM)
    return (node.text or "").strip().replace("\n", " ") if node is not None else ""


def parse_entries(xml: str) -> list[Paper]:
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = _text(entry, "a:id")
        if "/abs/" not in raw_id:
            continue
        tail = raw_id.split("/abs/")[-1]
        arxiv_id, _, version = tail.partition("v")

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                version=f"v{version}" if version else "v1",
                title=" ".join(_text(entry, "a:title").split()),
                authors=[
                    (a.find("a:name", ATOM).text or "").strip()
                    for a in entry.findall("a:author", ATOM)
                    if a.find("a:name", ATOM) is not None
                ],
                published=_text(entry, "a:published"),
                updated=_text(entry, "a:updated"),
                categories=[
                    c.attrib.get("term", "") for c in entry.findall("a:category", ATOM)
                ],
                summary=" ".join(_text(entry, "a:summary").split()),
                pdf_url=f"https://arxiv.org/pdf/{tail}",
            )
        )
    return papers


def search(client: httpx.Client, query: str, limit: int) -> list[Paper]:
    response = client.get(
        ARXIV_API,
        params={
            "search_query": query,
            "start": 0,
            "max_results": limit,
            # Relevance ordering gives recognisable, well-cited work rather
            # than whatever was posted most recently.
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return parse_entries(response.text)


def collect(target: int, per_query: int) -> list[Paper]:
    seen: dict[str, Paper] = {}
    headers = {"User-Agent": "aicore7-corpus/0.1 (student project; polite crawler)"}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for i, query in enumerate(QUERIES):
            if len(seen) >= target:
                break
            if i:
                time.sleep(API_DELAY)
            try:
                for paper in search(client, query, per_query):
                    seen.setdefault(paper.arxiv_id, paper)
            except Exception as exc:
                print(f"  query failed ({query[:40]}...): {exc}", file=sys.stderr)
            print(f"  [{len(seen):3d}] after: {query[:55]}")

    return list(seen.values())[:target]


def download(papers: list[Paper], pdf_dir: Path) -> tuple[int, int]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aicore7-corpus/0.1 (student project; polite crawler)"}
    fetched = skipped = 0

    with httpx.Client(headers=headers, follow_redirects=True, timeout=120.0) as client:
        for n, paper in enumerate(papers, 1):
            pdf_path = pdf_dir / paper.filename
            sidecar = pdf_path.with_suffix(".json")

            if pdf_path.exists() and pdf_path.stat().st_size > 1024:
                skipped += 1
                continue

            try:
                response = client.get(paper.pdf_url)
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("response was not a PDF")

                pdf_path.write_bytes(response.content)
                # Sidecar carries the metadata the PDF itself does not reliably
                # embed; the loader reads it alongside the file.
                sidecar.write_text(json.dumps(asdict(paper), indent=2), encoding="utf-8")
                fetched += 1
                print(f"  [{n:3d}/{len(papers)}] {paper.arxiv_id} {paper.title[:58]}")
                time.sleep(PDF_DELAY)
            except Exception as exc:
                print(f"  [{n:3d}] FAILED {paper.arxiv_id}: {exc}", file=sys.stderr)

    return fetched, skipped


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="aicore-corpus")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--out", type=Path, default=repo_root / "data" / "corpus")
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help="re-download exactly what manifest.json pins, ignoring the search queries",
    )
    args = parser.parse_args()

    corpus_dir = args.out
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.json"

    if args.from_manifest:
        if not manifest_path.exists():
            print(f"no manifest at {manifest_path}", file=sys.stderr)
            return 1
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        papers = [Paper(**p) for p in payload["papers"]]
        print(f"manifest pins {len(papers)} papers")
    else:
        print("searching arXiv...")
        papers = collect(args.target, args.per_query)
        if not papers:
            print("no papers found", file=sys.stderr)
            return 1
        manifest_path.write_text(
            json.dumps(
                {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "queries": QUERIES,
                    "count": len(papers),
                    "papers": [asdict(p) for p in papers],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote manifest: {len(papers)} papers")

    print("downloading PDFs...")
    fetched, skipped = download(papers, corpus_dir / "pdf")
    print(f"\ndone: {fetched} downloaded, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
