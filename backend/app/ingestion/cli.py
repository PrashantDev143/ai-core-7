import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import REPO_ROOT
from app.ingestion.pipeline import ingest
from app.runtime import use_compatible_event_loop

DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser(prog="aicore-ingest")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="keep documents whose source file has disappeared",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if not args.corpus.exists():
        print(f"corpus directory not found: {args.corpus}", file=sys.stderr)
        print("run: python scripts/fetch_corpus.py", file=sys.stderr)
        return 1

    stats = asyncio.run(ingest(args.corpus, prune=not args.no_prune))

    print(
        f"seen={stats.seen} new={stats.ingested} updated={stats.updated} "
        f"skipped={stats.skipped} failed={stats.failed} "
        f"deleted={stats.deleted} chunks={stats.chunks_written}"
    )
    for err in stats.errors[:10]:
        print(f"  ! {err}", file=sys.stderr)

    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
