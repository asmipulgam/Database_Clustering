#!/usr/bin/env python3
"""Load a corpus into Postgres. Run once per corpus; downstream scripts read from the DB.

    python scripts/01_ingest.py                          # 20 Newsgroups (default)
    python scripts/01_ingest.py --corpus pubmed          # 20 MeSH topics x 2,500 = 50,000 abstracts
    python scripts/01_ingest.py --corpus pubmed --refresh   # re-select PMIDs from PubMed

Re-running replaces that corpus's rows, so it is safe to repeat.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

# Allow `python scripts/01_ingest.py` (and the VS Code Run button) from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swm import db  # noqa: E402
from swm.ingest import load_20ng, load_pubmed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", choices=["20ng", "pubmed"], default="20ng")
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--subset", choices=["all", "train", "test"], default="all",
                        help="20ng only")
    parser.add_argument("--per-topic", type=int, default=2500,
                        help="pubmed only: abstracts per MeSH topic")
    parser.add_argument("--refresh", action="store_true",
                        help="pubmed only: ignore data/pubmed_pmids.tsv and search PubMed again")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.corpus == "20ng":
        docs = load_20ng(subset=args.subset, min_tokens=args.min_tokens)
    else:
        docs = load_pubmed(per_topic=args.per_topic, min_tokens=args.min_tokens,
                           refresh=args.refresh)

    with db.connect() as conn:
        db.init_schema(conn)
        n = db.write_documents(conn, args.corpus, docs)

    per_label = Counter(d["label"] for d in docs)
    tokens = [d["n_tokens"] for d in docs]
    width = max(map(len, per_label)) + 2
    print(f"\nWrote {n} documents to Postgres table 'documents' (corpus='{args.corpus}').")
    print(f"Tokens per doc: median {statistics.median(tokens):.0f}, max {max(tokens)}")
    print(f"Docs per label: min {min(per_label.values())}, max {max(per_label.values())}")
    for label, count in sorted(per_label.items()):
        print(f"  {label:<{width}} {count}")


if __name__ == "__main__":
    main()
