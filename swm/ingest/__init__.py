"""Corpus loaders. Each loader returns a list[dict] in the common document schema:

    {id, corpus, text, label, split, n_tokens, meta}

`label` is the corpus's own category (20NG newsgroup, PubMed MeSH) and is used
only as ground truth for evaluation, never as a clustering input. Adding a
corpus (Optional B) means adding a loader here; nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_INLINE_SPACE = re.compile(r"[ \t\f\v]+")
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse spaces within lines and runs of blank lines; keep paragraph breaks.

    Paragraphs are kept (not flattened to one line) so the dashboard and the
    KG extractor's evidence spans stay readable.
    """
    lines = (_INLINE_SPACE.sub(" ", ln).strip() for ln in text.splitlines())
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def load_20ng(subset: str = "all", min_tokens: int = 20) -> list[dict]:
    """Load 20news-bydate, strip headers/footers/quotes, drop short posts and duplicates.

    ponytail: scikit-learn's fetch_20newsgroups does the header/footer/quote
    stripping for us (remove=('headers','footers','quotes')) — no custom
    parser needed. First call downloads ~14 MB to ~/scikit_learn_data.

    IDs are "20ng/<newsgroup>/<article number>", which is stable across runs
    and points back to the original file (provenance for the KG layer).
    """
    from sklearn.datasets import fetch_20newsgroups

    splits = ["train", "test"] if subset == "all" else [subset]
    docs: list[dict] = []
    seen: set[str] = set()
    n_raw = n_short = n_dup = 0

    for split in splits:
        ds = fetch_20newsgroups(
            subset=split, remove=("headers", "footers", "quotes"), shuffle=False
        )
        for raw, target, fname in zip(ds.data, ds.target, ds.filenames):
            n_raw += 1
            text = normalize_whitespace(raw)
            n_tokens = len(text.split())
            if n_tokens < min_tokens:
                n_short += 1
                continue
            # After quote stripping, cross-posts and reposts collapse to the
            # same body; keep the first copy only.
            digest = hashlib.sha1(text.lower().encode()).hexdigest()
            if digest in seen:
                n_dup += 1
                continue
            seen.add(digest)
            group = ds.target_names[target]
            docs.append(
                {
                    "id": f"20ng/{group}/{Path(fname).name}",
                    "corpus": "20ng",
                    "text": text,
                    "label": group,
                    "split": split,
                    "n_tokens": n_tokens,
                    "meta": {},
                }
            )

    log.info(
        "20ng: %d raw posts -> %d kept (%d under %d tokens, %d duplicates)",
        n_raw, len(docs), n_short, min_tokens, n_dup,
    )
    return docs


# ==========================================================
# PubMed (Optional B)
# ==========================================================

# 20 MeSH disease headings, one per ground-truth class, mirroring 20NG's 20 groups.
# Some are deliberately close (three cancers; Alzheimer/Parkinson; asthma/COPD;
# hypertension/MI/stroke; TB/HIV/malaria) so the benchmark keeps hard cases,
# the way 20NG does with its five comp.* groups.
PUBMED_TOPICS = [
    "Breast Neoplasms",
    "Colorectal Neoplasms",
    "Lung Neoplasms",
    "Alzheimer Disease",
    "Parkinson Disease",
    "Stroke",
    "Schizophrenia",
    "Major Depressive Disorder",  # MeSH renamed this from "Depressive Disorder, Major"
    "Diabetes Mellitus, Type 2",
    "Hypertension",
    "Myocardial Infarction",
    "Asthma",
    "Pulmonary Disease, Chronic Obstructive",
    "COVID-19",
    "Tuberculosis",
    "Malaria",
    "HIV Infections",
    "Osteoarthritis",
    "Psoriasis",
    "Glaucoma",
]

# The selected PMIDs are written here and committed, so every teammate ingests
# exactly the same abstracts even though the live PubMed search keeps changing.
PUBMED_PMID_FILE = PROJECT_ROOT / "data" / "pubmed_pmids.tsv"


def _entrez():
    from Bio import Entrez
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    # NCBI asks for a contact email; without one Biopython only warns.
    # An API key raises the rate limit from 3 to 10 requests/second.
    Entrez.email = os.getenv("NCBI_EMAIL") or None
    Entrez.api_key = os.getenv("NCBI_API_KEY") or None
    return Entrez


def _entrez_read(Entrez, handle, **kwargs):
    try:
        return Entrez.read(handle, **kwargs)
    finally:
        handle.close()


# Markup NCBI leaves inside titles/abstracts (<i>, <sup>, <sub>, MathML, ...).
# Matched by exact tag name with only key="value" attributes, so plain-text
# comparisons like "p < 0.05" or "a<b and c>d" are never touched.
_TAG_ATTRS = r'(?:\s+[\w:.-]+\s*=\s*"[^"]*")*\s*/?>'
_BLOCK_TAGS = re.compile(r"</?(?:sec|title|p)" + _TAG_ATTRS, re.IGNORECASE)
_INLINE_TAGS = re.compile(
    r"</?(?:i|b|u|em|strong|sup|sub|inf|italic|bold|sc|"
    r"math|mi|mo|mn|ms|mrow|msub|msup|msubsup|mover|munder|munderover|mtext|mspace|"
    r"mfenced|mfrac|msqrt|mroot|mstyle|mtable|mtr|mtd|semantics|annotation)" + _TAG_ATTRS,
    re.IGNORECASE,
)


def _clean_pubmed_text(text: str) -> str:
    """Drop NCBI markup: block tags become line breaks, inline tags vanish (CO<sub>2</sub> -> CO2)."""
    return normalize_whitespace(_INLINE_TAGS.sub("", _BLOCK_TAGS.sub("\n", text)))


def _is_major(element) -> bool:
    return getattr(element, "attributes", {}).get("MajorTopicYN") == "Y"


def parse_pubmed_article(rec) -> dict:
    """One `PubmedArticle` from Entrez.read -> flat dict. Public so it can be unit-tested."""
    cit = rec["MedlineCitation"]
    art = cit["Article"]
    pub_date = art["Journal"]["JournalIssue"]["PubDate"]
    mesh, major = [], []
    for heading in cit.get("MeshHeadingList", []):
        name = str(heading["DescriptorName"])
        mesh.append(name)
        # MEDLINE marks a heading major on the descriptor or on any of its qualifiers.
        if _is_major(heading["DescriptorName"]) or any(
            _is_major(q) for q in heading.get("QualifierName", [])
        ):
            major.append(name)
    return {
        "pmid": str(cit["PMID"]),
        "title": _clean_pubmed_text(str(art.get("ArticleTitle", ""))),
        # Structured abstracts arrive as labelled sections (BACKGROUND, METHODS, ...).
        # Keep the section text, drop the labels: they are boilerplate shared by every doc.
        "abstract": _clean_pubmed_text(
            "\n\n".join(str(p) for p in art.get("Abstract", {}).get("AbstractText", []))
        ),
        "journal": str(art["Journal"].get("Title", "")),
        "year": str(pub_date.get("Year") or pub_date.get("MedlineDate", ""))[:4],
        "mesh": mesh,
        "mesh_major": major,
    }


def _fetch_pubmed(Entrez, pmids: list[str], batch: int = 200) -> dict[str, dict]:
    from tqdm import tqdm

    records: dict[str, dict] = {}
    for i in tqdm(range(0, len(pmids), batch), desc="PubMed efetch", unit="batch"):
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids[i : i + batch]), retmode="xml")
        for rec in _entrez_read(Entrez, handle)["PubmedArticle"]:
            parsed = parse_pubmed_article(rec)
            records[parsed["pmid"]] = parsed
    return records


def _pubmed_doc(rec: dict, topic: str) -> dict:
    text = f"{rec['title']}\n\n{rec['abstract']}".strip()
    return {
        "id": f"pubmed/{rec['pmid']}",
        "corpus": "pubmed",
        "text": text,
        "label": topic,
        "split": None,
        "n_tokens": len(text.split()),
        "meta": {k: rec[k] for k in ("pmid", "journal", "year", "mesh_major", "mesh")},
    }


def _select_pubmed(Entrez, topics, per_topic, min_tokens, date_range) -> list[dict]:
    """Search each topic, keep single-topic articles with a real abstract, trim to per_topic."""
    candidates: dict[str, list[str]] = {}
    for topic in topics:
        # :noexp = this exact heading, not its narrower terms, so the record-level
        # check below can see every selected topic by name.
        term = (
            f'"{topic}"[MeSH Major Topic:noexp] AND hasabstract AND english[lang] '
            f"AND {date_range}[dp]"
        )
        # Over-fetch: some candidates are dropped below as multi-topic or too short
        # (Tuberculosis loses ~20% to TB/HIV and TB/COVID-19 co-topic papers).
        handle = Entrez.esearch(db="pubmed", term=term, retmax=int(per_topic * 1.5))
        candidates[topic] = list(_entrez_read(Entrez, handle)["IdList"])
        if not candidates[topic]:
            # NLM renames MeSH headings every year; an old name silently matches nothing.
            raise ValueError(
                f"PubMed search for {topic!r} found nothing. Check the current MeSH "
                "heading name at https://meshb.nlm.nih.gov and update PUBMED_TOPICS."
            )

    all_pmids = list(dict.fromkeys(p for ids in candidates.values() for p in ids))
    records = _fetch_pubmed(Entrez, all_pmids)

    topic_set = set(topics)
    docs: list[dict] = []
    n_multi = n_short = 0
    for topic, ids in candidates.items():
        kept = []
        for pmid in ids:
            rec = records.get(pmid)
            if rec is None:
                continue
            # Single-label ground truth: drop articles where another selected
            # topic is also a major heading (e.g. Alzheimer AND Parkinson).
            if (set(rec["mesh_major"]) & topic_set) - {topic}:
                n_multi += 1
                continue
            doc = _pubmed_doc(rec, topic)
            if doc["n_tokens"] < min_tokens:
                n_short += 1
                continue
            kept.append(doc)
        if len(kept) < per_topic:
            log.warning("pubmed: only %d usable abstracts for %r", len(kept), topic)
        docs.extend(kept[:per_topic])

    log.info(
        "pubmed: %d candidates -> %d kept (%d multi-topic, %d under %d tokens)",
        len(all_pmids), len(docs), n_multi, n_short, min_tokens,
    )
    return docs


def load_pubmed(
    per_topic: int = 2500,
    min_tokens: int = 20,
    topics: list[str] | None = None,
    date_range: str = "2020:2024",
    refresh: bool = False,
    pmid_file: Path = PUBMED_PMID_FILE,
) -> list[dict]:
    """Load PubMed abstracts for PUBMED_TOPICS via NCBI E-utilities (Bio.Entrez).

    Each doc is title + abstract, labelled with the one MeSH topic it was
    selected for. First run searches PubMed and writes the chosen PMIDs to
    `pmid_file`; later runs re-fetch exactly that list (pass refresh=True to
    re-select). 20 topics x 2,500 = 50,000 abstracts: about 15 minutes to select,
    about 10 to re-fetch (an NCBI_API_KEY in .env makes both roughly 3x faster).
    """
    Entrez = _entrez()
    pmid_file = Path(pmid_file)

    if pmid_file.exists() and not refresh:
        rows = [ln.split("\t") for ln in pmid_file.read_text().splitlines()[1:] if ln]
        log.info("pubmed: re-fetching the %d PMIDs in %s (refresh=True to re-select)",
                 len(rows), pmid_file.relative_to(PROJECT_ROOT))
        records = _fetch_pubmed(Entrez, [pmid for pmid, _ in rows])
        missing = [pmid for pmid, _ in rows if pmid not in records]
        if missing:
            log.warning("pubmed: %d listed PMIDs no longer returned by NCBI", len(missing))
        return [_pubmed_doc(records[pmid], topic) for pmid, topic in rows if pmid in records]

    docs = _select_pubmed(Entrez, topics or PUBMED_TOPICS, per_topic, min_tokens, date_range)
    pmid_file.parent.mkdir(parents=True, exist_ok=True)
    pmid_file.write_text(
        "pmid\ttopic\n" + "".join(f"{d['meta']['pmid']}\t{d['label']}\n" for d in docs)
    )
    log.info("pubmed: wrote %d PMIDs to %s", len(docs), pmid_file.relative_to(PROJECT_ROOT))
    return docs
