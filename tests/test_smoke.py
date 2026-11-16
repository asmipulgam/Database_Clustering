"""ponytail: one smoke test per stage as it lands, not a full suite up front."""

import pytest


def test_ontology_yaml_loads():
    import yaml

    with open("ontology/news.yaml") as f:
        onto = yaml.safe_load(f)
    assert "entities" in onto and "relations" in onto
    assert "Person" in onto["entities"]
    assert "mentions" in onto["relations"]


def test_normalize_whitespace_keeps_paragraphs():
    from swm.ingest import normalize_whitespace

    raw = "  RSA   is\tslow.  \n\n\n\n  Use   the Clipper chip?\n"
    assert normalize_whitespace(raw) == "RSA is slow.\n\nUse the Clipper chip?"


def test_documents_ingested():
    """Stage 1: needs Postgres up and scripts/01_ingest.py run; skips otherwise."""
    import psycopg

    from swm import db

    try:
        conn = db.connect()
    except psycopg.OperationalError:
        pytest.skip("Postgres not running (docker compose up -d postgres)")
    with conn:
        try:
            n, n_ids, n_labels, n_short = conn.execute(
                "SELECT count(*), count(DISTINCT id), count(DISTINCT label), "
                "count(*) FILTER (WHERE n_tokens < 20) FROM documents WHERE corpus = '20ng'"
            ).fetchone()
        except psycopg.errors.UndefinedTable:
            pytest.skip("run scripts/01_ingest.py first")
    if n == 0:
        pytest.skip("run scripts/01_ingest.py first")
    assert 15_000 < n <= 18_846
    assert n_ids == n
    assert n_labels == 20
    assert n_short == 0


def test_parse_pubmed_article():
    """PubMed parser on a hand-built Entrez.read record (no network)."""
    from swm.ingest import parse_pubmed_article

    class E(str):  # mimics Bio.Entrez StringElement: a str with .attributes
        def __new__(cls, value, **attributes):
            obj = super().__new__(cls, value)
            obj.attributes = attributes
            return obj

    rec = {
        "MedlineCitation": {
            "PMID": E("12345"),
            "Article": {
                "ArticleTitle": E("Metformin  and <i>E. coli</i> CO<sub>2</sub>."),
                "Abstract": {"AbstractText": [E("Background text.", Label="BACKGROUND"),
                                              E("HbA1c fell (p < 0.05; a<b and c>d).",
                                                Label="RESULTS")]},
                "Journal": {"Title": E("Diabetes Care"),
                            "JournalIssue": {"PubDate": {"MedlineDate": "2021 Jan-Feb"}}},
            },
            "MeshHeadingList": [
                {"DescriptorName": E("Diabetes Mellitus, Type 2", MajorTopicYN="N"),
                 "QualifierName": [E("drug therapy", MajorTopicYN="Y")]},
                {"DescriptorName": E("Humans", MajorTopicYN="N")},
            ],
        }
    }
    out = parse_pubmed_article(rec)
    assert out["pmid"] == "12345"
    assert out["title"] == "Metformin and E. coli CO2."  # markup stripped
    # section labels dropped; plain-text "<" comparisons left alone
    assert out["abstract"] == "Background text.\n\nHbA1c fell (p < 0.05; a<b and c>d)."
    assert out["year"] == "2021"
    assert out["mesh"] == ["Diabetes Mellitus, Type 2", "Humans"]
    assert out["mesh_major"] == ["Diabetes Mellitus, Type 2"]  # major via its qualifier
