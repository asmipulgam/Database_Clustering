"""ponytail: one smoke test per stage as it lands, not a full suite up front."""

import pytest


def test_ontology_yaml_loads():
    import yaml

    with open("ontology/news.yaml") as f:
        onto = yaml.safe_load(f)
    assert "entities" in onto and "relations" in onto
    assert "Person" in onto["entities"]
    assert "mentions" in onto["relations"]


@pytest.mark.parametrize("path", ["ontology/news.yaml", "ontology/medical.yaml"])
def test_ontology_is_consistent(path):
    """Every relation endpoint is a defined entity; Document exists but is never LLM-extracted."""
    import yaml

    with open(path) as f:
        onto = yaml.safe_load(f)
    entities = onto["entities"]
    assert entities["Document"].get("extracted") is False
    for name, ent in entities.items():
        assert ent.get("descriptor"), f"{path}: entity {name} has no descriptor"
    for name, rel in onto["relations"].items():
        assert rel.get("descriptor"), f"{path}: relation {name} has no descriptor"
        for end in ("source", "target"):
            types = rel[end] if isinstance(rel[end], list) else [rel[end]]
            undefined = set(types) - set(entities)
            assert not undefined, f"{path}: {name}.{end} uses undefined {sorted(undefined)}"


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


def test_extraction_validation():
    """KG validator on hand-written LLM output (no Ollama needed): keeps grounded,
    in-ontology triples with verbatim evidence; discards the rest with a reason."""
    from swm.kg import LLMExtraction, load_ontology, validate_extraction

    onto = load_ontology("ontology/medical.yaml")
    text = ("Metformin improved glycaemic control in type 2 diabetes. "
            "Plasmodium falciparum causes severe malaria.")
    s1 = "Metformin improved glycaemic control in type 2 diabetes."
    raw = LLMExtraction(
        entities=[{"name": "metformin", "type": "Treatment"},
                  {"name": "Plasmodium falciparum", "type": "Pathogen"}],
        triples=[
            dict(evidence=s1, subject="Metformin", subject_type="Treatment",
                 relation="treats", object="type 2 diabetes", object_type="Disease"),
            # small copying slip (glycemic vs glycaemic, no period): cited verbatim instead
            dict(evidence="Metformin improved glycemic control in type 2 diabetes",
                 subject="DOCUMENT", subject_type="Document",
                 relation="about", object="type 2 diabetes", object_type="Disease"),
            dict(evidence="Plasmodium falciparum causes severe malaria.",
                 subject="Plasmodium falciparum", subject_type="Pathogen",
                 relation="causes", object="malaria", object_type="Disease"),
            dict(evidence=s1, subject="type 2 diabetes", subject_type="Disease",
                 relation="treats", object="metformin", object_type="Treatment"),
            dict(evidence="Insulin cures diabetes.", subject="insulin", subject_type="Treatment",
                 relation="treats", object="type 2 diabetes", object_type="Disease"),
            dict(evidence="Metformin cures everyone with diabetes forever.",
                 subject="Metformin", subject_type="Treatment",
                 relation="treats", object="type 2 diabetes", object_type="Disease"),
        ],
    )
    entities, triples, discarded = validate_extraction(raw, "pubmed/1", text, onto)

    llm = [t for t in triples if t["source"] == "llm"]
    assert {(t["subject"], t["relation"], t["object"]) for t in llm} == {
        ("Metformin", "treats", "type 2 diabetes"),
        ("pubmed/1", "about", "type 2 diabetes"),
    }
    for t in llm:
        assert t["evidence"] == s1 and text[slice(*t["span"])] == s1
    assert sorted(d["reason"] for d in discarded) == [
        "entity_not_in_text",        # insulin is not in the text
        "evidence_not_in_text",      # made-up sentence
        "type_not_in_ontology",      # Pathogen entity
        "type_not_in_ontology",      # Pathogen triple
        "wrong_types_for_relation",  # Disease treats Treatment
    ]
    assert {(e["name"], e["type"]) for e in entities} == {
        ("metformin", "Treatment"), ("type 2 diabetes", "Disease")}
    mentions = [t for t in triples if t["relation"] == "mentions"]
    assert {t["object"] for t in mentions} == {"metformin", "type 2 diabetes"}
    assert all(t["evidence"] == s1 for t in mentions)


def test_salvage_cut_off_reply():
    """A reply cut off at the output limit keeps its complete triples."""
    from swm.kg import _salvage

    t = ('{"evidence": "A uses B.", "subject": "A", "subject_type": "Organization", '
         '"relation": "produces", "object": "B", "object_type": "Artifact"}')
    cut = '{"entities": [{"name": "A", "type": "Organization"}], "triples": [' + t + ", " + t[:40]
    out = _salvage(cut)
    assert out is not None and len(out.entities) == 1 and len(out.triples) == 1
    assert out.triples[0].evidence == "A uses B."
    # cut inside the entity list: entities so far, no triples
    out = _salvage('{"entities": [{"name": "A", "type": "Organization"}, {"name": "B", "ty')
    assert out is not None and len(out.entities) == 1 and out.triples == []
    assert _salvage('{"entities": [{"na') is None


def test_llm_schema_requires_both_lists():
    """Optional lists let the model answer "{}" (18 of 40 docs did); keep them required."""
    from swm.kg import LLMExtraction

    assert set(LLMExtraction.model_json_schema()["required"]) == {"entities", "triples"}


def test_generic_and_substring_entities_rejected():
    """'I' is not grounded just because the letter i occurs; pronouns and type names are not entities."""
    from swm.kg import LLMExtraction, load_ontology, validate_extraction

    onto = load_ontology("ontology/news.yaml")
    text = "I am not claiming citizens need the NSA's algorithms. Something big will happen."
    raw = LLMExtraction(entities=[], triples=[
        dict(evidence="I am not claiming citizens need the NSA's algorithms.", subject="I",
             subject_type="Person", relation="affiliatedWith", object="NSA", object_type="Organization"),
        dict(evidence="Something big will happen.", subject="NSA", subject_type="Organization",
             relation="participatesIn", object="Event", object_type="Event"),
    ])
    _, triples, discarded = validate_extraction(raw, "d1", text, onto)
    assert [t for t in triples if t["source"] == "llm"] == []
    assert [d["reason"] for d in discarded] == ["generic_name", "generic_name"]
    from swm.kg import _generic, _grounded
    assert not _grounded("Bi", "the nsa is big", {"the", "nsa", "big"})  # "bi" only inside "big"
    assert not _generic("US", onto) and _generic("us", onto)  # the country is not the pronoun
