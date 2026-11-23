#!/usr/bin/env python3
"""Try KG extraction on a small sample, before clustering exists. Writes nothing to Neo4j.

    python scripts/try_extract.py --corpus 20ng --label sci.crypt   # 20 sci.crypt posts
    python scripts/try_extract.py --corpus pubmed                   # 1 abstract per topic + MeSH check
    python scripts/try_extract.py --corpus pubmed --model qwen2.5:7b

Full results (kept and discarded triples) go to data/cache/extract_sample_<corpus>[_<model>].json.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swm import db  # noqa: E402
from swm.kg import extract, load_ontology  # noqa: E402

ONTOLOGY = {"20ng": "ontology/news.yaml", "pubmed": "ontology/medical.yaml"}


def pick_sample(docs, corpus, label, n, min_words, max_words, seed):
    rng = random.Random(seed)
    pool = [d for d in docs
            if (label is None or d["label"] == label) and min_words <= d["n_tokens"] <= max_words]
    if corpus == "pubmed" and label is None:
        # One abstract per topic, so the MeSH check covers all 20 diseases.
        by_label = defaultdict(list)
        for d in pool:
            by_label[d["label"]].append(d)
        return [rng.choice(by_label[lab]) for lab in sorted(by_label)][:n]
    return rng.sample(pool, min(n, len(pool)))


def short(s, n=150):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize(results, onto):
    ok = [r for r in results if not r["error"]]
    llm = [t for r in ok for t in r["triples"] if t["source"] == "llm"]
    auto = [t for r in ok for t in r["triples"] if t["source"] == "auto"]
    discarded = [d for r in ok for d in r["discarded"]]
    bad_triples = [d for d in discarded if "relation" in d["item"]]
    ents = [e for r in ok for e in r["entities"]]
    known = {t.lower() for t in onto.entities}
    unknown = Counter(
        v for d in discarded if d["reason"] == "type_not_in_ontology"
        for k in ("type", "subject_type", "object_type")
        if (v := d["item"].get(k)) and v.lower() not in known
    )

    print(f"\n=== {onto.name} ontology | {results[0]['model']} ===")
    salvaged = sum(r.get("salvaged", False) for r in results)
    print(f"Docs: {len(results)} | LLM failures: {len(results) - len(ok)} | "
          f"cut-off replies salvaged: {salvaged} | "
          f"{statistics.mean(r['seconds'] for r in results):.0f} s/doc"
          + (f" | truncated: {sum(r['truncated'] for r in results)}" if any(r["truncated"] for r in results) else ""))
    print(f"Entities kept: {len(ents)}  ("
          + ", ".join(f"{t} {c}" for t, c in Counter(e["type"] for e in ents).most_common()) + ")")
    total = len(llm) + len(bad_triples)
    print(f"LLM triples: {len(llm)} kept, {len(bad_triples)} discarded"
          + (f"  -> {len(llm) / total:.0%} kept" if total else ""))
    print("  kept by relation:    " + ", ".join(f"{r} {c}" for r, c in Counter(t["relation"] for t in llm).most_common()))
    print("  discarded by reason: " + (", ".join(f"{r} {c}" for r, c in Counter(d["reason"] for d in discarded).most_common()) or "none"))
    print("  (discards include entities as well as triples)")
    if unknown:
        print("Types the model wanted that the ontology lacks: "
              + ", ".join(f"{t} x{c}" for t, c in unknown.most_common(8)))
    print(f"Provenance edges added (mentions): {len(auto)}")

    print("\nSample kept triples:")
    for r in ok[:6]:
        for t in [t for t in r["triples"] if t["source"] == "llm"][:3]:
            print(f"  [{r['doc_id']}] {t['subject']} --{t['relation']}--> {t['object']}")
            print(f"      \"{short(t['evidence'])}\"")
    print("\nSample discarded:")
    for d in discarded[:8]:
        it = d["item"]
        what = (f"{it['subject']} ({it['subject_type']}) --{it['relation']}--> {it['object']} ({it['object_type']})"
                if "relation" in it else f"entity {it['name']} ({it['type']})")
        print(f"  {d['reason']:<26} {short(what, 110)}")


def _mnorm(s):
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"['’]s\b", "", s.lower())))


def _matches(name, terms):
    padded = f" {_mnorm(name)} "
    return any(t and f" {t} " in padded for t in map(_mnorm, terms))


def mesh_check(results, sample):
    """Compare extracted Disease entities with each abstract's MeSH indexing (headings NLM
    indexers assigned by hand), using MeSH synonyms so 'breast cancer' matches 'Breast Neoplasms'."""
    from swm.ingest import _entrez, _entrez_read

    Entrez = _entrez()
    pmids = [d["meta"]["pmid"] for d in sample]
    headings = {}  # pmid -> [(heading, mesh uid)]
    recs = _entrez_read(Entrez, Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml"))
    for rec in recs["PubmedArticle"]:
        cit = rec["MedlineCitation"]
        headings[str(cit["PMID"])] = [
            (str(h["DescriptorName"]), "68" + h["DescriptorName"].attributes["UI"][1:])
            for h in cit.get("MeshHeadingList", [])
        ]
    uids = sorted({uid for hs in headings.values() for _, uid in hs})
    synonyms, disease_uids = {}, set()
    for i in range(0, len(uids), 200):
        handle = Entrez.esummary(db="mesh", id=",".join(uids[i:i + 200]))
        for s in _entrez_read(Entrez, handle, ignore_errors=True):
            if not hasattr(s, "get") or "Id" not in s:
                continue  # NCBI has no summary for this heading: it is matched by name only
            uid = str(s["Id"])
            synonyms[uid] = [str(x) for x in s.get("DS_MeshTerms", [])]
            trees = [str(link.get("TreeNum", "")) for link in s.get("DS_IdxLinks", [])]
            # MeSH disease branches: C (Diseases) and F03 (Mental Disorders).
            if any(t.startswith(("C", "F03")) for t in trees):
                disease_uids.add(uid)

    rows = []
    for d, r in zip(sample, results):
        hs = headings.get(d["meta"]["pmid"], [])
        label_uid = next((uid for name, uid in hs if name == d["label"]), None)
        label_terms = synonyms.get(label_uid, [d["label"]])
        diseases = [e["name"] for e in r["entities"] if e["type"] == "Disease"]
        about = [t["object"] for t in r["triples"] if t["relation"] == "about"]
        mesh_diseases = [(name, uid) for name, uid in hs if uid in disease_uids]
        rows.append({
            "doc_id": d["id"], "label": d["label"],
            "about": about,
            "about_matches_topic": any(_matches(a, label_terms) for a in about),
            "topic_extracted": any(_matches(x, label_terms) for x in diseases),
            "diseases": diseases,
            "diseases_in_mesh": [x for x in diseases
                                 if any(_matches(x, synonyms.get(uid, [n])) for n, uid in hs)],
            "mesh_diseases": [n for n, _ in mesh_diseases],
            "mesh_diseases_found": [n for n, uid in mesh_diseases
                                    if any(_matches(x, synonyms.get(uid, [n])) for x in diseases)],
        })

    print("\n=== PubMed: extracted diseases vs MeSH indexing ===")
    print(f"{'doc':<17} {'topic (label)':<32} {'about=topic':<12} {'topic found':<12} "
          f"{'diseases in MeSH':<17} MeSH diseases found")
    for x in rows:
        print(f"{x['doc_id']:<17} {short(x['label'], 31):<32} "
              f"{'yes' if x['about_matches_topic'] else 'NO':<12} {'yes' if x['topic_extracted'] else 'NO':<12} "
              f"{len(x['diseases_in_mesh'])}/{len(x['diseases']):<15} "
              f"{len(x['mesh_diseases_found'])}/{len(x['mesh_diseases'])}")
    n = len(rows)
    n_dis = sum(len(x["diseases"]) for x in rows)
    n_mesh = sum(len(x["mesh_diseases"]) for x in rows)
    print(f"\n'about' names the study's topic:        {sum(x['about_matches_topic'] for x in rows)}/{n}")
    print(f"Topic disease extracted at all:         {sum(x['topic_extracted'] for x in rows)}/{n}")
    if n_dis:
        print(f"Extracted diseases also in MeSH:        {sum(len(x['diseases_in_mesh']) for x in rows)}/{n_dis} "
              f"({sum(len(x['diseases_in_mesh']) for x in rows) / n_dis:.0%})")
    if n_mesh:
        print(f"MeSH disease headings extracted:        {sum(len(x['mesh_diseases_found']) for x in rows)}/{n_mesh} "
              f"({sum(len(x['mesh_diseases_found']) for x in rows) / n_mesh:.0%})")
    extra = [(x["doc_id"], dz) for x in rows for dz in x["diseases"] if dz not in x["diseases_in_mesh"]]
    if extra:
        print("Extracted diseases with no MeSH heading (not necessarily wrong: MeSH lists main "
              "subjects only): " + "; ".join(dz for _, dz in extra[:12]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", choices=list(ONTOLOGY), default="20ng")
    p.add_argument("--label", help="only docs with this label, e.g. sci.crypt")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--model", help="Ollama model (default: OLLAMA_MODEL from .env)")
    p.add_argument("--min-words", type=int, default=60)
    p.add_argument("--max-words", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    with db.connect() as conn:
        docs = db.read_documents(conn, args.corpus)
    sample = pick_sample(docs, args.corpus, args.label, args.n, args.min_words, args.max_words, args.seed)
    if not sample:
        raise SystemExit(f"No {args.corpus} docs match label={args.label!r} and the word limits.")

    onto = load_ontology(ONTOLOGY[args.corpus])
    results = extract(sample, ONTOLOGY[args.corpus], model=args.model)

    suffix = f"_{re.sub(r'[^a-z0-9]+', '-', args.model.lower())}" if args.model else ""
    out = ROOT / "data" / "cache" / f"extract_sample_{args.corpus}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(mesh=None):
        out.write_text(json.dumps({
            "corpus": args.corpus, "ontology": ONTOLOGY[args.corpus], "args": vars(args),
            "sample": [{"id": d["id"], "label": d["label"], "text": d["text"]} for d in sample],
            "results": results, "mesh_check": mesh,
        }, indent=2, ensure_ascii=False))

    save()  # before the MeSH check, so a network error there never loses LLM results
    summarize(results, onto)
    if args.corpus == "pubmed":
        save(mesh_check(results, sample))
    print(f"\nFull results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
