"""Ontology-anchored cluster knowledge graphs: extract -> normalize -> load -> label.

Only extract() is implemented so far. It works per document, so it can be tested
before clustering exists; 05_kg.py will later pass each cluster's representative
documents (those closest to the HDBSCAN exemplars).
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DOCUMENT = "Document"
# The ontology's catch-all Document -> entity relation. The pipeline adds one per
# extracted entity (with the sentence that names it), so the LLM is never asked for it.
AUTO_RELATION = "mentions"


# ==========================================================
# Ontology
# ==========================================================

@dataclass
class Ontology:
    name: str
    entities: dict[str, str]          # type -> descriptor, including Document
    extractable: list[str]            # types the LLM may extract (extracted != false)
    relations: dict[str, dict]        # name -> {"source": set, "target": set, "descriptor"}


def load_ontology(path: str | Path) -> Ontology:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text())

    def as_set(v):
        return set(v) if isinstance(v, list) else {v}

    return Ontology(
        name=path.stem,
        entities={name: e["descriptor"] for name, e in raw["entities"].items()},
        extractable=[n for n, e in raw["entities"].items() if e.get("extracted", True)],
        relations={
            name: {"source": as_set(r["source"]), "target": as_set(r["target"]),
                   "descriptor": " ".join(r["descriptor"].split())}
            for name, r in raw["relations"].items()
        },
    )


# ==========================================================
# LLM output schema
# ==========================================================
# Types and relations are plain strings on purpose, not enums. Forcing the model
# to pick from the ontology would hide gaps (a pathogen squeezed into "Disease");
# instead out-of-ontology output is discarded with a reason, so mismatches show up
# in the discard counts (proposal, Risk 5).

class LLMEntity(BaseModel):
    name: str
    type: str


class LLMTriple(BaseModel):
    # Evidence first: the model copies a sentence, then reads the triple off it.
    evidence: str = Field(description="One sentence copied word for word from the document")
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str


class LLMExtraction(BaseModel):
    # Both fields must stay required, with no defaults: optional fields let the
    # model answer "{}" (it did, for 18 of 40 docs). And no max length: Ollama
    # enforces maxItems, but the model then crams the rest of its answer into the
    # last string and loops. Overlong output is salvaged instead (see _salvage).
    entities: list[LLMEntity]
    triples: list[LLMTriple]


def _system_prompt(onto: Ontology) -> str:
    entity_lines = "\n".join(
        f"- {t}: {' '.join(onto.entities[t].split())}" for t in onto.extractable
    )
    relation_lines = "\n".join(
        f"- {name}: {' | '.join(sorted(r['source']))} -> {' | '.join(sorted(r['target']))}. "
        f"{r['descriptor']}"
        for name, r in onto.relations.items() if name != AUTO_RELATION
    )
    doc_relations = " or ".join(
        f'"{name}"' for name, r in onto.relations.items()
        if DOCUMENT in r["source"] and name != AUTO_RELATION
    )
    return f"""You build a knowledge graph from ONE document, using ONLY the ontology below.

Entity types:
{entity_lines}

Relations (subject type -> object type):
{relation_lines}

Rules:
1. Extract only entities explicitly named in the document. Copy each name exactly as written; keep abbreviations as written. Never use pronouns or roles ("I", "we", "the author", "patients") as entities.
2. Use only the entity types and relations listed above, with the subject and object types each relation allows.
3. Start with one triple whose subject is the document itself: subject "DOCUMENT", subject type {DOCUMENT}, relation {doc_relations}, naming the document's main subject. {DOCUMENT} is never the subject of any other relation.
4. "evidence" is ONE sentence copied word for word from the document that states the triple.
5. State only what the document says, not background knowledge. Fewer correct triples beat many guesses. Return at most 20 triples, most important first, or empty lists if nothing fits."""


def _closing_brackets(prefix: str) -> str | None:
    """Brackets that would close a cut-off JSON prefix; None if it ends inside a string."""
    stack, in_string, escaped = [], False, False
    for ch in prefix:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    return None if in_string else "".join(reversed(stack))


def _salvage(raw: str) -> LLMExtraction | None:
    """Keep the complete items of a reply cut off at the output limit (num_predict).

    The prompt asks for the most important triples first, so the lost tail is the
    least valuable part. Tries each '}' from the end until the prefix parses.
    """
    ends = [m.end() for m in re.finditer(r"\}", raw)]
    for end in reversed(ends[-300:]):
        closers = _closing_brackets(raw[:end])
        if closers is None:
            continue
        try:
            data = json.loads(raw[:end] + closers)
            if isinstance(data, dict):  # cut off before "triples" began
                data.setdefault("entities", [])
                data.setdefault("triples", [])
            return LLMExtraction.model_validate(data)
        except ValueError:
            continue
    return None


# ==========================================================
# Text matching (grounding + evidence spans)
# ==========================================================

_WORD = re.compile(r"[a-z0-9]+")


def _ws(s: str) -> str:
    return " ".join(s.split())


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n+", text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


# Pronouns and roles are never entities (prompt rule 1); this enforces it.
_GENERIC_NAMES = {
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "it", "its", "this", "that", "author", "the author",
    "authors", "people", "someone", "everyone", "patients", "participants", "subjects",
}


def _grounded(name: str, text_lower: str, text_words: set[str]) -> bool:
    """The entity is named in the text: verbatim as whole words, or every word of 3+ letters appears."""
    name = _ws(name).lower()
    if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text_lower):
        return True
    words = [w for w in _WORD.findall(name) if len(w) >= 3]
    return bool(words) and all(w in text_words for w in words)


def _generic(name: str, onto: Ontology) -> bool:
    """A pronoun, a role, or a bare type name ("Event"), not a real entity."""
    written = _ws(name)
    if written.lower() in {t.lower() for t in onto.entities}:
        return True
    # All-caps acronyms are names, not pronouns: "US" (country) vs "us", "IT" vs "it".
    return written.lower() in _GENERIC_NAMES and not (len(written) > 1 and written.isupper())


def _find_evidence(evidence: str, text: str, spans: list[tuple[int, int]]):
    """(start, end) of the evidence in the text, or None if the model made it up."""
    words = evidence.split()
    if not words:
        return None
    m = re.search(r"\s+".join(map(re.escape, words)), text, re.IGNORECASE)
    if m:
        return m.span()
    # Small copying slips (a dropped word, changed quotes or punctuation): accept the
    # sentence that contains >= 90% of the evidence's characters in order, and cite
    # that sentence verbatim instead of the model's version.
    ev = _ws(evidence).lower()
    if len(ev) < 20:
        return None
    best, best_cover = None, 0.0
    for start, end in spans:
        matcher = difflib.SequenceMatcher(None, ev, _ws(text[start:end]).lower(), autojunk=False)
        cover = sum(b.size for b in matcher.get_matching_blocks()) / len(ev)
        if cover > best_cover:
            best, best_cover = (start, end), cover
    return best if best_cover >= 0.9 else None


def _sentence_naming(name: str, text: str, spans: list[tuple[int, int]]):
    m = re.search(r"\s+".join(map(re.escape, name.split())), text, re.IGNORECASE)
    if not m:
        return None
    return next(((s, e) for s, e in spans if s <= m.start() < e), m.span())


# ==========================================================
# Validation
# ==========================================================

def validate_extraction(raw: LLMExtraction, doc_id: str, text: str, onto: Ontology):
    """Keep only ontology-compliant, text-grounded output. Returns (entities, triples, discarded).

    Every kept triple carries verbatim evidence and its character span in `text`.
    Every discarded item carries the reason, so schema mismatches stay visible.
    """
    text_lower = _ws(text).lower()
    text_words = set(_WORD.findall(text_lower))
    spans = _sentence_spans(text)
    type_of = {t.lower(): t for t in onto.entities}
    relation_of = {r.lower(): r for r in onto.relations}

    entities: dict[str, dict] = {}
    discarded: list[dict] = []

    def add_entity(name: str, etype: str) -> None:
        entities.setdefault(_ws(name).lower(), {"name": _ws(name), "type": etype})

    for e in raw.entities:
        etype = type_of.get(e.type.strip().lower())
        if etype is None or etype not in onto.extractable:
            discarded.append({"item": e.model_dump(), "reason": "type_not_in_ontology"})
        elif _generic(e.name, onto):
            discarded.append({"item": e.model_dump(), "reason": "generic_name"})
        elif not e.name.strip() or not _grounded(e.name, text_lower, text_words):
            discarded.append({"item": e.model_dump(), "reason": "entity_not_in_text"})
        else:
            add_entity(e.name, etype)

    triples: list[dict] = []
    seen: set[tuple] = set()
    for t in raw.triples:
        rel = relation_of.get(t.relation.strip().lower())
        s_type = type_of.get(t.subject_type.strip().lower())
        o_type = type_of.get(t.object_type.strip().lower())
        span = None
        if rel is None:
            reason = "relation_not_in_ontology"
        elif s_type is None or o_type is None or o_type == DOCUMENT:
            reason = "type_not_in_ontology"
        elif s_type not in onto.relations[rel]["source"] or o_type not in onto.relations[rel]["target"]:
            reason = "wrong_types_for_relation"
        elif (s_type != DOCUMENT and _generic(t.subject, onto)) or _generic(t.object, onto):
            reason = "generic_name"
        elif (s_type != DOCUMENT and not _grounded(t.subject, text_lower, text_words)) or \
                not _grounded(t.object, text_lower, text_words):
            reason = "entity_not_in_text"
        elif _ws(t.subject).lower() == _ws(t.object).lower():
            reason = "self_loop"
        elif (span := _find_evidence(t.evidence, text, spans)) is None:
            reason = "evidence_not_in_text"
        else:
            reason = None
        if reason:
            discarded.append({"item": t.model_dump(), "reason": reason})
            continue

        subject = doc_id if s_type == DOCUMENT else _ws(t.subject)
        key = (subject.lower(), rel, _ws(t.object).lower())
        if key in seen:
            continue
        seen.add(key)
        if s_type != DOCUMENT:
            add_entity(t.subject, s_type)
        add_entity(t.object, o_type)
        triples.append({
            "doc_id": doc_id, "subject": subject, "subject_type": s_type, "relation": rel,
            "object": _ws(t.object), "object_type": o_type,
            "evidence": text[span[0]:span[1]], "span": list(span), "source": "llm",
        })

    # Provenance edge for every entity: Document -mentions-> entity, citing the
    # first sentence that names it (proposal: every edge traceable to source text).
    auto = onto.relations.get(AUTO_RELATION)
    if auto:
        for ent in entities.values():
            key = (doc_id.lower(), AUTO_RELATION, ent["name"].lower())
            if ent["type"] not in auto["target"] or key in seen:
                continue
            seen.add(key)
            span = _sentence_naming(ent["name"], text, spans)
            triples.append({
                "doc_id": doc_id, "subject": doc_id, "subject_type": DOCUMENT,
                "relation": AUTO_RELATION, "object": ent["name"], "object_type": ent["type"],
                "evidence": text[span[0]:span[1]] if span else None,
                "span": list(span) if span else None, "source": "auto",
            })

    return list(entities.values()), triples, discarded


# ==========================================================
# Extraction
# ==========================================================

class Extractor:
    """One ontology + one Ollama model, reused across documents."""

    def __init__(self, ontology_path: str | Path = "ontology/news.yaml",
                 model: str | None = None, max_chars: int = 6000):
        from langchain_ollama import ChatOllama

        self.onto = load_ontology(ontology_path)
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.max_chars = max_chars
        self.system = _system_prompt(self.onto)
        chat = ChatOllama(
            model=self.model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
            num_ctx=8192,
            num_predict=2048,
        )
        # json_schema = Ollama constrained decoding: the reply is always valid JSON
        # of this shape; whether its content fits the ontology is checked afterwards.
        self.llm = chat.with_structured_output(LLMExtraction, method="json_schema",
                                               include_raw=True)

    def __call__(self, doc: dict) -> dict:
        text = doc["text"][: self.max_chars]
        result = {"doc_id": doc["id"], "ontology": self.onto.name, "model": self.model,
                  "truncated": len(doc["text"]) > self.max_chars, "salvaged": False,
                  "entities": [], "triples": [], "discarded": [], "error": None}
        start = time.time()
        try:
            out = self.llm.invoke([
                ("system", self.system),
                ("human", f'Document:\n"""\n{text}\n"""'),
            ])
            raw, error = out["parsed"], out["parsing_error"]
            if raw is None and out.get("raw") is not None:
                raw = _salvage(str(out["raw"].content))
                result["salvaged"] = raw is not None
        except Exception as exc:  # Ollama down, timeout, ...
            raw, error = None, exc
        result["seconds"] = round(time.time() - start, 1)
        if raw is None:
            result["error"] = f"{type(error).__name__}: {error}"
            log.warning("extract %s failed: %s", doc["id"], result["error"])
            return result
        result["entities"], result["triples"], result["discarded"] = validate_extraction(
            raw, doc["id"], text, self.onto
        )
        return result


def extract(docs: list[dict], ontology_path: str | Path = "ontology/news.yaml",
            model: str | None = None) -> list[dict]:
    """LLM triple extraction for each doc, constrained to the ontology. One result per doc:

        {doc_id, ontology, model, seconds, truncated, error,
         entities: [{name, type}],
         triples: [{doc_id, subject, subject_type, relation, object, object_type,
                    evidence, span, source: "llm" | "auto"}],
         discarded: [{item, reason}]}
    """
    from tqdm import tqdm

    extractor = Extractor(ontology_path, model)
    return [extractor(d) for d in tqdm(docs, desc=f"extract ({extractor.model})", unit="doc")]


def normalize(triples: list[dict]) -> list[dict]:
    """Lowercase + alias merge via embedding similarity > 0.9."""
    raise NotImplementedError


def load_neo4j(triples: list[dict]) -> None:
    """MERGE triples into Neo4j using the ontology-compiled schema."""
    raise NotImplementedError


def pagerank_label(cluster_id: str) -> str:
    """Top-3 PageRank entities + top c-TF-IDF term."""
    raise NotImplementedError
