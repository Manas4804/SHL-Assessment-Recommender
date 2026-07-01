"""
Stage-1 retrieval: builds a TF-IDF index over the catalog (pure Python, no
external deps) and composes a candidate set from three sources:
  1. Core products that are always relevant (OPQ32r, Verify G+, DSI, …)
  2. Acronym / substring hits against the full dialogue history
  3. Top-N TF-IDF results
"""
import math
import re
from collections import Counter

from .catalog import Catalog, get_catalog

# ── Stop-word list ────────────────────────────────────────────────────────────
_STOP = frozenset(
    """a an the is are was were be been being have has had do does did will would
    could should may might shall can need i me my we our you your he she it they
    them their this that these those and but or nor so yet both either neither not
    only same than too very just because if while although though since unless until
    when where which who whom what how we're we'll i'm i'll we've they're i'd we'd
    use used to for of in on with at by from as into through during before after
    above below up down out off over under again further then once""".split()
)

# ── Core products always seeded into every candidate set ─────────────────────
_CORE_NAMES = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "Dependability and Safety Instrument (DSI)",
    "Graduate Scenarios",
    "Entry Level Customer Serv-Retail & Contact Center",
]


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9+#.]*", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _product_text(p: dict) -> str:
    return " ".join(
        [
            p.get("name", ""),
            p.get("description", ""),
            " ".join(p.get("keys", [])),
            " ".join(p.get("job_levels", [])),
        ]
    )


class _TFIDFIndex:
    def __init__(self, products: list[dict]) -> None:
        self._products = products
        n = len(products)

        doc_tokens: list[list[str]] = [_tokenize(_product_text(p)) for p in products]
        doc_tfs: list[Counter] = [Counter(toks) for toks in doc_tokens]

        df: Counter = Counter()
        for toks in doc_tokens:
            for t in set(toks):
                df[t] += 1

        self._idf: dict[str, float] = {
            t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()
        }
        self._doc_tfs = doc_tfs

    def search(self, query: str, top_k: int = 35) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._doc_tfs):
            score = sum(tf[t] * self._idf.get(t, 1.0) for t in q_tokens)
            scores.append((score, idx))

        scores.sort(reverse=True)
        return [
            self._products[idx]
            for score, idx in scores[:top_k]
            if score > 0
        ]


# ── Module-level singletons ───────────────────────────────────────────────────
_index: _TFIDFIndex | None = None
_core_ids: set[str] = set()


def _get_index(catalog: Catalog) -> _TFIDFIndex:
    global _index, _core_ids
    if _index is None:
        products = catalog.get_all()
        _index = _TFIDFIndex(products)
        for name in _CORE_NAMES:
            hit = catalog.find_by_name(name)
            if hit:
                _core_ids.add(hit["entity_id"])
    return _index


# ── Public API ────────────────────────────────────────────────────────────────

def build_index() -> None:
    """Pre-warm the index at startup so the first /chat call isn't slow."""
    _get_index(get_catalog())


def retrieve_candidates(messages: list[dict], top_k: int = 20) -> list[dict]:
    """
    Return a de-duplicated list of candidate products for the given
    conversation history, drawn from core products + acronym hits + TF-IDF.
    """
    catalog = get_catalog()
    index = _get_index(catalog)

    # Build a single query string from the full dialogue history.
    query = " ".join(m.get("content", "") for m in messages)

    # 1. Core products (always included)
    core = [catalog.products[eid] for eid in _core_ids if eid in catalog.products]

    # 2. Acronym / short-token hits: for each catalog product, check whether
    #    any of its name-tokens (≥2 chars) appears verbatim in the query.
    query_tokens = set(_tokenize(query))
    # Also keep raw short uppercase tokens (OPQ, SVAR, DSI, GSA, …)
    raw_upper = set(re.findall(r'\b[A-Z][A-Z0-9+]{1,4}\b', query))
    raw_lower = {t.lower() for t in raw_upper}
    all_query_tokens = query_tokens | raw_lower

    acronym_hits: list[dict] = []
    for product in catalog.get_all():
        name_tokens = set(_tokenize(product["name"]))
        if name_tokens & all_query_tokens:
            acronym_hits.append(product)

    # 3. TF-IDF top-N
    tfidf_hits = index.search(query, top_k=top_k)

    # Merge – preserve insertion order, deduplicate by entity_id.
    # Hard cap: core products always make it in; rest trimmed to keep the
    # system prompt under ~2 000 tokens (well within the 30 s latency budget).
    MAX_TOTAL = 25
    seen: set[str] = set()
    candidates: list[dict] = []
    for product in core + acronym_hits + tfidf_hits:
        if len(candidates) >= MAX_TOTAL:
            break
        eid = product["entity_id"]
        if eid not in seen:
            seen.add(eid)
            candidates.append(product)

    return candidates
