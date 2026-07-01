"""
Orchestrates the two-stage pipeline:
  Stage 1 – Retriever produces candidate products from the catalog.
  Stage 2 – LLM generates a reply and a list of recommended product names.
  Stage 3 – Post-processor maps LLM names → verified catalog entries,
             enforcing catalog-only URLs and deterministic test_type values.
"""
from .catalog import get_catalog
from .llm_client import call_llm
from .models import ChatResponse, Recommendation
from .retriever import retrieve_candidates


def process_chat(messages: list[dict]) -> ChatResponse:
    catalog = get_catalog()

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    candidates = retrieve_candidates(messages)

    # ── Stage 2: LLM ─────────────────────────────────────────────────────────
    llm_out = call_llm(messages, candidates)

    reply: str = llm_out["reply"]
    raw_names: list[str] = llm_out["recommended_names"]
    end: bool = llm_out["end_of_conversation"]

    # ── Stage 3: Post-process ─────────────────────────────────────────────────
    # Map each LLM-returned name to a verified catalog entry.
    # Items that cannot be matched are silently dropped (hallucination guard).
    seen_ids: set[str] = set()
    recommendations: list[Recommendation] = []

    for name in raw_names:
        if len(recommendations) >= 10:
            break
        product = catalog.find_by_name_fuzzy(name)
        if product is None:
            continue
        eid = product["entity_id"]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        recommendations.append(
            Recommendation(
                name=product["name"],
                url=product["link"],
                test_type=product["test_type"],
            )
        )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end,
    )
