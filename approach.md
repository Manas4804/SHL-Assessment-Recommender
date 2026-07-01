# SHL Assessment Recommender — Approach Document

## Problem Decomposition

The task is to build a stateless conversational agent that maps vague hiring intent to a grounded shortlist from a 377-item SHL catalog. Three sub-problems need to be solved simultaneously: (1) retrieving the right candidates from the catalog given partial and evolving information, (2) conducting a multi-turn dialogue that clarifies intent without over-asking, and (3) guaranteeing that every URL and product name in the response is verifiably from the catalog — no hallucinations allowed.

---

## Architecture: Two-Stage RAG + Post-Processing

### Stage 1 — Retrieval (`retriever.py`, `catalog.py`)

A pure-Python TF-IDF index is built over all 377 products at startup (no external vector store). Each document is indexed on name + description + keys + job levels. The query is the full dialogue history concatenated — both user and assistant turns — which ensures that context from earlier turns influences retrieval.

Three retrieval sources are composed per call:

- **Core products (always seeded):** OPQ32r, Verify G+, DSI, Graduate Scenarios, Entry Level Customer Serv. These appear in roughly 80% of the sample conversations and must always be in the LLM's context window regardless of query.
- **Acronym and substring matching:** Short uppercase tokens in the user's query (OPQ, SVAR, DSI, G+) are matched against product name tokens. This handles the common case where users reference products by acronym.
- **TF-IDF top-35:** Captures semantically relevant products that haven't been named explicitly.

The catalog JSON contained embedded control characters (scraping artifact from newlines inside string values). Handled with `json.loads(strict=False)` after stripping `\r\n`.

### Stage 2 — LLM (`llm_client.py`)

**Model:** Groq `llama-3.3-70b-versatile`. Chosen for the free tier and fast inference (~1–2 s per call), well within the 30 s timeout requirement.

The system prompt encodes:
- **Scope rules:** Refuse legal/compliance questions, general hiring advice, non-SHL topics, and prompt injection attempts.
- **Clarification discipline:** Ask at most 1–2 targeted questions before recommending. Do not recommend on turn 1 if intent is vague.
- **Domain knowledge:** OPQ32r is included in virtually every professional/leadership battery; Verify G+ for graduate and above; DSI for safety-critical or trust-sensitive roles; SVAR + Contact Center Sim for high-volume contact centre hiring.
- **Structured output:** The LLM returns only JSON `{reply, recommended_names[], end_of_conversation}`. `response_format={"type": "json_object"}` is set so Groq enforces valid JSON structure. A regex-based fallback handles any edge cases where the model wraps output in a code fence.

### Stage 3 — Post-Processing (`agent.py`)

The LLM returns product names as strings. These are resolved to catalog entries using a three-pass fuzzy matcher: exact case-insensitive → substring containment → word-overlap (≥ 2 shared non-trivial words). Any name that does not resolve to a catalog entry is silently dropped. This is the hallucination guard: the `url` and `test_type` in the API response always come from the catalog, never from the LLM's weights. `test_type` is computed deterministically from the product's `keys` field using a fixed letter map (A/B/C/D/K/P/S).

---

## What Didn't Work

1. **Initial prompt without domain guidance:** The LLM excluded OPQ32r from leadership scenarios, recommending niche products instead. Fixed by injecting explicit domain knowledge about when each flagship product applies.
2. **Extracting recommendations from markdown tables:** An early approach asked the LLM to format a markdown table and then parsed product names from it. Table formatting was inconsistent across turns. Replaced with a separate `recommended_names` JSON list that is only set when the LLM has committed to a shortlist.
3. **Over-clarification:** Without an explicit turn budget in the prompt, the LLM asked up to 3–4 clarifying questions before recommending. Fixed by noting the 8-turn cap in the prompt and instructing it to converge within 2 turns.

---

## Evaluation Approach

A replay harness (`tests/evaluate.py`) sends the user turns from each of the 10 public traces sequentially to the API, collects the final recommendation list, and computes Recall@10 against the expected shortlist parsed from the trace file. Behavior probes (out-of-scope refusal, vague-query clarification, prompt injection, `end_of_conversation` detection) were verified manually via curl.

---

## AI Tools Used

Claude Code (Anthropic) was used as an agentic coding assistant for: initial project scaffolding, TF-IDF implementation, system prompt iteration, and debugging the JSON catalog parsing issue. All design decisions (retrieval strategy, post-processing logic, prompt structure) were reviewed and understood before implementation.
