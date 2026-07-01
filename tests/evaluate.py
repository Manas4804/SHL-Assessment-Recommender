#!/usr/bin/env python3
"""
Offline replay evaluator for the SHL Assessment Recommender.

Replays all 10 sample conversations, sends user turns sequentially to the
API, and computes Recall@10 against the expected final shortlist from each
trace.

Usage:
    python -m tests.evaluate [--base-url http://localhost:8000]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

CONV_DIR = Path(__file__).parent.parent / "GenAI_SampleConversations"
CONV_FILES = sorted(CONV_DIR.glob("C*.md"))


# ── Conversation parser ───────────────────────────────────────────────────────

def parse_md(path: Path) -> tuple[list[str], list[str]]:
    """
    Returns (user_turns, expected_names).
    - user_turns: list of raw user message strings in dialogue order.
    - expected_names: product names from the LAST recommendation table in the file.
    """
    text = path.read_text(encoding="utf-8")
    turns = re.split(r"### Turn \d+", text)[1:]

    user_msgs: list[str] = []
    last_names: list[str] = []

    for turn in turns:
        # User message block: lines starting with "> "
        um = re.search(r"\*\*User\*\*\s*\n+((?:>.*\n?)+)", turn)
        if um:
            lines = [l.lstrip("> ").strip() for l in um.group(1).splitlines()]
            user_msgs.append(" ".join(l for l in lines if l))

        # Recommendation table — grab Name column (second pipe-delimited cell)
        if "| # | Name |" in turn:
            names = re.findall(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", turn, re.MULTILINE)
            if names:
                last_names = [n.strip() for n in names]

    return user_msgs, last_names


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post_chat(base_url: str, messages: list[dict], timeout: int = 33) -> dict:
    payload = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(expected: list[str], actual: list[str], k: int = 10) -> float:
    """Fraction of expected items that appear in the top-k actual items."""
    if not expected:
        return 1.0
    expected_lower = {n.lower() for n in expected}
    hits = sum(1 for n in actual[:k] if n.lower() in expected_lower)
    return hits / len(expected_lower)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(base_url: str) -> None:
    recalls: list[float] = []
    errors: list[str] = []

    for path in CONV_FILES:
        print(f"\n{'='*64}")
        print(f"  {path.name}")
        print(f"{'='*64}")

        user_msgs, expected_names = parse_md(path)
        print(f"  Expected ({len(expected_names)}): {expected_names}")

        history: list[dict] = []
        final_recs: list[str] = []
        failed = False

        for i, msg in enumerate(user_msgs, 1):
            history.append({"role": "user", "content": msg})
            print(f"\n  [Turn {i}] User: {msg[:100]}")

            try:
                t0 = time.time()
                resp = post_chat(base_url, history)
                elapsed = time.time() - t0
            except Exception as e:
                print(f"  ERROR calling API: {e}")
                errors.append(f"{path.name} turn {i}: {e}")
                failed = True
                break

            recs = resp.get("recommendations", [])
            reply_preview = resp.get("reply", "")[:120].replace("\n", " ")
            eoc = resp.get("end_of_conversation", False)

            print(f"  [Turn {i}] Agent ({elapsed:.1f}s): {reply_preview}")

            if recs:
                names = [r["name"] for r in recs]
                final_recs = names
                print(f"  Recommendations: {names}")
            else:
                print(f"  Recommendations: []")

            history.append({"role": "assistant", "content": resp["reply"]})

            if eoc:
                print(f"  → end_of_conversation=true")
                break

        if not failed:
            r = recall_at_k(expected_names, final_recs)
            recalls.append(r)
            hits = [n for n in final_recs if n.lower() in {e.lower() for e in expected_names}]
            print(f"\n  Recall@10 = {r:.2f}  (hits: {hits})")
        else:
            recalls.append(0.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    mean = sum(recalls) / len(recalls) if recalls else 0
    print(f"\n{'='*64}")
    print(f"  RESULTS ACROSS {len(recalls)} CONVERSATIONS")
    print(f"{'='*64}")
    for path, r in zip(CONV_FILES, recalls):
        bar = "█" * int(r * 20)
        print(f"  {path.name}  {bar:<20}  {r:.2f}")
    print(f"{'='*64}")
    print(f"  Mean Recall@10:  {mean:.3f}")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e}")
    print()

    sys.exit(0 if mean >= 0.5 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay evaluator for SHL recommender")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running service")
    args = parser.parse_args()
    run(args.base_url)
