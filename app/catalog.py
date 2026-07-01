import json
import re
from pathlib import Path

# Maps catalog key names to single-letter test type codes.
# "Assessment Exercises" has no letter in the standard scheme and is skipped.
_KEY_TO_LETTER: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Letters in the canonical display order used in SHL tables.
_LETTER_ORDER = "ABCDKPS"

_CATALOG_PATH = Path(__file__).parent.parent / "shl_product_catalog.json"


def _compute_test_type(keys: list[str]) -> str:
    letters = {_KEY_TO_LETTER[k] for k in keys if k in _KEY_TO_LETTER}
    ordered = [l for l in _LETTER_ORDER if l in letters]
    return ",".join(ordered) if ordered else "K"


def _normalize_name(name: str) -> str:
    """Collapse any whitespace / newline artefacts from scraping."""
    return re.sub(r"\s+", " ", name).strip()


class Catalog:
    def __init__(self, path: Path = _CATALOG_PATH) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # The scraper left raw control characters (newlines, tabs) inside JSON
        # string values. json.loads strict=False accepts them; we also strip
        # the most common offenders explicitly.
        text = text.replace("\r\n", " ").replace("\r", " ")
        raw: list[dict] = json.loads(text, strict=False)

        self.products: dict[str, dict] = {}
        for item in raw:
            item = dict(item)
            item["name"] = _normalize_name(item["name"])
            item["test_type"] = _compute_test_type(item.get("keys", []))
            self.products[item["entity_id"]] = item

        # Lowercase-name → product for O(1) exact lookups.
        self._by_name: dict[str, dict] = {
            p["name"].lower(): p for p in self.products.values()
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict]:
        return list(self.products.values())

    def find_by_name(self, name: str) -> dict | None:
        return self._by_name.get(name.lower())

    def find_by_name_fuzzy(self, name: str) -> dict | None:
        """
        Try to match a name returned by the LLM to a real catalog entry.
        Sequence: exact → substring → word-overlap (≥2 shared words).
        """
        cleaned = _normalize_name(name)

        # 1. Exact (case-insensitive)
        hit = self._by_name.get(cleaned.lower())
        if hit:
            return hit

        # 2. One side is a substring of the other
        needle = cleaned.lower()
        for catalog_name, product in self._by_name.items():
            if needle in catalog_name or catalog_name in needle:
                return product

        # 3. Word overlap – at least 2 shared non-trivial words
        needle_words = set(w for w in needle.split() if len(w) > 2)
        best: dict | None = None
        best_score = 1  # require at least 2 to match
        for catalog_name, product in self._by_name.items():
            catalog_words = set(w for w in catalog_name.split() if len(w) > 2)
            score = len(needle_words & catalog_words)
            if score > best_score:
                best_score = score
                best = product

        return best


# Module-level singleton – loaded once at startup.
_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog
