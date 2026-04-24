"""
pinecone_search.py — Expert Search with Re-ranking & Scoring
=============================================================
Two-stage retrieval:
  Stage 1 — Vector search: retrieve top-K candidates by cosine similarity
             from Pinecone using a query embedding.
  Stage 2 — Structured re-ranking: apply deliberate signal weights
             based on what the query emphasises (seniority, geography,
             function, industry, trajectory).

Why two stages?
  Embeddings capture semantic similarity but treat all signals equally.
  A query like "former CPO at a Saudi petrochemical company" should
  weight seniority + geography + industry more than generic semantic
  similarity would. The re-ranking layer makes those weights explicit
  and adjustable.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query decomposition
# ─────────────────────────────────────────────────────────────────────────────

SENIORITY_KEYWORDS = {
    "c-suite", "cpo", "cto", "cfo", "ceo", "coo", "chief", "vp", "vice president",
    "director", "head of", "senior", "lead", "principal", "partner",
    "junior", "entry-level", "associate", "intern",
}

GEOGRAPHY_KEYWORDS = {
    "saudi", "saudi arabia", "uae", "dubai", "middle east", "mena", "qatar",
    "egypt", "uk", "europe", "usa", "us", "asia",
}

def _extract_signals(query: str) -> dict[str, Any]:
    """
    Heuristically detect which signals the query emphasises.
    Returns a dict of detected signals for use in weight calculation.
    """
    q = query.lower()
    signals: dict[str, Any] = {
        "emphasises_seniority":  any(kw in q for kw in SENIORITY_KEYWORDS),
        "emphasises_geography":  any(kw in q for kw in GEOGRAPHY_KEYWORDS),
        "emphasises_industry":   any(kw in q for kw in ("pharma", "petrochem", "financial", "tech", "healthcare", "energy")),
        "emphasises_language":   any(kw in q for kw in ("arabic", "french", "german", "mandarin", "language")),
        "min_experience":        _extract_min_experience(q),
        "geography_hint":        _extract_geography(q),
        "industry_hint":         _extract_industry(q),
    }
    return signals


def _extract_min_experience(query: str) -> int | None:
    m = re.search(r"(\d+)\+?\s+year", query)
    return int(m.group(1)) if m else None


def _extract_geography(query: str) -> str | None:
    q = query.lower()
    country_hints = {
        "saudi": "Saudi Arabia",
        "uae": "United Arab Emirates",
        "dubai": "United Arab Emirates",
        "qatar": "Qatar",
        "egypt": "Egypt",
        "uk": "United Kingdom",
        "germany": "Germany",
    }
    for kw, country in country_hints.items():
        if kw in q:
            return country
    return None


def _extract_industry(query: str) -> str | None:
    q = query.lower()
    for kw in ("pharma", "petrochem", "financial", "tech", "healthcare", "energy", "telecom"):
        if kw in q:
            return kw
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_structured_score(
    candidate_meta: dict,
    signals: dict,
    vector_score: float,
) -> tuple[float, list[str]]:
    """
    Compute a final relevance score from:
      - vector_score:   raw cosine similarity (0–1)
      - structured signals: geography match, experience threshold, etc.

    Returns (final_score, explanation_bullets).
    """
    score = vector_score  # baseline
    reasons: list[str] = [f"Semantic similarity: {vector_score:.2f}"]

    # ── Geography match ──────────────────────────────────────────────────────
    if signals.get("emphasises_geography"):
        geo_hint = signals.get("geography_hint")
        candidate_country = (candidate_meta.get("country_name") or "").lower()
        if geo_hint and geo_hint.lower() in candidate_country:
            score += 0.20
            reasons.append(f"✓ Location match: {candidate_meta.get('country_name')}")
        elif signals.get("emphasises_geography"):
            score -= 0.10
            reasons.append("✗ Location mismatch — query specifies a region")

    # ── Experience threshold ─────────────────────────────────────────────────
    min_exp = signals.get("min_experience")
    if min_exp is not None:
        yoe = candidate_meta.get("years_of_experience", 0) or 0
        if yoe >= min_exp:
            score += 0.10
            reasons.append(f"✓ Experience: {yoe} years (meets {min_exp}+ requirement)")
        else:
            score -= 0.15
            reasons.append(f"✗ Experience: {yoe} years (below {min_exp}+ requirement)")

    # ── Seniority boost ──────────────────────────────────────────────────────
    if signals.get("emphasises_seniority"):
        title = (candidate_meta.get("current_title") or "").lower()
        headline = (candidate_meta.get("headline") or "").lower()
        senior_terms = {"chief", "cpo", "cto", "cfo", "vp", "director", "head", "senior", "lead", "partner"}
        if any(t in title or t in headline for t in senior_terms):
            score += 0.10
            reasons.append(f"✓ Senior-level role: {candidate_meta.get('current_title', '')}")
        else:
            score -= 0.05
            reasons.append("✗ No senior seniority signal detected")

    # ── Industry match ───────────────────────────────────────────────────────
    if signals.get("emphasises_industry"):
        hint = signals.get("industry_hint", "").lower()
        candidate_industries = [i.lower() for i in candidate_meta.get("industries", [])]
        if any(hint in ind for ind in candidate_industries):
            score += 0.12
            reasons.append(f"✓ Industry match: {hint}")
        else:
            score -= 0.08
            reasons.append(f"✗ No {hint} industry signal detected")

    # ── Language match ───────────────────────────────────────────────────────
    if signals.get("emphasises_language"):
        candidate_langs = [l.lower() for l in candidate_meta.get("languages", [])]
        query_langs = ["arabic", "french", "german", "mandarin"]
        for ql in query_langs:
            if ql in " ".join(candidate_meta.get("languages", [])).lower():
                score += 0.08
                reasons.append(f"✓ Language: speaks {ql.capitalize()}")

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return round(score, 4), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main search function
# ─────────────────────────────────────────────────────────────────────────────

def search_experts(
    query: str,
    pinecone_api_key: str,
    index_name: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    top_k: int = 20,
    return_top_n: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """
    Full expert search pipeline:
      1. Embed the query
      2. Vector search Pinecone (top_k candidates)
      3. Extract signals from the query
      4. Structured re-ranking of retrieved candidates
      5. Return top_n with scores and explanations

    Args:
        query:              Natural language search query
        filters:            Optional Pinecone metadata filters to narrow search
                            e.g. {"country_name": {"$eq": "Saudi Arabia"}}
        top_k:              How many vectors to retrieve from Pinecone
        return_top_n:       How many final results to return after re-ranking
    """
    import openai
    from pinecone import Pinecone

    # ── Step 1: Embed query ──────────────────────────────────────────────────
    oa = openai.OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
    emb_resp = oa.embeddings.create(model=embedding_model, input=[query])
    query_vector = emb_resp.data[0].embedding

    # ── Step 2: Vector search ────────────────────────────────────────────────
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    search_kwargs: dict[str, Any] = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filters:
        search_kwargs["filter"] = filters

    results = index.query(**search_kwargs)
    matches = results.get("matches", [])
    logger.info("Pinecone returned %d matches for query: %s", len(matches), query[:80])

    # ── Step 3: Extract query signals ───────────────────────────────────────
    signals = _extract_signals(query)
    logger.debug("Detected signals: %s", signals)

    # ── Step 4: Re-rank ──────────────────────────────────────────────────────
    scored: list[dict] = []
    for match in matches:
        meta = match.get("metadata", {})
        vector_score = float(match.get("score", 0.0))
        final_score, reasons = compute_structured_score(meta, signals, vector_score)

        scored.append({
            "candidate_id":       meta.get("candidate_id"),
            "first_name":         meta.get("first_name"),
            "last_name":          meta.get("last_name"),
            "email":              meta.get("email"),
            "headline":           meta.get("headline"),
            "current_title":      meta.get("current_title"),
            "years_of_experience": meta.get("years_of_experience"),
            "city_name":          meta.get("city_name"),
            "country_name":       meta.get("country_name"),
            "skills":             meta.get("skills", []),
            "industries":         meta.get("industries", []),
            "companies":          meta.get("companies", []),
            "languages":          meta.get("languages", []),
            "degrees":            meta.get("degrees", []),
            "vector_score":       round(vector_score, 4),
            "relevance_score":    final_score,
            "match_explanation": {
                "summary":   _build_summary(meta, reasons, final_score),
                "signals":   reasons,
                "top_factor": reasons[1] if len(reasons) > 1 else reasons[0],
            },
        })

    # Sort by final score descending
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:return_top_n]


def _build_summary(meta: dict, reasons: list[str], score: float) -> str:
    name = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
    yoe = meta.get("years_of_experience", "?")
    country = meta.get("country_name", "Unknown location")
    title = meta.get("current_title") or meta.get("headline", "")[:80]
    positive = [r for r in reasons if r.startswith("✓")][:2]
    positive_str = "; ".join(positive) if positive else "Semantic relevance"
    return (
        f"{name} — {title} | {yoe} yrs exp | {country}. "
        f"Score {score:.2f}. Key signals: {positive_str}."
    )
