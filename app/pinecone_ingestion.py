"""
pinecone_ingestion.py — Expert Profile Ingestion into Pinecone
==============================================================
Reads candidate data from PostgreSQL, builds rich text chunks per expert,
generates embeddings via OpenAI-compatible API, and upserts into Pinecone.

Embedding strategy (justified in README):
  - One vector per candidate representing their FULL professional profile.
  - Text includes: headline, skills, work experience (titles + companies + industries),
    education (degree + field + institution), languages, and location.
  - Metadata stores ALL structured fields so re-ranking can use them directly
    without a second DB lookup.

Why one-vector-per-candidate (not per-field):
  - Expert search queries are holistic ("regulatory affairs expert in pharma, ME")
  - We want a single cosine distance to rank against, not multi-field fusion noise.
  - Structured re-ranking on top of vector retrieval handles field-specific weighting.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_list(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _safe(v: Any, fallback: str = "") -> str:
    return str(v).strip() if v is not None else fallback


# ─────────────────────────────────────────────────────────────────────────────
# Profile builder
# ─────────────────────────────────────────────────────────────────────────────

def build_profile_text(row: dict) -> str:
    """
    Compose a single rich-text string for a candidate that captures all
    semantically meaningful fields the embedding model should encode.
    """
    parts: list[str] = []

    name = f"{_safe(row.get('first_name'))} {_safe(row.get('last_name'))}".strip()
    if name:
        parts.append(f"Name: {name}")

    if row.get("headline"):
        parts.append(f"Professional Summary: {_safe(row['headline'])}")

    yoe = row.get("years_of_experience")
    if yoe is not None:
        parts.append(f"Years of Experience: {yoe}")

    if row.get("city_name") or row.get("country_name"):
        loc = ", ".join(filter(None, [_safe(row.get("city_name")), _safe(row.get("country_name"))]))
        parts.append(f"Location: {loc}")

    if row.get("skills"):
        parts.append(f"Skills: {', '.join(row['skills'])}")

    if row.get("work_experiences"):
        exp_lines = []
        for w in row["work_experiences"]:
            line = f"{_safe(w.get('job_title'))} at {_safe(w.get('company_name'))}"
            if w.get("industry"):
                line += f" ({_safe(w['industry'])})"
            if w.get("description"):
                line += f" — {_safe(w['description'])[:200]}"
            exp_lines.append(line)
        parts.append("Work Experience:\n" + "\n".join(f"  • {e}" for e in exp_lines))

    if row.get("education"):
        edu_lines = []
        for e in row["education"]:
            line = f"{_safe(e.get('degree'))} in {_safe(e.get('field_of_study'))} from {_safe(e.get('institution'))}"
            edu_lines.append(line)
        parts.append("Education:\n" + "\n".join(f"  • {e}" for e in edu_lines))

    if row.get("languages"):
        lang_lines = [f"{l['name']} ({l.get('proficiency', '')})" for l in row["languages"]]
        parts.append(f"Languages: {', '.join(lang_lines)}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_candidates(db_url: str) -> list[dict]:
    """
    Pull full enriched candidate profiles from PostgreSQL.
    Joins across all relevant tables to produce one dict per candidate.
    """
    from db import PostgreSQLClient

    with PostgreSQLClient(db_url) as client:
        # ── Base candidates + location ────────────────────────────────────
        candidates = client.query("""
            SELECT
                c.id,
                c.first_name,
                c.last_name,
                c.email,
                c.headline,
                c.years_of_experience,
                c.gender,
                ci.name  AS city_name,
                co.name  AS country_name,
                co.code  AS country_code
            FROM candidates c
            LEFT JOIN cities   ci ON ci.id = c.city_id
            LEFT JOIN countries co ON co.id = ci.country_id
        """)
        logger.info("Loaded %d candidates", len(candidates))

        # ── Skills ───────────────────────────────────────────────────────
        skills_rows = client.query("""
            SELECT cs.candidate_id, s.name AS skill_name
            FROM candidate_skills cs
            JOIN skills s ON s.id = cs.skill_id
        """)
        skills_map: dict[str, list[str]] = {}
        for r in skills_rows:
            skills_map.setdefault(str(r["candidate_id"]), []).append(r["skill_name"])

        # ── Work experience ───────────────────────────────────────────────
        work_rows = client.query("""
            SELECT
                we.candidate_id,
                we.job_title,
                we.start_date,
                we.end_date,
                we.is_current,
                we.description,
                co.name     AS company_name,
                co.industry AS industry
            FROM work_experience we
            LEFT JOIN companies co ON co.id = we.company_id
            ORDER BY we.start_date DESC NULLS LAST
        """)
        work_map: dict[str, list[dict]] = {}
        for r in work_rows:
            work_map.setdefault(str(r["candidate_id"]), []).append(dict(r))

        # ── Education ────────────────────────────────────────────────────
        edu_rows = client.query("""
            SELECT
                e.candidate_id,
                d.name  AS degree,
                f.name  AS field_of_study,
                i.name  AS institution,
                e.graduation_year
            FROM education e
            LEFT JOIN degrees         d ON d.id = e.degree_id
            LEFT JOIN fields_of_study f ON f.id = e.field_of_study_id
            LEFT JOIN institutions    i ON i.id = e.institution_id
        """)
        edu_map: dict[str, list[dict]] = {}
        for r in edu_rows:
            edu_map.setdefault(str(r["candidate_id"]), []).append(dict(r))

        # ── Languages ────────────────────────────────────────────────────
        lang_rows = client.query("""
            SELECT
                cl.candidate_id,
                l.name  AS name,
                pl.name AS proficiency
            FROM candidate_languages cl
            JOIN languages         l  ON l.id  = cl.language_id
            JOIN proficiency_levels pl ON pl.id = cl.proficiency_level_id
        """)
        lang_map: dict[str, list[dict]] = {}
        for r in lang_rows:
            lang_map.setdefault(str(r["candidate_id"]), []).append(dict(r))

    # ── Merge ──────────────────────────────────────────────────────────────
    enriched = []
    for c in candidates:
        cid = str(c["id"])
        row = dict(c)
        row["skills"]           = skills_map.get(cid, [])
        row["work_experiences"] = work_map.get(cid, [])
        row["education"]        = edu_map.get(cid, [])
        row["languages"]        = lang_map.get(cid, [])

        # Derived fields for metadata (used in re-ranking)
        row["industries"] = list({
            w.get("industry", "") for w in row["work_experiences"]
            if w.get("industry")
        })
        row["companies"] = list({
            w.get("company_name", "") for w in row["work_experiences"]
            if w.get("company_name")
        })
        row["current_title"] = next(
            (w["job_title"] for w in row["work_experiences"] if w.get("is_current")),
            None,
        )
        row["degrees"] = [e.get("degree", "") for e in row["education"] if e.get("degree")]
        enriched.append(row)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts: list[str], api_key: str, base_url: str, model: str) -> list[list[float]]:
    """
    Generate embeddings using the OpenAI-compatible endpoint (OpenRouter or direct).
    Falls back gracefully and retries on rate limits.
    """
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    all_embeddings: list[list[float]] = []

    for batch in _chunk_list(texts, 100):
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                all_embeddings.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** attempt * 2
                    logger.warning("Embedding error (attempt %d): %s — retrying in %ds", attempt + 1, exc, wait)
                    time.sleep(wait)
                else:
                    raise

    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone upsert
# ─────────────────────────────────────────────────────────────────────────────

def build_pinecone_metadata(row: dict) -> dict:
    """
    Flatten the enriched row into Pinecone-compatible metadata (strings/numbers/lists only).
    """
    return {
        "candidate_id":       str(row["id"]),
        "first_name":         _safe(row.get("first_name")),
        "last_name":          _safe(row.get("last_name")),
        "email":              _safe(row.get("email")),
        "headline":           _safe(row.get("headline"))[:500],
        "years_of_experience": int(row["years_of_experience"]) if row.get("years_of_experience") is not None else -1,
        "city_name":          _safe(row.get("city_name")),
        "country_name":       _safe(row.get("country_name")),
        "country_code":       _safe(row.get("country_code")),
        "skills":             row.get("skills", [])[:30],
        "industries":         row.get("industries", [])[:10],
        "companies":          row.get("companies", [])[:10],
        "current_title":      _safe(row.get("current_title")),
        "degrees":            row.get("degrees", [])[:5],
        "languages":          [l["name"] for l in row.get("languages", [])][:10],
        "gender":             _safe(row.get("gender")),
        "profile_text":       build_profile_text(row)[:1000],
    }


def ingest_to_pinecone(
    candidates: list[dict],
    index_name: str,
    pinecone_api_key: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Main ingestion pipeline:
    1. Build profile text for each candidate
    2. Batch-embed all texts
    3. Upsert vectors + metadata into Pinecone

    Returns summary dict.
    """
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=pinecone_api_key)

    # ── Determine embedding dimension from a test call ──────────────────────
    logger.info("Determining embedding dimension...")
    test_emb = get_embeddings(["test"], embedding_api_key, embedding_base_url, embedding_model)
    dimension = len(test_emb[0])
    logger.info("Embedding dimension: %d", dimension)

    # ── Create or connect to index ───────────────────────────────────────────
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d)", index_name, dimension)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        time.sleep(10)
    else:
        logger.info("Index '%s' already exists — upserting", index_name)

    index = pc.Index(index_name)

    # ── Embed + upsert in batches ────────────────────────────────────────────
    total_upserted = 0
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        texts = [build_profile_text(c) for c in batch]

        logger.info("Embedding batch %d–%d...", i, i + len(batch))
        embeddings = get_embeddings(texts, embedding_api_key, embedding_base_url, embedding_model)

        vectors = []
        for cand, emb in zip(batch, embeddings):
            meta = build_pinecone_metadata(cand)
            vectors.append({
                "id":       str(cand["id"]),
                "values":   emb,
                "metadata": meta,
            })

        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        logger.info("Upserted %d vectors (total so far: %d)", len(vectors), total_upserted)

    stats = index.describe_index_stats()
    logger.info("Ingestion complete. Index stats: %s", stats)

    # Pinecone returns a protobuf-style object — safely extract what we need
    # dict(stats) raises TypeError; use getattr instead
    try:
        stats_dict = {
            "total_vector_count": getattr(stats, "total_vector_count", 0),
            "dimension":          getattr(stats, "dimension", dimension),
            "namespaces":         str(getattr(stats, "namespaces", {})),
        }
    except Exception:
        stats_dict = {"total_vector_count": total_upserted}

    return {
        "total_upserted": total_upserted,
        "index_name":     index_name,
        "dimension":      dimension,
        "index_stats":    stats_dict,
    }
