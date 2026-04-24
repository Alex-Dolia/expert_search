"""
routes.py — FastAPI route handlers
===================================
Endpoints:
  POST /ingest          — run Pinecone ingestion pipeline
  POST /chat            — natural-language expert search + scoring
  GET  /health          — health check
  GET  /conversations   — list active sessions
  GET  /expert/{id}     — full profile for a single expert
  POST /query           — pass-through to text-to-SQL agent (analytics queries)
  GET  /schema          — DB schema info
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.models import (
    ChatRequest, ChatResponse,
    IngestRequest, IngestResponse,
    ConversationsResponse, ConversationSummary,
    ExpertProfileResponse, ExpertResult, MatchExplanation,
)
from app.session_store import get_store
from app.schema import TABLE_DESCRIPTIONS, ROW_COUNTS, RELATIONSHIPS

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy singleton: Text-to-SQL agent ───────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from app.agent import TextToSQLAgent
        _agent = TextToSQLAgent()
    return _agent


# ── Config helpers ────────────────────────────────────────────────────────────

def _pinecone_cfg() -> dict:
    return {
        "pinecone_api_key":   os.environ["PINECONE_API_KEY"],
        "index_name":         os.environ.get("PINECONE_INDEX_NAME", "expert-profiles"),
    }


def _embedding_cfg() -> dict:
    # OpenRouter supports embeddings via /api/v1/embeddings — same key as LLM.
    # See: https://openrouter.ai/docs/features/embeddings
    return {
        "embedding_api_key":   os.environ.get("EMBEDDING_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        "embedding_base_url":  os.environ.get("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"),
        "embedding_model":     os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    }


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health():
    """Health check."""
    return {"status": "ok"}


# ── /ingest ───────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(req: IngestRequest):
    """
    Trigger the Pinecone ingestion pipeline.

    Reads all candidate profiles from PostgreSQL, generates embeddings,
    and upserts vectors + metadata into Pinecone.

    This is idempotent — running it again will overwrite existing vectors.
    Expect ~2–5 minutes for a full dataset depending on size.
    """
    from app.pinecone_ingestion import load_all_candidates, ingest_to_pinecone
    from pinecone import Pinecone

    pc_cfg = _pinecone_cfg()
    emb_cfg = _embedding_cfg()
    db_url = os.environ["DATABASE_URL"]

    try:
        # Optionally delete + recreate index
        if req.force_recreate:
            pc = Pinecone(api_key=pc_cfg["pinecone_api_key"])
            existing = [idx.name for idx in pc.list_indexes()]
            if pc_cfg["index_name"] in existing:
                logger.info("Deleting index '%s' for recreation", pc_cfg["index_name"])
                pc.delete_index(pc_cfg["index_name"])

        logger.info("Loading candidates from PostgreSQL...")
        candidates = load_all_candidates(db_url)

        logger.info("Starting Pinecone ingestion for %d candidates...", len(candidates))
        result = ingest_to_pinecone(
            candidates=candidates,
            index_name=pc_cfg["index_name"],
            batch_size=req.batch_size,
            **pc_cfg,
            **emb_cfg,
        )

        return IngestResponse(
            total_upserted=result["total_upserted"],
            index_name=result["index_name"],
            dimension=result["dimension"],
            message=f"Successfully ingested {result['total_upserted']} expert profiles into Pinecone.",
        )

    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /chat ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Expert Search"])
async def chat(req: ChatRequest):
    """
    Natural-language expert search with re-ranking and explainable scoring.

    Combines:
    1. **Pinecone vector search** — semantic similarity over expert profiles
    2. **Structured re-ranking** — signal-weighted scoring (geography, seniority, industry, experience)
    3. **Text-to-SQL fallback** — for analytical/count queries
    4. **Conversational context** — follow-up queries reference prior results

    Example queries:
    - "Find me regulatory affairs experts with pharma experience in the Middle East"
    - "Senior data scientists in Saudi Arabia with 10+ years"
    - "Filter those to people who speak Arabic" (follow-up)
    """
    from app.pinecone_search import search_experts

    store = get_store()
    session = store.get_or_create(req.conversation_id)

    # ── Enrich query with conversation context ──────────────────────────────
    context = session.to_context_string()
    enriched_query = req.query
    if context and session.last_results:
        # If this looks like a follow-up (short query referencing "those", "them", etc.)
        followup_indicators = ["those", "them", "filter", "narrow", "only", "same", "above"]
        if any(ind in req.query.lower() for ind in followup_indicators) and session.last_results:
            # Extract additional filter from the prior result context
            enriched_query = f"{context}\n\nFollow-up: {req.query}"
            logger.info("Follow-up query detected; enriching with context")

    # ── Pinecone vector search + re-ranking ─────────────────────────────────
    pc_cfg = _pinecone_cfg()
    emb_cfg = _embedding_cfg()

    try:
        raw_results = search_experts(
            query=enriched_query,
            top_k=min(req.top_k * 3, 50),   # retrieve 3× for re-ranking
            return_top_n=req.top_k,
            filters=req.filters,
            **pc_cfg,
            **emb_cfg,
        )
    except Exception as exc:
        logger.exception("Vector search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Vector search error: {exc}")

    # ── Convert to response models ───────────────────────────────────────────
    expert_results = []
    for r in raw_results:
        expert_results.append(ExpertResult(
            candidate_id=r["candidate_id"],
            first_name=r["first_name"],
            last_name=r["last_name"],
            email=r.get("email"),
            headline=r.get("headline"),
            current_title=r.get("current_title"),
            years_of_experience=r.get("years_of_experience"),
            city_name=r.get("city_name"),
            country_name=r.get("country_name"),
            skills=r.get("skills", []),
            industries=r.get("industries", []),
            companies=r.get("companies", []),
            languages=r.get("languages", []),
            degrees=r.get("degrees", []),
            vector_score=r["vector_score"],
            relevance_score=r["relevance_score"],
            match_explanation=MatchExplanation(**r["match_explanation"]),
        ))

    # ── Text-to-SQL: run in parallel for analytical queries ──────────────────
    sql_answer: str | None = None
    analytical_triggers = ["how many", "count", "average", "distribution", "list all", "top "]
    if any(t in req.query.lower() for t in analytical_triggers):
        try:
            agent = get_agent()
            sql_result = agent.query(req.query)
            sql_answer = sql_result.get("answer")
        except Exception as exc:
            logger.warning("SQL agent failed (non-fatal): %s", exc)

    # ── Build summary ────────────────────────────────────────────────────────
    answer_summary = _build_answer_summary(req.query, expert_results, sql_answer)

    # ── Update session ───────────────────────────────────────────────────────
    session.add_turn("user", req.query)
    session.add_turn("assistant", answer_summary)
    session.last_results = raw_results
    session.last_query = req.query

    return ChatResponse(
        conversation_id=session.session_id,
        query=req.query,
        results=expert_results,
        total_returned=len(expert_results),
        answer_summary=answer_summary,
        sql_answer=sql_answer,
    )


def _build_answer_summary(query: str, results: list[ExpertResult], sql_answer: str | None) -> str:
    if not results:
        return f"No matching experts found for: '{query}'"
    top = results[0]
    top_name = f"{top.first_name} {top.last_name}"
    top_score = top.relevance_score
    n = len(results)
    summary = (
        f"Found {n} matching expert{'s' if n != 1 else ''} for your query. "
        f"Top match: {top_name} (score {top_score:.2f}) — {top.match_explanation.summary}"
    )
    if sql_answer:
        summary += f"\n\nAnalytical insight: {sql_answer}"
    return summary


# ── /conversations ────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=ConversationsResponse, tags=["Conversations"])
def list_conversations():
    """List all active conversation sessions."""
    store = get_store()
    return ConversationsResponse(
        sessions=[ConversationSummary(**s) for s in store.list_sessions()]
    )


# ── /expert/{candidate_id} ────────────────────────────────────────────────────

@router.get("/expert/{candidate_id}", response_model=ExpertProfileResponse, tags=["Expert Search"])
def get_expert(candidate_id: str):
    """
    Retrieve the full profile for a specific expert by their candidate ID.
    Fetches directly from PostgreSQL for complete, up-to-date data.
    """
    from app.pinecone_ingestion import load_all_candidates
    import os

    db_url = os.environ["DATABASE_URL"]
    from db import PostgreSQLClient
    with PostgreSQLClient(db_url) as client:
        rows = client.query("""
            SELECT
                c.id, c.first_name, c.last_name, c.email, c.headline,
                c.years_of_experience, c.gender,
                ci.name AS city_name, co.name AS country_name
            FROM candidates c
            LEFT JOIN cities    ci ON ci.id = c.city_id
            LEFT JOIN countries co ON co.id = ci.country_id
            WHERE c.id = %s
        """, (candidate_id,))

        if not rows:
            raise HTTPException(status_code=404, detail=f"Expert {candidate_id} not found")

        c = rows[0]

        skills = [r["skill_name"] for r in client.query("""
            SELECT s.name AS skill_name FROM candidate_skills cs
            JOIN skills s ON s.id = cs.skill_id WHERE cs.candidate_id = %s
        """, (candidate_id,))]

        industries = list({r["industry"] for r in client.query("""
            SELECT co.industry FROM work_experience we
            JOIN companies co ON co.id = we.company_id
            WHERE we.candidate_id = %s AND co.industry IS NOT NULL
        """, (candidate_id,))})

        companies = [r["company_name"] for r in client.query("""
            SELECT co.name AS company_name FROM work_experience we
            JOIN companies co ON co.id = we.company_id WHERE we.candidate_id = %s
        """, (candidate_id,))]

        languages = [r["lang_name"] for r in client.query("""
            SELECT l.name AS lang_name FROM candidate_languages cl
            JOIN languages l ON l.id = cl.language_id WHERE cl.candidate_id = %s
        """, (candidate_id,))]

        work_exp = client.query("""
            SELECT we.job_title, we.start_date, we.end_date, we.is_current,
                   we.description, co.name AS company_name, co.industry
            FROM work_experience we
            LEFT JOIN companies co ON co.id = we.company_id
            WHERE we.candidate_id = %s ORDER BY we.start_date DESC NULLS LAST
        """, (candidate_id,))

        education = client.query("""
            SELECT d.name AS degree, f.name AS field_of_study,
                   i.name AS institution, e.graduation_year
            FROM education e
            LEFT JOIN degrees         d ON d.id = e.degree_id
            LEFT JOIN fields_of_study f ON f.id = e.field_of_study_id
            LEFT JOIN institutions    i ON i.id = e.institution_id
            WHERE e.candidate_id = %s
        """, (candidate_id,))

    return ExpertProfileResponse(
        candidate_id=str(c["id"]),
        first_name=c["first_name"],
        last_name=c["last_name"],
        email=c.get("email"),
        headline=c.get("headline"),
        years_of_experience=c.get("years_of_experience"),
        city_name=c.get("city_name"),
        country_name=c.get("country_name"),
        skills=skills,
        industries=industries,
        companies=companies,
        languages=languages,
        work_experiences=[dict(r) for r in work_exp],
        education=[dict(r) for r in education],
    )


# ── /query (text-to-SQL pass-through) ────────────────────────────────────────

from pydantic import BaseModel as _BM

class _QueryReq(_BM):
    question: str
    include_sql:  bool = True
    include_rows: bool = False
    include_plan: bool = False

class _QueryResp(_BM):
    answer: str
    sql: str | None = None
    rows: list[dict] | None = None
    plan: str | None = None
    row_count: int
    warnings: list[str]
    retry_count: int

@router.post("/query", response_model=_QueryResp, tags=["Text-to-SQL"])
async def sql_query(req: _QueryReq):
    """
    Direct pass-through to the Text-to-SQL LangGraph agent.
    Use this for analytical/aggregate queries (counts, distributions, etc.)
    rather than expert search.
    """
    agent = get_agent()
    try:
        result = agent.query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _QueryResp(
        answer=result["answer"],
        sql=result["sql"] if req.include_sql else None,
        rows=result["rows"] if req.include_rows else None,
        plan=result["plan"] if req.include_plan else None,
        row_count=len(result["rows"]),
        warnings=result["warnings"],
        retry_count=result["retry_count"],
    )


# ── /schema ───────────────────────────────────────────────────────────────────

from pydantic import BaseModel as _BM2
class _SchemaResp(_BM2):
    tables: dict
    row_counts: dict
    relationships: list[str]

@router.get("/schema", response_model=_SchemaResp, tags=["Schema"])
def schema():
    """Return the database schema."""
    return _SchemaResp(
        tables=TABLE_DESCRIPTIONS,
        row_counts=ROW_COUNTS,
        relationships=RELATIONSHIPS,
    )
