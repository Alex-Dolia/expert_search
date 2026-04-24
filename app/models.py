"""
models.py — Pydantic request/response models
=============================================
All API inputs/outputs are validated and documented here.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MatchExplanation(BaseModel):
    summary:     str = Field(description="One-sentence summary of why this expert matched")
    signals:     list[str] = Field(description="List of scoring signals (positive/negative)")
    top_factor:  str = Field(description="The single most impactful scoring signal")


class ExpertResult(BaseModel):
    candidate_id:         str
    first_name:           str
    last_name:            str
    email:                Optional[str] = None
    headline:             Optional[str] = None
    current_title:        Optional[str] = None
    years_of_experience:  Optional[int] = None
    city_name:            Optional[str] = None
    country_name:         Optional[str] = None
    skills:               list[str] = Field(default_factory=list)
    industries:           list[str] = Field(default_factory=list)
    companies:            list[str] = Field(default_factory=list)
    languages:            list[str] = Field(default_factory=list)
    degrees:              list[str] = Field(default_factory=list)
    vector_score:         float = Field(description="Raw cosine similarity (0–1)")
    relevance_score:      float = Field(description="Final re-ranked score (0–1)")
    match_explanation:    MatchExplanation


# ─────────────────────────────────────────────────────────────────────────────
# /chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        description="Natural language search query for experts",
        examples=[
            "Find me regulatory affairs experts with experience in the pharmaceutical industry in the Middle East.",
            "Senior data scientists in Saudi Arabia with 10+ years of experience",
        ],
        min_length=3,
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Pass a previous conversation_id to enable follow-up queries",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of experts to return",
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Pinecone metadata filters, e.g. {\"country_name\": {\"$eq\": \"Saudi Arabia\"}}",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    query:           str
    results:         list[ExpertResult]
    total_returned:  int
    answer_summary:  str = Field(description="LLM-generated natural language summary of results")
    sql_answer:      Optional[str] = Field(
        default=None,
        description="If the query has an analytical component, the Text-to-SQL agent answer is included here",
    )


# ─────────────────────────────────────────────────────────────────────────────
# /ingest
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of candidates to embed and upsert per batch",
    )
    force_recreate: bool = Field(
        default=False,
        description="If true, delete and recreate the Pinecone index",
    )


class IngestResponse(BaseModel):
    total_upserted: int
    index_name:     str
    dimension:      int
    message:        str


# ─────────────────────────────────────────────────────────────────────────────
# /conversations
# ─────────────────────────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    session_id:  str
    turn_count:  int
    last_active: float


class ConversationsResponse(BaseModel):
    sessions: list[ConversationSummary]


# ─────────────────────────────────────────────────────────────────────────────
# /expert/{id}
# ─────────────────────────────────────────────────────────────────────────────

class ExpertProfileResponse(BaseModel):
    candidate_id:        str
    first_name:          str
    last_name:           str
    email:               Optional[str] = None
    headline:            Optional[str] = None
    years_of_experience: Optional[int] = None
    city_name:           Optional[str] = None
    country_name:        Optional[str] = None
    skills:              list[str] = Field(default_factory=list)
    industries:          list[str] = Field(default_factory=list)
    companies:           list[str] = Field(default_factory=list)
    languages:           list[str] = Field(default_factory=list)
    work_experiences:    list[dict[str, Any]] = Field(default_factory=list)
    education:           list[dict[str, Any]] = Field(default_factory=list)
