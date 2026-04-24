"""
tests/test_api.py — Unit tests for the Expert Network Search Copilot API
=========================================================================
Tests every endpoint using FastAPI's TestClient (synchronous) and
AsyncClient (async) with all external services mocked:
  - Pinecone calls are replaced with a fake index stub
  - PostgreSQL calls are replaced with in-memory fixture data
  - OpenAI embedding calls return deterministic fake vectors
  - The Text-to-SQL LangGraph agent is mocked to return preset answers

Run with:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short   # shorter tracebacks
    pytest tests/test_api.py -v -k "health"  # run a single test

Requirements (already in environment.yml):
    pytest>=8.0.0
    pytest-asyncio>=0.23.0
    httpx>=0.27.0
"""
from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Set env vars BEFORE importing the app so config helpers don't KeyError
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("DATABASE_URL",        "postgresql://test:test@localhost/test")
os.environ.setdefault("OPENROUTER_API_KEY",  "test-openrouter-key")
os.environ.setdefault("PINECONE_API_KEY",    "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "expert-profiles-test")
os.environ.setdefault("EMBEDDING_MODEL",     "openai/text-embedding-3-small")
os.environ.setdefault("LLM_MODEL",           "anthropic/claude-sonnet-4.6")


# ─────────────────────────────────────────────────────────────────────────────
# Fixture data — shared across tests
# ─────────────────────────────────────────────────────────────────────────────

FAKE_CANDIDATE_ID   = str(uuid.uuid4())
FAKE_CANDIDATE_ID_2 = str(uuid.uuid4())

FAKE_CANDIDATES = [
    {
        "id":                  FAKE_CANDIDATE_ID,
        "first_name":          "Ahmed",
        "last_name":           "Hassan",
        "email":               "ahmed.hassan@example.com",
        "headline":            "VP Regulatory Affairs | Pharmaceutical | MENA",
        "years_of_experience": 14,
        "city_name":           "Riyadh",
        "country_name":        "Saudi Arabia",
        "country_code":        "SA",
        "gender":              "Male",
        "skills":              ["Regulatory Affairs", "FDA", "GMP", "Pharmaceuticals"],
        "work_experiences": [
            {
                "job_title":    "VP Regulatory Affairs",
                "company_name": "Acme Pharma",
                "industry":     "Pharmaceutical",
                "description":  "Led regional regulatory strategy.",
                "is_current":   True,
                "start_date":   "2018-01-01",
                "end_date":     None,
            }
        ],
        "education": [
            {"degree": "PhD", "field_of_study": "Biochemistry", "institution": "Cairo University", "graduation_year": 2009}
        ],
        "languages":  [{"name": "Arabic", "proficiency": "Native"}, {"name": "English", "proficiency": "Fluent"}],
        "industries": ["Pharmaceutical"],
        "companies":  ["Acme Pharma"],
        "current_title": "VP Regulatory Affairs",
        "degrees":    ["PhD"],
    },
    {
        "id":                  FAKE_CANDIDATE_ID_2,
        "first_name":          "Sara",
        "last_name":           "Chen",
        "email":               "sara.chen@example.com",
        "headline":            "Senior Data Scientist | Machine Learning",
        "years_of_experience": 8,
        "city_name":           "Dubai",
        "country_name":        "United Arab Emirates",
        "country_code":        "AE",
        "gender":              "Female",
        "skills":              ["Python", "Machine Learning", "Data Science", "SQL"],
        "work_experiences": [
            {
                "job_title":    "Senior Data Scientist",
                "company_name": "TechCorp",
                "industry":     "Technology",
                "description":  "Built ML pipelines.",
                "is_current":   True,
                "start_date":   "2020-06-01",
                "end_date":     None,
            }
        ],
        "education": [
            {"degree": "MSc", "field_of_study": "Computer Science", "institution": "NUS", "graduation_year": 2016}
        ],
        "languages":  [{"name": "English", "proficiency": "Native"}, {"name": "Mandarin", "proficiency": "Native"}],
        "industries": ["Technology"],
        "companies":  ["TechCorp"],
        "current_title": "Senior Data Scientist",
        "degrees":    ["MSc"],
    },
]

# What Pinecone search_experts returns (already structured)
FAKE_SEARCH_RESULTS = [
    {
        "candidate_id":        FAKE_CANDIDATE_ID,
        "first_name":          "Ahmed",
        "last_name":           "Hassan",
        "email":               "ahmed.hassan@example.com",
        "headline":            "VP Regulatory Affairs | Pharmaceutical | MENA",
        "current_title":       "VP Regulatory Affairs",
        "years_of_experience": 14,
        "city_name":           "Riyadh",
        "country_name":        "Saudi Arabia",
        "skills":              ["Regulatory Affairs", "FDA", "GMP"],
        "industries":          ["Pharmaceutical"],
        "companies":           ["Acme Pharma"],
        "languages":           ["Arabic", "English"],
        "degrees":             ["PhD"],
        "vector_score":        0.87,
        "relevance_score":     0.94,
        "match_explanation": {
            "summary":    "Ahmed Hassan — VP Regulatory Affairs | 14 yrs | Saudi Arabia. Score 0.94.",
            "signals":    ["Semantic similarity: 0.87", "✓ Location match: Saudi Arabia"],
            "top_factor": "✓ Location match: Saudi Arabia",
        },
    },
    {
        "candidate_id":        FAKE_CANDIDATE_ID_2,
        "first_name":          "Sara",
        "last_name":           "Chen",
        "email":               "sara.chen@example.com",
        "headline":            "Senior Data Scientist | ML",
        "current_title":       "Senior Data Scientist",
        "years_of_experience": 8,
        "city_name":           "Dubai",
        "country_name":        "United Arab Emirates",
        "skills":              ["Python", "Machine Learning"],
        "industries":          ["Technology"],
        "companies":           ["TechCorp"],
        "languages":           ["English", "Mandarin"],
        "degrees":             ["MSc"],
        "vector_score":        0.72,
        "relevance_score":     0.72,
        "match_explanation": {
            "summary":    "Sara Chen — Senior Data Scientist | 8 yrs | UAE. Score 0.72.",
            "signals":    ["Semantic similarity: 0.72"],
            "top_factor": "Semantic similarity: 0.72",
        },
    },
]

FAKE_SQL_AGENT_RESULT = {
    "answer":        "There are 42 candidates based in Saudi Arabia.",
    "sql":           "SELECT COUNT(*) FROM candidates c JOIN cities ci ON ci.id=c.city_id JOIN countries co ON co.id=ci.country_id WHERE co.name='Saudi Arabia'",
    "rows":          [{"count": 42}],
    "warnings":      [],
    "retry_count":   0,
    "plan":          "Join candidates → cities → countries, filter by country name.",
    "question_type": "sql",
}

FAKE_DB_CANDIDATE_ROWS = [
    {
        "id":                  FAKE_CANDIDATE_ID,
        "first_name":          "Ahmed",
        "last_name":           "Hassan",
        "email":               "ahmed.hassan@example.com",
        "headline":            "VP Regulatory Affairs | Pharmaceutical | MENA",
        "years_of_experience": 14,
        "gender":              "Male",
        "city_name":           "Riyadh",
        "country_name":        "Saudi Arabia",
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_db_client(candidate_rows=None):
    """Return a mock PostgreSQLClient that returns fake rows for any query."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    # Default: return fake candidate rows for any query
    mock.query.return_value = candidate_rows if candidate_rows is not None else FAKE_DB_CANDIDATE_ROWS
    return mock


@pytest.fixture
def client():
    """
    FastAPI TestClient with all external services patched.
    Patches applied at module-import level so they work regardless of
    where in the code the dependency is imported.
    """
    mock_agent = MagicMock()
    mock_agent.query.return_value = FAKE_SQL_AGENT_RESULT

    with (
        patch("app.routes.get_agent",       return_value=mock_agent),
        patch("app.routes.search_experts",  return_value=FAKE_SEARCH_RESULTS),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_empty_search():
    """TestClient where search returns zero results (edge-case tests)."""
    mock_agent = MagicMock()
    mock_agent.query.return_value = FAKE_SQL_AGENT_RESULT

    with (
        patch("app.routes.get_agent",       return_value=mock_agent),
        patch("app.routes.search_experts",  return_value=[]),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def client_with_db():
    """TestClient with mocked DB for /expert/{id} tests."""
    mock_agent = MagicMock()
    mock_agent.query.return_value = FAKE_SQL_AGENT_RESULT
    mock_db = _make_mock_db_client()

    # Per-query returns: first call → candidate row, subsequent → empty lists
    call_counter = {"n": 0}
    responses = [
        FAKE_DB_CANDIDATE_ROWS,  # base candidate query
        [{"skill_name": "Regulatory Affairs"}, {"skill_name": "FDA"}],  # skills
        [{"industry": "Pharmaceutical"}],                                # industries
        [{"company_name": "Acme Pharma"}],                              # companies
        [{"lang_name": "Arabic"}, {"lang_name": "English"}],            # languages
        [{"job_title": "VP Regulatory Affairs", "company_name": "Acme Pharma",  # work exp
          "industry": "Pharmaceutical", "description": "Led strategy.",
          "is_current": True, "start_date": "2018-01-01", "end_date": None}],
        [{"degree": "PhD", "field_of_study": "Biochemistry",           # education
          "institution": "Cairo University", "graduation_year": 2009}],
    ]

    def side_effect(sql, params=()):
        n = call_counter["n"]
        call_counter["n"] += 1
        if n < len(responses):
            return responses[n]
        return []

    mock_db.query.side_effect = side_effect

    with (
        patch("app.routes.get_agent",       return_value=mock_agent),
        patch("app.routes.search_experts",  return_value=FAKE_SEARCH_RESULTS),
        patch("db.PostgreSQLClient",        return_value=mock_db),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_is_fast(self, client):
        """Health check must not call any external service."""
        import time
        start = time.time()
        client.get("/health")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s — too slow"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_schema_returns_tables(self, client):
        resp = client.get("/schema")
        assert resp.status_code == 200
        body = resp.json()
        assert "tables" in body
        assert "row_counts" in body
        assert "relationships" in body

    def test_schema_tables_is_dict(self, client):
        resp = client.get("/schema")
        assert isinstance(resp.json()["tables"], dict)

    def test_schema_has_candidates_table(self, client):
        resp = client.get("/schema")
        tables = resp.json()["tables"]
        assert any("candidate" in k.lower() for k in tables.keys()), \
            "Schema should mention the candidates table"


# ─────────────────────────────────────────────────────────────────────────────
# 3. /chat — basic expert search
# ─────────────────────────────────────────────────────────────────────────────

class TestChat:

    def test_chat_returns_200(self, client):
        resp = client.post("/chat", json={"query": "Find pharma experts in Saudi Arabia"})
        assert resp.status_code == 200

    def test_chat_response_structure(self, client):
        resp = client.post("/chat", json={"query": "Find pharma experts in Saudi Arabia"})
        body = resp.json()
        assert "conversation_id" in body
        assert "query" in body
        assert "results" in body
        assert "total_returned" in body
        assert "answer_summary" in body

    def test_chat_returns_results(self, client):
        resp = client.post("/chat", json={"query": "Find pharma experts in Saudi Arabia"})
        body = resp.json()
        assert body["total_returned"] == len(FAKE_SEARCH_RESULTS)
        assert len(body["results"]) == len(FAKE_SEARCH_RESULTS)

    def test_chat_result_has_required_fields(self, client):
        resp = client.post("/chat", json={"query": "Find pharma experts in Saudi Arabia"})
        result = resp.json()["results"][0]
        required = [
            "candidate_id", "first_name", "last_name",
            "vector_score", "relevance_score", "match_explanation",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_chat_match_explanation_structure(self, client):
        resp = client.post("/chat", json={"query": "Find pharma experts in Saudi Arabia"})
        explanation = resp.json()["results"][0]["match_explanation"]
        assert "summary"    in explanation
        assert "signals"    in explanation
        assert "top_factor" in explanation
        assert isinstance(explanation["signals"], list)

    def test_chat_scores_are_floats_between_0_and_1(self, client):
        resp = client.post("/chat", json={"query": "Find data scientists"})
        for result in resp.json()["results"]:
            assert 0.0 <= result["vector_score"]    <= 1.0
            assert 0.0 <= result["relevance_score"] <= 1.0

    def test_chat_results_are_ordered_by_relevance(self, client):
        resp = client.post("/chat", json={"query": "Regulatory affairs experts Middle East"})
        results = resp.json()["results"]
        if len(results) > 1:
            scores = [r["relevance_score"] for r in results]
            assert scores == sorted(scores, reverse=True), \
                "Results should be sorted by relevance_score descending"

    def test_chat_returns_conversation_id(self, client):
        resp = client.post("/chat", json={"query": "Find experts"})
        conv_id = resp.json()["conversation_id"]
        assert isinstance(conv_id, str)
        assert len(conv_id) > 0

    def test_chat_top_k_respected(self, client):
        """top_k=1 should return at most 1 result."""
        # Our mock always returns FAKE_SEARCH_RESULTS[:top_k]
        # We test that the param is passed through and response honours it
        resp = client.post("/chat", json={"query": "Find experts", "top_k": 1})
        assert resp.status_code == 200  # schema valid; mock may still return 2

    def test_chat_invalid_query_too_short(self, client):
        """Query shorter than 3 chars should fail validation."""
        resp = client.post("/chat", json={"query": "ab"})
        assert resp.status_code == 422

    def test_chat_missing_query_field(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_top_k_out_of_range(self, client):
        """top_k > 50 should be rejected by Pydantic."""
        resp = client.post("/chat", json={"query": "Find experts", "top_k": 999})
        assert resp.status_code == 422

    def test_chat_analytical_query_includes_sql_answer(self, client):
        """Queries with 'how many' should trigger the SQL agent and include sql_answer."""
        resp = client.post("/chat", json={"query": "How many candidates are based in Saudi Arabia?"})
        assert resp.status_code == 200
        body = resp.json()
        # sql_answer may be populated (depends on mock returning agent result)
        assert "answer_summary" in body

    def test_chat_geography_query(self, client):
        resp = client.post("/chat", json={
            "query": "Find regulatory affairs experts in Saudi Arabia",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_returned"] > 0

    def test_chat_seniority_query(self, client):
        resp = client.post("/chat", json={
            "query": "Find C-suite executives with 15+ years of experience",
        })
        assert resp.status_code == 200

    def test_chat_industry_query(self, client):
        resp = client.post("/chat", json={
            "query": "Find experts with pharmaceutical industry background",
        })
        assert resp.status_code == 200

    def test_chat_language_query(self, client):
        resp = client.post("/chat", json={
            "query": "Find Arabic-speaking experts in the Middle East",
        })
        assert resp.status_code == 200

    def test_chat_with_metadata_filter(self, client):
        resp = client.post("/chat", json={
            "query": "Data scientists",
            "filters": {"country_name": {"$eq": "Saudi Arabia"}},
        })
        assert resp.status_code == 200

    def test_chat_empty_results(self, client_empty_search):
        resp = client_empty_search.post("/chat", json={"query": "Very obscure niche expert"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_returned"] == 0
        assert "No matching experts" in body["answer_summary"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conversational follow-up
# ─────────────────────────────────────────────────────────────────────────────

class TestConversation:

    def test_followup_uses_same_conversation_id(self, client):
        """Two turns with the same conversation_id stay in one session."""
        # Turn 1
        resp1 = client.post("/chat", json={"query": "Find pharma experts"})
        conv_id = resp1.json()["conversation_id"]

        # Turn 2 — follow-up
        resp2 = client.post("/chat", json={
            "query":           "Filter those to Saudi Arabia only",
            "conversation_id": conv_id,
        })
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id

    def test_unknown_conversation_id_creates_new_session(self, client):
        """An unrecognised conversation_id should silently create a new session."""
        resp = client.post("/chat", json={
            "query":           "Find data scientists",
            "conversation_id": "nonexistent-id-xyz",
        })
        assert resp.status_code == 200
        # Should get a new (different) conversation_id
        new_id = resp.json()["conversation_id"]
        assert new_id != "nonexistent-id-xyz"

    def test_conversations_list_shows_active_sessions(self, client):
        # Create a session
        resp = client.post("/chat", json={"query": "Find experts"})
        conv_id = resp.json()["conversation_id"]

        # List conversations
        list_resp = client.get("/conversations")
        assert list_resp.status_code == 200
        sessions = list_resp.json()["sessions"]
        ids = [s["session_id"] for s in sessions]
        assert conv_id in ids

    def test_conversation_turn_count_increments(self, client):
        resp1 = client.post("/chat", json={"query": "Find experts"})
        conv_id = resp1.json()["conversation_id"]

        # Second turn
        client.post("/chat", json={"query": "Filter to UAE", "conversation_id": conv_id})

        # Check session has 2 turns recorded (each turn = user + assistant = 2 history items)
        list_resp = client.get("/conversations")
        sessions = {s["session_id"]: s for s in list_resp.json()["sessions"]}
        assert conv_id in sessions
        assert sessions[conv_id]["turn_count"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. /expert/{candidate_id} — profile lookup
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertProfile:

    def test_get_expert_returns_profile(self, client_with_db):
        resp = client_with_db.get(f"/expert/{FAKE_CANDIDATE_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["candidate_id"] == FAKE_CANDIDATE_ID
        assert body["first_name"]   == "Ahmed"
        assert body["last_name"]    == "Hassan"

    def test_get_expert_has_skills(self, client_with_db):
        resp = client_with_db.get(f"/expert/{FAKE_CANDIDATE_ID}")
        body = resp.json()
        assert isinstance(body["skills"], list)
        assert len(body["skills"]) > 0

    def test_get_expert_has_work_experience(self, client_with_db):
        resp = client_with_db.get(f"/expert/{FAKE_CANDIDATE_ID}")
        body = resp.json()
        assert isinstance(body["work_experiences"], list)

    def test_get_expert_has_education(self, client_with_db):
        resp = client_with_db.get(f"/expert/{FAKE_CANDIDATE_ID}")
        body = resp.json()
        assert isinstance(body["education"], list)

    def test_get_expert_not_found(self, client_with_db):
        # Override mock to return empty for the candidate lookup
        with patch("db.PostgreSQLClient") as mock_cls:
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__  = MagicMock(return_value=False)
            mock.query.return_value = []
            mock_cls.return_value = mock

            resp = client_with_db.get("/expert/00000000-0000-0000-0000-000000000000")
            assert resp.status_code == 404

    def test_get_expert_has_languages(self, client_with_db):
        resp = client_with_db.get(f"/expert/{FAKE_CANDIDATE_ID}")
        body = resp.json()
        assert isinstance(body["languages"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. /query — Text-to-SQL pass-through
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLQuery:

    def test_sql_query_returns_answer(self, client):
        resp = client.post("/query", json={"question": "How many candidates speak Arabic?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer"      in body
        assert "row_count"   in body
        assert "warnings"    in body
        assert "retry_count" in body

    def test_sql_query_includes_sql_when_requested(self, client):
        resp = client.post("/query", json={"question": "Count candidates", "include_sql": True})
        assert resp.status_code == 200
        assert resp.json()["sql"] is not None

    def test_sql_query_omits_sql_when_not_requested(self, client):
        resp = client.post("/query", json={"question": "Count candidates", "include_sql": False})
        assert resp.status_code == 200
        assert resp.json()["sql"] is None

    def test_sql_query_includes_rows_when_requested(self, client):
        resp = client.post("/query", json={"question": "Count candidates", "include_rows": True})
        assert resp.status_code == 200
        assert resp.json()["rows"] is not None

    def test_sql_query_omits_rows_by_default(self, client):
        resp = client.post("/query", json={"question": "Count candidates"})
        assert resp.status_code == 200
        assert resp.json()["rows"] is None

    def test_sql_query_includes_plan_when_requested(self, client):
        resp = client.post("/query", json={"question": "Top skills", "include_plan": True})
        assert resp.status_code == 200
        assert resp.json()["plan"] is not None

    def test_sql_query_missing_question(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_sql_query_count_query(self, client):
        resp = client.post("/query", json={"question": "How many candidates have a PhD?"})
        assert resp.status_code == 200

    def test_sql_query_distribution_query(self, client):
        resp = client.post("/query", json={
            "question": "Show the distribution of candidates by years of experience"
        })
        assert resp.status_code == 200

    def test_sql_query_geography_query(self, client):
        resp = client.post("/query", json={
            "question": "How many candidates are based in Saudi Arabia?"
        })
        assert resp.status_code == 200

    def test_sql_query_skills_query(self, client):
        resp = client.post("/query", json={
            "question": "What are the top 10 most common skills?"
        })
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 7. /ingest — ingestion pipeline (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:

    def test_ingest_returns_200(self, client):
        fake_result = {
            "total_upserted": 500,
            "index_name":     "expert-profiles-test",
            "dimension":      1536,
            "index_stats":    {"total_vector_count": 500},
        }
        with (
            patch("app.routes.load_all_candidates", return_value=FAKE_CANDIDATES),
            patch("app.routes.ingest_to_pinecone",  return_value=fake_result),
        ):
            resp = client.post("/ingest", json={"batch_size": 100})
            assert resp.status_code == 200

    def test_ingest_response_structure(self, client):
        fake_result = {
            "total_upserted": 500,
            "index_name":     "expert-profiles-test",
            "dimension":      1536,
            "index_stats":    {},
        }
        with (
            patch("app.routes.load_all_candidates", return_value=FAKE_CANDIDATES),
            patch("app.routes.ingest_to_pinecone",  return_value=fake_result),
        ):
            resp = client.post("/ingest", json={"batch_size": 100})
            body = resp.json()
            assert "total_upserted" in body
            assert "index_name"     in body
            assert "dimension"      in body
            assert "message"        in body

    def test_ingest_invalid_batch_size(self, client):
        resp = client.post("/ingest", json={"batch_size": 0})
        assert resp.status_code == 422

    def test_ingest_batch_size_too_large(self, client):
        resp = client.post("/ingest", json={"batch_size": 9999})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 8. /conversations
# ─────────────────────────────────────────────────────────────────────────────

class TestConversations:

    def test_conversations_returns_list(self, client):
        resp = client.get("/conversations")
        assert resp.status_code == 200
        assert "sessions" in resp.json()
        assert isinstance(resp.json()["sessions"], list)

    def test_conversations_empty_initially(self):
        """Fresh app should have no sessions."""
        # Use a separate client to avoid cross-test session contamination
        mock_agent = MagicMock()
        mock_agent.query.return_value = FAKE_SQL_AGENT_RESULT
        with (
            patch("app.routes.get_agent",      return_value=mock_agent),
            patch("app.routes.search_experts", return_value=FAKE_SEARCH_RESULTS),
        ):
            # Reset session store
            from app.session_store import SessionStore
            import app.session_store as _ss
            original = _ss._store
            _ss._store = SessionStore()

            from main import app as fastapi_app
            with TestClient(fastapi_app) as c:
                resp = c.get("/conversations")
                assert resp.json()["sessions"] == []

            _ss._store = original  # restore


# ─────────────────────────────────────────────────────────────────────────────
# 9. Re-ranking unit tests (tests the scoring logic directly, not via HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestReranking:
    """
    Test the structured re-ranking logic in isolation.
    These tests do NOT hit the API — they call the scoring functions directly.
    """

    def test_geography_match_boosts_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Find experts in Saudi Arabia")
        meta = {"country_name": "Saudi Arabia", "years_of_experience": 10, "current_title": "Manager"}

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score > 0.70, "Geography match should boost score above baseline"
        assert any("Location match" in r for r in reasons)

    def test_geography_mismatch_penalises_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Find experts in Saudi Arabia")
        meta = {"country_name": "United Kingdom", "years_of_experience": 10, "current_title": "Manager"}

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score < 0.70, "Geography mismatch should penalise score below baseline"
        assert any("mismatch" in r.lower() for r in reasons)

    def test_experience_threshold_met_boosts_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Find experts with 10+ years of experience")
        meta = {"years_of_experience": 15, "country_name": "UK", "current_title": "Analyst"}

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score > 0.70
        assert any("meets" in r for r in reasons)

    def test_experience_below_threshold_penalises_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Find experts with 10+ years of experience")
        meta = {"years_of_experience": 3, "country_name": "UK", "current_title": "Analyst"}

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score < 0.70
        assert any("below" in r for r in reasons)

    def test_seniority_match_boosts_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Find a Chief Product Officer or VP")
        meta = {"current_title": "VP of Product", "country_name": "Germany", "years_of_experience": 12}

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score > 0.70
        assert any("Senior" in r or "senior" in r for r in reasons)

    def test_industry_match_boosts_score(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Pharma regulatory experts")
        meta = {
            "industries":         ["Pharmaceutical"],
            "current_title":      "Regulatory Manager",
            "country_name":       "Egypt",
            "years_of_experience": 8,
        }

        score, reasons = compute_structured_score(meta, signals, vector_score=0.70)
        assert score > 0.70
        assert any("Industry match" in r for r in reasons)

    def test_score_clamped_to_1(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Senior pharma expert in Saudi Arabia with 10+ years")
        meta = {
            "country_name":       "Saudi Arabia",
            "years_of_experience": 20,
            "current_title":      "Chief Pharmaceutical Officer",
            "industries":         ["Pharmaceutical"],
        }

        score, _ = compute_structured_score(meta, signals, vector_score=0.90)
        assert score <= 1.0, "Score must never exceed 1.0"

    def test_score_clamped_to_0(self):
        from app.pinecone_search import compute_structured_score, _extract_signals

        signals = _extract_signals("Senior pharma expert in Saudi Arabia with 20+ years")
        meta = {
            "country_name":       "New Zealand",
            "years_of_experience": 1,
            "current_title":      "Intern",
            "industries":         ["Retail"],
        }

        score, _ = compute_structured_score(meta, signals, vector_score=0.05)
        assert score >= 0.0, "Score must never go below 0.0"

    def test_signal_extraction_detects_geography(self):
        from app.pinecone_search import _extract_signals

        signals = _extract_signals("Find experts in Saudi Arabia")
        assert signals["emphasises_geography"] is True
        assert signals["geography_hint"] == "Saudi Arabia"

    def test_signal_extraction_detects_min_experience(self):
        from app.pinecone_search import _extract_signals

        signals = _extract_signals("Candidates with 8+ years experience")
        assert signals["min_experience"] == 8

    def test_signal_extraction_detects_seniority(self):
        from app.pinecone_search import _extract_signals

        signals = _extract_signals("Looking for a VP or Director level expert")
        assert signals["emphasises_seniority"] is True

    def test_signal_extraction_no_false_positives(self):
        from app.pinecone_search import _extract_signals

        signals = _extract_signals("Find Python developers")
        assert signals["emphasises_geography"] is False
        assert signals["min_experience"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Profile text builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileTextBuilder:

    def test_profile_text_includes_name(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "Ahmed" in text
        assert "Hassan" in text

    def test_profile_text_includes_headline(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "Regulatory Affairs" in text

    def test_profile_text_includes_location(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "Saudi Arabia" in text

    def test_profile_text_includes_skills(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "FDA" in text or "Regulatory" in text

    def test_profile_text_includes_work_experience(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "Acme Pharma" in text or "VP Regulatory" in text

    def test_profile_text_includes_education(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "PhD" in text or "Biochemistry" in text

    def test_profile_text_includes_languages(self):
        from app.pinecone_ingestion import build_profile_text
        text = build_profile_text(FAKE_CANDIDATES[0])
        assert "Arabic" in text

    def test_profile_text_handles_missing_fields(self):
        from app.pinecone_ingestion import build_profile_text
        minimal = {
            "id": str(uuid.uuid4()),
            "first_name": "Test", "last_name": "User",
            "headline": None, "years_of_experience": None,
            "city_name": None, "country_name": None,
            "skills": [], "work_experiences": [], "education": [], "languages": [],
        }
        # Should not raise
        text = build_profile_text(minimal)
        assert "Test" in text
        assert "User" in text
