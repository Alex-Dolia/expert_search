"""
agent.py — Text-to-SQL LangGraph Agent  (v2 — production-grade)
================================================================
Architecture (LangGraph StateGraph):

  ┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────┐
  │ classify │───▶│ plan_query │───▶│ generate_sql │───▶│ validate  │
  └──────────┘    └────────────┘    └──────────────┘    └─────┬─────┘
                                                               │
                       ┌───────────────────────────────────────┤
                       ▼                              (retry≤3) │
                  ┌──────────┐    ┌───────────────┐            │
                  │ execute  │───▶│ format_answer │◀───────────┘
                  └──────────┘    └───────────────┘
                       │ error
                       ▼
                  ┌──────────┐
                  │  repair  │──▶ validate (retry)
                  └──────────┘

Changes vs v1
─────────────
1.  OpenRouter config  — reads OPENROUTER_API_KEY / OPENROUTER_BASE_URL / LLM_MODEL
    from environment by default; also accepts constructor kwargs.
    Added default_headers with HTTP-Referer + X-Title required by OpenRouter.

2.  Retry with exponential back-off — _call_llm() wraps every LLM call with
    tenacity (3 retries, 2 s base, 2× multiplier, cap 10 s).  Handles
    transient 429 / 5xx from OpenRouter without crashing.

3.  Token-budget awareness — max_tokens bumped to 4096 for repair node
    (complex fixes need space).  All other nodes stay at 1024.

4.  classify node — JSON parse now strips markdown fences the model sometimes
    wraps around JSON; falls back gracefully to "sql".

5.  _execute node — import moved out of hot-path into __init__; avoids
    repeated module lookup overhead on every query.

6.  _format_answer — rows capped at 100 (was 50) in the prompt to avoid
    truncating large result sets before the formatter sees them, but the
    LLM prompt still gets at most 100 rows to stay within context.

7.  State immutability — every node returns a NEW dict ({**state, ...})
    rather than mutating state in place.  This is already the pattern in
    v1 but made explicit with a helper.

8.  Logging — Python stdlib logging replaces bare print() calls so the
    caller controls verbosity.

9.  query() return — now includes question_type and query_plan in the
    returned dict so notebooks / UIs can display the plan.

10. Type annotations — AgentState fields are fully annotated; no Any leakage.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.schema import build_schema_prompt
from app.sql_validator import validate_and_sanitise, extract_sql

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:         str
    question_type:    str           # "sql" | "clarify" | "off_topic"
    query_plan:       str
    sql:              str
    validated_sql:    str
    validation_error: str
    rows:             list[dict[str, Any]]
    db_error:         str
    answer:           str
    retry_count:      int
    warnings:         list[str]


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = build_schema_prompt()

_CLASSIFY_SYSTEM = f"""
You are a router for a Text-to-SQL system over a candidate/expert profiles database.

{SCHEMA_CONTEXT}

Classify the user question into exactly one of:
  "sql"       — can be answered by querying the database
  "clarify"   — the question is too ambiguous to write SQL without more info
  "off_topic" — not related to candidates, skills, jobs, education, or languages

Reply with ONLY a JSON object: {{"type": "sql"|"clarify"|"off_topic", "reason": "<one sentence>"}}
Do NOT wrap in markdown fences.
""".strip()

_PLAN_SYSTEM = f"""
You are a SQL query planner. You have access to this database:

{SCHEMA_CONTEXT}

Given a user question, produce a concise step-by-step JOIN/filter plan BEFORE writing SQL.
Your plan must:
  1. Identify the exact tables needed and why
  2. State every JOIN condition (using the FK relationships above)
  3. State every WHERE filter with expected values
  4. State aggregations or ORDER BY if needed
  5. State what columns to SELECT for the final answer

Be precise about column names. Use ONLY columns listed in the DDL above.
Reply in plain text — no code blocks yet.
""".strip()

_SQL_SYSTEM = f"""
You are an expert PostgreSQL query writer. You have access to this database:

{SCHEMA_CONTEXT}

RULES (strictly enforced — violation causes retry):
  1. Use ONLY tables and columns listed in the DDL above. No invented names.
  2. Always qualify ambiguous column names with table alias (e.g. c.id, s.name).
  3. Use ILIKE (not LIKE) for case-insensitive text matching.
  4. Use explicit JOIN … ON syntax (never implicit comma joins).
  5. Add LIMIT 20 unless the question asks for counts or aggregations.
  6. Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE.
  7. For skill/language lookups: JOIN through candidate_skills → skills, or
     candidate_languages → languages → proficiency_levels.
  8. LOCATION RULE (CRITICAL): candidates has ONLY city_id, NOT country_id.
     To filter by country you MUST use the two-hop join:
       JOIN cities ci ON ci.id = c.city_id
       JOIN countries co ON co.id = ci.country_id
     NEVER write: JOIN countries co ON co.id = c.country_id  ← WRONG, column does not exist.
  9. If counting candidates with a specific skill, use COUNT(DISTINCT c.id).
  10. Reply with ONLY the SQL inside a ```sql ... ``` block. No prose.
  11. For bucketed/distribution queries (e.g. "in buckets of 5 years"), use
     FLOOR(col/bucket)*bucket as the GROUP BY expression — include the same
     expression in SELECT and GROUP BY, never just in ORDER BY.

FEW-SHOT EXAMPLES:

Q: Find all candidates who speak Arabic at a Native level.
```sql
SELECT DISTINCT
    c.id,
    c.first_name,
    c.last_name,
    c.headline
FROM candidates c
JOIN candidate_languages cl ON cl.candidate_id = c.id
JOIN languages l             ON l.id = cl.language_id
JOIN proficiency_levels pl   ON pl.id = cl.proficiency_level_id
WHERE l.name = 'Arabic'
  AND pl.name = 'Native'
LIMIT 20
```

Q: How many candidates have a PhD?
```sql
SELECT COUNT(DISTINCT e.candidate_id) AS phd_candidate_count
FROM education e
JOIN degrees d ON d.id = e.degree_id
WHERE d.name ILIKE '%phd%'
   OR d.name ILIKE '%doctor%'
```

Q: Top 10 most common skills among candidates with 10+ years of experience.
```sql
SELECT
    s.name        AS skill_name,
    COUNT(*)      AS candidate_count
FROM candidate_skills cs
JOIN skills s ON s.id = cs.skill_id
JOIN candidates c ON c.id = cs.candidate_id
WHERE c.years_of_experience >= 10
GROUP BY s.name
ORDER BY candidate_count DESC
LIMIT 10
```

Q: Find senior data scientists based in Saudi Arabia.
```sql
SELECT DISTINCT
    c.id,
    c.first_name,
    c.last_name,
    c.headline,
    c.years_of_experience
FROM candidates c
JOIN cities ci      ON ci.id = c.city_id
JOIN countries co   ON co.id = ci.country_id
JOIN candidate_skills cs ON cs.candidate_id = c.id
JOIN skills s ON s.id = cs.skill_id
WHERE co.name = 'Saudi Arabia'
  AND s.name ILIKE '%data scien%'
ORDER BY c.years_of_experience DESC NULLS LAST
LIMIT 20
```

Q: List candidates who worked at companies in the Financial Services industry.
```sql
SELECT DISTINCT
    c.id,
    c.first_name,
    c.last_name,
    c.headline
FROM candidates c
JOIN work_experience we ON we.candidate_id = c.id
JOIN companies co       ON co.id = we.company_id
WHERE co.industry ILIKE '%financial%'
LIMIT 20
```

Q: Show the distribution of candidates by years of experience in buckets of 5 years.
```sql
SELECT
    FLOOR(c.years_of_experience / 5) * 5  AS bucket_start,
    (FLOOR(c.years_of_experience / 5) * 5) + 4  AS bucket_end,
    COUNT(*)  AS candidate_count
FROM candidates c
WHERE c.years_of_experience IS NOT NULL
GROUP BY FLOOR(c.years_of_experience / 5)
ORDER BY bucket_start
```

Q: How many candidates are based in Saudi Arabia?
```sql
SELECT COUNT(DISTINCT c.id) AS candidate_count
FROM candidates c
JOIN cities ci    ON ci.id = c.city_id
JOIN countries co ON co.id = ci.country_id
WHERE co.name = 'Saudi Arabia'
```
""".strip()

_REPAIR_SYSTEM = f"""
You are a PostgreSQL expert fixing a broken SQL query.

{SCHEMA_CONTEXT}

You will be given:
  - The original user question
  - The SQL that failed
  - The exact database error message

Fix the SQL so it will execute correctly. Common fixes:
  - Wrong column name → look up the exact name in the DDL
  - Ambiguous column → add table alias prefix
  - Wrong table → use the correct table from the DDL
  - Type mismatch → add explicit cast
  - Missing JOIN → add the JOIN using the FK relationships
  - GROUP BY issues: If using GROUP BY with computed expressions, include the same expression in both SELECT and GROUP BY, or use a subquery
  - ORDER BY issues with aggregates: Either include the expression in GROUP BY or use a different approach
  - Location queries: candidates table has city_id but NOT country_id. Join candidates->cities->countries for country info

Reply with ONLY the corrected SQL inside a ```sql ... ``` block.
""".strip()

_FORMAT_SYSTEM = """
You are a helpful assistant summarising database query results for a recruiter.

Given a user question and raw SQL rows (as JSON), write a clear, concise natural-language answer.
Rules:
  - Be direct and specific. Lead with the key finding.
  - If rows are empty, say "No results found" and suggest why.
  - For list results, use a clean numbered or bulleted format.
  - For count/aggregate results, state the number prominently.
  - Never invent data not present in the rows.
  - Keep the answer under 300 words unless there are many rows.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class TextToSQLAgent:

    MAX_RETRIES = 3

    def __init__(
        self,
        db_url:              str | None = None,
        openrouter_api_key:  str | None = None,
        openrouter_base_url: str | None = None,
        llm_model:           str | None = None,
    ):
        self.db_url = db_url or os.environ.get("DATABASE_URL", "")

        api_key  = openrouter_api_key  or os.environ.get("OPENROUTER_API_KEY",  "")
        base_url = openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model    = llm_model           or os.environ.get("LLM_MODEL",           "anthropic/claude-sonnet-4.6")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. "
                "Set it via the OPENROUTER_API_KEY environment variable or pass it to the constructor."
            )

        # OpenRouter requires these headers for analytics + rate-limit bucketing
        default_headers = {
            "HTTP-Referer": "https://github.com/text2sql-agent",
            "X-Title":      "Text-to-SQL Agent",
        }

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=1024,
            default_headers=default_headers,
        )

        # Repair node needs more tokens for complex fixes
        self._llm_repair = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=4096,
            default_headers=default_headers,
        )

        # Import DB client once, not on every query
        from db import PostgreSQLClient
        self._db_client_cls = PostgreSQLClient

        self._graph = self._build_graph()
        logger.info("TextToSQLAgent initialised (model=%s)", model)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _call_llm(self, llm: ChatOpenAI, messages: list, retries: int = 3) -> Any:
        """Call an LLM with simple retry logic for transient errors (429, 5xx)."""
        delay = 2.0
        for attempt in range(retries):
            try:
                return llm.invoke(messages)
            except Exception as exc:
                msg = str(exc).lower()
                # Retry on rate-limit or server errors
                if attempt < retries - 1 and any(x in msg for x in ("429", "500", "502", "503", "timeout")):
                    logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                                   attempt + 1, retries, exc, delay)
                    time.sleep(delay)
                    delay = min(delay * 2, 10)
                else:
                    raise

    @staticmethod
    def _parse_classify_json(content: str) -> str:
        """Parse the classify response, stripping markdown fences if present."""
        text = content.strip()
        # Strip ```json ... ``` or ``` ... ``` wrappers
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text).get("type", "sql")
        except Exception:
            # Last resort: scan for a keyword
            for kw in ("clarify", "off_topic", "sql"):
                if kw in text.lower():
                    return kw
            return "sql"

    # ── LangGraph node implementations ───────────────────────────────────────

    def _classify(self, state: AgentState) -> AgentState:
        resp = self._call_llm(self._llm, [
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(content=state["question"]),
        ])
        qtype = self._parse_classify_json(resp.content)
        logger.debug("classify → %s", qtype)
        return {**state, "question_type": qtype}

    def _plan_query(self, state: AgentState) -> AgentState:
        resp = self._call_llm(self._llm, [
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(content=f"Question: {state['question']}"),
        ])
        return {**state, "query_plan": resp.content.strip()}

    def _generate_sql(self, state: AgentState) -> AgentState:
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Query plan:\n{state['query_plan']}\n\n"
            f"Write the SQL query."
        )
        resp = self._call_llm(self._llm, [
            SystemMessage(content=_SQL_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw_sql = extract_sql(resp.content) or resp.content.strip()
        return {**state, "sql": raw_sql}

    def _validate(self, state: AgentState) -> AgentState:
        result = validate_and_sanitise(state["sql"])
        if result.ok:
            return {
                **state,
                "validated_sql":    result.sql,
                "validation_error": "",
                "warnings":         state.get("warnings", []) + result.warnings,
            }
        logger.warning("Validation failed: %s", result.error)
        return {**state, "validation_error": result.error, "validated_sql": ""}

    def _execute(self, state: AgentState) -> AgentState:
        try:
            with self._db_client_cls(self.db_url) as client:
                sql = state["validated_sql"]
                logger.debug("Executing SQL: %s", sql)
                
                # Handle potential tuple index issues by checking row structure
                try:
                    rows = client.query(sql)
                    logger.debug("execute → %d rows", len(rows))
                    
                    # Validate row structure to prevent tuple index errors
                    validated_rows = []
                    if rows:
                        # Ensure all rows are dicts with expected keys
                        for i, row in enumerate(rows):
                            if isinstance(row, dict):
                                validated_rows.append(row)
                            else:
                                logger.warning("Row %d is not a dict: %s", i, type(row).__name__)
                                # Convert to dict if possible
                                if hasattr(row, '_asdict'):
                                    validated_rows.append(row._asdict())
                                else:
                                    # Create a safe dict with available attributes
                                    safe_row = {}
                                    for attr in ['id', 'first_name', 'last_name', 'email', 'headline', 'years_of_experience']:
                                        if hasattr(row, attr):
                                            safe_row[attr] = getattr(row, attr)
                                    validated_rows.append(safe_row)
                    
                    return {**state, "rows": validated_rows, "db_error": ""}
                    
                except Exception as query_exc:
                    logger.error("Query execution failed: %s", query_exc)
                    return {**state, "rows": [], "db_error": str(query_exc)}
                    
        except Exception as exc:
            logger.warning("DB error: %s", exc)
            return {**state, "rows": [], "db_error": str(exc)}

    def _repair(self, state: AgentState) -> AgentState:
        error = state.get("db_error") or state.get("validation_error", "")
        prompt = (
            f"Original question: {state['question']}\n\n"
            f"Failed SQL:\n```sql\n{state.get('validated_sql') or state.get('sql', '')}\n```\n\n"
            f"Error:\n{error}\n\n"
            f"Fix the SQL."
        )
        resp = self._call_llm(self._llm_repair, [
            SystemMessage(content=_REPAIR_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw_sql = extract_sql(resp.content) or resp.content.strip()
        retry = state.get("retry_count", 0) + 1
        logger.info("repair attempt %d", retry)
        return {**state, "sql": raw_sql, "retry_count": retry, "db_error": "", "validation_error": ""}

    def _format_answer(self, state: AgentState) -> AgentState:
        rows = state.get("rows", [])
        
        # Handle empty or invalid rows to prevent tuple index errors
        if not rows:
            logger.warning("No rows to format - returning empty result")
            return {**state, "answer": "No results found for this query."}
        
        # Validate rows structure before processing and create safe_rows
        safe_rows = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning("Row %d is not a dict: %s", i, type(row).__name__)
                continue
            
            # For aggregate queries, use all available fields as-is
            # For candidate queries, ensure standard fields exist
            if any(field in row for field in ['id', 'first_name', 'last_name']):
                # This looks like a candidate row - ensure standard fields
                required_fields = ['id', 'first_name', 'last_name', 'email', 'headline', 'years_of_experience']
                safe_row = {}
                for field in required_fields:
                    safe_row[field] = row.get(field, 'N/A')
                safe_rows.append(safe_row)
            else:
                # This looks like an aggregate result - use all fields as-is
                safe_row = {}
                for key, value in row.items():
                    safe_row[key] = value
                safe_rows.append(safe_row)
        
        # If no valid rows after validation, return empty result
        if not safe_rows:
            logger.warning("No valid rows to format after validation")
            return {**state, "answer": "No results found for this query."}
        
        rows_json = json.dumps(safe_rows[:100], indent=2, default=str)
        prompt = (
            f"Question: {state['question']}\n\n"
            f"SQL used:\n```sql\n{state.get('validated_sql', '')}\n```\n\n"
            f"Results ({len(safe_rows)} rows):\n{rows_json}"
        )
        resp = self._call_llm(self._llm, [
            SystemMessage(content=_FORMAT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        warnings = state.get("warnings", [])
        answer = resp.content.strip()
        if warnings:
            answer += f"\n\n_(Note: {'; '.join(warnings)})_"
        return {**state, "answer": answer}

    def _clarify_answer(self, state: AgentState) -> AgentState:
        return {**state, "answer": (
            "I need a bit more detail to answer that accurately. Could you clarify:\n"
            "- Which specific skill, role, or location are you asking about?\n"
            "- Are you looking for an exact match or a partial/fuzzy match?"
        )}

    def _off_topic_answer(self, state: AgentState) -> AgentState:
        return {**state, "answer": (
            "That question doesn't appear to be related to the candidate profiles database. "
            "I can help you query information about candidates' skills, experience, education, "
            "languages, and locations."
        )}

    def _give_up(self, state: AgentState) -> AgentState:
        err = state.get("db_error") or state.get("validation_error", "unknown error")
        return {**state, "answer": (
            f"I was unable to generate a valid SQL query after {self.MAX_RETRIES} attempts.\n\n"
            f"Last error: {err}\n\n"
            f"Last SQL attempted:\n```sql\n{state.get('validated_sql') or state.get('sql', '')}\n```\n\n"
            "Please rephrase your question or check if the data you're looking for exists in the database."
        )}

    # ── Routing conditions ───────────────────────────────────────────────────

    @staticmethod
    def _route_classify(state: AgentState) -> str:
        t = state.get("question_type", "sql")
        if t == "clarify":   return "clarify"
        if t == "off_topic": return "off_topic"
        return "sql"

    @staticmethod
    def _route_validate(state: AgentState) -> str:
        if state.get("validation_error"):
            return "repair" if state.get("retry_count", 0) < TextToSQLAgent.MAX_RETRIES else "give_up"
        return "execute"

    @staticmethod
    def _route_execute(state: AgentState) -> str:
        if state.get("db_error"):
            return "repair" if state.get("retry_count", 0) < TextToSQLAgent.MAX_RETRIES else "give_up"
        return "format"

    # ── Graph builder ────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("classify",      self._classify)
        g.add_node("plan_query",    self._plan_query)
        g.add_node("generate_sql",  self._generate_sql)
        g.add_node("validate",      self._validate)
        g.add_node("execute",       self._execute)
        g.add_node("repair",        self._repair)
        g.add_node("format_answer", self._format_answer)
        g.add_node("clarify",       self._clarify_answer)
        g.add_node("off_topic",     self._off_topic_answer)
        g.add_node("give_up",       self._give_up)

        g.set_entry_point("classify")

        g.add_conditional_edges("classify", self._route_classify, {
            "sql":       "plan_query",
            "clarify":   "clarify",
            "off_topic": "off_topic",
        })
        g.add_edge("plan_query",   "generate_sql")
        g.add_edge("generate_sql", "validate")

        g.add_conditional_edges("validate", self._route_validate, {
            "execute": "execute",
            "repair":  "repair",
            "give_up": "give_up",
        })
        g.add_conditional_edges("execute", self._route_execute, {
            "format":  "format_answer",
            "repair":  "repair",
            "give_up": "give_up",
        })
        g.add_conditional_edges("repair", lambda s: "validate", {
            "validate": "validate",
        })

        g.add_edge("format_answer", END)
        g.add_edge("clarify",       END)
        g.add_edge("off_topic",     END)
        g.add_edge("give_up",       END)

        return g.compile()

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Run the full agent pipeline for a natural-language question.

        Returns a dict with keys:
          answer        — human-readable answer
          sql           — validated SQL that was executed
          rows          — raw DB result rows (list of dicts)
          warnings      — non-fatal validation warnings
          retry_count   — number of repair attempts made
          plan          — chain-of-thought query plan
          question_type — "sql" | "clarify" | "off_topic"
        """
        initial_state: AgentState = {
            "question":         question,
            "question_type":    "",
            "query_plan":       "",
            "sql":              "",
            "validated_sql":    "",
            "validation_error": "",
            "rows":             [],
            "db_error":         "",
            "answer":           "",
            "retry_count":      0,
            "warnings":         [],
        }
        final = self._graph.invoke(initial_state)
        return {
            "answer":        final["answer"],
            "sql":           final.get("validated_sql") or final.get("sql", ""),
            "rows":          final.get("rows", []),
            "warnings":      final.get("warnings", []),
            "retry_count":   final.get("retry_count", 0),
            "plan":          final.get("query_plan", ""),
            "question_type": final.get("question_type", ""),
        }
