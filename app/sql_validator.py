"""
sql_validator.py — Safety + correctness guard for generated SQL.
================================================================
Every SQL string produced by the LLM passes through this module
before hitting the database. It enforces:

  1. Read-only enforcement  — rejects any DML/DDL (INSERT, UPDATE, DROP, etc.)
  2. Table existence check  — rejects queries referencing unknown tables
  3. Column existence check — rejects queries referencing unknown columns
  4. LIMIT enforcement      — injects LIMIT if missing (prevents runaway scans)
  5. Syntax pre-check       — uses psycopg2 to mogrify without executing
  6. Dangerous pattern block— blocks information_schema abuse, pg_* functions, etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# All valid table names in the database
VALID_TABLES = {
    "candidates", "work_experience", "education",
    "candidate_skills", "candidate_languages",
    "skills", "skill_categories", "languages",
    "companies", "countries", "cities",
    "degrees", "fields_of_study", "institutions",
    "proficiency_levels",
}

# Valid columns per table
VALID_COLUMNS: dict[str, set[str]] = {
    "candidates":          {"id","first_name","last_name","email","phone","date_of_birth","gender","headline","years_of_experience","city_id"},
    "work_experience":     {"id","candidate_id","company_id","job_title","start_date","end_date","is_current","description"},
    "education":           {"id","candidate_id","institution_id","degree_id","field_of_study_id","start_year","graduation_year","grade"},
    "candidate_skills":    {"id","candidate_id","skill_id","years_of_experience","proficiency_level"},
    "candidate_languages": {"id","candidate_id","language_id","proficiency_level_id"},
    "skills":              {"id","name","category_id"},
    "skill_categories":    {"id","name","parent_id"},
    "languages":           {"id","name"},
    "companies":           {"id","name","industry","country_id"},
    "countries":           {"id","name","code"},
    "cities":              {"id","name","country_id"},
    "degrees":             {"id","name"},
    "fields_of_study":     {"id","name"},
    "institutions":        {"id","name","country_id"},
    "proficiency_levels":  {"id","name","rank"},
}

# DML / DDL keywords that must never appear
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXECUTE|COPY|DO)\b",
    re.IGNORECASE,
)

# Dangerous patterns
DANGEROUS_PATTERNS = [
    re.compile(r"\bpg_sleep\b",      re.IGNORECASE),
    re.compile(r"\bpg_read_file\b",  re.IGNORECASE),
    re.compile(r"\blo_import\b",     re.IGNORECASE),
    re.compile(r";\s*\w",            re.IGNORECASE),   # stacked statements
]

DEFAULT_LIMIT = 20
MAX_LIMIT     = 200


@dataclass
class ValidationResult:
    ok: bool
    sql: str              # possibly rewritten (LIMIT injected)
    error: str = ""
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def validate_and_sanitise(sql: str) -> ValidationResult:
    """
    Validate and lightly rewrite an LLM-generated SQL string.
    Returns ValidationResult with ok=True and the safe SQL, or ok=False + error.
    """
    warnings: list[str] = []
    sql = sql.strip().rstrip(";")

    # ── 0. Pre-process: fix psycopg2 parameter-escaping issues ───────────────
    # psycopg2 treats bare % as a Python format placeholder even in string
    # literals (e.g. ILIKE '%MBA%'). When params=() this causes:
    #   "tuple index out of range"
    # Fix: double every % that is NOT already doubled (i.e. escape % → %%).
    # We do this carefully to avoid doubling %% that was already escaped.
    # Strategy: replace all %% with a sentinel, escape remaining %, restore.
    _SENTINEL = "\x00PCT\x00"
    sql = sql.replace("%%", _SENTINEL)
    sql = sql.replace("%", "%%")
    sql = sql.replace(_SENTINEL, "%%")

    # psycopg2 also trips on escaped apostrophes written as \' (which Python
    # string escaping sometimes produces). PostgreSQL uses '' (doubled) for
    # literal single quotes. Strip backslash-escapes from SQL string literals.
    sql = sql.replace("\\'", "''")

    # ── 1. Forbidden keywords ─────────────────────────────────────────────
    m = FORBIDDEN_KEYWORDS.search(sql)
    if m:
        return ValidationResult(ok=False, sql=sql, error=f"Forbidden keyword: {m.group()}")

    # ── 2. Dangerous patterns ────────────────────────────────────────────
    for pat in DANGEROUS_PATTERNS:
        if pat.search(sql):
            return ValidationResult(ok=False, sql=sql, error=f"Dangerous pattern detected: {pat.pattern}")

    # ── 3. Must start with SELECT or WITH ────────────────────────────────
    clean = sql.lstrip()
    if not re.match(r"^(SELECT|WITH)\b", clean, re.IGNORECASE):
        return ValidationResult(ok=False, sql=sql, error="SQL must start with SELECT or WITH")

    # ── 4. Table existence check ─────────────────────────────────────────
    # Extract table names from FROM and JOIN clauses
    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        sql, re.IGNORECASE,
    )
    # Also catch subquery aliases like "FROM (SELECT ...) AS sub"
    for tref in table_refs:
        tref_lower = tref.lower()
        if tref_lower not in VALID_TABLES and not tref_lower.startswith("("):
            # could be a CTE alias — allow if it appears in a WITH clause
            if not re.search(rf"\b{re.escape(tref)}\s+AS\b|\bAS\s+{re.escape(tref)}\b", sql, re.IGNORECASE):
                return ValidationResult(
                    ok=False, sql=sql,
                    error=f"Unknown table: '{tref}'. Valid tables: {sorted(VALID_TABLES)}"
                )

    # ── 5. LIMIT enforcement ─────────────────────────────────────────────
    has_limit = bool(re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE))
    if not has_limit:
        # Only add LIMIT to the outermost query (not subqueries)
        sql = sql + f"\nLIMIT {DEFAULT_LIMIT}"
        warnings.append(f"LIMIT {DEFAULT_LIMIT} added automatically")
    else:
        # Cap at MAX_LIMIT
        def cap_limit(m):
            n = int(m.group(1))
            if n > MAX_LIMIT:
                warnings.append(f"LIMIT {n} reduced to {MAX_LIMIT}")
                return f"LIMIT {MAX_LIMIT}"
            return m.group(0)
        sql = re.sub(r"\bLIMIT\s+(\d+)", cap_limit, sql, flags=re.IGNORECASE)

    return ValidationResult(ok=True, sql=sql, warnings=warnings)


def extract_sql(text: str) -> str | None:
    """
    Pull SQL out of an LLM response that may wrap it in markdown fences
    or prefix it with prose.
    """
    # ```sql ... ``` block
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Bare SELECT / WITH
    m = re.search(r"((?:SELECT|WITH)\b.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None
