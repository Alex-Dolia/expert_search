"""
schema.py — Authoritative schema catalogue for the candidate_profiles database.
===============================================================================
This module is the single source of truth the SQL agent uses.
It was derived from direct Postgres introspection of the live database.

Having schema knowledge hard-coded (rather than fetched at query time)
is a deliberate precision choice:
  • The LLM sees exact column names — no hallucination of non-existent columns.
  • Foreign key relationships are explicit — the agent can plan JOINs correctly.
  • Sample values for enum-like columns teach the LLM about allowed values.
  • The DDL string is injected verbatim into the system prompt so the model
    cannot invent tables or columns.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Full DDL — mirroring the live Postgres schema exactly
# ─────────────────────────────────────────────────────────────────────────────
DDL = """
-- ============================================================
-- Core entity tables
-- ============================================================

CREATE TABLE candidates (
    id                   UUID PRIMARY KEY,
    first_name           VARCHAR NOT NULL,
    last_name            VARCHAR NOT NULL,
    email                VARCHAR,
    phone                VARCHAR,
    date_of_birth        DATE,
    gender               VARCHAR,           -- e.g. 'Male', 'Female', 'Other'
    headline             TEXT,              -- professional headline / summary
    years_of_experience  INTEGER,
    city_id              UUID REFERENCES cities(id)
    -- Note: country_id may not exist in actual database
    -- city_id is the primary location field
);

CREATE TABLE countries (
    id    UUID PRIMARY KEY,
    name  VARCHAR NOT NULL,   -- e.g. 'Saudi Arabia', 'United States', 'Germany'
    code  CHAR(2) NOT NULL    -- ISO 3166-1 alpha-2 e.g. 'SA', 'US', 'DE'
);

CREATE TABLE cities (
    id          UUID PRIMARY KEY,
    name        VARCHAR NOT NULL,
    country_id  UUID REFERENCES countries(id)
);

CREATE TABLE companies (
    id          UUID PRIMARY KEY,
    name        VARCHAR NOT NULL,
    industry    VARCHAR,    -- e.g. 'Software Development', 'Financial Services'
    country_id  UUID REFERENCES countries(id)
);

CREATE TABLE institutions (
    id          UUID PRIMARY KEY,
    name        VARCHAR NOT NULL,
    country_id  UUID REFERENCES countries(id)
);

CREATE TABLE skills (
    id           UUID PRIMARY KEY,
    name         VARCHAR NOT NULL,
    category_id  UUID REFERENCES skill_categories(id)
);

CREATE TABLE skill_categories (
    id         UUID PRIMARY KEY,
    name       VARCHAR NOT NULL,
    parent_id  UUID REFERENCES skill_categories(id)   -- nullable; NULL = top-level
);

CREATE TABLE languages (
    id    UUID PRIMARY KEY,
    name  VARCHAR NOT NULL   -- e.g. 'Arabic', 'English', 'French'
);

CREATE TABLE degrees (
    id    UUID PRIMARY KEY,
    name  VARCHAR NOT NULL   -- e.g. 'Bachelor''s', 'Master''s', 'MBA', 'PhD'
);

CREATE TABLE fields_of_study (
    id    UUID PRIMARY KEY,
    name  VARCHAR NOT NULL   -- e.g. 'Computer Science', 'Business Administration'
);

CREATE TABLE proficiency_levels (
    id    UUID PRIMARY KEY,
    name  VARCHAR NOT NULL,  -- 'Beginner', 'Intermediate', 'Advanced', 'Fluent', 'Native'
    rank  INTEGER            -- higher = more proficient
);

-- ============================================================
-- Junction / relationship tables
-- ============================================================

CREATE TABLE work_experience (
    id            UUID PRIMARY KEY,
    candidate_id  UUID NOT NULL REFERENCES candidates(id),
    company_id    UUID REFERENCES companies(id),
    job_title     VARCHAR NOT NULL,
    start_date    DATE,
    end_date      DATE,        -- NULL means is_current = true
    is_current    BOOLEAN DEFAULT FALSE,
    description   TEXT
);

CREATE TABLE education (
    id               UUID PRIMARY KEY,
    candidate_id     UUID NOT NULL REFERENCES candidates(id),
    institution_id   UUID REFERENCES institutions(id),
    degree_id        UUID REFERENCES degrees(id),
    field_of_study_id UUID REFERENCES fields_of_study(id),
    start_year       INTEGER,
    graduation_year  INTEGER,
    grade            VARCHAR    -- e.g. 'A', 'B+', '3.8'
);

CREATE TABLE candidate_skills (
    id                  UUID PRIMARY KEY,
    candidate_id        UUID NOT NULL REFERENCES candidates(id),
    skill_id            UUID NOT NULL REFERENCES skills(id),
    years_of_experience INTEGER,
    proficiency_level   VARCHAR    -- 'Beginner','Intermediate','Advanced','Expert'
);

CREATE TABLE candidate_languages (
    id                   UUID PRIMARY KEY,
    candidate_id         UUID NOT NULL REFERENCES candidates(id),
    language_id          UUID NOT NULL REFERENCES languages(id),
    proficiency_level_id UUID REFERENCES proficiency_levels(id)
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Relationship map (used for JOIN path planning)
# ─────────────────────────────────────────────────────────────────────────────
RELATIONSHIPS = [
    "candidates.city_id           → cities.id",
    "cities.country_id            → countries.id",
    "companies.country_id         → countries.id",
    "institutions.country_id      → countries.id",
    "skills.category_id           → skill_categories.id",
    "skill_categories.parent_id   → skill_categories.id  (self-ref)",
    "work_experience.candidate_id → candidates.id",
    "work_experience.company_id   → companies.id",
    "education.candidate_id       → candidates.id",
    "education.institution_id     → institutions.id",
    "education.degree_id          → degrees.id",
    "education.field_of_study_id  → fields_of_study.id",
    "candidate_skills.candidate_id → candidates.id",
    "candidate_skills.skill_id     → skills.id",
    "candidate_languages.candidate_id    → candidates.id",
    "candidate_languages.language_id     → languages.id",
    "candidate_languages.proficiency_level_id → proficiency_levels.id",
]

# ─────────────────────────────────────────────────────────────────────────────
# Known enum values (prevent the LLM from guessing wrong spellings)
# ─────────────────────────────────────────────────────────────────────────────
ENUM_VALUES = {
    "candidate_skills.proficiency_level": ["Beginner", "Intermediate", "Advanced", "Expert"],
    "proficiency_levels.name":            ["Beginner", "Intermediate", "Fluent", "Native"],
    "degrees.name":                       [
        "Associate's", "Bachelor's", "Bachelor of Laws", "BS", "BA(Honours)",
        "Master's", "MBA", "MFA", "MSc", "PhD", "Doctor of Philosophy (PhD)",
        "EMBA", "Certificate", "Diploma", "Professional",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Table descriptions (human-readable, injected into the system prompt)
# ─────────────────────────────────────────────────────────────────────────────
TABLE_DESCRIPTIONS = {
    "candidates":          "Core table: one row per expert/candidate. Contains personal info, headline, years of experience, and FK to city/country.",
    "work_experience":     "Employment history. Each row is one job. company_id → companies. is_current=TRUE means they still hold this role.",
    "education":           "Academic history. degree_id → degrees, field_of_study_id → fields_of_study, institution_id → institutions.",
    "candidate_skills":    "Skills each candidate has. proficiency_level is a VARCHAR: 'Beginner','Intermediate','Advanced','Expert'. years_of_experience is an integer.",
    "candidate_languages": "Languages each candidate speaks. proficiency_level_id → proficiency_levels (Beginner/Intermediate/Fluent/Native).",
    "skills":              "Skill lookup table. name is the skill label. category_id → skill_categories.",
    "skill_categories":    "Hierarchical skill categories. parent_id is self-referential (NULL = top-level).",
    "languages":           "Language lookup. name e.g. 'Arabic', 'English', 'French'.",
    "companies":           "Company lookup. industry field e.g. 'Software Development', 'Financial Services'.",
    "countries":           "Country lookup. code is ISO-2 e.g. 'SA', 'US'.",
    "cities":              "City lookup. country_id → countries.",
    "degrees":             "Degree-type lookup e.g. 'MBA', 'PhD', 'Bachelor\\'s'.",
    "fields_of_study":     "Field-of-study lookup e.g. 'Computer Science', 'Business Administration'.",
    "institutions":        "University / school lookup.",
    "proficiency_levels":  "Language proficiency scale: Beginner < Intermediate < Fluent < Native (rank column).",
}

# ─────────────────────────────────────────────────────────────────────────────
# Row counts (approximate, from last introspection — used in system prompt)
# ─────────────────────────────────────────────────────────────────────────────
ROW_COUNTS = {
    "candidates":          10_120,
    "work_experience":     "~30,000+",
    "education":           20_171,
    "candidate_skills":    50_638,
    "candidate_languages": 25_276,
    "skills":              1_551,
    "skill_categories":    112,
    "languages":           49,
    "companies":           2_263,
    "countries":           56,
    "cities":              318,
    "degrees":             242,
    "fields_of_study":     348,
    "institutions":        725,
    "proficiency_levels":  4,
}


def build_schema_prompt() -> str:
    """Return the full schema context injected into the LLM system prompt."""
    rel_block = "\n".join(f"  {r}" for r in RELATIONSHIPS)
    enum_block = "\n".join(
        f"  {col}: {vals}" for col, vals in ENUM_VALUES.items()
    )
    desc_block = "\n".join(
        f"  {t}: {d}" for t, d in TABLE_DESCRIPTIONS.items()
    )
    count_block = "\n".join(
        f"  {t:35}: {n:>10}" for t, n in ROW_COUNTS.items()
    )
    return f"""
=== DATABASE: candidate_profiles (PostgreSQL) ===

--- TABLE DESCRIPTIONS ---
{desc_block}

--- ROW COUNTS (approximate) ---
{count_block}

--- FOREIGN KEY RELATIONSHIPS ---
{rel_block}

--- KNOWN ENUM VALUES (use exact spelling, case-sensitive) ---
{enum_block}

--- FULL DDL ---
{DDL}
""".strip()
