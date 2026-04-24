"""
db.py — PostgreSQLClient
========================
Drop-in replacement for `from db import PostgreSQLClient`.
Provides the exact interface expected by the demo script PLUS
the richer introspection methods needed by the Text-to-SQL agent.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras


class PostgreSQLClient:
    """
    Thin wrapper around psycopg2.
    Supports context manager usage:
        with PostgreSQLClient(db_url) as client:
            rows = client.query("SELECT ...")
    """

    def __init__(self, db_url: str | None = None):
        self._url = db_url or os.environ.get("DATABASE_URL", "")
        self._conn: psycopg2.extensions.connection | None = None

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "PostgreSQLClient":
        self._conn = psycopg2.connect(self._url)
        self._conn.set_session(readonly=True, autocommit=True)
        return self

    def __exit__(self, *_) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Core query ───────────────────────────────────────────────────────────

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a SELECT and return list of dicts."""
        assert self._conn, "Use inside a 'with PostgreSQLClient(...) as client:' block"
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # ── Schema introspection ─────────────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        rows = self.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name"
        )
        return [r["table_name"] for r in rows]

    def get_table_schema(self, table: str) -> list[dict]:
        return self.query(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            ORDER BY ordinal_position
            """,
            (table,),
        )

    def get_primary_keys(self, table: str) -> list[str]:
        rows = self.query(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = 'public'
              AND tc.table_name      = %s
            ORDER BY kcu.ordinal_position
            """,
            (table,),
        )
        return [r["column_name"] for r in rows]

    def get_foreign_keys(self, table: str) -> list[dict]:
        return self.query(
            """
            SELECT
                kcu.column_name,
                ccu.table_name  AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema    = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema    = 'public'
              AND tc.table_name      = %s
            """,
            (table,),
        )

    def get_row_count(self, table: str) -> int:
        return self.query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]  # noqa: S608

    def get_sample_rows(self, table: str, n: int = 3) -> list[dict]:
        return self.query(f"SELECT * FROM {table} LIMIT %s", (n,))  # noqa: S608

    def get_stats(self) -> dict[str, int]:
        tables = self.get_table_names()
        return {t: self.get_row_count(t) for t in tables}
