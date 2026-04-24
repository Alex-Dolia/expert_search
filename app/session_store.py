"""
session_store.py — In-process conversation session store
=========================================================
Maintains conversation context (prior results + query history) so
follow-up queries like "filter those to Saudi Arabia" work correctly.

In production, swap the in-memory dict for Redis / DynamoDB.
"""
from __future__ import annotations

import time
import uuid
from typing import Any


class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.history: list[dict] = []      # list of {role, content}
        self.last_results: list[dict] = [] # last search results for follow-up filtering
        self.last_query: str = ""

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.last_active = time.time()

    def to_context_string(self) -> str:
        """Produce a summary of prior turns for the LLM."""
        if not self.history:
            return ""
        lines = []
        for turn in self.history[-6:]:  # last 3 exchanges
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content'][:300]}")
        return "\n".join(lines)


class SessionStore:
    """Simple in-memory session store with TTL-based cleanup."""

    TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def create(self) -> ConversationSession:
        sid = str(uuid.uuid4())
        session = ConversationSession(sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> ConversationSession | None:
        self._cleanup()
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> ConversationSession:
        if session_id:
            s = self.get(session_id)
            if s:
                return s
        return self.create()

    def _cleanup(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items()
                if now - s.last_active > self.TTL_SECONDS]
        for sid in dead:
            del self._sessions[sid]

    def list_sessions(self) -> list[dict[str, Any]]:
        self._cleanup()
        return [
            {
                "session_id": s.session_id,
                "turn_count": len(s.history),
                "last_active": s.last_active,
            }
            for s in self._sessions.values()
        ]


# Singleton
_store = SessionStore()


def get_store() -> SessionStore:
    return _store
