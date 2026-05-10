from __future__ import annotations

from datetime import datetime

from app.schemas import SessionState


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def save(self, session: SessionState) -> SessionState:
        session.updated_at = datetime.utcnow()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def restore(self, session: SessionState) -> SessionState:
        self._sessions[session.session_id] = session
        return session


store = SessionStore()

