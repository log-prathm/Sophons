from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.schemas import SessionListItem, SessionState
from backend.app.core.settings import Settings


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def save(self, session: SessionState) -> None:
        target = self._path(session.session_id)
        target.write_text(
            session.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> SessionState | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionState.model_validate(payload)

    def list_sessions(self, live_ids: set[str] | None = None) -> list[SessionListItem]:
        items: list[SessionListItem] = []
        for path in sorted(self.settings.sessions_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            session = SessionState.model_validate(payload)
            is_live = session.session_id in (live_ids or set())
            status = session.status
            if not is_live and status in {"ready", "warming"}:
                status = "closed"
            first_user_message = next(
                (message.content for message in session.conversation_history if message.role == "user"),
                None,
            )
            preview = (session.conversation_history[-1].content if session.conversation_history else session.system_prompt)
            items.append(
                SessionListItem(
                    session_id=session.session_id,
                    title=_truncate(first_user_message or f"{session.llm_model.label} conversation", 42),
                    preview=_truncate(preview, 80),
                    updated_at=session.updated_at,
                    status=status,
                    is_live=is_live,
                )
            )

        return sorted(items, key=lambda item: item.updated_at, reverse=True)

    def _path(self, session_id: str) -> Path:
        return self.settings.sessions_dir / f"{session_id}.json"


def _truncate(value: str, size: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= size:
        return compact
    return f"{compact[: size - 1].rstrip()}..."
