from __future__ import annotations

import asyncio
from contextlib import suppress
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.schemas import (
    ChatMessage,
    ModelManifest,
    PipelineStartRequest,
    SessionListItem,
    SessionState,
    TurnMetrics,
    TurnResponse,
)
from backend.app.core.settings import Settings
from backend.app.pipeline.base import LLMEngine, STTEngine, TTSEngine
from backend.app.pipeline.llm.llama_cpp_engine import LlamaCppServerEngine
from backend.app.pipeline.llm.transformers_engine import TransformersQwenEngine
from backend.app.pipeline.stt.faster_whisper_engine import FasterWhisperEngine
from backend.app.pipeline.tts.kokoro_engine import KokoroTTSEngine
from backend.app.pipeline.tts.melo_engine import MeloTTSEngine
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.session_store import SessionStore


@dataclass
class SessionRuntime:
    state: SessionState
    stt_engine: STTEngine
    llm_engine: LLMEngine
    tts_engine: TTSEngine
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PipelineService:
    def __init__(
        self,
        settings: Settings,
        model_registry: ModelRegistry,
        session_store: SessionStore,
    ) -> None:
        self.settings = settings
        self.model_registry = model_registry
        self.session_store = session_store
        self.sessions: dict[str, SessionRuntime] = {}
        self._engine_cache: dict[str, Any] = {}
        self._engine_refs: dict[str, int] = {}
        self._warmed_engines: set[str] = set()
        self._engine_idle_since: dict[str, float] = {}
        self._invalidated_engines: set[str] = set()

    async def start_session(self, payload: PipelineStartRequest) -> SessionState:
        now = _utcnow()
        stt_model = self.model_registry.find("stt", payload.stt_model_id)
        llm_model = self.model_registry.find("llm", payload.llm_model_id)
        tts_model = self.model_registry.find("tts", payload.tts_model_id)

        session_state = SessionState(
            session_id=str(uuid4()),
            stt_model=stt_model,
            llm_model=llm_model,
            tts_model=tts_model,
            system_prompt=payload.system_prompt or DEFAULT_SYSTEM_PROMPT,
            status="warming",
            created_at=now,
            updated_at=now,
            conversation_history=[],
        )
        self.session_store.save(session_state)

        try:
            stt_engine = await asyncio.to_thread(self._acquire_engine, "stt", stt_model)
            llm_engine = await asyncio.to_thread(self._acquire_engine, "llm", llm_model)
            tts_engine = await asyncio.to_thread(self._acquire_engine, "tts", tts_model)
            await asyncio.to_thread(self._warm_engine, "stt", stt_model, stt_engine)
            await asyncio.to_thread(self._warm_engine, "llm", llm_model, llm_engine)
            await asyncio.to_thread(self._warm_engine, "tts", tts_model, tts_engine)
            session_state.status = "ready"
            session_state.updated_at = _utcnow()
            runtime = SessionRuntime(
                state=session_state,
                stt_engine=stt_engine,
                llm_engine=llm_engine,
                tts_engine=tts_engine,
            )
            self.sessions[session_state.session_id] = runtime
            self.session_store.save(session_state)
            return session_state
        except Exception as exc:
            session_state.status = "error"
            session_state.last_error = str(exc)
            session_state.updated_at = _utcnow()
            self.session_store.save(session_state)
            raise

    async def stop_session(self, session_id: str) -> SessionState:
        runtime = self._get_runtime(session_id)
        runtime.state.status = "closed"
        runtime.state.updated_at = _utcnow()
        self.session_store.save(runtime.state)

        await asyncio.to_thread(self._release_engine, "stt", runtime.state.stt_model)
        await asyncio.to_thread(self._release_engine, "llm", runtime.state.llm_model)
        await asyncio.to_thread(self._release_engine, "tts", runtime.state.tts_model)
        self.sessions.pop(session_id, None)
        return runtime.state

    def list_sessions(self) -> list[SessionListItem]:
        return self.session_store.list_sessions(set(self.sessions.keys()))

    def get_session(self, session_id: str) -> SessionState:
        runtime = self.sessions.get(session_id)
        if runtime:
            return runtime.state
        session = self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' was not found.")
        return session

    def invalidate_engine(self, component: str, model_id: str) -> None:
        key = f"{component}:{model_id}"
        if self._engine_refs.get(key, 0) > 0:
            self._invalidated_engines.add(key)
            return
        if key in self._engine_cache:
            self._dispose_engine(key)

    async def process_audio_turn(self, session_id: str, audio_file: UploadFile) -> TurnResponse:
        session = self._get_runtime(session_id)
        async with session.lock:
            pipeline_started = perf_counter()
            audio_path = await self._save_audio_upload(session.state.session_id, audio_file)
            stt_started = perf_counter()
            transcript = await asyncio.to_thread(session.stt_engine.transcribe, audio_path)
            stt_ms = (perf_counter() - stt_started) * 1000
            return await self._complete_turn(session, transcript, stt_ms=stt_ms, pipeline_started=pipeline_started)

    async def process_text_turn(self, session_id: str, text: str) -> TurnResponse:
        session = self._get_runtime(session_id)
        async with session.lock:
            pipeline_started = perf_counter()
            return await self._complete_turn(
                session,
                text.strip(),
                stt_ms=0.0,
                pipeline_started=pipeline_started,
            )

    def audio_path(self, session_id: str, filename: str) -> Path:
        audio_path = self.settings.audio_dir / session_id / filename
        if not audio_path.exists():
            raise FileNotFoundError(filename)
        return audio_path

    async def _complete_turn(
        self,
        session: SessionRuntime,
        transcript: str,
        *,
        stt_ms: float,
        pipeline_started: float,
    ) -> TurnResponse:
        if not transcript:
            raise ValueError("No speech was detected in the provided audio.")

        now = _utcnow()
        session.state.conversation_history.append(
            ChatMessage(role="user", content=transcript, created_at=now)
        )

        llm_history = session.state.conversation_history[-self.settings.conversation_window :]
        llm_started = perf_counter()
        effective_system_prompt = _compose_system_prompt(
            session.state.system_prompt,
            session.state.tts_model,
        )
        assistant_text = await asyncio.to_thread(
            session.llm_engine.generate,
            effective_system_prompt,
            llm_history,
        )
        llm_ms = (perf_counter() - llm_started) * 1000
        assistant_message = ChatMessage(role="assistant", content=assistant_text, created_at=_utcnow())
        session.state.conversation_history.append(assistant_message)

        turn_index = len(session.state.conversation_history)
        output_dir = self.settings.audio_dir / session.state.session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"turn-{turn_index}-assistant.wav"
        tts_started = perf_counter()
        await asyncio.to_thread(session.tts_engine.synthesize, assistant_text, output_path)
        tts_ms = (perf_counter() - tts_started) * 1000

        session.state.updated_at = _utcnow()
        self.session_store.save(session.state)

        metrics = TurnMetrics(
            stt_ms=round(stt_ms, 1),
            llm_ms=round(llm_ms, 1),
            tts_ms=round(tts_ms, 1),
            pipeline_ms=round((perf_counter() - pipeline_started) * 1000, 1),
        )

        return TurnResponse(
            session_id=session.state.session_id,
            transcript=transcript,
            assistant_text=assistant_text,
            assistant_audio_url=f"/api/sessions/{session.state.session_id}/audio/{output_path.name}",
            conversation_history=session.state.conversation_history,
            metrics=metrics,
        )

    async def _save_audio_upload(self, session_id: str, audio_file: UploadFile) -> Path:
        suffix = Path(audio_file.filename or "input.wav").suffix or ".wav"
        target_dir = self.settings.audio_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"turn-{uuid4().hex}{suffix}"

        with target_path.open("wb") as handle:
            shutil.copyfileobj(audio_file.file, handle)

        return target_path

    def _acquire_engine(self, component: str, manifest: ModelManifest) -> Any:
        key = self._engine_key(component, manifest)
        self._evict_idle_engines(component, keep_key=key)
        if key not in self._engine_cache:
            self._engine_cache[key] = self._build_engine(component, manifest)
            self._engine_refs[key] = 0
        self._engine_idle_since.pop(key, None)
        self._engine_refs[key] += 1
        return self._engine_cache[key]

    def _release_engine(self, component: str, manifest: ModelManifest) -> None:
        key = self._engine_key(component, manifest)
        refs = self._engine_refs.get(key, 0)
        if refs <= 1:
            if key in self._invalidated_engines:
                self._invalidated_engines.discard(key)
                self._dispose_engine(key)
                return
            if key in self._engine_cache:
                self._engine_refs[key] = 0
                self._engine_idle_since[key] = perf_counter()
            return
        self._engine_refs[key] = refs - 1

    def _warm_engine(self, component: str, manifest: ModelManifest, engine: Any) -> None:
        key = self._engine_key(component, manifest)
        if key in self._warmed_engines:
            return
        if hasattr(engine, "warmup"):
            engine.warmup()
        self._warmed_engines.add(key)

    def _engine_key(self, component: str, manifest: ModelManifest) -> str:
        return f"{component}:{manifest.id}"

    def _evict_idle_engines(self, component: str, *, keep_key: str) -> None:
        now = perf_counter()
        ttl_seconds = max(self.settings.engine_idle_ttl_seconds, 0)
        for key in list(self._engine_cache.keys()):
            if not key.startswith(f"{component}:") or key == keep_key:
                continue
            if self._engine_refs.get(key, 0) > 0:
                continue
            self._dispose_engine(key)

        if keep_key in self._engine_cache:
            return

        idle_since = self._engine_idle_since.get(keep_key)
        if idle_since is not None and (ttl_seconds == 0 or now - idle_since >= ttl_seconds):
            self._dispose_engine(keep_key)

    def _dispose_engine(self, key: str) -> None:
        engine = self._engine_cache.pop(key, None)
        self._engine_refs.pop(key, None)
        self._engine_idle_since.pop(key, None)
        self._invalidated_engines.discard(key)
        self._warmed_engines.discard(key)
        if engine and hasattr(engine, "close"):
            with suppress(Exception):
                engine.close()

    def _build_engine(self, component: str, manifest: ModelManifest) -> Any:
        if component == "stt" and manifest.provider == "faster_whisper":
            return FasterWhisperEngine(manifest, self.settings)
        if component == "llm" and manifest.provider == "transformers_qwen":
            return TransformersQwenEngine(manifest, self.settings)
        if component == "llm" and manifest.provider == "llama_cpp_gguf":
            return LlamaCppServerEngine(manifest, self.settings)
        if component == "tts" and manifest.provider == "kokoro_onnx":
            return KokoroTTSEngine(manifest)
        if component == "tts" and manifest.provider == "melo":
            return MeloTTSEngine(manifest)
        raise ValueError(f"Unsupported {component} provider '{manifest.provider}'.")

    def _get_runtime(self, session_id: str) -> SessionRuntime:
        runtime = self.sessions.get(session_id)
        if runtime is None:
            raise KeyError(f"Session '{session_id}' is not active. Start the pipeline first.")
        return runtime


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, warm local voice assistant running on a personal laptop. "
    "Answer in one or two short sentences unless the user explicitly asks for more detail. "
    "Avoid emojis, keep latency low, and preserve useful conversational context across turns."
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _compose_system_prompt(system_prompt: str, tts_model: ModelManifest) -> str:
    style_prompt = str(tts_model.config.get("style_prompt", "") or "").strip()
    if not style_prompt:
        return system_prompt

    return "\n".join(
        [
            system_prompt.strip(),
            "",
            "Speech delivery guidance for the active TTS voice:",
            style_prompt,
            (
                "Write the answer so it sounds natural when spoken aloud. "
                "Use wording, rhythm, and punctuation that support the requested delivery style, "
                "but do not mention these instructions explicitly."
            ),
        ]
    ).strip()
