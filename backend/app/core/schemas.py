from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ComponentKind = Literal["stt", "llm", "tts"]
MessageRole = Literal["system", "user", "assistant"]


class HardwareProfile(BaseModel):
    cpu_model: str
    cpu_threads: int
    gpu_name: str
    gpu_target: str
    llm_quantization: str
    notes: list[str] = Field(default_factory=list)


class ModelManifest(BaseModel):
    id: str
    label: str
    component: ComponentKind
    provider: str
    path: str | None = None
    source: str | None = None
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ModelCatalog(BaseModel):
    stt: list[ModelManifest]
    llm: list[ModelManifest]
    tts: list[ModelManifest]
    hardware: HardwareProfile


class PipelineStartRequest(BaseModel):
    stt_model_id: str
    llm_model_id: str
    tts_model_id: str
    system_prompt: str | None = None


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime


class SessionState(BaseModel):
    session_id: str
    stt_model: ModelManifest
    llm_model: ModelManifest
    tts_model: ModelManifest
    system_prompt: str
    status: Literal["warming", "ready", "error", "closed"]
    created_at: datetime
    updated_at: datetime
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    last_error: str | None = None


class PipelineStartResponse(BaseModel):
    session: SessionState


class TextTurnRequest(BaseModel):
    text: str


class TurnResponse(BaseModel):
    session_id: str
    transcript: str
    assistant_text: str
    assistant_audio_url: str
    conversation_history: list[ChatMessage]
    metrics: TurnMetrics


class SessionSummary(BaseModel):
    session: SessionState


class SessionListItem(BaseModel):
    session_id: str
    title: str
    preview: str
    updated_at: datetime
    status: Literal["warming", "ready", "error", "closed"]
    is_live: bool = False


class TurnMetrics(BaseModel):
    stt_ms: float
    llm_ms: float
    tts_ms: float
    pipeline_ms: float
    heard_to_response_ms: float | None = None


class LiveMetrics(BaseModel):
    gpu_memory_used_mb: int | None = None
    gpu_memory_total_mb: int | None = None
    gpu_memory_free_mb: int | None = None
    gpu_memory_percent: float | None = None
    gpu_name: str | None = None
    updated_at: datetime
