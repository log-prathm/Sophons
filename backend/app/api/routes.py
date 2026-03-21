from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.schemas import (
    LiveMetrics,
    ModelCatalog,
    ModelConfigEditor,
    ModelConfigSaveResponse,
    ModelConfigUpdateRequest,
    PipelineStartRequest,
    PipelineStartResponse,
    SessionListItem,
    SessionSummary,
    TextTurnRequest,
    TurnResponse,
)
from backend.app.core.settings import Settings, get_settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.hardware import live_metrics
from backend.app.services.pipeline_service import PipelineService
from backend.app.services.session_store import SessionStore


router = APIRouter()


def get_model_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    return ModelRegistry(settings)


def get_pipeline_service(settings: Settings = Depends(get_settings)) -> PipelineService:
    if not hasattr(get_pipeline_service, "_instance"):
        get_pipeline_service._instance = PipelineService(  # type: ignore[attr-defined]
            settings=settings,
            model_registry=ModelRegistry(settings),
            session_store=SessionStore(settings),
        )
    return get_pipeline_service._instance  # type: ignore[attr-defined]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics/live", response_model=LiveMetrics)
def get_live_metrics() -> LiveMetrics:
    return live_metrics()


@router.get("/models", response_model=ModelCatalog)
def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelCatalog:
    return registry.discover()


@router.get("/models/{component}/{model_id}/hyperparameters", response_model=ModelConfigEditor)
def get_model_hyperparameters(
    component: str,
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelConfigEditor:
    try:
        return registry.config_editor(component, model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/models/{component}/{model_id}/hyperparameters", response_model=ModelConfigSaveResponse)
def update_model_hyperparameters(
    component: str,
    model_id: str,
    payload: ModelConfigUpdateRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    service: PipelineService = Depends(get_pipeline_service),
) -> ModelConfigSaveResponse:
    try:
        editor = registry.save_config(component, model_id, payload.config)
        service.invalidate_engine(component, model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelConfigSaveResponse(editor=editor)


@router.get("/sessions", response_model=list[SessionListItem])
def list_sessions(service: PipelineService = Depends(get_pipeline_service)) -> list[SessionListItem]:
    return service.list_sessions()


@router.post("/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(
    payload: PipelineStartRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineStartResponse:
    try:
        session = await service.start_session(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PipelineStartResponse(session=session)


@router.post("/sessions/{session_id}/stop", response_model=SessionSummary)
async def stop_pipeline(
    session_id: str,
    service: PipelineService = Depends(get_pipeline_service),
) -> SessionSummary:
    try:
        session = await service.stop_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionSummary(session=session)


@router.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(
    session_id: str,
    service: PipelineService = Depends(get_pipeline_service),
) -> SessionSummary:
    try:
        session = service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionSummary(session=session)


@router.post("/sessions/{session_id}/turns/audio", response_model=TurnResponse)
async def create_audio_turn(
    session_id: str,
    audio: UploadFile = File(...),
    service: PipelineService = Depends(get_pipeline_service),
) -> TurnResponse:
    try:
        return await service.process_audio_turn(session_id, audio)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/turns/text", response_model=TurnResponse)
async def create_text_turn(
    session_id: str,
    payload: TextTurnRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> TurnResponse:
    try:
        return await service.process_text_turn(session_id, payload.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/audio/{filename}")
def get_audio(
    session_id: str,
    filename: str,
    service: PipelineService = Depends(get_pipeline_service),
) -> FileResponse:
    try:
        target = service.audio_path(session_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(target, media_type="audio/wav", filename=filename)
