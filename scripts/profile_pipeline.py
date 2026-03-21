from __future__ import annotations

import argparse
import asyncio
import json
from io import BytesIO
from pathlib import Path
import sys
from statistics import mean
from time import perf_counter

from fastapi import UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.schemas import PipelineStartRequest
from backend.app.core.settings import get_settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.pipeline_service import PipelineService
from backend.app.services.session_store import SessionStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the local STT -> LLM -> TTS pipeline.")
    parser.add_argument("--audio", required=True, help="Path to a WAV file to replay through the pipeline.")
    parser.add_argument("--stt", default="faster-whisper-small-en", help="STT model id.")
    parser.add_argument("--llm", default="Qwen3.5-2B-int8", help="LLM model id.")
    parser.add_argument("--tts", default="melo-en-india", help="TTS model id.")
    parser.add_argument("--runs", type=int, default=2, help="Number of repeated audio turns to execute.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    settings = get_settings()
    service = PipelineService(settings, ModelRegistry(settings), SessionStore(settings))
    start_payload = PipelineStartRequest(
        stt_model_id=args.stt,
        llm_model_id=args.llm,
        tts_model_id=args.tts,
    )

    session_started = perf_counter()
    session = await service.start_session(start_payload)
    start_session_ms = round((perf_counter() - session_started) * 1000, 1)

    audio_bytes = audio_path.read_bytes()
    turns: list[dict[str, object]] = []

    try:
        for index in range(args.runs):
            upload = UploadFile(
                filename=f"{audio_path.stem}-{index + 1}{audio_path.suffix}",
                file=BytesIO(audio_bytes),
            )
            turn_started = perf_counter()
            response = await service.process_audio_turn(session.session_id, upload)
            measured_ms = round((perf_counter() - turn_started) * 1000, 1)
            turns.append(
                {
                    "run": index + 1,
                    "metrics": response.metrics.model_dump(),
                    "measured_ms": measured_ms,
                    "transcript": response.transcript,
                    "assistant_text": response.assistant_text,
                }
            )
    finally:
        await service.stop_session(session.session_id)

    metric_names = ("stt_ms", "llm_ms", "tts_ms", "pipeline_ms")
    averages = {
        name: round(mean(float(turn["metrics"][name]) for turn in turns), 1)
        for name in metric_names
    }

    report = {
        "audio": str(audio_path),
        "models": {"stt": args.stt, "llm": args.llm, "tts": args.tts},
        "start_session_ms": start_session_ms,
        "runs": turns,
        "averages": averages,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
