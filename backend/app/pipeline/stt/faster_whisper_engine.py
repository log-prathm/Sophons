from __future__ import annotations

from pathlib import Path

from backend.app.core.schemas import ModelManifest
from backend.app.core.settings import Settings
from backend.app.pipeline.base import STTEngine


class FasterWhisperEngine(STTEngine):
    def __init__(self, manifest: ModelManifest, settings: Settings) -> None:
        super().__init__(manifest)
        from faster_whisper import WhisperModel

        config = manifest.config
        model_source = manifest.path or manifest.source or "small.en"
        self.transcribe_kwargs = {
            "beam_size": config.get("beam_size", 1),
            "language": config.get("language", "en"),
            "initial_prompt": config.get(
                "initial_prompt",
                "Indian English, Hindi accent, customer support conversation",
            ),
            "vad_filter": config.get("vad_filter", True),
            "vad_parameters": config.get(
                "vad_parameters",
                {"min_silence_duration_ms": 500},
            ),
        }
        self.model = WhisperModel(
            model_source,
            device=config.get("device", "cpu"),
            compute_type=config.get("compute_type", "int8"),
            cpu_threads=config.get("cpu_threads", settings.cpu_threads),
            download_root=str(settings.stt_dir),
        )

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), **self.transcribe_kwargs)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text.strip()

