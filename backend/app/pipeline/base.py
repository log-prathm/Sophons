from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.core.schemas import ChatMessage, ModelManifest


class STTEngine(ABC):
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class LLMEngine(ABC):
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def generate(self, system_prompt: str, history: list[ChatMessage]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class TTSEngine(ABC):
    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> Path:
        raise NotImplementedError

    def close(self) -> None:
        return None
