from __future__ import annotations

import os
from pathlib import Path

import torch

from backend.app.core.schemas import ModelManifest
from backend.app.pipeline.base import TTSEngine


class MeloTTSEngine(TTSEngine):
    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._prime_unidic()
        try:
            from melo.api import TTS
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "Melo TTS is currently experimental in this shared environment. "
                "The main stable GPU TTS path is Kokoro."
            ) from exc

        config = manifest.config
        self.language = config.get("language", "EN")
        self.speed = float(config.get("speed", 0.95))
        self.speaker = config.get("speaker", "EN_INDIA")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = TTS(language=self.language, device=device)
        self.speaker_ids = self.engine.hps.data.spk2id
        if self.speaker not in self.speaker_ids:
            normalized = self.speaker.replace("-", "_").upper()
            if normalized in self.speaker_ids:
                self.speaker = normalized
            elif "EN_INDIA" in self.speaker_ids:
                self.speaker = "EN_INDIA"
            else:
                self.speaker = next(iter(self.speaker_ids.keys()))

    def synthesize(self, text: str, output_path: Path) -> Path:
        self.engine.tts_to_file(
            text,
            self.speaker_ids[self.speaker],
            str(output_path),
            speed=self.speed,
        )
        return output_path

    def _prime_unidic(self) -> None:
        try:
            import unidic_lite
        except Exception:
            return

        dic_dir = Path(unidic_lite.DICDIR)
        if dic_dir.exists():
            os.environ.setdefault("MECABRC", str(dic_dir / "mecabrc"))
            os.environ.setdefault("UNIDICDIR", str(dic_dir))
