from __future__ import annotations

import os
from pathlib import Path

import torch

from backend.app.core.schemas import ModelManifest
from backend.app.pipeline.base import TTSEngine


class KokoroTTSEngine(TTSEngine):
    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        from kokoro_onnx import Kokoro

        root = Path(manifest.path or "")
        if not root.exists():
            raise FileNotFoundError(
                f"Kokoro model directory was not found at '{root}'. Put the ONNX files under models/TTSs."
            )

        config = manifest.config
        model_file = root / config.get("model_file", "kokoro-v0_19.onnx")
        voices_file = root / config.get("voices_file", "voices.bin")
        if not model_file.exists() or not voices_file.exists():
            raise FileNotFoundError(
                "Kokoro requires both the ONNX model file and voices.bin inside the selected TTS folder."
            )

        os.environ.setdefault(
            "ONNX_PROVIDER",
            "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider",
        )
        self.voice = config.get("voice", "af")
        self.speed = float(config.get("speed", 1.0))
        self.lang = config.get("lang", "en-us")
        self.engine = Kokoro(str(model_file), str(voices_file))

    def synthesize(self, text: str, output_path: Path) -> Path:
        import soundfile as sf

        samples, sample_rate = self.engine.create(
            text=text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )
        sf.write(output_path, samples, sample_rate)
        return output_path

