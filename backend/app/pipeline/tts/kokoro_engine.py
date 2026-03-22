from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.app.core.runtime_env import load_runtime_env
from backend.app.core.schemas import ModelManifest
from backend.app.pipeline.base import TTSEngine
from backend.app.pipeline.device_utils import normalize_device_name, onnx_cuda_available, preload_onnx_cuda_runtime


load_runtime_env()
preload_onnx_cuda_runtime()


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
        requested_device = normalize_device_name(config.get("device", "gpu"), default="gpu")
        model_file = root / config.get("model_file", "kokoro-v0_19.onnx")
        voices_file = root / config.get("voices_file", "voices.bin")
        if not model_file.exists() or not voices_file.exists():
            raise FileNotFoundError(
                "Kokoro requires both the ONNX model file and voices.bin inside the selected TTS folder."
            )

        providers: list[str | tuple[str, dict[str, int]]] = ["CPUExecutionProvider"]
        if requested_device == "cuda":
            preload_onnx_cuda_runtime()
            if not onnx_cuda_available():
                raise RuntimeError(
                    "GPU was selected for Kokoro, but ONNX Runtime still does not expose CUDAExecutionProvider in this environment."
                )
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider" if requested_device == "cuda" else "CPUExecutionProvider"
        self.voice = config.get("voice", "af")
        self.speed = float(config.get("speed", 1.0))
        self.lang = config.get("lang", "en-us")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_file), sess_options=session_options, providers=providers)
        self.providers = session.get_providers()
        self.engine = Kokoro.from_session(session, str(voices_file))
        self.voice_style = self.engine.get_voice_style(self.voice)

    def synthesize(self, text: str, output_path: Path) -> Path:
        import soundfile as sf

        prepared_text = _compact_tts_text(text)
        phonemes = self.engine.tokenizer.phonemize(prepared_text, self.lang)
        batched_phonemes = self.engine._split_phonemes(phonemes)
        audio_segments: list[np.ndarray] = []
        sample_rate = 24000

        for batch in batched_phonemes:
            audio_part, sample_rate = self.engine._create_audio(batch, self.voice_style, self.speed)
            audio_segments.append(_fast_trim_audio(audio_part))

        if audio_segments:
            samples = np.concatenate(audio_segments).astype(np.float32, copy=False)
        else:
            samples = np.zeros(1, dtype=np.float32)
        sf.write(output_path, samples, sample_rate)
        return output_path

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            self.synthesize("Ready.", Path(handle.name))


def _compact_tts_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = normalized.replace("•", ", ").replace("*", "")
    return normalized.strip() or "."


def _fast_trim_audio(audio: np.ndarray, *, threshold: float = 1e-3, padding_samples: int = 480) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    non_silent = np.flatnonzero(np.abs(samples) > threshold)
    if non_silent.size == 0:
        return samples

    start = max(int(non_silent[0]) - padding_samples, 0)
    end = min(int(non_silent[-1]) + padding_samples + 1, samples.shape[0])
    return samples[start:end]
