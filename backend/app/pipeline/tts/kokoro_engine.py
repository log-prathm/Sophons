from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from backend.app.core.runtime_env import load_runtime_env
from backend.app.core.schemas import ModelManifest
from backend.app.pipeline.base import TTSEngine
from backend.app.pipeline.device_utils import normalize_device_name, onnx_cuda_available, preload_onnx_cuda_runtime, torch_cuda_available


load_runtime_env()
preload_onnx_cuda_runtime()

_NATIVE_VOICE_DEFAULTS = {
    "af": "af_heart",
    "am": "am_adam",
    "bf": "bf_emma",
    "bm": "bm_george",
}

_NATIVE_LANG_CODE_DEFAULTS = {
    "en": "a",
    "en-us": "a",
    "en_us": "a",
    "en-gb": "b",
    "en_gb": "b",
}


class KokoroTTSEngine(TTSEngine):
    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        config = manifest.config
        self.voice = str(config.get("voice", "af") or "af").strip()
        self.speed = float(config.get("speed", 1.0))
        self.lang = str(config.get("lang", "en-us") or "en-us").strip()
        requested_device = normalize_device_name(config.get("device", "gpu"), default="gpu")
        runtime_backend = _normalize_runtime_backend(config.get("runtime_backend", "auto"))
        self._runtime_backend = "onnx"
        self.engine = None
        self.voice_style = None
        self.native_voice = None
        self.native_lang_code = None
        self.providers: list[str] = []

        if runtime_backend in {"auto", "native"} and _native_kokoro_available():
            self._init_native_runtime(config, requested_device)
            return

        if runtime_backend == "native":
            raise RuntimeError(
                "Kokoro native runtime was selected, but the 'kokoro' package is not installed. "
                "Install 'kokoro>=0.9.2' and 'misaki[en]' to use the newer KPipeline path."
            )

        self._init_onnx_runtime(manifest, config, requested_device)

    def _init_onnx_runtime(
        self,
        manifest: ModelManifest,
        config: dict[str, Any],
        requested_device: str,
    ) -> None:
        from kokoro_onnx import Kokoro

        root = Path(manifest.path or "")
        if not root.exists():
            raise FileNotFoundError(
                f"Kokoro model directory was not found at '{root}'. Put the ONNX files under models/TTSs."
            )

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
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_file), sess_options=session_options, providers=providers)
        self.providers = session.get_providers()
        self.engine = Kokoro.from_session(session, str(voices_file))
        self.voice_style = self.engine.get_voice_style(self.voice)
        self._runtime_backend = "onnx"

    def _init_native_runtime(self, config: dict[str, Any], requested_device: str) -> None:
        from kokoro import KPipeline

        native_device = "cuda" if requested_device == "cuda" else "cpu"
        if native_device == "cuda" and not torch_cuda_available():
            raise RuntimeError(
                "GPU was selected for Kokoro native runtime, but PyTorch CUDA is not available in this environment."
            )

        lang_code = _resolve_native_lang_code(config.get("native_lang_code"), self.lang, self.voice)
        native_voice = _resolve_native_voice(config.get("native_voice"), self.voice, lang_code)
        try:
            self.engine = KPipeline(lang_code=lang_code, device=native_device)
        except TypeError:
            self.engine = KPipeline(lang_code=lang_code)
        self.native_voice = native_voice
        self.native_lang_code = lang_code
        self.providers = [native_device]
        self._runtime_backend = "native"

    def synthesize(self, text: str, output_path: Path) -> Path:
        if self._runtime_backend == "native":
            return self._synthesize_native(text, output_path)
        return self._synthesize_onnx(text, output_path)

    def _synthesize_onnx(self, text: str, output_path: Path) -> Path:
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

    def _synthesize_native(self, text: str, output_path: Path) -> Path:
        import soundfile as sf

        prepared_text = _compact_tts_text(text)
        chunks = self.engine(prepared_text, voice=self.native_voice, speed=self.speed)
        audio_segments: list[np.ndarray] = []
        sample_rate = 24000

        for chunk in _iterate_native_chunks(chunks):
            audio_part = _extract_native_audio(chunk)
            sample_rate = _extract_native_sample_rate(chunk, default=sample_rate)
            audio_segments.append(audio_part)

        if not audio_segments:
            raise RuntimeError("Kokoro native runtime did not produce any audio chunks.")

        samples = np.concatenate(audio_segments).astype(np.float32, copy=False)
        sf.write(output_path, samples, sample_rate)
        return output_path

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            self.synthesize("Ready.", Path(handle.name))

    def close(self) -> None:
        engine = self.engine
        self.engine = None
        self.voice_style = None
        self.native_voice = None
        gc.collect()
        if self._runtime_backend == "native":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        del engine


def _native_kokoro_available() -> bool:
    try:
        from kokoro import KPipeline  # noqa: F401
        return True
    except Exception:
        return False


def _normalize_runtime_backend(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"native", "kpipeline", "kokoro"}:
        return "native"
    if normalized in {"onnx", "legacy", "kokoro_onnx"}:
        return "onnx"
    return "auto"


def _resolve_native_lang_code(explicit: Any, lang: str, voice: str) -> str:
    explicit_code = str(explicit or "").strip().lower()
    if explicit_code:
        return explicit_code

    normalized_lang = str(lang or "").strip().lower().replace("_", "-")
    if normalized_lang in _NATIVE_LANG_CODE_DEFAULTS:
        return _NATIVE_LANG_CODE_DEFAULTS[normalized_lang]

    voice_prefix = str(voice or "").strip().lower()
    if len(voice_prefix) >= 1 and voice_prefix[0] in {"a", "b"}:
        return voice_prefix[0]

    return "a"


def _resolve_native_voice(explicit: Any, voice: str, lang_code: str) -> str:
    explicit_voice = str(explicit or "").strip()
    if explicit_voice:
        return explicit_voice

    normalized_voice = str(voice or "").strip().lower()
    if "_" in normalized_voice:
        return normalized_voice
    if normalized_voice in _NATIVE_VOICE_DEFAULTS:
        return _NATIVE_VOICE_DEFAULTS[normalized_voice]

    if lang_code == "b":
        return "bf_emma"
    return "af_heart"


def _iterate_native_chunks(chunks: Any) -> Any:
    if chunks is None:
        return []
    if isinstance(chunks, np.ndarray):
        return [chunks]
    if isinstance(chunks, (list, tuple)):
        if chunks and _looks_like_audio(chunks):
            return [chunks]
        return chunks
    return chunks


def _extract_native_audio(chunk: Any) -> np.ndarray:
    if hasattr(chunk, "audio"):
        return np.asarray(chunk.audio, dtype=np.float32).reshape(-1)

    if isinstance(chunk, (tuple, list)):
        for item in reversed(chunk):
            if _looks_like_audio(item):
                return np.asarray(item, dtype=np.float32).reshape(-1)

    if _looks_like_audio(chunk):
        return np.asarray(chunk, dtype=np.float32).reshape(-1)

    raise RuntimeError("Kokoro native runtime returned an unsupported audio chunk shape.")


def _extract_native_sample_rate(chunk: Any, *, default: int) -> int:
    sample_rate = getattr(chunk, "sample_rate", None)
    if isinstance(sample_rate, (int, np.integer)):
        return int(sample_rate)

    if isinstance(chunk, (tuple, list)):
        for item in chunk:
            if isinstance(item, (int, np.integer)) and 8000 <= int(item) <= 96000:
                return int(item)

    return default


def _looks_like_audio(value: Any) -> bool:
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return bool(array.size) and array.ndim >= 1 and np.issubdtype(array.dtype, np.number)


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
