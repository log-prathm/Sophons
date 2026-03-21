from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.schemas import ComponentKind, ModelCatalog, ModelManifest
from backend.app.core.settings import Settings
from backend.app.services.hardware import detect_hardware


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def discover(self) -> ModelCatalog:
        return ModelCatalog(
            stt=self._discover_component("stt", self.settings.stt_dir),
            llm=self._discover_component("llm", self.settings.llm_dir),
            tts=self._discover_component("tts", self.settings.tts_dir),
            hardware=detect_hardware(self.settings),
        )

    def find(self, component: ComponentKind, model_id: str) -> ModelManifest:
        options = self._discover_component(component, self._component_dir(component))
        for option in options:
            if option.id == model_id:
                return option
        raise KeyError(f"{component} model '{model_id}' was not found in models/")

    def _component_dir(self, component: ComponentKind) -> Path:
        return {
            "stt": self.settings.stt_dir,
            "llm": self.settings.llm_dir,
            "tts": self.settings.tts_dir,
        }[component]

    def _discover_component(self, component: ComponentKind, root: Path) -> list[ModelManifest]:
        manifests: list[ModelManifest] = []
        seen_ids: set[str] = set()

        root.mkdir(parents=True, exist_ok=True)

        for path in sorted(root.iterdir()):
            if path.name.startswith("."):
                continue
            manifest = None
            if path.is_file() and path.suffix == ".json":
                manifest = self._load_manifest(path, component)
            elif path.is_dir():
                manifest = self._infer_manifest(path, component)

            if manifest and manifest.id not in seen_ids:
                manifests.append(manifest)
                seen_ids.add(manifest.id)

        return manifests

    def _load_manifest(self, path: Path, component: ComponentKind) -> ModelManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("component", component)

        rel_path = payload.get("path")
        if rel_path:
            resolved = (path.parent / rel_path).resolve()
            payload["path"] = str(resolved)

        return ModelManifest.model_validate(payload)

    def _infer_manifest(self, path: Path, component: ComponentKind) -> ModelManifest | None:
        if component == "llm" and (path / "config.json").exists():
            return ModelManifest(
                id=path.name,
                label=f"{path.name} (8-bit)",
                component="llm",
                provider="transformers_qwen",
                path=str(path.resolve()),
                description="Local Hugging Face model folder loaded with bitsandbytes int8 on GPU.",
                config={
                    "quantization": "int8",
                    "max_new_tokens": self.settings.llm_max_new_tokens,
                    "temperature": self.settings.llm_temperature,
                    "top_p": self.settings.llm_top_p,
                    "repetition_penalty": self.settings.llm_repetition_penalty,
                },
            )

        if component == "tts" and any(child.suffix == ".onnx" for child in path.iterdir()):
            return ModelManifest(
                id=path.name,
                label=path.name,
                component="tts",
                provider="kokoro_onnx",
                path=str(path.resolve()),
                description="Inferred ONNX TTS directory.",
                config={
                    "model_file": "kokoro-v0_19.onnx",
                    "voices_file": "voices.bin",
                    "voice": "af",
                    "speed": 1.0,
                    "lang": "en-us",
                },
            )

        return None

