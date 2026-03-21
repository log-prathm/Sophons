from __future__ import annotations

from dataclasses import dataclass
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.app.core.schemas import (
    ComponentKind,
    ModelCatalog,
    ModelConfigEditor,
    ModelConfigField,
    ModelConfigOption,
    ModelManifest,
)
from backend.app.core.settings import Settings
from backend.app.services.hardware import detect_hardware


@dataclass
class ManifestEntry:
    manifest: ModelManifest
    manifest_path: Path | None
    directory: Path | None
    artifact_path: Path | None = None


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def discover(self) -> ModelCatalog:
        return ModelCatalog(
            stt=self._discover_component("stt"),
            llm=self._discover_component("llm"),
            tts=self._discover_component("tts"),
            hardware=detect_hardware(self.settings),
        )

    def find(self, component: ComponentKind, model_id: str) -> ModelManifest:
        return self._find_entry(component, model_id).manifest

    def config_editor(self, component: ComponentKind, model_id: str) -> ModelConfigEditor:
        entry = self._find_entry(component, model_id)
        return ModelConfigEditor(
            component=component,
            manifest=entry.manifest,
            fields=self._build_fields(entry.manifest),
            manifest_path=str(self._persist_path(component, entry)),
        )

    def save_config(
        self,
        component: ComponentKind,
        model_id: str,
        config: dict[str, Any],
    ) -> ModelConfigEditor:
        entry = self._find_entry(component, model_id)
        merged = deepcopy(entry.manifest.config)
        for field in self._build_fields(entry.manifest):
            if field.key not in config:
                continue
            _set_nested_value(merged, field.key, _coerce_field_value(field, config[field.key]))

        payload = entry.manifest.model_dump(mode="json")
        payload["config"] = merged

        persist_path = self._persist_path(component, entry)
        if entry.directory is not None and persist_path == entry.directory / "manifest.json":
            payload["path"] = "."
        elif entry.artifact_path is not None and entry.artifact_path.is_file():
            payload["path"] = entry.artifact_path.name

        persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.config_editor(component, model_id)

    def _component_dir(self, component: ComponentKind) -> Path:
        return {
            "stt": self.settings.stt_dir,
            "llm": self.settings.llm_dir,
            "tts": self.settings.tts_dir,
        }[component]

    def _discover_component(self, component: ComponentKind) -> list[ModelManifest]:
        manifests: list[ModelManifest] = []
        seen_ids: set[str] = set()
        for entry in self._iter_component_entries(component):
            if entry.manifest.id not in seen_ids:
                manifests.append(entry.manifest)
                seen_ids.add(entry.manifest.id)

        return manifests

    def _load_manifest(self, path: Path, component: ComponentKind) -> ModelManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("component", component)

        rel_path = payload.get("path")
        if rel_path:
            resolved = (path.parent / rel_path).resolve()
            payload["path"] = str(resolved)

        return ModelManifest.model_validate(payload)

    def _iter_component_entries(self, component: ComponentKind) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        root = self._component_dir(component)
        root.mkdir(parents=True, exist_ok=True)

        deferred_paths: list[Path] = []
        for path in sorted(root.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_file() and path.suffix == ".json":
                entries.append(
                    ManifestEntry(
                        manifest=self._load_manifest(path, component),
                        manifest_path=path,
                        directory=None,
                    )
                )
                continue

            if path.is_dir():
                embedded_manifest = path / "manifest.json"
                if embedded_manifest.exists():
                    entries.append(
                        ManifestEntry(
                            manifest=self._load_manifest(embedded_manifest, component),
                            manifest_path=embedded_manifest,
                            directory=path,
                            artifact_path=path,
                        )
                    )
                    continue

            deferred_paths.append(path)

        for path in deferred_paths:
            entry = None
            if path.is_dir():
                manifest = self._infer_manifest(path, component)
                if manifest:
                    entry = ManifestEntry(
                        manifest=manifest,
                        manifest_path=None,
                        directory=path,
                        artifact_path=path,
                    )
            elif path.is_file():
                manifest = self._infer_file_manifest(path, component)
                if manifest:
                    entry = ManifestEntry(
                        manifest=manifest,
                        manifest_path=None,
                        directory=None,
                        artifact_path=path,
                    )

            if entry:
                entries.append(entry)

        return entries

    def _find_entry(self, component: ComponentKind, model_id: str) -> ManifestEntry:
        for entry in self._iter_component_entries(component):
            if entry.manifest.id == model_id:
                return entry
        raise KeyError(f"{component} model '{model_id}' was not found in models/")

    def _persist_path(self, component: ComponentKind, entry: ManifestEntry) -> Path:
        if entry.manifest_path is not None:
            return entry.manifest_path
        if entry.directory is not None:
            return entry.directory / "manifest.json"
        if entry.artifact_path is not None and entry.artifact_path.suffix == ".gguf":
            return entry.artifact_path.with_suffix(".json")
        return self._component_dir(component) / f"{entry.manifest.id}.json"

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
                    "do_sample": self.settings.llm_do_sample,
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
                    "device": "gpu",
                    "model_file": "kokoro-v0_19.onnx",
                    "voices_file": "voices.bin",
                    "voice": "af",
                    "speed": 1.0,
                    "lang": "en-us",
                    "style_prompt": "",
                },
            )

        return None

    def _infer_file_manifest(self, path: Path, component: ComponentKind) -> ModelManifest | None:
        if component == "llm" and path.suffix == ".gguf":
            quant = _extract_gguf_quant(path.stem)
            quant_label = quant or "GGUF"
            base_label = path.stem[: -(len(quant) + 1)] if quant and path.stem.endswith(f"-{quant}") else path.stem
            return ModelManifest(
                id=path.stem,
                label=f"{base_label} ({quant_label})",
                component="llm",
                provider="llama_cpp_gguf",
                path=str(path.resolve()),
                description="Local GGUF model served through llama.cpp for low-VRAM GPU inference.",
                config={
                    "max_new_tokens": self.settings.llm_max_new_tokens,
                    "temperature": self.settings.llm_temperature,
                    "top_p": self.settings.llm_top_p,
                    "repetition_penalty": self.settings.llm_repetition_penalty,
                    "do_sample": self.settings.llm_do_sample,
                    "context_size": 4096,
                    "threads": self.settings.cpu_threads,
                    "threads_batch": self.settings.cpu_threads,
                    "batch_size": 1024,
                    "gpu_layers": "all",
                    "flash_attn": True,
                    "quantization": quant.lower() if quant else "gguf",
                },
            )

        return None

    def _build_fields(self, manifest: ModelManifest) -> list[ModelConfigField]:
        templates = _field_templates(manifest)
        fields: list[ModelConfigField] = []
        for template in templates:
            field = template.model_copy(deep=True)
            field.value = _get_nested_value(manifest.config, field.key)
            fields.append(field)
        return fields


def _field_templates(manifest: ModelManifest) -> list[ModelConfigField]:
    if manifest.provider == "faster_whisper":
        return [
            ModelConfigField(key="device", label="Device", input_type="select", options=[
                ModelConfigOption(label="CPU", value="cpu"),
                ModelConfigOption(label="GPU", value="cuda"),
            ]),
            ModelConfigField(key="compute_type", label="Compute Type", input_type="select", options=[
                ModelConfigOption(label="int8", value="int8"),
                ModelConfigOption(label="int8_float16", value="int8_float16"),
                ModelConfigOption(label="float16", value="float16"),
                ModelConfigOption(label="float32", value="float32"),
            ]),
            ModelConfigField(
                key="cpu_threads",
                label="CPU Threads",
                input_type="number",
                description="Threads reserved for Whisper inference on your CPU.",
                min=1,
                max=32,
                step=1,
            ),
            ModelConfigField(
                key="beam_size",
                label="Beam Size",
                input_type="number",
                description="Lower is faster. `1` is best for real-time latency.",
                min=1,
                max=5,
                step=1,
            ),
            ModelConfigField(
                key="language",
                label="Language",
                input_type="text",
                placeholder="en",
            ),
            ModelConfigField(
                key="initial_prompt",
                label="Recognition Prompt",
                input_type="textarea",
                description="Bias Whisper toward your accent, domain, or speaking style.",
                placeholder="Indian English, Hindi accent, customer support conversation",
            ),
            ModelConfigField(
                key="vad_filter",
                label="Enable VAD",
                input_type="boolean",
                description="Discard silence before transcription.",
            ),
            ModelConfigField(
                key="vad_parameters.min_silence_duration_ms",
                label="Min Silence Duration (ms)",
                input_type="number",
                description="How long silence must last before VAD treats it as a break.",
                min=100,
                max=3000,
                step=50,
            ),
        ]

    if manifest.provider == "transformers_qwen":
        return [
            ModelConfigField(
                key="quantization",
                label="Quantization",
                input_type="select",
                options=[
                    ModelConfigOption(label="8-bit", value="int8"),
                ],
            ),
            ModelConfigField(
                key="max_new_tokens",
                label="Max New Tokens",
                input_type="number",
                description="Upper bound for each assistant reply.",
                min=16,
                max=512,
                step=1,
            ),
            ModelConfigField(
                key="do_sample",
                label="Sampling",
                input_type="boolean",
                description="Turn on for more creative but slower and less deterministic replies.",
            ),
            ModelConfigField(
                key="temperature",
                label="Temperature",
                input_type="number",
                description="Higher values increase creativity when sampling is enabled.",
                min=0,
                max=2,
                step=0.05,
            ),
            ModelConfigField(
                key="top_p",
                label="Top P",
                input_type="number",
                description="Nucleus sampling threshold.",
                min=0,
                max=1,
                step=0.05,
            ),
            ModelConfigField(
                key="repetition_penalty",
                label="Repetition Penalty",
                input_type="number",
                description="Reduce looping or repeated phrases.",
                min=0.8,
                max=2,
                step=0.05,
            ),
        ]

    if manifest.provider == "llama_cpp_gguf":
        return [
            ModelConfigField(
                key="max_new_tokens",
                label="Max New Tokens",
                input_type="number",
                description="Upper bound for each assistant reply.",
                min=16,
                max=512,
                step=1,
            ),
            ModelConfigField(
                key="do_sample",
                label="Sampling",
                input_type="boolean",
                description="Turn on for more creative but less deterministic replies.",
            ),
            ModelConfigField(
                key="temperature",
                label="Temperature",
                input_type="number",
                description="Higher values increase creativity when sampling is enabled.",
                min=0,
                max=2,
                step=0.05,
            ),
            ModelConfigField(
                key="top_p",
                label="Top P",
                input_type="number",
                description="Nucleus sampling threshold.",
                min=0,
                max=1,
                step=0.05,
            ),
            ModelConfigField(
                key="repetition_penalty",
                label="Repetition Penalty",
                input_type="number",
                description="Reduce looping or repeated phrases.",
                min=0.8,
                max=2,
                step=0.05,
            ),
            ModelConfigField(
                key="context_size",
                label="Context Size",
                input_type="number",
                description="llama.cpp context window. Higher values use more VRAM.",
                min=1024,
                max=16384,
                step=256,
            ),
            ModelConfigField(
                key="threads",
                label="CPU Threads",
                input_type="number",
                description="CPU threads used by llama.cpp generation.",
                min=1,
                max=32,
                step=1,
            ),
            ModelConfigField(
                key="threads_batch",
                label="Batch Threads",
                input_type="number",
                description="CPU threads used while processing prompts.",
                min=1,
                max=32,
                step=1,
            ),
            ModelConfigField(
                key="batch_size",
                label="Batch Size",
                input_type="number",
                description="Prompt-processing batch size for llama.cpp.",
                min=128,
                max=4096,
                step=128,
            ),
            ModelConfigField(
                key="gpu_layers",
                label="GPU Layers",
                input_type="text",
                description="Use `all` to offload as much as possible, `auto` for fitted offload, or `0` for CPU only.",
                placeholder="all",
            ),
            ModelConfigField(
                key="flash_attn",
                label="Flash Attention",
                input_type="boolean",
                description="Use llama.cpp flash attention when available.",
            ),
        ]

    if manifest.provider == "melo":
        return [
            ModelConfigField(
                key="device",
                label="Device",
                input_type="select",
                options=[
                    ModelConfigOption(label="GPU", value="gpu"),
                    ModelConfigOption(label="CPU", value="cpu"),
                ],
                description="Choose where speech synthesis runs.",
            ),
            ModelConfigField(
                key="style_prompt",
                label="Speech Style Prompt",
                input_type="textarea",
                description=(
                    "Describe how replies should sound when spoken, such as warm, calm, "
                    "professional, energetic, or reassuring. This guides reply wording "
                    "for speech delivery across TTS engines."
                ),
                placeholder="Speak in a calm, warm, reassuring tone with short natural pauses.",
            ),
            ModelConfigField(
                key="language",
                label="Language",
                input_type="text",
                placeholder="EN",
            ),
            ModelConfigField(
                key="speaker",
                label="Speaker / Accent",
                input_type="text",
                description="Example: EN_INDIA for Indian English.",
                placeholder="EN_INDIA",
            ),
            ModelConfigField(
                key="speed",
                label="Speech Speed",
                input_type="number",
                description="Lower is slower and clearer. Higher is faster.",
                min=0.5,
                max=1.5,
                step=0.05,
            ),
            ModelConfigField(
                key="disable_bert",
                label="Disable BERT Prosody",
                input_type="boolean",
                description="Much faster on longer replies, with some loss in expressive phrasing.",
            ),
            ModelConfigField(
                key="half_precision",
                label="Use Half Precision",
                input_type="boolean",
                description="Use FP16 on the GPU for lower VRAM use and faster synthesis.",
            ),
            ModelConfigField(
                key="sentence_pause_ms",
                label="Sentence Pause (ms)",
                input_type="number",
                description="Pause inserted between synthesized sentences.",
                min=0,
                max=250,
                step=5,
            ),
        ]

    if manifest.provider == "kokoro_onnx":
        return [
            ModelConfigField(
                key="device",
                label="Device",
                input_type="select",
                options=[
                    ModelConfigOption(label="GPU", value="gpu"),
                    ModelConfigOption(label="CPU", value="cpu"),
                ],
                description="Choose the preferred ONNX execution target.",
            ),
            ModelConfigField(
                key="style_prompt",
                label="Speech Style Prompt",
                input_type="textarea",
                description=(
                    "Describe how replies should sound when spoken, such as warm, calm, "
                    "professional, energetic, or reassuring. This guides reply wording "
                    "for speech delivery across TTS engines."
                ),
                placeholder="Speak in a calm, warm, reassuring tone with short natural pauses.",
            ),
            ModelConfigField(
                key="voice",
                label="Voice",
                input_type="text",
                placeholder="af",
            ),
            ModelConfigField(
                key="speed",
                label="Speech Speed",
                input_type="number",
                min=0.5,
                max=1.5,
                step=0.05,
            ),
            ModelConfigField(
                key="lang",
                label="Language Variant",
                input_type="text",
                placeholder="en-us",
            ),
        ]

    return [
        ModelConfigField(
            key="notes",
            label="Config",
            input_type="textarea",
            description="This provider does not have a custom editor yet.",
        )
    ]


def _get_nested_value(payload: dict[str, Any], key: str) -> Any:
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested_value(payload: dict[str, Any], key: str, value: Any) -> None:
    current = payload
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _coerce_field_value(field: ModelConfigField, value: Any) -> Any:
    if field.input_type == "number":
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        return float(text) if "." in text else int(text)
    if field.input_type == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return None
    return str(value)


def _extract_gguf_quant(stem: str) -> str | None:
    last_segment = stem.rsplit("-", 1)[-1]
    if last_segment.upper().startswith(("Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ")):
        return last_segment
    return None
