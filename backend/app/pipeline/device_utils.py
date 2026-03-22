from __future__ import annotations

from typing import Any

from backend.app.core.runtime_env import load_runtime_env


GPU_DEVICE_ALIASES = {"gpu", "cuda", "cuda:0"}
CPU_DEVICE_ALIASES = {"cpu"}


load_runtime_env()


def normalize_device_name(value: Any, *, default: str = "cpu") -> str:
    normalized = str(value or default).strip().lower()
    if not normalized:
        normalized = default.strip().lower()

    if normalized in GPU_DEVICE_ALIASES:
        return "cuda"
    if normalized in CPU_DEVICE_ALIASES:
        return "cpu"
    if normalized == "auto":
        return "auto"

    fallback = str(default).strip().lower()
    if fallback in GPU_DEVICE_ALIASES:
        return "cuda"
    if fallback == "auto":
        return "auto"
    return "cpu"


def ctranslate2_cuda_available() -> bool:
    load_runtime_env()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def preload_onnx_cuda_runtime() -> None:
    load_runtime_env()
    try:
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload(cuda=True, cudnn=True)
    except Exception:
        pass


def onnx_cuda_available() -> bool:
    load_runtime_env()
    try:
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload(cuda=True, cudnn=True)
        return "CUDAExecutionProvider" in set(ort.get_available_providers())
    except Exception:
        return False


def torch_cuda_available() -> bool:
    load_runtime_env()
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
