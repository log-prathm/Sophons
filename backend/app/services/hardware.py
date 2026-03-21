from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import torch

from backend.app.core.schemas import HardwareProfile, LiveMetrics
from backend.app.core.settings import Settings


def _run_line(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def detect_hardware(settings: Settings) -> HardwareProfile:
    cpu_model = _run_line(["bash", "-lc", "lscpu | awk -F: '/Model name/ {print $2; exit}'"]) or "Intel i7-13620H class CPU"
    cpu_model = cpu_model.strip()

    gpu_name = "RTX 4060 8 GB target"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    notes = [
        "STT is pinned to CPU with 8 threads to keep the system responsive.",
        "LLM supports both Transformers int8 and llama.cpp GGUF Q4/Q6 paths for the RTX 4060 8 GB profile.",
        "Melo is the verified GPU TTS path on this machine, while Kokoro remains available as an ONNX option.",
    ]
    if not torch.cuda.is_available():
        notes.append("CUDA is not visible from the current process, so GPU checks are best-effort only in this environment.")

    return HardwareProfile(
        cpu_model=cpu_model,
        cpu_threads=settings.cpu_threads,
        gpu_name=gpu_name,
        gpu_target="NVIDIA GeForce RTX 4060 Laptop GPU 8 GB",
        llm_quantization="int8 via bitsandbytes or GGUF via llama.cpp",
        notes=notes,
    )


def live_metrics() -> LiveMetrics:
    gpu_name: str | None = None
    used_mb: int | None = None
    total_mb: int | None = None
    free_mb: int | None = None

    nvidia_smi = _run_line(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia_smi:
        first = nvidia_smi.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            used_mb = int(float(parts[1]))
            total_mb = int(float(parts[2]))
            free_mb = max(total_mb - used_mb, 0)

    if used_mb is None and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_mb = int(total_bytes / (1024 * 1024))
            free_mb = int(free_bytes / (1024 * 1024))
            used_mb = total_mb - free_mb
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    percent = None
    if used_mb is not None and total_mb:
        percent = round((used_mb / total_mb) * 100, 1)

    return LiveMetrics(
        gpu_memory_used_mb=used_mb,
        gpu_memory_total_mb=total_mb,
        gpu_memory_free_mb=free_mb,
        gpu_memory_percent=percent,
        gpu_name=gpu_name,
        updated_at=datetime.now(timezone.utc),
    )
