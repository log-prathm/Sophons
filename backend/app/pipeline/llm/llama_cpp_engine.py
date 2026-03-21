from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import error, request

from backend.app.core.schemas import ChatMessage, ModelManifest
from backend.app.core.settings import Settings
from backend.app.pipeline.base import LLMEngine


class LlamaCppServerEngine(LLMEngine):
    def __init__(self, manifest: ModelManifest, settings: Settings) -> None:
        super().__init__(manifest)
        self.settings = settings
        self.model_path = Path(manifest.path or manifest.source or "")
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model was not found at '{self.model_path}'.")

        self.server_bin = settings.llama_cpp_bin_dir / "llama-server"
        if not self.server_bin.exists():
            raise FileNotFoundError(
                f"llama-server was not found at '{self.server_bin}'. Build llama.cpp first."
            )

        config = manifest.config
        self.max_new_tokens = int(config.get("max_new_tokens", settings.llm_max_new_tokens))
        self.temperature = float(config.get("temperature", settings.llm_temperature))
        self.top_p = float(config.get("top_p", settings.llm_top_p))
        self.do_sample = bool(config.get("do_sample", settings.llm_do_sample))
        self.repeat_penalty = float(
            config.get("repetition_penalty", settings.llm_repetition_penalty)
        )
        self.context_size = int(config.get("context_size", 4096))
        self.threads = int(config.get("threads", settings.cpu_threads))
        self.threads_batch = int(config.get("threads_batch", self.threads))
        self.batch_size = int(config.get("batch_size", 1024))
        self.gpu_layers = str(config.get("gpu_layers", "all"))
        self.flash_attn = bool(config.get("flash_attn", True))

        self.process: subprocess.Popen[bytes] | None = None
        self.port: int | None = None
        self.base_url: str | None = None
        self._log_handle: Any | None = None

    def generate(self, system_prompt: str, history: list[ChatMessage]) -> str:
        self._ensure_server()
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend({"role": item.role, "content": item.content} for item in history)
        return self._chat(messages)

    def warmup(self) -> None:
        self._ensure_server()
        self._chat(
            [
                {"role": "system", "content": "You are a voice assistant."},
                {"role": "user", "content": "Reply with ready."},
            ],
            max_tokens=8,
            do_sample=False,
        )

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.port = None
        self.base_url = None
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("llama.cpp server is not ready.")

        sampling_enabled = self.do_sample if do_sample is None else do_sample
        payload = {
            "model": self.manifest.id,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens or self.max_new_tokens,
            "temperature": self.temperature if sampling_enabled else 0.0,
            "top_p": self.top_p if sampling_enabled else 1.0,
            "repeat_penalty": self.repeat_penalty,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama.cpp generation failed: {detail or exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach llama.cpp server: {exc.reason}") from exc

        message = data["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return _clean_response_text("".join(parts))
        return _clean_response_text(str(content))

    def _ensure_server(self) -> None:
        if self.process and self.process.poll() is None and self.base_url:
            return

        self.close()
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        log_path = self.settings.llama_cpp_logs_dir / f"{_safe_name(self.manifest.id)}.log"
        self._log_handle = log_path.open("a", encoding="utf-8")

        env = os.environ.copy()
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{self.settings.llama_cpp_bin_dir}:{existing_ld}"
            if existing_ld
            else str(self.settings.llama_cpp_bin_dir)
        )

        command = [
            str(self.server_bin),
            "--model",
            str(self.model_path),
            "--alias",
            self.manifest.id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--threads",
            str(self.threads),
            "--threads-batch",
            str(self.threads_batch),
            "--batch-size",
            str(self.batch_size),
            "--parallel",
            "1",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "none",
            "--cache-reuse",
            "256",
            "--no-webui",
            "--jinja",
            "--flash-attn",
            "on" if self.flash_attn else "off",
            "--gpu-layers",
            self.gpu_layers,
        ]

        self.process = subprocess.Popen(
            command,
            cwd=str(self.settings.llama_cpp_bin_dir),
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )

        deadline = time.time() + 120
        last_error: str | None = None
        while time.time() < deadline:
            if self.process.poll() is not None:
                break
            try:
                req = request.Request(f"{self.base_url}/v1/models", method="GET")
                with request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # pragma: no cover - startup race
                last_error = str(exc)
                time.sleep(0.5)

        log_tail = _tail_text(log_path)
        raise RuntimeError(
            "llama.cpp server did not become ready for "
            f"{self.manifest.id}. Last error: {last_error or 'startup failed'}\n{log_tail}"
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _tail_text(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return text[-limit:]


def _clean_response_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
