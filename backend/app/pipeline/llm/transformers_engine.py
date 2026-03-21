from __future__ import annotations

import os
from typing import Any

import torch

from backend.app.core.schemas import ChatMessage, ModelManifest
from backend.app.core.settings import Settings
from backend.app.pipeline.base import LLMEngine


class TransformersQwenEngine(LLMEngine):
    def __init__(self, manifest: ModelManifest, settings: Settings) -> None:
        super().__init__(manifest)
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        torch.backends.cuda.matmul.allow_tf32 = True

        self.settings = settings
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.path = manifest.path or manifest.source
        if not self.path:
            raise ValueError("LLM manifest must define a local path or source.")

        config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)
        self.model_type = getattr(config, "model_type", "")

        quantization = None
        if torch.cuda.is_available() and manifest.config.get("quantization", "int8") == "int8":
            quantization = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_cls: Any = AutoModelForCausalLM
        if self.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_cls = Qwen3_5ForConditionalGeneration

        self.model = model_cls.from_pretrained(
            self.path,
            trust_remote_code=True,
            quantization_config=quantization,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.max_new_tokens = int(manifest.config.get("max_new_tokens", settings.llm_max_new_tokens))
        self.temperature = float(manifest.config.get("temperature", settings.llm_temperature))
        self.top_p = float(manifest.config.get("top_p", settings.llm_top_p))
        self.do_sample = bool(manifest.config.get("do_sample", settings.llm_do_sample))
        self.repetition_penalty = float(
            manifest.config.get("repetition_penalty", settings.llm_repetition_penalty)
        )

    def generate(self, system_prompt: str, history: list[ChatMessage]) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend({"role": item.role, "content": item.content} for item in history)

        return self._generate_from_messages(messages)

    def warmup(self) -> None:
        self._generate_from_messages(
            [{"role": "system", "content": "You are a voice assistant."}, {"role": "user", "content": "Reply with ready."}],
            max_new_tokens=8,
            do_sample=False,
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        lines = []
        for message in messages:
            lines.append(f"{message['role'].upper()}: {message['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> str:
        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        sampling_enabled = self.do_sample if do_sample is None else do_sample
        generate_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": sampling_enabled,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }
        if sampling_enabled:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()
