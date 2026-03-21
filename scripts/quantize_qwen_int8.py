from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize an 8-bit Qwen checkpoint for this laptop.")
    parser.add_argument("--source", required=True, help="Source model directory")
    parser.add_argument("--target", required=True, help="Target directory for the quantized model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this process, so 8-bit GPU quantization cannot run.")

    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    if getattr(config, "model_type", "") == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration

    quantization = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = model_cls.from_pretrained(
        source,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.float16,
    )
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    print(f"Saved 8-bit model to {target}")


if __name__ == "__main__":
    main()

