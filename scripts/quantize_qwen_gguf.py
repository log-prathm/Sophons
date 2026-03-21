from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a local Qwen HF model to GGUF and quantize it.")
    parser.add_argument("--source", required=True, help="Source Hugging Face model directory")
    parser.add_argument(
        "--llama-cpp-root",
        default="/home/prathmesh/Desktop/llama.cpp/llama.cpp",
        help="Path to the local llama.cpp checkout",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate f16 GGUF file after quantization",
    )
    return parser.parse_args()


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    llama_cpp_root = Path(args.llama_cpp_root).resolve()
    convert_script = llama_cpp_root / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_root / "build" / "bin" / "llama-quantize"

    if not source.exists():
        raise SystemExit(f"Source model directory was not found: {source}")
    if not convert_script.exists():
        raise SystemExit(f"convert_hf_to_gguf.py was not found: {convert_script}")
    if not quantize_bin.exists():
        raise SystemExit(f"llama-quantize was not found: {quantize_bin}")

    model_name = source.name
    output_dir = source.parent
    f16_path = output_dir / f"{model_name}-f16.gguf"
    q6_path = output_dir / f"{model_name}-Q6_K.gguf"
    q4_path = output_dir / f"{model_name}-Q4_K_M.gguf"

    run_checked(
        [
            sys.executable,
            str(convert_script),
            str(source),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )

    run_checked([str(quantize_bin), str(f16_path), str(q6_path), "Q6_K"])
    run_checked([str(quantize_bin), str(f16_path), str(q4_path), "Q4_K_M"])

    if not args.keep_f16:
        f16_path.unlink(missing_ok=True)
        print(f"Removed intermediate file: {f16_path}")

    print(f"Saved 6-bit GGUF to {q6_path}")
    print(f"Saved 4-bit GGUF to {q4_path}")


if __name__ == "__main__":
    main()
