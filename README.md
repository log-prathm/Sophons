# Betelgeuse

Local-first voice pipeline tuned for an i7-13620H + RTX 4060 8 GB laptop.

## Stack

- `STT`: Faster-Whisper on CPU with `small.en`, `int8`, and 8 CPU threads.
- `LLM`: Hugging Face Transformers with 8-bit Qwen loading on GPU.
- `TTS`: Kokoro ONNX on GPU by default, with an experimental Melo adapter scaffold.
- `UI`: Vite + React + TypeScript.

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
cd frontend && npm install
```

## Run

Backend:

```bash
.venv/bin/uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

## Quantize Qwen To 8-Bit

```bash
.venv/bin/python scripts/quantize_qwen_int8.py \
  --source models/LLMs/Qwen3.5-2B \
  --target models/LLMs/Qwen3.5-2B-int8
```

The UI discovers models from `models/` at runtime, so replacing pipeline parts is mostly a matter of adding or editing manifests in those folders.

