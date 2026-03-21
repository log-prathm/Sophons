#!/usr/bin/env bash
set -euo pipefail

python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt

