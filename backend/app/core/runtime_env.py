from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_runtime_env() -> None:
    global _LOADED
    if _LOADED:
        return

    project_root = Path(__file__).resolve().parents[3]
    for candidate in (project_root / ".env", project_root / ".env.local"):
        _load_env_file(candidate)

    _LOADED = True


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, _strip_quotes(value.strip()))


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
