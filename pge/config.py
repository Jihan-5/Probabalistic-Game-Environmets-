"""Configuration loader for API keys.

Loads from a `.env` file at the repo root (preferred) or process env vars
(fallback). The .env format is plain `KEY=value` lines, one per line. The
file is gitignored.

This is deliberately tiny — no python-dotenv dependency — to keep the
package's runtime footprint minimal.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _find_env_file() -> Path | None:
    """Walk up from CWD looking for .env (max 4 levels)."""
    cur = Path.cwd().resolve()
    for _ in range(5):
        candidate = cur / ".env"
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


@lru_cache(maxsize=1)
def _load_dotenv() -> dict[str, str]:
    env_file = _find_env_file()
    if env_file is None:
        return {}
    out: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def get(key: str, default: str | None = None) -> str | None:
    """Return value from .env if present, else from os.environ, else default."""
    return _load_dotenv().get(key) or os.environ.get(key, default)


def require(key: str) -> str:
    """Same as get() but raises if missing — for keys the code can't run without."""
    value = get(key)
    if not value:
        raise RuntimeError(
            f"Missing required config key: {key}. "
            f"Set it in .env (preferred) or export it in your shell."
        )
    return value
