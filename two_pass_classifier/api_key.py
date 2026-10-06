"""Load a real OpenAI key for paid calls.

Importing this module does not require ``OPENAI_API_KEY``. Offline commands and
tests can import production code with the variable unset.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from .paths import PROJECT_ROOT

_PLACEHOLDER_KEYS = frozenset({"placeholder", "test", "your_openai_key_here"})


class MissingAPIKeyError(RuntimeError):
    """A paid call has no usable OpenAI key."""


def load_real_api_key(project_root: Path | None = None) -> str:
    """Return a non-placeholder key from the environment or ``keys/openai.env``."""
    root = PROJECT_ROOT if project_root is None else project_root
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        env_path = root / "keys" / "openai.env"
        if env_path.is_file():
            key = str(dotenv_values(env_path).get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is missing. Set it in your environment or in "
            f"{root / 'keys' / 'openai.env'} before a paid command"
        )
    if key.lower() in _PLACEHOLDER_KEYS:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is a placeholder. Provide a real key before a paid command"
        )
    return key
