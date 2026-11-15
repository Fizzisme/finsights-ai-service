"""Versioned system prompts: one Markdown+XML file per agent, kept static so the prefix stays cacheable.

Per-request data (today's date, the user's question) goes in the user message, never in these files.
"""

from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).parent


@lru_cache
def load_prompt(name: str) -> str:
    path = DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"unknown prompt {name!r}")
    # Normalise line endings so the prompt is byte-identical on Windows and Linux.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip() + "\n"
