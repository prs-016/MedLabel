"""
xAI API client (OpenAI-compatible chat completions).

Set ``XAI_API_KEY`` or ``xai_api_key`` in ``.env``.
"""

from __future__ import annotations

import os


def get_xai_api_key() -> str:
    return (
        os.getenv("XAI_API_KEY", "").strip()
        or os.getenv("xai_api_key", "").strip()
    )
