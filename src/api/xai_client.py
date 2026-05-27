"""
xAI API client (OpenAI-compatible chat completions).

Set ``XAI_API_KEY`` or ``xai_api_key`` in ``.env``.
Optional: ``XAI_MODEL_TEXT``, ``XAI_MODEL_VISION``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_TEXT_MODEL = "grok-3"
DEFAULT_VISION_MODEL = "grok-4"


def get_xai_api_key() -> str:
    """Read API key from environment (supports ``XAI_API_KEY`` or ``xai_api_key``)."""
    return (
        os.getenv("XAI_API_KEY", "").strip()
        or os.getenv("xai_api_key", "").strip()
    )


def get_xai_text_model() -> str:
    return os.getenv("XAI_MODEL_TEXT", DEFAULT_TEXT_MODEL).strip() or DEFAULT_TEXT_MODEL


def get_xai_vision_model() -> str:
    return os.getenv("XAI_MODEL_VISION", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL


def require_xai_api_key(explicit: Optional[str] = None) -> str:
    key = (explicit or get_xai_api_key()).strip()
    if not key:
        raise ValueError(
            "No xAI API key found. Add XAI_API_KEY=your_key to your .env file."
        )
    return key


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    model: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """Call xAI ``/v1/chat/completions`` and return assistant text."""
    payload = {
        "model": model or get_xai_text_model(),
        "messages": messages,
        "temperature": 0,
    }
    resp = requests.post(
        XAI_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"xAI API error {resp.status_code}: {resp.text[:500]}"
        )
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected xAI response shape: {e}\n{data}") from e


def vision_completion(
    image_b64: str,
    mime_type: str,
    prompt: str,
    *,
    api_key: str,
    model: Optional[str] = None,
) -> str:
    """Vision: image + text prompt → model text."""
    data_url = f"data:{mime_type};base64,{image_b64}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return chat_completion(
        messages,
        api_key=api_key,
        model=model or get_xai_vision_model(),
    )


def text_completion(prompt: str, *, api_key: str, model: Optional[str] = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(messages, api_key=api_key, model=model)
