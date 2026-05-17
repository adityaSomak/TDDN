"""Async VLM dispatch + image-encoding helpers for the eval scripts.

Public surface:
    encode(path)              base64 image bytes
    mime_for(path)            "image/png" or "image/jpeg"
    model_tag(model)          last segment of an HF-style model id
    call_openai(client, ...)  async OpenAI chat completion (retries + reasoning detect)
    call_vllm(session, ...)   async POST to a vLLM /v1/chat/completions server

Both call_openai and call_vllm take pre-built `messages` (OpenAI chat format).
Callers are responsible for encoding images and constructing the message list.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Optional

import aiohttp


REASONING_MODEL_PREFIXES = ("gpt-5.1", "gpt-5.4")


def encode(path: Path) -> str:
    """Return base64-encoded bytes of the file at path."""
    return base64.b64encode(path.read_bytes()).decode()


def mime_for(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def model_tag(model: str) -> str:
    """Strip the HuggingFace-style namespace prefix for use in output paths."""
    return model.split("/")[-1]


async def call_vllm(
    session: aiohttp.ClientSession,
    port: int,
    model: str,
    messages: list[dict],
    sem: asyncio.Semaphore,
) -> str:
    """POST to a vLLM /v1/chat/completions server. Returns assistant content or "" after 3 failures."""
    payload: dict = {
        "model": model, "messages": messages,
        "max_tokens": 4096, "temperature": 0, "seed": 42,
    }
    # Qwen3.6's default thinking-mode CoT routinely overruns max_tokens.
    if "Qwen3.6" in model:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as r:
                    return (await r.json())["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    return ""
                await asyncio.sleep(2)
    return ""


async def call_openai(
    client,
    model: str,
    messages: list[dict],
    sem: asyncio.Semaphore,
    reasoning: Optional[str] = None,
) -> str:
    """Call OpenAI chat completions. Sets reasoning_effort for gpt-5.x, temperature=0 otherwise."""
    kwargs: dict = {"model": model, "messages": messages, "max_completion_tokens": 4096}
    is_reasoning = any(model.startswith(p) for p in REASONING_MODEL_PREFIXES)
    if reasoning and reasoning != "default":
        kwargs["reasoning_effort"] = reasoning
    elif not is_reasoning:
        kwargs["temperature"] = 0
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    return ""
                await asyncio.sleep(3)
    return ""
