"""HTTP wrapper around llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.

Captures both the visible `content` and the `reasoning_content` (chain-of-thought)
fields returned by llama-server when `--jinja` is enabled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


# Known LLM endpoints (host:port → model alias). Add new servers here.
ENDPOINTS = {
    "glm-4.7-flash": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "alias": "glm-4.7-flash",
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "url": "http://127.0.0.1:8081/v1/chat/completions",
        "alias": "deepseek-r1-distill-qwen-1.5b",
    },
}


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str
    finish_reason: str
    usage: dict
    latency_seconds: float
    raw: dict  # the full response body from the server


class LLMClient:
    """Thin OpenAI-compatible client for a single llama.cpp server."""

    def __init__(self, model_key: str, timeout_seconds: float = 1800.0):
        # Default bumped from 600s → 1800s to handle worst-case reasoning chains
        # at max_tokens=8192 under resource contention (e.g., other GPU apps).
        if model_key not in ENDPOINTS:
            raise ValueError(f"Unknown model_key {model_key!r}; expected one of {list(ENDPOINTS)}")
        self.model_key = model_key
        cfg = ENDPOINTS[model_key]
        self.url = cfg["url"]
        self.alias = cfg["alias"]
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        body = {
            "model": self.alias,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask llama-server to split <think> blocks into reasoning_content.
            # Controlled by --jinja on the server; this is best-effort here.
        }
        t0 = time.perf_counter()
        resp = self._client.post(self.url, json=body)
        resp.raise_for_status()
        latency = time.perf_counter() - t0
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            reasoning_content=msg.get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage") or {},
            latency_seconds=latency,
            raw=data,
        )
