"""Shared OpenAI-compatible chat client for the LLM-driven mining methods.

Configured through the same environment variables the other LLM methods in
this repo use (``OPENAI_API_BASE`` / ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
``CHAT_MODEL``), so a single exported environment drives AlphaAgent,
QuantaAlpha, RD-Agent and this method alike.

The default sampling temperatures are Alpha Jungle's (its Appendix G): 1.0 to
generate, 0.8 to correct an invalid output, 0.1 to score. Callers that need
different values pass their own.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

TEMPERATURE_GENERATE = 1.0
TEMPERATURE_CORRECT = 0.8
TEMPERATURE_SCORE = 0.1

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_REPO_ROOT = Path(__file__).resolve().parents[1]
# RD-Agent and AlphaAgent go through LiteLLM, which wants a provider-prefixed
# model name ("openai/arc:nexus"). This client speaks the OpenAI HTTP API
# directly, where that prefix is part of the model id and 404s. Strip it so one
# .env drives both.
_LITELLM_PREFIX = "openai/"


def load_dotenv(path: Path | None = None) -> None:
    """Load the repo-root ``.env`` without clobbering the real environment.

    The other LLM methods get this file loaded for them by their own
    frameworks; doing it here keeps a bare ``python -m ...run`` working with
    the same configuration.
    """
    env_path = path or _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # An explicitly exported variable always wins over the file.
        os.environ.setdefault(key, value)


class LLMError(RuntimeError):
    """The model could not be reached, or did not return usable JSON."""


class ChatClient(Protocol):
    """Minimal interface the search needs; see :class:`FakeClient` for tests."""

    def complete(self, prompt: str, temperature: float) -> str: ...


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Callers may issue requests from several threads; `+=` on an int attribute
    # is not atomic, so every update goes through `record`.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False,
                                  compare=False)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class OpenAIChatClient:
    """Chat-completions client with retry on transient failures."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        load_dotenv()
        base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        if not base:
            raise LLMError(
                "No LLM endpoint configured. Set OPENAI_API_BASE (or OPENAI_BASE_URL), "
                "OPENAI_API_KEY and CHAT_MODEL in the environment or in the "
                "repo-root .env, the same variables the other LLM methods use."
            )
        self.base_url = base.rstrip("/")
        self.model = model or os.environ.get("CHAT_MODEL") or ""
        if not self.model:
            raise LLMError("No model configured. Set CHAT_MODEL or pass --model.")
        if self.model.startswith(_LITELLM_PREFIX):
            self.model = self.model[len(_LITELLM_PREFIX):]
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = UsageStats()

    def complete(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                self.usage.record(
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
                return body["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001 - retried, then re-raised below
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (2**attempt))
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last}")


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that this is worth doing
    properly rather than failing the whole expansion step.
    """
    if not text or not text.strip():
        raise LLMError("Empty response from the model.")
    fenced = _JSON_BLOCK.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"Model response was not valid JSON: {text[:300]!r}")


def complete_json(client: ChatClient, prompt: str, temperature: float,
                  attempts: int = 3, expect: type | None = dict) -> Any:
    """Call the model and parse JSON, retrying a malformed response.

    ``expect`` is enforced: a model that answers a "respond with a JSON object"
    prompt with a bare string or list produces valid JSON of the wrong shape,
    and callers then fail on ``.get``. Raising LLMError instead routes it into
    the retry-then-skip path every caller already handles.
    """
    last: Exception | None = None
    for i in range(attempts):
        raw = client.complete(prompt, temperature)
        try:
            payload = extract_json(raw)
            if expect is not None and not isinstance(payload, expect):
                raise LLMError(
                    f"expected a JSON {expect.__name__}, got "
                    f"{type(payload).__name__}: {str(payload)[:160]!r}"
                )
            return payload
        except LLMError as e:
            last = e
            prompt = (
                f"{prompt}\n\nYour previous response was rejected "
                f"({e}). Respond with a single JSON object only, no prose, "
                "no code fences."
            )
            if i == attempts - 1:
                break
    raise LLMError(str(last))
