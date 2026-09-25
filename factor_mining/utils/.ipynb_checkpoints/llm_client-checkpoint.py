"""LLM API client for factor mining methods."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMConfig:
    """Configuration for the LLM API."""
    api_url: str = "https://api.ai.create.kcl.ac.uk/v1/chat/completions"
    api_key: str = "sk-Fk0T8ZnaKERkOsKCR1hbmA"
    model: str = "arc:nano"
    temperature: float = 0.8
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: float = 2.0


class LLMClient:
    """Client for calling the LLM API."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.api_url,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM API failed after {self.config.max_retries} attempts: {e}")

    def generate_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = None,
    ) -> dict | list | None:
        """Call LLM and parse the response as JSON. Returns None on parse failure."""
        text = self.chat(messages, temperature=temperature)
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict | list | None:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object or array in the text
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                if start == -1:
                    continue
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i + 1])
                            except json.JSONDecodeError:
                                break
            return None
