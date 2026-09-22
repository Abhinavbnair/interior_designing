"""Provider-agnostic LLM/VLM interface (Phase 5/7).

The project brief requires the LLM/VLM provider to be swappable. Everything
downstream (constraint extraction, verification) depends only on this abstract
interface, so adding OpenAI/Gemini/self-hosted Qwen means writing one subclass —
not touching the pipeline.

A MockProvider is included so the whole constraint → prompt → verify → refine
pipeline can be exercised locally with no API key, no GPU and no network. That
matters for two reasons: the brief asks for lightweight stub modes, and the
pipeline logic needs tests that don't depend on a paid, non-deterministic API.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when a provider call or response parse fails.

    Deliberately not swallowed anywhere in the pipeline — the brief forbids
    silently catching errors, and a silently-empty constraint set would corrupt
    the compliance metric without anyone noticing.
    """


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model response, tolerating ```json fences and
    surrounding prose. Raises LLMError if nothing parseable is found."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} span.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse JSON from model response: {exc}") from exc

    raise LLMError(f"No JSON object found in model response: {text[:200]!r}")


class LLMProvider(ABC):
    """Interface every provider must implement."""

    @abstractmethod
    def complete(self, system: str, user: str, image_paths: Optional[list[Path]] = None) -> str:
        """Return the model's text response. `image_paths` enables VLM usage."""


class MockProvider(LLMProvider):
    """Deterministic offline stub.

    Returns canned-but-schema-valid responses so pipeline logic can be tested
    without an API key. It does NOT do real understanding — never use it to
    produce results reported as system performance.
    """

    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses or []
        self.calls: list[dict] = []
        self._index = 0

    def complete(self, system: str, user: str, image_paths: Optional[list[Path]] = None) -> str:
        self.calls.append({"system": system, "user": user, "image_paths": image_paths})
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response
        raise LLMError(
            "MockProvider ran out of scripted responses. Provide one response per "
            "expected call so the test stays explicit about how many calls it expects."
        )


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic API (text + vision)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError(
                "No API key found. Set LLM_API_KEY (or ANTHROPIC_API_KEY) in the "
                "environment — never hardcode it."
            )
        self.model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

    def complete(self, system: str, user: str, image_paths: Optional[list[Path]] = None) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is required: pip install anthropic") from exc

        content: list[dict] = []
        for path in image_paths or []:
            path = Path(path)
            if not path.exists():
                raise LLMError(f"Image not found: {path}")
            media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(path.read_bytes()).decode("utf-8"),
                    },
                }
            )
        content.append({"type": "text", "text": user})

        client = anthropic.Anthropic(api_key=self.api_key)
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # re-raised, not swallowed
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def get_provider(provider_name: Optional[str] = None, **kwargs: Any) -> LLMProvider:
    """Factory. LLM_PROVIDER env var selects the backend; 'mock' needs no key."""
    name = (provider_name or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise LLMError(
        f"Unknown LLM_PROVIDER {name!r}. Supported: 'anthropic', 'mock'. "
        "Add a new LLMProvider subclass to support another backend."
    )
