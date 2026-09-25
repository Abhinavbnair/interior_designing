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


class FakeProvider(LLMProvider):
    """Heuristic offline stand-in that GENERATES plausible responses, unlike
    MockProvider (which only replays pre-scripted ones).

    Exists so the Phase 9 harness (and run_full_pipeline.py) can be smoke-tested
    end-to-end — checking the plumbing, the CSV/JSON output, the refinement loop's
    control flow — without an API key or any real intelligence behind it.

    IMPORTANT: this does no real language or vision understanding. Constraint
    extraction is keyword matching; verification is a seeded pseudo-random guess
    with a mild bias toward "satisfied". Every response is tagged internally and
    a warning is logged on first use. Compliance numbers, extraction accuracy, or
    any other figure produced with this provider MUST NOT be reported as system
    performance — they validate the harness, not the model.
    """

    _COLOR_WORDS = {
        "cream", "white", "beige", "grey", "gray", "black", "blue", "green",
        "emerald", "terracotta", "navy", "olive", "mustard", "pale blue",
    }
    _MATERIAL_WORDS = {"wood", "wooden", "oak", "walnut", "brass", "metal", "stone", "rattan", "brick", "linen", "velvet"}
    _LIGHTING_WORDS = {"lighting", "light", "lamp", "daylight", "ambient", "warm", "bright"}
    _ROOM_TYPES = ["bedroom", "living room", "kitchen", "bathroom", "office", "dining room"]
    _NEGATION_MARKERS = ["no ", "not add", "do not", "don't", "without"]

    def __init__(self, seed: int = 0):
        self._rng_seed = seed
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(
                "FakeProvider is active: responses are keyword-heuristic, not real "
                "LLM/VLM output. Do not report these numbers as system performance."
            )
            self._warned = True

    def complete(self, system: str, user: str, image_paths: Optional[list[Path]] = None) -> str:
        self._warn_once()
        if "structured constraints" in system.lower():
            return self._fake_extraction(user)
        if "verdicts" in system.lower():
            return self._fake_verification(user)
        raise LLMError(f"FakeProvider does not recognize this task from the system prompt: {system[:80]!r}")

    def _fake_extraction(self, user_message: str) -> str:
        instruction = user_message.split("Instruction:", 1)[-1].strip()
        clauses = [
            c.strip(" .")
            for c in re.split(r",| and ", instruction)
            if c.strip(" .")
        ]

        room_type = next((rt for rt in self._ROOM_TYPES if rt in instruction.lower()), None)

        constraints = []
        for i, clause in enumerate(clauses, start=1):
            lower = clause.lower()
            is_negative = any(marker in lower for marker in self._NEGATION_MARKERS)
            if is_negative:
                category = "negative"
            elif any(w in lower for w in self._COLOR_WORDS):
                category = "color"
            elif any(w in lower for w in self._MATERIAL_WORDS):
                category = "material"
            elif any(w in lower for w in self._LIGHTING_WORDS):
                category = "lighting"
            elif any(w in lower for w in ("keep", "preserve", "position", "same place")):
                category = "structural"
            elif any(w in lower for w in ("style", "modern", "luxury", "minimalist", "rustic", "industrial")):
                category = "style"
            else:
                category = "appearance"

            constraints.append({
                "id": f"c{i}",
                "category": category,
                "description": clause,
                "target": category,
                "value": clause,
                "is_negative": is_negative,
                "verifiable": category != "structural",
                "priority": 1,
            })

        if not constraints:
            # Never emit zero constraints — extract_constraints() treats that as an
            # error, correctly, since it would silently zero out the KPI.
            constraints = [{
                "id": "c1", "category": "style", "description": instruction, "target": "style",
                "value": instruction, "is_negative": False, "verifiable": True, "priority": 1,
            }]

        payload = {
            "room_info": {"room_type": room_type, "detected_objects": [], "notes": "generated by FakeProvider"},
            "constraints": constraints,
        }
        return json.dumps(payload)

    def _fake_verification(self, user_message: str) -> str:
        ids = re.findall(r"^-\s+(\S+)\s+\[([^\]]*)\]", user_message, flags=re.MULTILINE)
        verdicts = []
        for constraint_id, tags in ids:
            is_structural = "structural" in tags.lower()
            # Deterministic (seeded by id, not wall clock) so repeated runs of the
            # same case are reproducible, with a mild bias toward "satisfied" so a
            # smoke-test run resembles a working system rather than pure noise.
            local_rng_value = (hash((self._rng_seed, constraint_id)) % 1000) / 1000.0
            satisfied = local_rng_value < 0.75
            verdicts.append({
                "constraint_id": constraint_id,
                "satisfied": satisfied,
                "confidence": 0.5,
                "reasoning": "FakeProvider heuristic — not a real judgement.",
                "unverifiable": is_structural,
            })
        return json.dumps({"verdicts": verdicts})


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
    """Factory. LLM_PROVIDER env var selects the backend.

    - 'anthropic' (default): real Claude API, needs LLM_API_KEY.
    - 'mock': strict test double — only replays responses YOU pass in via
      responses=[...]. Used by the test suite. Will raise immediately if called
      with no scripted responses, by design.
    - 'fake': heuristic offline generator, no API key needed. For smoke-testing
      the harness end-to-end. NEVER use its output as a reported result — see
      FakeProvider's docstring.
    """
    name = (provider_name or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "fake":
        return FakeProvider(**kwargs)
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise LLMError(
        f"Unknown LLM_PROVIDER {name!r}. Supported: 'anthropic', 'fake', 'mock'. "
        "Add a new LLMProvider subclass to support another backend."
    )
