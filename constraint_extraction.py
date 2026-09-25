"""Constraint extraction (Phase 5).

Turns a free-text design instruction (plus, optionally, the room image) into a
validated ConstraintSet. Pydantic validation is the guard rail: a malformed model
response raises rather than silently producing an empty constraint list, because
an empty list would score 0/0 compliance and quietly corrupt the primary KPI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.constraints.schema import ConstraintSet
from app.llm.provider import LLMError, LLMProvider, extract_json

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You convert interior-design instructions into structured constraints.

Return ONLY a JSON object. No prose, no markdown fences.

Schema:
{
  "room_info": {"room_type": str|null, "detected_objects": [str], "notes": str|null},
  "constraints": [
    {
      "id": "c1",
      "category": one of ["structural","appearance","object","style","lighting","material","color","spatial","negative","quantitative"],
      "description": "human-readable requirement",
      "target": "the object or attribute concerned",
      "value": "desired value or null",
      "is_negative": bool,
      "verifiable": bool,
      "priority": 1|2|3
    }
  ]
}

Rules:
- Split the instruction into ATOMIC constraints: one checkable fact each. "cream walls and
  wooden furniture" is TWO constraints, not one.
- Number ids sequentially c1, c2, c3...
- Set is_negative=true for prohibitions ("no television" -> category "negative").
- Set verifiable=false ONLY when a constraint cannot be judged from the generated image alone
  (e.g. "keep the bed in its exact original position" needs a before/after comparison).
  Preferring verifiable=true is correct for anything visible in a single image.
- priority: 1 for explicit must-haves, 2 for clear preferences, 3 for vague/aesthetic asides.
- Do NOT invent requirements the user did not state. Missing a stated constraint and
  inventing an unstated one are both errors.
- If the room image is provided, use it ONLY to populate room_info, not to add constraints.
"""


def extract_constraints(
    provider: LLMProvider,
    instruction: str,
    room_image_path: Optional[Path] = None,
) -> ConstraintSet:
    """Extract a validated ConstraintSet from a natural-language instruction.

    Args:
        provider: any LLMProvider (use MockProvider for offline testing).
        instruction: the user's free-text design request.
        room_image_path: optional room photo, used to populate room_info.

    Raises:
        LLMError: if the response can't be parsed or fails schema validation.
    """
    if not instruction or not instruction.strip():
        raise LLMError("Instruction is empty — nothing to extract.")

    user_message = f"Instruction:\n{instruction}"
    images = [room_image_path] if room_image_path else None

    raw = provider.complete(EXTRACTION_SYSTEM_PROMPT, user_message, image_paths=images)
    payload = extract_json(raw)
    payload["raw_instruction"] = instruction

    try:
        constraint_set = ConstraintSet.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"Extracted constraints failed schema validation: {exc}") from exc

    if not constraint_set.constraints:
        raise LLMError(
            "Extraction returned zero constraints. Treating this as an error rather "
            "than scoring 0/0 compliance, which would silently corrupt the KPI."
        )

    logger.info(
        "Extracted %d constraints (%d negative, %d unverifiable)",
        len(constraint_set.constraints),
        len(constraint_set.negative),
        len(constraint_set.constraints) - len(constraint_set.verifiable),
    )
    return constraint_set
