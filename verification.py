"""VLM constraint verification (Phase 7).

Checks a generated image against each constraint and returns a structured
VerificationReport. Unverifiable constraints are recorded as such rather than
guessed at — per success criteria §H, an unreliable automatic judgement is worse
than an acknowledged gap, because it inflates the headline compliance number.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.constraints.schema import ConstraintSet
from app.evaluation.compliance import ConstraintVerdict, VerificationReport
from app.llm.provider import LLMError, LLMProvider, extract_json

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """You verify whether a generated interior image satisfies specific constraints.

Return ONLY a JSON object. No prose, no markdown fences.

Schema:
{
  "verdicts": [
    {
      "constraint_id": "c1",
      "satisfied": bool,
      "confidence": 0.0-1.0,
      "reasoning": "one short sentence citing what you actually see",
      "unverifiable": bool
    }
  ]
}

Rules:
- Judge ONLY from the image. Do not assume a constraint is met because it was requested.
- You are an independent checker, not the generator's advocate. If something is absent,
  wrong, or ambiguous, say so. Being agreeable here corrupts the evaluation.
- For negative constraints ("no television"), satisfied=true means the forbidden thing is
  ABSENT from the image.
- Set unverifiable=true when the image genuinely cannot settle the question (e.g. comparing
  against an original you cannot see). Do not use it to avoid a hard judgement call.
- Give one verdict per constraint id provided, and no extras.
"""


def verify_constraints(
    provider: LLMProvider,
    image_path: Path,
    constraint_set: ConstraintSet,
    iteration: int = 0,
) -> VerificationReport:
    """Verify a generated image against a ConstraintSet.

    Raises:
        LLMError: if the response can't be parsed or fails validation.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise LLMError(f"Image to verify does not exist: {image_path}")

    lines = [
        f"- {c.id} [{c.category.value}{', NEGATIVE' if c.is_negative else ''}]: {c.description}"
        for c in constraint_set.constraints
    ]
    user_message = "Verify this image against these constraints:\n" + "\n".join(lines)

    raw = provider.complete(VERIFICATION_SYSTEM_PROMPT, user_message, image_paths=[image_path])
    payload = extract_json(raw)

    try:
        verdicts = [ConstraintVerdict.model_validate(v) for v in payload.get("verdicts", [])]
    except ValidationError as exc:
        raise LLMError(f"Verification response failed schema validation: {exc}") from exc

    if not verdicts:
        raise LLMError("Verifier returned no verdicts — refusing to report 0/0 compliance.")

    # Surface coverage gaps rather than silently scoring a partial check as complete.
    expected = {c.id for c in constraint_set.constraints}
    returned = {v.constraint_id for v in verdicts}
    missing = expected - returned
    unexpected = returned - expected
    if missing:
        logger.warning("Verifier omitted verdicts for constraints: %s", sorted(missing))
    if unexpected:
        logger.warning("Verifier returned unknown constraint ids (dropped): %s", sorted(unexpected))
        verdicts = [v for v in verdicts if v.constraint_id in expected]

    report = VerificationReport(image_path=str(image_path), verdicts=verdicts, iteration=iteration)
    logger.info(
        "Verification iteration %d: %.1f%% compliance (%d passed, %d failed, %d unverifiable)",
        iteration,
        report.compliance_percent,
        len(report.passed_ids),
        len(report.failed_ids),
        len(report.unverifiable_ids),
    )
    return report
