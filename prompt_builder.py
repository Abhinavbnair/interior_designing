"""Prompt generation from constraints (Phase 6).

Deliberately deterministic/template-based rather than another LLM call. Two
reasons: it keeps the ablation honest (Experiment D isolates "LLM prompt
enhancement" as its own condition, so the constrained pipeline shouldn't
secretly also be doing LLM prompt rewriting), and a deterministic mapping means
a compliance failure can be traced to either extraction or generation, instead
of a third stochastic stage in between.

Negative constraints become the diffusion negative_prompt, which is the
mechanically correct place for them — appending "no television" to a positive
prompt tends to *summon* televisions, since CLIP text encoders have no reliable
negation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from app.constraints.schema import Constraint, ConstraintCategory, ConstraintSet

logger = logging.getLogger(__name__)

QUALITY_SUFFIX = "realistic architectural photography, high-quality interior design visualization"

BASE_NEGATIVE = "blurry, distorted, low quality, watermark, text, deformed furniture, warped walls"


@dataclass(frozen=True)
class GenerationPrompt:
    """The prompt pair handed to the diffusion pipeline."""

    prompt: str
    negative_prompt: str

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "negative_prompt": self.negative_prompt}


def _phrase(constraint: Constraint) -> str:
    """Render one positive constraint as a prompt fragment."""
    if constraint.value:
        if constraint.category in (ConstraintCategory.COLOR, ConstraintCategory.MATERIAL):
            return f"{constraint.value} {constraint.target}"
        if constraint.category == ConstraintCategory.LIGHTING:
            return f"{constraint.value} lighting"
        if constraint.category == ConstraintCategory.STYLE:
            return f"{constraint.value} style"
        return f"{constraint.target}: {constraint.value}"
    return constraint.target


def build_prompt(constraint_set: ConstraintSet) -> GenerationPrompt:
    """Build the positive/negative prompt pair from a ConstraintSet.

    Ordering follows what SDXL weights most heavily: room type and style first,
    then appearance/material/colour, then objects, then the quality suffix.
    """
    room_type = (constraint_set.room_info.room_type if constraint_set.room_info else None) or "interior room"

    styles: List[str] = []
    appearance: List[str] = []
    objects: List[str] = []

    for constraint in constraint_set.positive:
        if constraint.category == ConstraintCategory.STYLE:
            styles.append(_phrase(constraint))
        elif constraint.category in (
            ConstraintCategory.APPEARANCE,
            ConstraintCategory.COLOR,
            ConstraintCategory.MATERIAL,
            ConstraintCategory.LIGHTING,
        ):
            appearance.append(_phrase(constraint))
        elif constraint.category in (
            ConstraintCategory.OBJECT,
            ConstraintCategory.QUANTITATIVE,
            ConstraintCategory.SPATIAL,
        ):
            objects.append(_phrase(constraint))
        # STRUCTURAL constraints are handled by ControlNet conditioning, not text —
        # restating "keep the window" in the prompt adds nothing the depth map
        # isn't already enforcing more reliably.

    parts = [f"{' '.join(styles)} {room_type}".strip() if styles else room_type]
    parts.extend(appearance)
    parts.extend(objects)
    parts.append(QUALITY_SUFFIX)

    prompt = ", ".join(p for p in parts if p)

    negative_terms = [BASE_NEGATIVE]
    for constraint in constraint_set.negative:
        negative_terms.append(constraint.value or constraint.target)
    negative_prompt = ", ".join(negative_terms)

    logger.info("Built prompt from %d constraints", len(constraint_set.constraints))
    return GenerationPrompt(prompt=prompt, negative_prompt=negative_prompt)


def build_refined_prompt(
    constraint_set: ConstraintSet,
    failed_constraint_ids: List[str],
) -> GenerationPrompt:
    """Rebuild the prompt with emphasis on constraints that failed verification (Phase 8).

    Uses SDXL's `(term:1.4)` attention-weighting syntax to push failed positive
    constraints harder, and moves failed negative constraints to the front of the
    negative prompt. This is intentionally a conservative edit of the original
    prompt rather than a free-form LLM rewrite: the refinement must not silently
    drop constraints that already passed, which §G of the success criteria
    explicitly asks us to measure.
    """
    base = build_prompt(constraint_set)
    failed = set(failed_constraint_ids)

    emphasised: List[str] = []
    extra_negative: List[str] = []
    for constraint in constraint_set.constraints:
        if constraint.id not in failed:
            continue
        if constraint.is_negative:
            extra_negative.append(constraint.value or constraint.target)
        else:
            emphasised.append(f"({_phrase(constraint)}:1.4)")

    prompt = base.prompt
    if emphasised:
        prompt = f"{prompt}, {', '.join(emphasised)}"

    negative_prompt = base.negative_prompt
    if extra_negative:
        # Failed negatives go first — earlier terms carry more weight.
        negative_prompt = f"{', '.join(extra_negative)}, {negative_prompt}"

    return GenerationPrompt(prompt=prompt, negative_prompt=negative_prompt)
