"""The full proposed pipeline with refinement loop (Phase 8).

UNDERSTAND -> CONSTRAIN -> GENERATE -> VERIFY -> REFINE

Design decisions worth stating, since they affect how the results should be read:

1. `max_iterations` is hard-capped (default 3, per success criteria §G) so the
   loop cannot spin forever on a constraint the generator simply cannot satisfy.
2. The loop keeps the BEST image by compliance, not the last one. Refinement can
   make things worse — §G explicitly asks us to measure whether refinement breaks
   previously-satisfied constraints, so regressions must be observable rather
   than hidden by always returning the final attempt.
3. Every iteration's full report is retained, so the "did refinement help?"
   analysis has the whole trajectory rather than just endpoints.
4. The generate step is injected as a callable. That keeps this module free of
   torch/diffusers, so the orchestration logic is testable offline with a fake
   generator — the GPU-bound part stays in the scripts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from app.constraints.schema import ConstraintSet
from app.evaluation.compliance import VerificationReport
from app.llm.constraint_extraction import extract_constraints
from app.llm.prompt_builder import GenerationPrompt, build_prompt, build_refined_prompt
from app.llm.provider import LLMProvider
from app.llm.verification import verify_constraints

logger = logging.getLogger(__name__)


class GenerateFn(Protocol):
    """A generation backend: takes a prompt pair + iteration, returns the image path.

    Implemented by the scripts using whichever diffusion pipeline is in play
    (Phase 1 text2img, Phase 2 img2img, Phase 3 ControlNet), which is what lets
    the same refinement loop drive every ablation condition.
    """

    def __call__(self, prompt: GenerationPrompt, iteration: int) -> Path: ...


@dataclass
class IterationRecord:
    """One pass through generate -> verify."""

    iteration: int
    prompt: GenerationPrompt
    image_path: Path
    report: VerificationReport


@dataclass
class PipelineResult:
    """Everything the run produced, including the full refinement trajectory."""

    constraint_set: ConstraintSet
    iterations: List[IterationRecord] = field(default_factory=list)

    @property
    def best(self) -> IterationRecord:
        return max(self.iterations, key=lambda r: r.report.compliance_rate)

    @property
    def initial(self) -> IterationRecord:
        return self.iterations[0]

    @property
    def compliance_improvement(self) -> float:
        """Final-best minus initial compliance, in percentage points (§G metric)."""
        return self.best.report.compliance_percent - self.initial.report.compliance_percent

    @property
    def regressed_constraint_ids(self) -> List[str]:
        """Constraints that passed initially but fail in the best-scoring image.

        §G asks whether refinement breaks previously-satisfied constraints; this
        makes that directly measurable instead of inferable.
        """
        if self.best is self.initial:
            return []
        initially_passed = set(self.initial.report.passed_ids)
        finally_failed = set(self.best.report.failed_ids)
        return sorted(initially_passed & finally_failed)

    def summary(self) -> str:
        lines = [
            f"Iterations run: {len(self.iterations)}",
            f"Initial compliance: {self.initial.report.compliance_percent:.1f}%",
            f"Best compliance:    {self.best.report.compliance_percent:.1f}% "
            f"(iteration {self.best.iteration})",
            f"Improvement:        {self.compliance_improvement:+.1f} percentage points",
            f"Best image:         {self.best.image_path}",
        ]
        if self.regressed_constraint_ids:
            lines.append(
                f"WARNING — refinement broke previously-satisfied constraints: "
                f"{', '.join(self.regressed_constraint_ids)}"
            )
        lines.append("")
        lines.append(self.best.report.summary(self.constraint_set))
        return "\n".join(lines)


def run_pipeline(
    provider: LLMProvider,
    instruction: str,
    generate_fn: GenerateFn,
    room_image_path: Optional[Path] = None,
    max_iterations: int = 3,
    target_compliance: float = 1.0,
    constraint_set: Optional[ConstraintSet] = None,
) -> PipelineResult:
    """Run the full constrained-generation pipeline with refinement.

    Args:
        provider: LLM/VLM provider (MockProvider works offline).
        instruction: the user's natural-language design request.
        generate_fn: callable producing an image from a GenerationPrompt.
        room_image_path: optional input room photo.
        max_iterations: hard cap on total generations (§G: 3).
        target_compliance: stop early once compliance reaches this (0-1).
        constraint_set: pre-extracted constraints, to skip the extraction call
            (used by the experiment runner so every ablation condition is scored
            against an identical constraint set).

    Returns:
        PipelineResult with every iteration recorded.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    if constraint_set is None:
        constraint_set = extract_constraints(provider, instruction, room_image_path)

    result = PipelineResult(constraint_set=constraint_set)
    prompt = build_prompt(constraint_set)

    for iteration in range(max_iterations):
        image_path = generate_fn(prompt, iteration)
        report = verify_constraints(provider, image_path, constraint_set, iteration=iteration)
        result.iterations.append(
            IterationRecord(iteration=iteration, prompt=prompt, image_path=Path(image_path), report=report)
        )

        if report.compliance_rate >= target_compliance:
            logger.info("Target compliance reached at iteration %d — stopping early.", iteration)
            break

        if iteration == max_iterations - 1:
            logger.info("Hit max_iterations (%d) — stopping with best result so far.", max_iterations)
            break

        failed = report.failed_ids
        if not failed:
            # Nothing actionable to emphasise (e.g. remaining gaps are unverifiable).
            logger.info("No failed verifiable constraints to refine — stopping.")
            break

        logger.info("Refining for failed constraints: %s", failed)
        prompt = build_refined_prompt(constraint_set, failed)

    return result
