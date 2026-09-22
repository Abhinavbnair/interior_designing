"""Verification results + compliance scoring (Phase 7 / primary KPI).

Constraint Compliance Rate (CCR) = satisfied / total explicit constraints × 100
is the project's single most important metric, so it's defined once, here, and
reused by both the live refinement loop and the offline experiment runner —
rather than being recomputed slightly differently in two places.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.constraints.schema import ConstraintSet


class ConstraintVerdict(BaseModel):
    """The verifier's judgement on one constraint."""

    constraint_id: str
    satisfied: bool
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    unverifiable: bool = Field(
        False,
        description=(
            "True if the verifier could not reliably judge this from the image. "
            "Per success criteria §H these are excluded from the compliance score "
            "rather than being counted as passes or failures."
        ),
    )


class VerificationReport(BaseModel):
    """Result of checking one generated image against one ConstraintSet."""

    image_path: str
    verdicts: List[ConstraintVerdict] = Field(default_factory=list)
    iteration: int = 0

    @property
    def scored(self) -> List[ConstraintVerdict]:
        """Verdicts that actually count toward compliance (excludes unverifiable)."""
        return [v for v in self.verdicts if not v.unverifiable]

    @property
    def passed_ids(self) -> List[str]:
        return [v.constraint_id for v in self.scored if v.satisfied]

    @property
    def failed_ids(self) -> List[str]:
        return [v.constraint_id for v in self.scored if not v.satisfied]

    @property
    def unverifiable_ids(self) -> List[str]:
        return [v.constraint_id for v in self.verdicts if v.unverifiable]

    @property
    def compliance_rate(self) -> float:
        """CCR as a 0-1 fraction. Returns 0.0 when nothing is scorable — callers
        should check `scored` before treating a 0.0 as a genuine failure."""
        if not self.scored:
            return 0.0
        return len(self.passed_ids) / len(self.scored)

    @property
    def compliance_percent(self) -> float:
        return self.compliance_rate * 100.0

    def summary(self, constraint_set: Optional[ConstraintSet] = None) -> str:
        """Human-readable compliance report."""
        lines = [
            f"Compliance: {self.compliance_percent:.1f}% "
            f"({len(self.passed_ids)}/{len(self.scored)} verifiable constraints satisfied)"
        ]
        for verdict in self.verdicts:
            if verdict.unverifiable:
                mark = "?"
            elif verdict.satisfied:
                mark = "PASS"
            else:
                mark = "FAIL"
            label = verdict.constraint_id
            if constraint_set is not None:
                constraint = constraint_set.by_id(verdict.constraint_id)
                if constraint is not None:
                    label = f"{verdict.constraint_id}: {constraint.description}"
            lines.append(f"  [{mark}] {label}")
            if verdict.reasoning:
                lines.append(f"         {verdict.reasoning}")
        if self.unverifiable_ids:
            lines.append(
                f"  Note: {len(self.unverifiable_ids)} constraint(s) excluded as "
                f"not reliably verifiable from a single image."
            )
        return "\n".join(lines)
