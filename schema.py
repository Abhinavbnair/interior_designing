"""Structured constraint schema (Phase 5).

This is the project's core data structure: the machine-checkable representation of
what the user asked for. Every downstream stage depends on it — prompt generation
(Phase 6) reads it, VLM verification (Phase 7) checks the generated image against
it one constraint at a time, and the compliance score (the project's primary KPI)
is computed over it.

Design notes:
- Every constraint is a *separately verifiable atom* with a stable `id`. That's
  deliberate: the primary KPI (Constraint Compliance Rate = satisfied/total) and
  the refinement loop both need to point at individual constraints, which is
  impossible if constraints are collapsed into free text.
- `verifiable` marks constraints a VLM can realistically check from a single image.
  The success criteria (§H) explicitly require marking constraints as unsuitable
  for automatic evaluation rather than pretending the verification is reliable —
  this field is how that gets recorded rather than silently fudged.
- Categories mirror the brief's schema (structural/appearance/object/style/negative)
  but as a flat list of typed atoms rather than nested dicts, because nested dicts
  can't carry per-constraint verification results.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ConstraintCategory(str, Enum):
    """The constraint taxonomy from the project brief."""

    STRUCTURAL = "structural"      # preserve the window / keep bed position
    APPEARANCE = "appearance"      # cream walls, wooden furniture
    OBJECT = "object"              # add a reading chair
    STYLE = "style"                # modern luxury
    LIGHTING = "lighting"          # warm lighting
    MATERIAL = "material"          # oak, marble
    COLOR = "color"                # cream, navy
    SPATIAL = "spatial"            # nightstand beside the bed
    NEGATIVE = "negative"          # do NOT add a television
    QUANTITATIVE = "quantitative"  # exactly two nightstands


class Constraint(BaseModel):
    """One atomically verifiable requirement."""

    id: str = Field(..., description="Stable id, e.g. 'c1' — used to track pass/fail across refinement iterations.")
    category: ConstraintCategory
    description: str = Field(..., description="Human-readable statement of the requirement.")
    target: str = Field(..., description="The object/attribute this concerns, e.g. 'window', 'wall_color'.")
    value: Optional[str] = Field(None, description="Desired value where applicable, e.g. 'cream'.")
    is_negative: bool = Field(False, description="True if this forbids something rather than requires it.")
    verifiable: bool = Field(
        True,
        description=(
            "Whether a VLM can realistically check this from a single generated image. "
            "Set False for things like 'preserve the exact original bed position', which "
            "needs a reference comparison rather than single-image judgement."
        ),
    )
    priority: int = Field(1, ge=1, le=3, description="1=must-have, 2=important, 3=nice-to-have.")

    @field_validator("description", "target")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class RoomInfo(BaseModel):
    """What the VLM observed about the *input* room (Phase 5 understanding step)."""

    room_type: Optional[str] = None
    detected_objects: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ConstraintSet(BaseModel):
    """The full structured representation of one user request."""

    raw_instruction: str
    room_info: Optional[RoomInfo] = None
    constraints: List[Constraint] = Field(default_factory=list)

    @field_validator("constraints")
    @classmethod
    def _unique_ids(cls, v: List[Constraint]) -> List[Constraint]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint ids must be unique")
        return v

    @property
    def positive(self) -> List[Constraint]:
        return [c for c in self.constraints if not c.is_negative]

    @property
    def negative(self) -> List[Constraint]:
        return [c for c in self.constraints if c.is_negative]

    @property
    def verifiable(self) -> List[Constraint]:
        return [c for c in self.constraints if c.verifiable]

    def by_id(self, constraint_id: str) -> Optional[Constraint]:
        return next((c for c in self.constraints if c.id == constraint_id), None)
