"""Experiment runner for the A-E ablation (Phase 9).

Runs the same test cases through every experimental condition and writes
structured CSV + JSON results, which is what the project's KPI comparison
(baseline vs. proposed constraint compliance) is computed from.

The generation backends are injected, so this module contains no torch/diffusers
and its aggregation logic is testable offline. `scripts/run_phase9_experiments.py`
supplies the real GPU-backed backends.

Conditions (from the brief §8 / §I):
    A  text-only            prompt -> diffusion
    B  img2img              room image + prompt -> img2img diffusion
    C  controlnet           room image -> depth ControlNet -> diffusion
    D  llm_prompt           LLM-enhanced prompt -> diffusion (no verification)
    E  full                 constraints -> ControlNet -> verify -> refine
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.constraints.schema import ConstraintSet
from app.core.pipeline import PipelineResult

logger = logging.getLogger(__name__)

CONDITIONS = ["A_text_only", "B_img2img", "C_controlnet", "D_llm_prompt", "E_full"]


@dataclass
class TestCase:
    """One room + instruction scenario."""

    case_id: str
    instruction: str
    room_image_path: Optional[str] = None


@dataclass
class ExperimentRecord:
    """One (test case, condition) result row — the unit of analysis."""

    case_id: str
    condition: str
    instruction: str
    room_image_path: Optional[str]
    generated_image_path: Optional[str]
    total_constraints: int
    scored_constraints: int
    satisfied_constraints: int
    compliance_percent: float
    negative_constraints: int
    negative_satisfied: int
    refinement_iterations: int
    compliance_improvement: float
    regressed_constraints: str
    generation_time_seconds: float
    error: Optional[str] = None


@dataclass
class ExperimentResults:
    """All rows plus aggregation helpers."""

    records: List[ExperimentRecord] = field(default_factory=list)

    def by_condition(self, condition: str) -> List[ExperimentRecord]:
        return [r for r in self.records if r.condition == condition and r.error is None]

    def mean_compliance(self, condition: str) -> Optional[float]:
        values = [r.compliance_percent for r in self.by_condition(condition)]
        return statistics.mean(values) if values else None

    def stdev_compliance(self, condition: str) -> Optional[float]:
        values = [r.compliance_percent for r in self.by_condition(condition)]
        return statistics.stdev(values) if len(values) > 1 else None

    def negative_constraint_accuracy(self, condition: str) -> Optional[float]:
        """§F: correctly avoided negative objects / total negative constraints."""
        rows = self.by_condition(condition)
        total = sum(r.negative_constraints for r in rows)
        satisfied = sum(r.negative_satisfied for r in rows)
        return (satisfied / total * 100.0) if total else None

    def mean_generation_time(self, condition: str) -> Optional[float]:
        values = [r.generation_time_seconds for r in self.by_condition(condition)]
        return statistics.mean(values) if values else None

    def failure_count(self, condition: str) -> int:
        return len([r for r in self.records if r.condition == condition and r.error is not None])

    def improvement_over(self, baseline: str, proposed: str) -> Optional[float]:
        """The headline KPI: percentage-point improvement of proposed over baseline."""
        base = self.mean_compliance(baseline)
        prop = self.mean_compliance(proposed)
        if base is None or prop is None:
            return None
        return prop - base

    def refinement_improvement_rate(self, condition: str = "E_full") -> Optional[float]:
        """§G: share of initially-failed cases that improved after refinement."""
        rows = [r for r in self.by_condition(condition) if r.refinement_iterations > 1]
        if not rows:
            return None
        improved = [r for r in rows if r.compliance_improvement > 0]
        return len(improved) / len(rows) * 100.0

    def summary(self) -> str:
        lines = ["=" * 72, "EXPERIMENT SUMMARY", "=" * 72]
        header = f"{'Condition':<16}{'Mean CCR':>10}{'StDev':>9}{'NegAcc':>9}{'Time(s)':>10}{'N':>5}{'Fail':>6}"
        lines.append(header)
        lines.append("-" * 72)
        for condition in CONDITIONS:
            rows = self.by_condition(condition)
            if not rows and self.failure_count(condition) == 0:
                continue
            mean = self.mean_compliance(condition)
            stdev = self.stdev_compliance(condition)
            neg = self.negative_constraint_accuracy(condition)
            gtime = self.mean_generation_time(condition)
            lines.append(
                f"{condition:<16}"
                f"{(f'{mean:.1f}%' if mean is not None else '—'):>10}"
                f"{(f'{stdev:.1f}' if stdev is not None else '—'):>9}"
                f"{(f'{neg:.1f}%' if neg is not None else '—'):>9}"
                f"{(f'{gtime:.1f}' if gtime is not None else '—'):>10}"
                f"{len(rows):>5}"
                f"{self.failure_count(condition):>6}"
            )
        lines.append("-" * 72)

        improvement = self.improvement_over("A_text_only", "E_full")
        if improvement is not None:
            lines.append(f"PRIMARY KPI — E_full vs A_text_only: {improvement:+.1f} percentage points")
            target_met = improvement >= 10.0
            lines.append(f"  Target (>= +10pp): {'MET' if target_met else 'NOT MET'}")
            if not target_met:
                lines.append(
                    "  Reporting the measured value as-is. Per the project brief, the target "
                    "is not to be reached by tuning the evaluation."
                )

        refine_rate = self.refinement_improvement_rate()
        if refine_rate is not None:
            lines.append(f"Refinement improvement rate: {refine_rate:.1f}% of refined cases improved")

        n_cases = len({r.case_id for r in self.records})
        if n_cases < 50:
            lines.append("")
            lines.append(
                f"NOTE: {n_cases} test case(s) — below the 50 specified in the success criteria. "
                "Treat these numbers as preliminary and report the sample size as a limitation."
            )
        return "\n".join(lines)

    def save(self, output_dir: Path, prefix: str = "experiment") -> Dict[str, Path]:
        """Write CSV + JSON results for reproducibility."""
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{prefix}_results.csv"
        json_path = output_dir / f"{prefix}_results.json"

        rows = [asdict(r) for r in self.records]
        if rows:
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        json_path.write_text(json.dumps({
            "records": rows,
            "aggregates": {
                condition: {
                    "mean_compliance_percent": self.mean_compliance(condition),
                    "stdev_compliance": self.stdev_compliance(condition),
                    "negative_constraint_accuracy": self.negative_constraint_accuracy(condition),
                    "mean_generation_time_seconds": self.mean_generation_time(condition),
                    "n": len(self.by_condition(condition)),
                    "failures": self.failure_count(condition),
                }
                for condition in CONDITIONS
            },
            "primary_kpi_improvement_pp": self.improvement_over("A_text_only", "E_full"),
            "refinement_improvement_rate": self.refinement_improvement_rate(),
        }, indent=2))

        return {"csv": csv_path, "json": json_path}


def record_from_pipeline_result(
    case: TestCase,
    condition: str,
    result: PipelineResult,
    generation_time: float,
) -> ExperimentRecord:
    """Convert a PipelineResult into a flat ExperimentRecord row."""
    best = result.best
    report = best.report
    cs = result.constraint_set

    negative_ids = {c.id for c in cs.negative}
    negative_satisfied = len([vid for vid in report.passed_ids if vid in negative_ids])

    return ExperimentRecord(
        case_id=case.case_id,
        condition=condition,
        instruction=case.instruction,
        room_image_path=case.room_image_path,
        generated_image_path=str(best.image_path),
        total_constraints=len(cs.constraints),
        scored_constraints=len(report.scored),
        satisfied_constraints=len(report.passed_ids),
        compliance_percent=report.compliance_percent,
        negative_constraints=len(negative_ids),
        negative_satisfied=negative_satisfied,
        refinement_iterations=len(result.iterations),
        compliance_improvement=result.compliance_improvement,
        regressed_constraints=",".join(result.regressed_constraint_ids),
        generation_time_seconds=generation_time,
    )


def run_experiment(
    cases: List[TestCase],
    condition_runners: Dict[str, Callable[[TestCase, ConstraintSet], PipelineResult]],
    constraint_sets: Dict[str, ConstraintSet],
) -> ExperimentResults:
    """Run every case through every condition.

    All conditions are scored against the SAME pre-extracted ConstraintSet per
    case, so differences in compliance reflect the generation pipeline rather
    than variation in how the instruction was parsed that run.

    Generation failures are recorded as error rows rather than aborting the run —
    §C requires failures to be logged, and one bad case shouldn't destroy a
    multi-hour experiment.
    """
    results = ExperimentResults()

    for case in cases:
        constraint_set = constraint_sets[case.case_id]
        for condition, runner in condition_runners.items():
            start = time.perf_counter()
            try:
                pipeline_result = runner(case, constraint_set)
                elapsed = time.perf_counter() - start
                results.records.append(
                    record_from_pipeline_result(case, condition, pipeline_result, elapsed)
                )
                logger.info(
                    "%s / %s: %.1f%% compliance",
                    case.case_id, condition,
                    results.records[-1].compliance_percent,
                )
            except Exception as exc:  # logged, not silenced
                elapsed = time.perf_counter() - start
                logger.error("%s / %s FAILED: %s", case.case_id, condition, exc)
                results.records.append(
                    ExperimentRecord(
                        case_id=case.case_id, condition=condition, instruction=case.instruction,
                        room_image_path=case.room_image_path, generated_image_path=None,
                        total_constraints=len(constraint_set.constraints), scored_constraints=0,
                        satisfied_constraints=0, compliance_percent=0.0,
                        negative_constraints=len(constraint_set.negative), negative_satisfied=0,
                        refinement_iterations=0, compliance_improvement=0.0,
                        regressed_constraints="", generation_time_seconds=elapsed,
                        error=str(exc),
                    )
                )

    return results
