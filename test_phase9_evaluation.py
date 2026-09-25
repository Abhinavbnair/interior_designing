"""Tests for Phase 9: experiment aggregation, results I/O, and structural metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.constraints.schema import Constraint, ConstraintCategory, ConstraintSet
from app.core.pipeline import IterationRecord, PipelineResult
from app.evaluation.compliance import ConstraintVerdict, VerificationReport
from app.evaluation.experiment import (
    ExperimentRecord,
    ExperimentResults,
    TestCase,
    record_from_pipeline_result,
    run_experiment,
)
from app.evaluation.structural import edge_similarity, ssim, structural_scores
from app.llm.prompt_builder import GenerationPrompt


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _record(case_id, condition, compliance, neg_total=1, neg_ok=1, iterations=1, improvement=0.0, error=None):
    return ExperimentRecord(
        case_id=case_id, condition=condition, instruction="x", room_image_path=None,
        generated_image_path="img.png", total_constraints=5, scored_constraints=5,
        satisfied_constraints=int(compliance / 20), compliance_percent=compliance,
        negative_constraints=neg_total, negative_satisfied=neg_ok,
        refinement_iterations=iterations, compliance_improvement=improvement,
        regressed_constraints="", generation_time_seconds=20.0, error=error,
    )


def test_mean_compliance_and_primary_kpi():
    results = ExperimentResults(records=[
        _record("t1", "A_text_only", 60.0),
        _record("t2", "A_text_only", 70.0),
        _record("t1", "E_full", 85.0),
        _record("t2", "E_full", 95.0),
    ])

    assert results.mean_compliance("A_text_only") == pytest.approx(65.0)
    assert results.mean_compliance("E_full") == pytest.approx(90.0)
    assert results.improvement_over("A_text_only", "E_full") == pytest.approx(25.0)


def test_error_rows_excluded_from_means_but_counted_as_failures():
    results = ExperimentResults(records=[
        _record("t1", "E_full", 80.0),
        _record("t2", "E_full", 0.0, error="CUDA OOM"),
    ])

    assert results.mean_compliance("E_full") == pytest.approx(80.0)  # failure not averaged in
    assert results.failure_count("E_full") == 1
    assert len(results.by_condition("E_full")) == 1


def test_negative_constraint_accuracy():
    results = ExperimentResults(records=[
        _record("t1", "E_full", 80.0, neg_total=2, neg_ok=2),
        _record("t2", "E_full", 80.0, neg_total=2, neg_ok=1),
    ])
    # 3 of 4 negative constraints avoided
    assert results.negative_constraint_accuracy("E_full") == pytest.approx(75.0)


def test_refinement_improvement_rate_counts_only_refined_cases():
    results = ExperimentResults(records=[
        _record("t1", "E_full", 100.0, iterations=1, improvement=0.0),   # never refined
        _record("t2", "E_full", 80.0, iterations=2, improvement=20.0),   # refined, improved
        _record("t3", "E_full", 60.0, iterations=3, improvement=0.0),    # refined, no gain
    ])
    # 1 of the 2 refined cases improved
    assert results.refinement_improvement_rate() == pytest.approx(50.0)


def test_mean_compliance_none_when_no_data():
    assert ExperimentResults().mean_compliance("A_text_only") is None
    assert ExperimentResults().improvement_over("A_text_only", "E_full") is None


def test_summary_flags_small_sample_size():
    results = ExperimentResults(records=[_record("t1", "E_full", 80.0)])
    summary = results.summary()
    assert "below the 50 specified" in summary


def test_save_writes_csv_and_json(tmp_path):
    results = ExperimentResults(records=[
        _record("t1", "A_text_only", 60.0),
        _record("t1", "E_full", 90.0),
    ])
    paths = results.save(tmp_path)

    assert paths["csv"].exists() and paths["json"].exists()

    with paths["csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {r["condition"] for r in rows} == {"A_text_only", "E_full"}

    payload = json.loads(paths["json"].read_text())
    assert payload["primary_kpi_improvement_pp"] == pytest.approx(30.0)
    assert payload["aggregates"]["E_full"]["mean_compliance_percent"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# record_from_pipeline_result
# ---------------------------------------------------------------------------

def _pipeline_result(tmp_path):
    cs = ConstraintSet(
        raw_instruction="x",
        constraints=[
            Constraint(id="c1", category=ConstraintCategory.COLOR, description="cream walls", target="walls"),
            Constraint(id="c2", category=ConstraintCategory.NEGATIVE, description="no tv", target="television", is_negative=True),
        ],
    )
    image = tmp_path / "gen.png"
    Image.new("RGB", (16, 16)).save(image)

    report = VerificationReport(image_path=str(image), verdicts=[
        ConstraintVerdict(constraint_id="c1", satisfied=True),
        ConstraintVerdict(constraint_id="c2", satisfied=True),
    ])
    prompt = GenerationPrompt(prompt="p", negative_prompt="n")
    return PipelineResult(constraint_set=cs, iterations=[
        IterationRecord(iteration=0, prompt=prompt, image_path=image, report=report)
    ])


def test_record_from_pipeline_result_counts_negatives(tmp_path):
    result = _pipeline_result(tmp_path)
    case = TestCase(case_id="t1", instruction="x")

    record = record_from_pipeline_result(case, "E_full", result, generation_time=12.5)

    assert record.compliance_percent == 100.0
    assert record.negative_constraints == 1
    assert record.negative_satisfied == 1
    assert record.generation_time_seconds == 12.5


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def test_run_experiment_records_failures_without_aborting(tmp_path):
    cases = [TestCase(case_id="t1", instruction="x")]
    cs = _pipeline_result(tmp_path).constraint_set

    def good_runner(case, constraint_set):
        return _pipeline_result(tmp_path)

    def bad_runner(case, constraint_set):
        raise RuntimeError("CUDA out of memory")

    results = run_experiment(
        cases,
        {"A_text_only": bad_runner, "E_full": good_runner},
        {"t1": cs},
    )

    assert len(results.records) == 2  # the failure did NOT abort the run
    failed = [r for r in results.records if r.error]
    assert len(failed) == 1
    assert "CUDA out of memory" in failed[0].error
    assert results.mean_compliance("E_full") == 100.0


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def test_ssim_identical_images_is_one():
    array = np.random.default_rng(0).uniform(0, 255, (64, 64))
    assert ssim(array, array) == pytest.approx(1.0, abs=1e-6)


def test_ssim_differs_for_different_images():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, (64, 64))
    b = rng.uniform(0, 255, (64, 64))
    assert ssim(a, b) < 0.5


def test_ssim_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssim(np.zeros((8, 8)), np.zeros((16, 16)))


def test_edge_similarity_identical_is_one():
    array = np.zeros((32, 32))
    array[10:20, 10:20] = 255.0  # a square, so there are real edges to match
    assert edge_similarity(array, array) == pytest.approx(1.0)


def test_edge_similarity_disjoint_shapes_is_low():
    a = np.zeros((32, 32))
    a[2:10, 2:10] = 255.0
    b = np.zeros((32, 32))
    b[20:28, 20:28] = 255.0
    assert edge_similarity(a, b) < 0.1


def test_structural_scores_on_real_files(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(path_a)
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(path_b)

    scores = structural_scores(path_a, path_b, size=(32, 32))

    assert scores["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert set(scores) == {"ssim", "edge_similarity"}
