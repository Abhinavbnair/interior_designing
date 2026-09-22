"""Tests for Phases 5-8: constraint schema, extraction, prompt building,
verification, compliance scoring, and the refinement loop.

Unlike Phases 1-3, almost all of this is GPU-free and API-free by design, so
coverage here is real rather than mocked-around-the-edges. The MockProvider
supplies scripted model responses so the pipeline's control flow is deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.constraints.schema import Constraint, ConstraintCategory, ConstraintSet, RoomInfo
from app.core.pipeline import run_pipeline
from app.evaluation.compliance import ConstraintVerdict, VerificationReport
from app.llm.constraint_extraction import extract_constraints
from app.llm.prompt_builder import build_prompt, build_refined_prompt
from app.llm.provider import LLMError, MockProvider, extract_json
from app.llm.verification import verify_constraints


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _constraint(cid, category, description, target, value=None, is_negative=False, verifiable=True):
    return Constraint(
        id=cid, category=category, description=description, target=target,
        value=value, is_negative=is_negative, verifiable=verifiable,
    )


def _sample_set():
    return ConstraintSet(
        raw_instruction="modern luxury bedroom, cream walls, wooden furniture, warm lighting, no television",
        room_info=RoomInfo(room_type="bedroom", detected_objects=["bed", "window"]),
        constraints=[
            _constraint("c1", ConstraintCategory.STYLE, "modern luxury style", "style", "modern luxury"),
            _constraint("c2", ConstraintCategory.COLOR, "cream walls", "walls", "cream"),
            _constraint("c3", ConstraintCategory.MATERIAL, "wooden furniture", "furniture", "wooden"),
            _constraint("c4", ConstraintCategory.LIGHTING, "warm lighting", "lighting", "warm"),
            _constraint("c5", ConstraintCategory.NEGATIVE, "no television", "television", is_negative=True),
            _constraint("c6", ConstraintCategory.STRUCTURAL, "keep bed position", "bed", verifiable=False),
        ],
    )


def test_constraint_set_partitions_positive_negative_verifiable():
    cs = _sample_set()
    assert [c.id for c in cs.negative] == ["c5"]
    assert "c5" not in [c.id for c in cs.positive]
    assert "c6" not in [c.id for c in cs.verifiable]
    assert cs.by_id("c2").value == "cream"
    assert cs.by_id("nope") is None


def test_duplicate_constraint_ids_rejected():
    with pytest.raises(Exception):
        ConstraintSet(
            raw_instruction="x",
            constraints=[
                _constraint("c1", ConstraintCategory.COLOR, "a", "walls"),
                _constraint("c1", ConstraintCategory.COLOR, "b", "floor"),
            ],
        )


def test_empty_description_rejected():
    with pytest.raises(Exception):
        _constraint("c1", ConstraintCategory.COLOR, "   ", "walls")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go:\n{"a": 2}\nHope that helps!') == {"a": 2}


def test_extract_json_raises_on_garbage():
    with pytest.raises(LLMError):
        extract_json("no json at all here")


# ---------------------------------------------------------------------------
# Phase 5 — extraction
# ---------------------------------------------------------------------------

def test_extract_constraints_parses_valid_response():
    response = json.dumps({
        "room_info": {"room_type": "bedroom", "detected_objects": ["bed"], "notes": None},
        "constraints": [
            {"id": "c1", "category": "color", "description": "cream walls", "target": "walls",
             "value": "cream", "is_negative": False, "verifiable": True, "priority": 1},
            {"id": "c2", "category": "negative", "description": "no television", "target": "television",
             "value": None, "is_negative": True, "verifiable": True, "priority": 1},
        ],
    })
    provider = MockProvider([response])

    cs = extract_constraints(provider, "cream walls, no television")

    assert len(cs.constraints) == 2
    assert cs.raw_instruction == "cream walls, no television"
    assert cs.room_info.room_type == "bedroom"
    assert [c.id for c in cs.negative] == ["c2"]


def test_extract_constraints_raises_on_empty_result():
    """An empty extraction must raise, not silently yield 0/0 compliance."""
    provider = MockProvider([json.dumps({"constraints": []})])
    with pytest.raises(LLMError):
        extract_constraints(provider, "some instruction")


def test_extract_constraints_raises_on_invalid_schema():
    provider = MockProvider([json.dumps({"constraints": [{"id": "c1", "category": "not_a_category"}]})])
    with pytest.raises(LLMError):
        extract_constraints(provider, "some instruction")


def test_extract_constraints_rejects_empty_instruction():
    with pytest.raises(LLMError):
        extract_constraints(MockProvider([]), "   ")


# ---------------------------------------------------------------------------
# Phase 6 — prompt building
# ---------------------------------------------------------------------------

def test_build_prompt_includes_positive_and_excludes_negative():
    prompt = build_prompt(_sample_set())

    assert "bedroom" in prompt.prompt
    assert "cream walls" in prompt.prompt
    assert "wooden furniture" in prompt.prompt
    assert "warm lighting" in prompt.prompt
    # The forbidden object must NOT appear in the positive prompt.
    assert "television" not in prompt.prompt.lower()
    # ...it belongs in the negative prompt instead.
    assert "television" in prompt.negative_prompt.lower()


def test_build_refined_prompt_emphasises_failures_without_dropping_passes():
    cs = _sample_set()
    refined = build_refined_prompt(cs, failed_constraint_ids=["c2", "c5"])

    # Failed positive constraint gets attention weighting.
    assert "(cream walls:1.4)" in refined.prompt
    # Failed negative moves to the front of the negative prompt.
    assert refined.negative_prompt.lower().startswith("television")
    # Previously-satisfied constraints are still present.
    assert "wooden furniture" in refined.prompt
    assert "warm lighting" in refined.prompt


def test_structural_constraints_not_restated_in_prompt():
    """Structural preservation is ControlNet's job, not the text prompt's."""
    cs = _sample_set()
    prompt = build_prompt(cs)
    assert "keep bed position" not in prompt.prompt


# ---------------------------------------------------------------------------
# Compliance scoring (primary KPI)
# ---------------------------------------------------------------------------

def test_compliance_rate_excludes_unverifiable():
    report = VerificationReport(
        image_path="x.png",
        verdicts=[
            ConstraintVerdict(constraint_id="c1", satisfied=True),
            ConstraintVerdict(constraint_id="c2", satisfied=True),
            ConstraintVerdict(constraint_id="c3", satisfied=False),
            ConstraintVerdict(constraint_id="c4", satisfied=False, unverifiable=True),
        ],
    )
    # 2 of 3 scorable -> 66.7%, with the unverifiable one excluded entirely.
    assert report.compliance_percent == pytest.approx(66.666, abs=0.01)
    assert report.passed_ids == ["c1", "c2"]
    assert report.failed_ids == ["c3"]
    assert report.unverifiable_ids == ["c4"]


def test_compliance_rate_zero_when_nothing_scorable():
    report = VerificationReport(
        image_path="x.png",
        verdicts=[ConstraintVerdict(constraint_id="c1", satisfied=True, unverifiable=True)],
    )
    assert report.compliance_rate == 0.0
    assert report.scored == []


# ---------------------------------------------------------------------------
# Phase 7 — verification
# ---------------------------------------------------------------------------

def _image(tmp_path, name="gen.png"):
    path = tmp_path / name
    Image.new("RGB", (32, 32), color=(10, 10, 10)).save(path)
    return path


def test_verify_constraints_parses_verdicts(tmp_path):
    cs = _sample_set()
    response = json.dumps({"verdicts": [
        {"constraint_id": c.id, "satisfied": True, "confidence": 0.9, "reasoning": "ok", "unverifiable": False}
        for c in cs.constraints
    ]})
    report = verify_constraints(MockProvider([response]), _image(tmp_path), cs)

    assert len(report.verdicts) == 6
    assert report.compliance_percent == 100.0


def test_verify_constraints_drops_unknown_ids(tmp_path):
    cs = _sample_set()
    response = json.dumps({"verdicts": [
        {"constraint_id": "c1", "satisfied": True},
        {"constraint_id": "c999", "satisfied": True},
    ]})
    report = verify_constraints(MockProvider([response]), _image(tmp_path), cs)

    assert [v.constraint_id for v in report.verdicts] == ["c1"]


def test_verify_constraints_raises_on_missing_image(tmp_path):
    with pytest.raises(LLMError):
        verify_constraints(MockProvider([]), tmp_path / "nope.png", _sample_set())


def test_verify_constraints_raises_on_empty_verdicts(tmp_path):
    with pytest.raises(LLMError):
        verify_constraints(MockProvider([json.dumps({"verdicts": []})]), _image(tmp_path), _sample_set())


# ---------------------------------------------------------------------------
# Phase 8 — refinement loop
# ---------------------------------------------------------------------------

def _verdicts_response(cs, satisfied_ids):
    """Build a verifier response. Constraints marked verifiable=False in the schema
    come back as unverifiable, mirroring what a well-behaved VLM should do."""
    return json.dumps({"verdicts": [
        {
            "constraint_id": c.id,
            "satisfied": c.id in satisfied_ids,
            "confidence": 0.9,
            "unverifiable": not c.verifiable,
        }
        for c in cs.constraints
    ]})


def test_pipeline_stops_early_when_fully_compliant(tmp_path):
    cs = _sample_set()
    all_ids = {c.id for c in cs.constraints}
    provider = MockProvider([_verdicts_response(cs, all_ids)])

    generated = []

    def generate_fn(prompt, iteration):
        path = _image(tmp_path, f"gen{iteration}.png")
        generated.append(path)
        return path

    result = run_pipeline(provider, "x", generate_fn, max_iterations=3, constraint_set=cs)

    assert len(result.iterations) == 1  # stopped early, didn't burn the budget
    assert len(generated) == 1
    assert result.best.report.compliance_percent == 100.0


def test_pipeline_refines_and_improves(tmp_path):
    cs = _sample_set()
    provider = MockProvider([
        _verdicts_response(cs, {"c1", "c3", "c4"}),              # iter 0: 3/5
        _verdicts_response(cs, {"c1", "c2", "c3", "c4", "c5"}),  # iter 1: 5/5
    ])

    def generate_fn(prompt, iteration):
        return _image(tmp_path, f"gen{iteration}.png")

    result = run_pipeline(provider, "x", generate_fn, max_iterations=3, constraint_set=cs)

    assert len(result.iterations) == 2
    assert result.initial.report.compliance_percent == pytest.approx(60.0)
    assert result.best.report.compliance_percent == 100.0
    assert result.compliance_improvement == pytest.approx(40.0)
    assert result.regressed_constraint_ids == []


def test_pipeline_respects_max_iterations(tmp_path):
    cs = _sample_set()
    never_passes = _verdicts_response(cs, set())
    provider = MockProvider([never_passes, never_passes, never_passes])

    calls = []

    def generate_fn(prompt, iteration):
        calls.append(iteration)
        return _image(tmp_path, f"gen{iteration}.png")

    result = run_pipeline(provider, "x", generate_fn, max_iterations=3, constraint_set=cs)

    assert calls == [0, 1, 2]  # hard cap honoured, no infinite loop
    assert len(result.iterations) == 3


def test_pipeline_keeps_best_not_last_and_reports_regression(tmp_path):
    """Refinement can make things worse; the loop must return the best image
    and surface which constraints regressed."""
    cs = _sample_set()
    provider = MockProvider([
        _verdicts_response(cs, {"c1", "c2", "c3", "c4"}),  # iter 0: 4/5 = 80%
        _verdicts_response(cs, {"c5"}),                     # iter 1: 1/5 = 20% (worse)
        _verdicts_response(cs, {"c5"}),                     # iter 2: still worse
    ])

    def generate_fn(prompt, iteration):
        return _image(tmp_path, f"gen{iteration}.png")

    result = run_pipeline(provider, "x", generate_fn, max_iterations=3, constraint_set=cs)

    assert result.best.iteration == 0                      # kept the best, not the last
    assert result.compliance_improvement == pytest.approx(0.0)
    assert result.regressed_constraint_ids == []           # best IS initial, so no regression


def test_pipeline_rejects_invalid_max_iterations(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline(MockProvider([]), "x", lambda p, i: _image(tmp_path), max_iterations=0,
                     constraint_set=_sample_set())
