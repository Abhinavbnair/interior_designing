"""Tests for FakeProvider — proves it can actually drive extract_constraints(),
verify_constraints(), and the full refinement loop with zero scripted responses
and no API key, unlike MockProvider.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.pipeline import run_pipeline
from app.llm.constraint_extraction import extract_constraints
from app.llm.prompt_builder import build_prompt
from app.llm.provider import FakeProvider, LLMError, get_provider
from app.llm.verification import verify_constraints


def test_get_provider_fake_needs_no_scripted_responses():
    provider = get_provider("fake")
    assert isinstance(provider, FakeProvider)
    # Unlike MockProvider(), this does not raise immediately.


def test_fake_extraction_produces_valid_constraint_set():
    provider = FakeProvider(seed=1)
    instruction = (
        "Convert this into a modern luxury bedroom, cream-colored walls, "
        "wooden furniture, warm lighting, and do not add a television"
    )
    cs = extract_constraints(provider, instruction)

    assert len(cs.constraints) >= 4
    assert cs.raw_instruction == instruction
    # At least one negative constraint should be picked up from "do not add".
    assert any(c.is_negative for c in cs.constraints)
    assert any("television" in c.description.lower() for c in cs.negative)


def test_fake_extraction_never_returns_zero_constraints():
    provider = FakeProvider(seed=1)
    # A single clause with none of the keyword triggers.
    cs = extract_constraints(provider, "make it nice")
    assert len(cs.constraints) >= 1


def test_fake_extraction_detects_room_type():
    provider = FakeProvider(seed=1)
    cs = extract_constraints(provider, "Convert this into a modern bedroom with cream walls")
    assert cs.room_info.room_type == "bedroom"


def test_fake_verification_is_deterministic_for_same_seed():
    provider_a = FakeProvider(seed=42)
    provider_b = FakeProvider(seed=42)
    cs = extract_constraints(FakeProvider(seed=1), "cream walls, wooden furniture, no television")

    image = Path("/tmp/fake_provider_test_image.png")
    Image.new("RGB", (16, 16)).save(image)

    report_a = verify_constraints(provider_a, image, cs)
    report_b = verify_constraints(provider_b, image, cs)

    assert [v.satisfied for v in report_a.verdicts] == [v.satisfied for v in report_b.verdicts]


def test_fake_verification_marks_structural_as_unverifiable():
    provider = FakeProvider(seed=1)
    cs = extract_constraints(provider, "keep the bed in its exact position, cream walls")

    image = Path("/tmp/fake_provider_test_image2.png")
    Image.new("RGB", (16, 16)).save(image)

    report = verify_constraints(FakeProvider(seed=1), image, cs)
    structural_ids = {c.id for c in cs.constraints if c.category.value == "structural"}
    for verdict in report.verdicts:
        if verdict.constraint_id in structural_ids:
            assert verdict.unverifiable is True


def test_fake_provider_rejects_unrecognized_task():
    provider = FakeProvider(seed=1)
    with pytest.raises(LLMError):
        provider.complete("some unrelated system prompt", "hello")


def test_full_pipeline_runs_with_fake_provider_only(tmp_path):
    """The whole point: this must work with LLM_PROVIDER=fake and nothing else —
    no scripted responses, no API key, no mocking beyond the diffusion backend."""
    provider = FakeProvider(seed=7)

    def generate_fn(prompt, iteration):
        path = tmp_path / f"gen{iteration}.png"
        Image.new("RGB", (16, 16), color=(iteration * 30, 100, 100)).save(path)
        return path

    result = run_pipeline(
        provider,
        "Convert this into a modern luxury bedroom, cream walls, wooden furniture, no television",
        generate_fn,
        max_iterations=3,
    )

    assert len(result.iterations) >= 1
    assert 0.0 <= result.best.report.compliance_rate <= 1.0
    # Prompt building must also succeed on FakeProvider's constraint output.
    assert build_prompt(result.constraint_set).prompt
