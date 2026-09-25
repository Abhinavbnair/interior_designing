"""Phase 9 — run the A-E ablation experiment.

Runs every test case through every condition and writes CSV/JSON results plus a
summary table containing the project's primary KPI.

Requires a GPU runtime and (for conditions D and E) LLM_API_KEY.

Usage:
    export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png
    export LLM_API_KEY=sk-...
    export SEED=42
    export TEST_CASES=data/test_cases.json
    export MAX_CASES=3          # start small; each case runs 5 conditions
    python scripts/run_phase9_experiments.py

Runtime warning: each case runs 5 conditions, and condition E may generate up to
3 times. At roughly 20-26s per generation on a T4, 10 cases is on the order of
40+ minutes. Start with MAX_CASES=2 to validate the harness before committing to
a full run, and be aware Colab/Kaggle sessions time out.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_controlnet_config  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.core.pipeline import IterationRecord, PipelineResult, run_pipeline  # noqa: E402
from app.evaluation.experiment import TestCase, run_experiment  # noqa: E402
from app.generation.model_loader import (  # noqa: E402
    load_sdxl_controlnet_pipeline,
    load_sdxl_img2img_pipeline,
)
from app.generation.result_io import resolve_output_path, save_image  # noqa: E402
from app.llm.constraint_extraction import extract_constraints  # noqa: E402
from app.llm.prompt_builder import GenerationPrompt, build_prompt  # noqa: E402
from app.llm.provider import get_provider  # noqa: E402
from app.llm.verification import verify_constraints  # noqa: E402
from app.vision.depth import depth_array_to_image, load_depth_estimator, predict_depth_array  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    config = load_controlnet_config()

    if not str(config.input_image_path) or not config.input_image_path.exists():
        raise FileNotFoundError(f"INPUT_IMAGE_PATH must exist (got: {config.input_image_path!r})")

    cases_path = Path(os.environ.get("TEST_CASES", "data/test_cases.json"))
    raw_cases = json.loads(cases_path.read_text())
    max_cases = int(os.environ.get("MAX_CASES", "0")) or len(raw_cases)
    cases = [
        TestCase(case_id=c["case_id"], instruction=c["instruction"],
                 room_image_path=str(config.input_image_path))
        for c in raw_cases[:max_cases]
    ]
    logger.info("Running %d test case(s) x 5 conditions", len(cases))

    import torch
    from PIL import Image

    provider = get_provider()

    # --- Extract constraints ONCE per case, shared across all conditions ---
    constraint_sets = {}
    for case in cases:
        logger.info("Extracting constraints for %s", case.case_id)
        constraint_sets[case.case_id] = extract_constraints(
            provider, case.instruction, config.input_image_path
        )

    # --- Shared inputs ---
    input_image = Image.open(config.input_image_path).convert("RGB").resize(
        (config.base.resolution, config.base.resolution)
    )
    feature_extractor, depth_model = load_depth_estimator(config.depth_model_id, device=config.base.device)
    depth_array = predict_depth_array(
        feature_extractor, depth_model, input_image, config.base.resolution, device=config.base.device
    )
    depth_image = depth_array_to_image(depth_array)
    del depth_model
    torch.cuda.empty_cache()

    controlnet_pipe = load_sdxl_controlnet_pipeline(
        config.base.model_id, config.controlnet_model_id, config.vae_model_id,
        device=config.base.device, dtype=config.base.dtype,
    )
    img2img_pipe = load_sdxl_img2img_pipeline(
        config.base.model_id, device=config.base.device, dtype=config.base.dtype
    )

    def _generator(iteration: int = 0):
        if config.base.seed is None:
            return None
        return torch.Generator(device=config.base.device).manual_seed(config.base.seed + iteration)

    def _save(image, case_id: str, condition: str, iteration: int) -> Path:
        path = resolve_output_path(config.base.output_dir, f"{case_id}_{condition}_it{iteration}.png")
        save_image(image, path)
        return path

    def _single_shot(case, constraint_set, condition, render):
        """Conditions A-D: generate once, verify once, no refinement."""
        prompt = build_prompt(constraint_set)
        image = render(prompt)
        path = _save(image, case.case_id, condition, 0)
        report = verify_constraints(provider, path, constraint_set, iteration=0)
        return PipelineResult(
            constraint_set=constraint_set,
            iterations=[IterationRecord(iteration=0, prompt=prompt, image_path=path, report=report)],
        )

    # --- Condition definitions ---
    def condition_a(case, constraint_set):
        """Text-only: raw instruction, no constraint structuring, no image."""
        def render(_prompt):
            # Uses the raw user instruction, NOT the built prompt — this is the
            # naive baseline the proposed system must beat.
            return controlnet_pipe(
                prompt=case.instruction,
                image=Image.new("RGB", (config.base.resolution, config.base.resolution), (128, 128, 128)),
                height=config.base.resolution, width=config.base.resolution,
                num_inference_steps=config.base.num_inference_steps,
                guidance_scale=config.base.guidance_scale,
                controlnet_conditioning_scale=0.0,  # disabled -> effectively text-to-image
                generator=_generator(),
            ).images[0]
        return _single_shot(case, constraint_set, "A_text_only", render)

    def condition_b(case, constraint_set):
        """Image-to-image from the room photo, raw instruction."""
        def render(_prompt):
            return img2img_pipe(
                prompt=case.instruction, image=input_image, strength=0.6,
                num_inference_steps=config.base.num_inference_steps,
                guidance_scale=config.base.guidance_scale, generator=_generator(),
            ).images[0]
        return _single_shot(case, constraint_set, "B_img2img", render)

    def condition_c(case, constraint_set):
        """ControlNet structural conditioning, raw instruction."""
        def render(_prompt):
            return controlnet_pipe(
                prompt=case.instruction, image=depth_image,
                height=config.base.resolution, width=config.base.resolution,
                num_inference_steps=config.base.num_inference_steps,
                guidance_scale=config.base.guidance_scale,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                generator=_generator(),
            ).images[0]
        return _single_shot(case, constraint_set, "C_controlnet", render)

    def condition_d(case, constraint_set):
        """LLM-structured prompt (incl. negative prompt), no ControlNet, no refinement."""
        def render(prompt: GenerationPrompt):
            return controlnet_pipe(
                prompt=prompt.prompt, negative_prompt=prompt.negative_prompt,
                image=Image.new("RGB", (config.base.resolution, config.base.resolution), (128, 128, 128)),
                height=config.base.resolution, width=config.base.resolution,
                num_inference_steps=config.base.num_inference_steps,
                guidance_scale=config.base.guidance_scale,
                controlnet_conditioning_scale=0.0,
                generator=_generator(),
            ).images[0]
        return _single_shot(case, constraint_set, "D_llm_prompt", render)

    def condition_e(case, constraint_set):
        """Full proposed system: constraints + ControlNet + verification + refinement."""
        def generate_fn(prompt: GenerationPrompt, iteration: int) -> Path:
            image = controlnet_pipe(
                prompt=prompt.prompt, negative_prompt=prompt.negative_prompt,
                image=depth_image,
                height=config.base.resolution, width=config.base.resolution,
                num_inference_steps=config.base.num_inference_steps,
                guidance_scale=config.base.guidance_scale,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                generator=_generator(iteration),
            ).images[0]
            return _save(image, case.case_id, "E_full", iteration)

        return run_pipeline(
            provider=provider, instruction=case.instruction, generate_fn=generate_fn,
            room_image_path=config.input_image_path,
            max_iterations=int(os.environ.get("MAX_REFINEMENT_ITERATIONS", "3")),
            constraint_set=constraint_set,
        )

    results = run_experiment(
        cases,
        {
            "A_text_only": condition_a,
            "B_img2img": condition_b,
            "C_controlnet": condition_c,
            "D_llm_prompt": condition_d,
            "E_full": condition_e,
        },
        constraint_sets,
    )

    paths = results.save(config.base.output_dir)
    print("\n" + results.summary())
    print(f"\nResults written to:\n  {paths['csv']}\n  {paths['json']}")


if __name__ == "__main__":
    main()
