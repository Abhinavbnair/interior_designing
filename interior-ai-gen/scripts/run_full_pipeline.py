"""Full proposed pipeline — UNDERSTAND -> CONSTRAIN -> GENERATE -> VERIFY -> REFINE.

This is the complete system (Phases 5-8) running on real models.

Requires:
    - A GPU runtime (Kaggle/Colab) for the diffusion side.
    - LLM_API_KEY in the environment for constraint extraction + verification.
      Set LLM_PROVIDER=mock to exercise the control flow without an API key
      (the images will still be real; the constraints/verdicts will be canned
      and must NOT be reported as system performance).

Usage:
    export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png
    export LLM_API_KEY=sk-...
    export SEED=42
    python scripts/run_full_pipeline.py "Convert this into a modern luxury bedroom..."
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_controlnet_config  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.core.pipeline import run_pipeline  # noqa: E402
from app.generation.model_loader import load_sdxl_controlnet_pipeline  # noqa: E402
from app.generation.result_io import resolve_output_path, save_image  # noqa: E402
from app.llm.prompt_builder import GenerationPrompt  # noqa: E402
from app.llm.provider import get_provider  # noqa: E402
from app.vision.depth import depth_array_to_image, load_depth_estimator, predict_depth_array  # noqa: E402

DEFAULT_INSTRUCTION = (
    "Convert this into a modern luxury bedroom. Keep the existing bed and window "
    "positions. Use cream-colored walls, wooden furniture, warm lighting, and do "
    "not add a television."
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    instruction = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INSTRUCTION
    config = load_controlnet_config()

    if not str(config.input_image_path) or not config.input_image_path.exists():
        raise FileNotFoundError(
            f"INPUT_IMAGE_PATH must point at an existing room photo (got: {config.input_image_path!r})"
        )

    import torch
    from PIL import Image

    provider = get_provider()
    logger.info("Using LLM provider: %s", type(provider).__name__)

    # --- Depth map (structural conditioning) ---
    input_image = Image.open(config.input_image_path).convert("RGB")
    feature_extractor, depth_model = load_depth_estimator(config.depth_model_id, device=config.base.device)
    depth_array = predict_depth_array(
        feature_extractor, depth_model, input_image, config.base.resolution, device=config.base.device
    )
    depth_image = depth_array_to_image(depth_array)
    depth_path = resolve_output_path(config.base.output_dir, "pipeline_depth_map.png")
    save_image(depth_image, depth_path)

    # --- Diffusion pipeline ---
    pipe = load_sdxl_controlnet_pipeline(
        config.base.model_id, config.controlnet_model_id, config.vae_model_id,
        device=config.base.device, dtype=config.base.dtype,
    )

    def generate_fn(prompt: GenerationPrompt, iteration: int) -> Path:
        """Generation backend injected into the refinement loop."""
        generator = None
        if config.base.seed is not None:
            # Vary the seed per iteration: regenerating with an identical seed AND
            # a barely-changed prompt tends to reproduce the same failure.
            generator = torch.Generator(device=config.base.device).manual_seed(
                config.base.seed + iteration
            )
        logger.info("Generating iteration %d: %s", iteration, prompt.prompt[:100])
        result = pipe(
            prompt=prompt.prompt,
            negative_prompt=prompt.negative_prompt,
            image=depth_image,
            height=config.base.resolution,
            width=config.base.resolution,
            num_inference_steps=config.base.num_inference_steps,
            guidance_scale=config.base.guidance_scale,
            controlnet_conditioning_scale=config.controlnet_conditioning_scale,
            generator=generator,
        )
        path = resolve_output_path(config.base.output_dir, f"pipeline_iter{iteration}.png")
        save_image(result.images[0], path)
        return path

    result = run_pipeline(
        provider=provider,
        instruction=instruction,
        generate_fn=generate_fn,
        room_image_path=config.input_image_path,
        max_iterations=int(__import__("os").environ.get("MAX_REFINEMENT_ITERATIONS", "3")),
    )

    print("\n" + "=" * 72)
    print("EXTRACTED CONSTRAINTS")
    print("=" * 72)
    for constraint in result.constraint_set.constraints:
        flags = []
        if constraint.is_negative:
            flags.append("NEGATIVE")
        if not constraint.verifiable:
            flags.append("not auto-verifiable")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {constraint.id} ({constraint.category.value}): {constraint.description}{suffix}")

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(result.summary())

    if config.base.device == "cuda":
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU:       {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
