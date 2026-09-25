"""Text-to-image generation + result/metadata recording (Phase 1 baseline).

Deliberately does not import torch/diffusers directly: the diffusion `pipe` and an
optional pre-built `generator` (e.g. torch.Generator) are passed in by the caller
(see scripts/run_phase1_baseline.py). This keeps the generation-result logic —
building the output path, writing metadata, timing — unit-testable with a fake
pipe object, without requiring torch/diffusers/a GPU to be present.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import GenerationConfig
from app.generation.result_io import resolve_output_path, save_image, write_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """Everything Phase 1's success criteria require us to record about one run."""

    image_path: str
    metadata_path: str
    prompt: str
    model_id: str
    resolution: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    device: str
    dtype: str
    generation_time_seconds: float


def generate_image(
    pipe: Any,
    prompt: str,
    config: GenerationConfig,
    generator: Any = None,
    filename: Optional[str] = None,
) -> GenerationResult:
    """Run one text-to-image generation and persist the image + its metadata.

    Args:
        pipe: a callable diffusers-style pipeline: pipe(prompt=..., height=..., width=...,
            num_inference_steps=..., guidance_scale=..., generator=...) -> object with
            an `.images` list of PIL Images.
        prompt: the text prompt.
        config: GenerationConfig (see app.core.config).
        generator: optional pre-built RNG (e.g. torch.Generator(...).manual_seed(seed)).
            Building this is the caller's job, specifically so this function has no
            torch dependency.
        filename: optional explicit output filename; a timestamped name is used
            otherwise.

    Returns:
        GenerationResult with the saved image path and recorded parameters.
    """
    filename = filename or f"gen_{int(time.time() * 1000)}.png"
    image_path = resolve_output_path(config.output_dir, filename)
    metadata_path = config.output_dir / f"{image_path.stem}_metadata.json"

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        height=config.resolution,
        width=config.resolution,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    )
    elapsed = time.perf_counter() - start

    save_image(result.images[0], image_path)

    gen_result = GenerationResult(
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        prompt=prompt,
        model_id=config.model_id,
        resolution=config.resolution,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
        generation_time_seconds=elapsed,
    )
    write_metadata(gen_result, metadata_path)

    logger.info(
        "Generated image saved: %s (%.2fs, steps=%d, guidance=%.1f, seed=%s)",
        image_path,
        elapsed,
        config.num_inference_steps,
        config.guidance_scale,
        config.seed,
    )
    return gen_result
