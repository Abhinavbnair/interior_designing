"""Image-to-image generation + result/metadata recording (Phase 2).

Same design principle as Phase 1's text_to_image.py: no direct torch/diffusers
import here. The `pipe`, the loaded `init_image` (a PIL Image), and an optional
pre-built `generator` are supplied by the caller
(see scripts/run_phase2_image_to_image.py), so this module's logic is unit-testable
with a fake pipe and a synthetic PIL image, without a GPU.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.core.config import ImageToImageConfig
from app.generation.result_io import resolve_output_path, save_image, write_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageToImageResult:
    """Everything worth recording about one image-to-image run."""

    image_path: str
    metadata_path: str
    input_image_path: str
    prompt: str
    model_id: str
    resolution: int
    num_inference_steps: int
    guidance_scale: float
    strength: float
    seed: Optional[int]
    device: str
    dtype: str
    generation_time_seconds: float


def load_init_image(path: Path, resolution: int) -> Image.Image:
    """Load and resize the input room image to a square `resolution x resolution`
    canvas. Phase 2 keeps this simple (direct resize) — aspect-ratio-preserving
    crop/pad is a reasonable later improvement, not needed for the baseline."""
    image = Image.open(path).convert("RGB")
    return image.resize((resolution, resolution))


def generate_image_to_image(
    pipe: Any,
    prompt: str,
    init_image: Image.Image,
    config: ImageToImageConfig,
    generator: Any = None,
    filename: Optional[str] = None,
) -> ImageToImageResult:
    """Run one image-to-image generation and persist the image + its metadata.

    Args:
        pipe: a callable diffusers-style img2img pipeline: pipe(prompt=..., image=...,
            strength=..., num_inference_steps=..., guidance_scale=..., generator=...)
            -> object with an `.images` list of PIL Images.
        prompt: the text instruction describing the desired transformation.
        init_image: the existing room photo, already loaded/resized (see load_init_image).
        config: ImageToImageConfig (see app.core.config).
        generator: optional pre-built RNG, built by the caller (keeps this module
            torch-free).
        filename: optional explicit output filename; a timestamped name is used
            otherwise.

    Returns:
        ImageToImageResult with the saved image path and recorded parameters.
    """
    base = config.base
    filename = filename or f"img2img_{int(time.time() * 1000)}.png"
    image_path = resolve_output_path(base.output_dir, filename)
    metadata_path = base.output_dir / f"{image_path.stem}_metadata.json"

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=config.strength,
        num_inference_steps=base.num_inference_steps,
        guidance_scale=base.guidance_scale,
        generator=generator,
    )
    elapsed = time.perf_counter() - start

    save_image(result.images[0], image_path)

    gen_result = ImageToImageResult(
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        input_image_path=str(config.input_image_path),
        prompt=prompt,
        model_id=base.model_id,
        resolution=base.resolution,
        num_inference_steps=base.num_inference_steps,
        guidance_scale=base.guidance_scale,
        strength=config.strength,
        seed=base.seed,
        device=base.device,
        dtype=base.dtype,
        generation_time_seconds=elapsed,
    )
    write_metadata(gen_result, metadata_path)

    logger.info(
        "Image-to-image result saved: %s (%.2fs, strength=%.2f, steps=%d, guidance=%.1f, seed=%s)",
        image_path,
        elapsed,
        config.strength,
        base.num_inference_steps,
        base.guidance_scale,
        base.seed,
    )
    return gen_result
