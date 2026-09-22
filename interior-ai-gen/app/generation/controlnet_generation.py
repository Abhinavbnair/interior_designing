"""ControlNet-conditioned generation + result/metadata recording (Phase 3).

Same design as Phase 1/2: no direct torch/diffusers import here. The `pipe`, the
`control_image` (a depth map, already computed by app.vision.depth), and an
optional pre-built `generator` are supplied by the caller
(see scripts/run_phase3_controlnet.py), so this module is unit-testable with a
fake pipe and a synthetic depth image, without a GPU.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

from app.core.config import ControlNetConfig
from app.generation.result_io import resolve_output_path, save_image, write_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlNetGenerationResult:
    """Everything worth recording about one ControlNet-conditioned run."""

    image_path: str
    metadata_path: str
    depth_map_path: str
    input_image_path: str
    prompt: str
    model_id: str
    controlnet_model_id: str
    depth_model_id: str
    resolution: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    seed: Optional[int]
    device: str
    dtype: str
    generation_time_seconds: float


def generate_with_controlnet(
    pipe: Any,
    prompt: str,
    control_image: Image.Image,
    config: ControlNetConfig,
    depth_map_path: str,
    generator: Any = None,
    filename: Optional[str] = None,
) -> ControlNetGenerationResult:
    """Run one depth-ControlNet-conditioned generation and persist the image + metadata.

    Args:
        pipe: a callable diffusers-style ControlNet pipeline: pipe(prompt=...,
            image=<control image>, num_inference_steps=..., guidance_scale=...,
            controlnet_conditioning_scale=..., generator=...) -> object with an
            `.images` list of PIL Images.
        prompt: the text instruction describing the desired room.
        control_image: the depth map (already computed — see app.vision.depth).
        config: ControlNetConfig (see app.core.config).
        depth_map_path: where the depth map was saved, recorded for traceability.
        generator: optional pre-built RNG, built by the caller.
        filename: optional explicit output filename; a timestamped name is used
            otherwise.

    Returns:
        ControlNetGenerationResult with the saved image path and recorded parameters.
    """
    base = config.base
    filename = filename or f"controlnet_{int(time.time() * 1000)}.png"
    image_path = resolve_output_path(base.output_dir, filename)
    metadata_path = base.output_dir / f"{image_path.stem}_metadata.json"

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        image=control_image,
        height=base.resolution,
        width=base.resolution,
        num_inference_steps=base.num_inference_steps,
        guidance_scale=base.guidance_scale,
        controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        generator=generator,
    )
    elapsed = time.perf_counter() - start

    save_image(result.images[0], image_path)

    gen_result = ControlNetGenerationResult(
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        depth_map_path=str(depth_map_path),
        input_image_path=str(config.input_image_path),
        prompt=prompt,
        model_id=base.model_id,
        controlnet_model_id=config.controlnet_model_id,
        depth_model_id=config.depth_model_id,
        resolution=base.resolution,
        num_inference_steps=base.num_inference_steps,
        guidance_scale=base.guidance_scale,
        controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        seed=base.seed,
        device=base.device,
        dtype=base.dtype,
        generation_time_seconds=elapsed,
    )
    write_metadata(gen_result, metadata_path)

    logger.info(
        "ControlNet result saved: %s (%.2fs, cond_scale=%.2f, steps=%d, guidance=%.1f, seed=%s)",
        image_path,
        elapsed,
        config.controlnet_conditioning_scale,
        base.num_inference_steps,
        base.guidance_scale,
        base.seed,
    )
    return gen_result
