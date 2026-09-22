"""Phase 1 configuration.

Deliberately dependency-light (stdlib only) so it can be imported and unit-tested
without torch/diffusers installed. All values are overridable via environment
variables (see .env.example) — nothing is hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    """All parameters that affect a single text-to-image generation run."""

    model_id: str
    resolution: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    output_dir: Path
    device: str
    dtype: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["output_dir"] = str(self.output_dir)
        return d


def load_config(env: Optional[dict] = None) -> GenerationConfig:
    """Build a GenerationConfig from environment variables (or an injected mapping,
    for testing) with sensible Phase-1 defaults.

    Env vars:
        DIFFUSION_BASE_MODEL   default: stabilityai/stable-diffusion-xl-base-1.0
        IMAGE_RESOLUTION       default: 1024
        NUM_INFERENCE_STEPS    default: 30
        GUIDANCE_SCALE         default: 7.5
        SEED                   default: unset (None -> non-reproducible run)
        OUTPUT_DIR             default: ./outputs
        DEVICE                 default: cuda   (Phase 1 targets Kaggle/Colab GPU)
        TORCH_DTYPE            default: float16
    """
    source = env if env is not None else os.environ

    seed_raw = source.get("SEED")
    seed = int(seed_raw) if seed_raw not in (None, "") else None

    return GenerationConfig(
        model_id=source.get("DIFFUSION_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
        resolution=int(source.get("IMAGE_RESOLUTION", "1024")),
        num_inference_steps=int(source.get("NUM_INFERENCE_STEPS", "30")),
        guidance_scale=float(source.get("GUIDANCE_SCALE", "7.5")),
        seed=seed,
        output_dir=Path(source.get("OUTPUT_DIR", "./outputs")),
        device=source.get("DEVICE", "cuda"),
        dtype=source.get("TORCH_DTYPE", "float16"),
    )


@dataclass(frozen=True)
class ImageToImageConfig:
    """Phase 2 config: wraps the same base generation parameters plus the two
    things image-to-image needs on top — an input image and a denoising strength."""

    base: GenerationConfig
    input_image_path: Path
    strength: float

    def as_dict(self) -> dict:
        d = self.base.as_dict()
        d["input_image_path"] = str(self.input_image_path)
        d["strength"] = self.strength
        return d


def load_img2img_config(env: Optional[dict] = None) -> ImageToImageConfig:
    """Build an ImageToImageConfig from environment variables (or an injected
    mapping, for testing).

    Additional env vars on top of load_config()'s:
        INPUT_IMAGE_PATH   required — path to the existing room photo to transform.
        STRENGTH           default: 0.6  (0=ignore prompt/keep input image,
                                           1=ignore input image/pure text-to-image;
                                           diffusers' img2img denoising strength.)
    """
    source = env if env is not None else os.environ
    base = load_config(env=source)
    return ImageToImageConfig(
        base=base,
        input_image_path=Path(source.get("INPUT_IMAGE_PATH", "")),
        strength=float(source.get("STRENGTH", "0.6")),
    )


@dataclass(frozen=True)
class ControlNetConfig:
    """Phase 3 config: base generation parameters plus the input room image and
    the depth-ControlNet-specific settings."""

    base: GenerationConfig
    input_image_path: Path
    controlnet_model_id: str
    depth_model_id: str
    vae_model_id: str
    controlnet_conditioning_scale: float

    def as_dict(self) -> dict:
        d = self.base.as_dict()
        d.update(
            {
                "input_image_path": str(self.input_image_path),
                "controlnet_model_id": self.controlnet_model_id,
                "depth_model_id": self.depth_model_id,
                "vae_model_id": self.vae_model_id,
                "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            }
        )
        return d


def load_controlnet_config(env: Optional[dict] = None) -> ControlNetConfig:
    """Build a ControlNetConfig from environment variables (or an injected mapping,
    for testing).

    Additional env vars on top of load_config()'s:
        INPUT_IMAGE_PATH               required — path to the existing room photo.
        CONTROLNET_DEPTH_MODEL         default: diffusers/controlnet-depth-sdxl-1.0
        DEPTH_MODEL                    default: Intel/dpt-hybrid-midas
        VAE_MODEL_ID                   default: madebyollin/sdxl-vae-fp16-fix
                                        (fixes the AutoencoderKL fp16 NaN/casting
                                        issue noted in RESEARCH.md's Phase 1/2 addenda)
        CONTROLNET_CONDITIONING_SCALE  default: 0.8 (0=no structural conditioning,
                                        1=strict adherence to the depth map)
    """
    source = env if env is not None else os.environ
    base = load_config(env=source)
    return ControlNetConfig(
        base=base,
        input_image_path=Path(source.get("INPUT_IMAGE_PATH", "")),
        controlnet_model_id=source.get("CONTROLNET_DEPTH_MODEL", "diffusers/controlnet-depth-sdxl-1.0"),
        depth_model_id=source.get("DEPTH_MODEL", "Intel/dpt-hybrid-midas"),
        vae_model_id=source.get("VAE_MODEL_ID", "madebyollin/sdxl-vae-fp16-fix"),
        controlnet_conditioning_scale=float(source.get("CONTROLNET_CONDITIONING_SCALE", "0.8")),
    )
