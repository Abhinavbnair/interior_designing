"""Loads the fixed SDXL 1.0 baseline pipeline.

Phase 1 constraint: SDXL 1.0 base only — no alternative checkpoints (e.g. RealVisXL)
yet, to keep the first experiment a fixed, reproducible baseline.

This module requires torch + diffusers + (in practice) a CUDA GPU, so it is exercised
on Kaggle/Colab, not in a CPU-only/no-GPU sandbox. It is intentionally the *only*
module in Phase 1 that imports torch/diffusers at module load time, so the rest of
the codebase (config, generation-result logic) stays importable and unit-testable
without those heavy/GPU-bound dependencies installed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_sdxl_pipeline(model_id: str, device: str = "cuda", dtype: str = "float16"):
    """Load the SDXL 1.0 base text-to-image pipeline.

    Args:
        model_id: Hugging Face model id or local path. Phase 1 default is
            "stabilityai/stable-diffusion-xl-base-1.0" (see app.core.config).
        device: "cuda" (expected on Kaggle/Colab) or "cpu" (works but is very slow
            and only useful for a smoke test, not real generation).
        dtype: torch dtype name, e.g. "float16" (recommended on GPU) or "float32".

    Returns:
        A diffusers StableDiffusionXLPipeline moved to `device`.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is False. "
            "Phase 1 targets a Kaggle/Colab GPU runtime — check Runtime > Change "
            "runtime type > GPU (Colab) or the Accelerator setting (Kaggle)."
        )

    torch_dtype = getattr(torch, dtype)
    logger.info("Loading SDXL pipeline: model_id=%s dtype=%s device=%s", model_id, dtype, device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if dtype == "float16" else None,
    )
    pipe = pipe.to(device)

    logger.info("SDXL pipeline loaded on %s", device)
    return pipe


def load_sdxl_img2img_pipeline(model_id: str, device: str = "cuda", dtype: str = "float16"):
    """Load the SDXL 1.0 base image-to-image pipeline (Phase 2).

    Loaded independently from load_sdxl_pipeline() rather than converted from an
    already-loaded text-to-image pipe, to keep Phase 2 simple and its GPU-memory
    footprint predictable/measurable on its own. Sharing weights between the two
    pipelines (via `StableDiffusionXLImg2ImgPipeline(**text2img_pipe.components)`)
    is a reasonable later optimization once both are used together in one process
    (e.g. inside the Phase 10 FastAPI app) — not needed for Phase 2's standalone
    baseline.
    """
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is False. "
            "Phase 2 targets a Kaggle/Colab GPU runtime — check Runtime > Change "
            "runtime type > GPU (Colab) or the Accelerator setting (Kaggle)."
        )

    torch_dtype = getattr(torch, dtype)
    logger.info("Loading SDXL img2img pipeline: model_id=%s dtype=%s device=%s", model_id, dtype, device)

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if dtype == "float16" else None,
    )
    pipe = pipe.to(device)

    logger.info("SDXL img2img pipeline loaded on %s", device)
    return pipe


def load_sdxl_controlnet_pipeline(
    model_id: str,
    controlnet_model_id: str,
    vae_model_id: str,
    device: str = "cuda",
    dtype: str = "float16",
):
    """Load the SDXL 1.0 base + depth-ControlNet pipeline (Phase 3).

    Uses `vae_model_id` (default madebyollin/sdxl-vae-fp16-fix — see RESEARCH.md)
    instead of the base model's own VAE, to avoid the AutoencoderKL fp16
    casting/NaN warning seen in Phase 1/2's runs. Not yet backported to Phase 1/2's
    loaders — see RESEARCH.md Phase 3 addendum for why.
    """
    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is False. "
            "Phase 3 targets a Kaggle/Colab GPU runtime — check Runtime > Change "
            "runtime type > GPU (Colab) or the Accelerator setting (Kaggle)."
        )

    torch_dtype = getattr(torch, dtype)
    logger.info(
        "Loading SDXL ControlNet pipeline: model_id=%s controlnet=%s vae=%s dtype=%s device=%s",
        model_id,
        controlnet_model_id,
        vae_model_id,
        dtype,
        device,
    )

    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=torch_dtype)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if dtype == "float16" else None,
    )
    pipe = pipe.to(device)

    logger.info("SDXL ControlNet pipeline loaded on %s", device)
    return pipe
