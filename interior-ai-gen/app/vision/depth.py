"""Depth estimation for ControlNet structural conditioning (Phase 3).

Split the same way as Phase 1/2's torch-dependent vs. torch-free code:
- load_depth_estimator / predict_depth_array: require torch + transformers + (in
  practice) a GPU. Exercised on Kaggle/Colab, not tested here.
- depth_array_to_image: pure numpy/PIL postprocessing (raw depth array -> the
  normalized 3-channel image SDXL's depth ControlNet expects). No torch dependency,
  so it's directly unit-testable.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_depth_estimator(model_id: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load a DPT depth-estimation feature extractor + model.

    Returns (feature_extractor, model).
    """
    import torch
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    logger.info("Loading depth estimator: %s (device=%s)", model_id, device)
    feature_extractor = DPTImageProcessor.from_pretrained(model_id)
    model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return feature_extractor, model


def predict_depth_array(
    feature_extractor: Any,
    model: Any,
    image: Image.Image,
    resolution: int,
    device: str = "cuda",
) -> np.ndarray:
    """Run the depth model and return a raw (resolution x resolution) float array —
    NOT yet normalized to 0-255 (see depth_array_to_image for that step)."""
    import torch

    inputs = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        predicted = model(inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=(resolution, resolution),
        mode="bicubic",
        align_corners=False,
    )
    return depth.squeeze().cpu().numpy()


def depth_array_to_image(depth_array: np.ndarray) -> Image.Image:
    """Normalize a raw depth array to the 3-channel, 0-255 grayscale-as-RGB image
    format SDXL's depth ControlNet expects. Pure numpy — no torch — so this is
    directly unit-testable without a GPU."""
    depth_min = depth_array.min()
    depth_max = depth_array.max()

    if depth_max - depth_min < 1e-8:
        # Degenerate case (e.g. a flat/blank input) — avoid a divide-by-zero and
        # fall back to a mid-gray depth map rather than propagating NaNs.
        normalized = np.full_like(depth_array, 0.5)
    else:
        normalized = (depth_array - depth_min) / (depth_max - depth_min)

    stacked = np.stack([normalized] * 3, axis=-1)
    return Image.fromarray((stacked * 255.0).clip(0, 255).astype(np.uint8))
