"""Structural preservation metrics (Phase 9, §E).

Implemented in pure numpy so they run anywhere — no torch, no GPU, no scikit-image
dependency — which matters because these are computed over the whole experiment
grid and should not require the GPU runtime.

Deliberately more than one metric, per the brief's instruction not to assume a
single metric suffices: SSIM is sensitive to local structural/contrast change but
tolerates global restyling poorly, while edge similarity more directly reflects
"did the walls/windows/furniture outlines stay put", which is what structural
preservation actually means here. LPIPS is the obvious third metric but needs
torch + pretrained weights, so it belongs in the GPU notebook rather than here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _to_grayscale_array(path: Path, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size)
    return np.asarray(image, dtype=np.float64)


def ssim(image_a: np.ndarray, image_b: np.ndarray, window_size: int = 8) -> float:
    """Structural Similarity Index over non-overlapping windows.

    Returns a value in roughly [-1, 1]; 1 means identical. This is the standard
    windowed formulation with the usual C1/C2 stabilisers, simplified to
    non-overlapping blocks (rather than a Gaussian-weighted sliding window) to
    keep it dependency-free and fast. Absolute values therefore won't match
    scikit-image exactly — fine here, since every condition is compared using
    this same implementation, but worth stating in the write-up rather than
    citing the numbers as canonical SSIM.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Shape mismatch: {image_a.shape} vs {image_b.shape}")

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    height, width = image_a.shape
    scores = []
    for y in range(0, height - window_size + 1, window_size):
        for x in range(0, width - window_size + 1, window_size):
            block_a = image_a[y : y + window_size, x : x + window_size]
            block_b = image_b[y : y + window_size, x : x + window_size]

            mu_a, mu_b = block_a.mean(), block_b.mean()
            var_a, var_b = block_a.var(), block_b.var()
            covar = ((block_a - mu_a) * (block_b - mu_b)).mean()

            numerator = (2 * mu_a * mu_b + c1) * (2 * covar + c2)
            denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
            scores.append(numerator / denominator)

    return float(np.mean(scores)) if scores else 0.0


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude, normalized to 0-1."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = kernel_x.T

    padded = np.pad(image, 1, mode="edge")
    height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            region = padded[y : y + 3, x : x + 3]
            grad_x[y, x] = np.sum(region * kernel_x)
            grad_y[y, x] = np.sum(region * kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    peak = magnitude.max()
    return magnitude / peak if peak > 0 else magnitude


def edge_similarity(image_a: np.ndarray, image_b: np.ndarray, threshold: float = 0.2) -> float:
    """Intersection-over-union of thresholded edge maps.

    More directly interpretable than SSIM for this project's purpose: it asks
    "do the same structural outlines exist in both images", which is close to
    what "preserve the window/wall positions" actually means.
    """
    edges_a = _sobel_edges(image_a) > threshold
    edges_b = _sobel_edges(image_b) > threshold

    intersection = np.logical_and(edges_a, edges_b).sum()
    union = np.logical_or(edges_a, edges_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def structural_scores(original_path: Path, generated_path: Path, size: Tuple[int, int] = (256, 256)) -> dict:
    """Compute all structural-preservation metrics for one image pair.

    Default size is 256x256 rather than full resolution because the pure-numpy
    Sobel is O(pixels) with a Python loop — adequate for a few hundred image
    pairs, but full 1024x1024 would be needlessly slow.
    """
    array_a = _to_grayscale_array(Path(original_path), size)
    array_b = _to_grayscale_array(Path(generated_path), size)
    return {
        "ssim": ssim(array_a, array_b),
        "edge_similarity": edge_similarity(array_a, array_b),
    }
