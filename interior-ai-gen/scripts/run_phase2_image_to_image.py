"""Phase 2 — Image-to-Image.

Takes an existing room photo + a text instruction and transforms it via SDXL
image-to-image (no structural conditioning yet — that's Phase 3's ControlNet).

Run this on a GPU runtime (Kaggle or Google Colab), same as Phase 1. Requires
INPUT_IMAGE_PATH to point at a real room photo (e.g. the Phase 1 output, or your
own photo uploaded to the runtime).

Usage:
    export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png
    export SEED=42
    python scripts/run_phase2_image_to_image.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_img2img_config  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.generation.model_loader import load_sdxl_img2img_pipeline  # noqa: E402
from app.generation.image_to_image import generate_image_to_image, load_init_image  # noqa: E402

TEST_INSTRUCTION = (
    "Convert this into a modern luxury bedroom. Keep the existing furniture "
    "positions. Use cream-colored walls, wooden furniture, and warm lighting."
)

logger = logging.getLogger(__name__)


def _image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    setup_logging()
    config = load_img2img_config()

    if not str(config.input_image_path):
        raise ValueError(
            "INPUT_IMAGE_PATH is not set. Point it at a room photo, e.g. the Phase 1 "
            "output: export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png"
        )
    if not config.input_image_path.exists():
        raise FileNotFoundError(f"INPUT_IMAGE_PATH does not exist: {config.input_image_path}")

    logger.info("Phase 2 img2img run starting with config: %s", config.as_dict())

    import torch  # local import: only needed on the GPU runtime

    init_image = load_init_image(config.input_image_path, config.base.resolution)
    pipe = load_sdxl_img2img_pipeline(config.base.model_id, device=config.base.device, dtype=config.base.dtype)

    if config.base.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Run 1 ---
    generator_1 = None
    if config.base.seed is not None:
        generator_1 = torch.Generator(device=config.base.device).manual_seed(config.base.seed)
    result_1 = generate_image_to_image(
        pipe, TEST_INSTRUCTION, init_image, config, generator=generator_1, filename="phase2_run1.png"
    )

    peak_vram_gb = None
    if config.base.device == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n=== Phase 2 — Run 1 ===")
    print(f"Input image:      {config.input_image_path}")
    print(f"Output image:     {result_1.image_path}")
    print(f"Metadata:         {result_1.metadata_path}")
    print(f"Strength:         {result_1.strength}")
    print(f"Generation time:  {result_1.generation_time_seconds:.2f}s")
    if peak_vram_gb is not None:
        print(f"Peak VRAM:        {peak_vram_gb:.2f} GB")
        print(f"GPU:              {torch.cuda.get_device_name(0)}")

    # --- Reproducibility check: same seed, second run ---
    if config.base.seed is not None:
        generator_2 = torch.Generator(device=config.base.device).manual_seed(config.base.seed)
        result_2 = generate_image_to_image(
            pipe, TEST_INSTRUCTION, init_image, config, generator=generator_2, filename="phase2_run2.png"
        )

        hash_1 = _image_sha256(Path(result_1.image_path))
        hash_2 = _image_sha256(Path(result_2.image_path))
        reproducible = hash_1 == hash_2

        print("\n=== Reproducibility check (fixed seed) ===")
        print(f"Seed:             {config.base.seed}")
        print(f"Run 1 sha256:     {hash_1}")
        print(f"Run 2 sha256:     {hash_2}")
        print(f"Reproducible:     {reproducible}")
    else:
        print("\nSEED is not set — skipping reproducibility check.")

    print("\nPhase 2 image-to-image run complete.")


if __name__ == "__main__":
    main()
