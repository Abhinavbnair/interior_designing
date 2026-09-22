"""Phase 3 — ControlNet (structural conditioning).

Room image -> depth map -> ControlNet + SDXL text-to-image (guided by the depth
map, not by starting from the room image's own noisy latents the way Phase 2's
img2img did) -> generated image.

This is the standard ControlNet usage pattern: generation starts from pure noise,
and the depth map only *guides* structure via conditioning — it's a different
mechanism from Phase 2's img2img, and the point of Phase 3 is to compare how much
better it preserves room layout (eyeballed now; measured properly in Phase 9).

Run this on a GPU runtime (Kaggle or Google Colab), same as Phases 1-2. Requires
INPUT_IMAGE_PATH to point at a real room photo.

Usage:
    export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png
    export SEED=42
    python scripts/run_phase3_controlnet.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_controlnet_config  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.generation.controlnet_generation import generate_with_controlnet  # noqa: E402
from app.generation.model_loader import load_sdxl_controlnet_pipeline  # noqa: E402
from app.generation.result_io import resolve_output_path, save_image  # noqa: E402
from app.vision.depth import depth_array_to_image, load_depth_estimator, predict_depth_array  # noqa: E402

TEST_INSTRUCTION = (
    "Convert this into a modern luxury bedroom. Keep the existing furniture "
    "positions. Use cream-colored walls, wooden furniture, and warm lighting."
)

logger = logging.getLogger(__name__)


def _image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    setup_logging()
    config = load_controlnet_config()

    if not str(config.input_image_path):
        raise ValueError(
            "INPUT_IMAGE_PATH is not set. Point it at a room photo, e.g. the Phase 1 "
            "output: export INPUT_IMAGE_PATH=/content/outputs/phase1_run1.png"
        )
    if not config.input_image_path.exists():
        raise FileNotFoundError(f"INPUT_IMAGE_PATH does not exist: {config.input_image_path}")

    logger.info("Phase 3 ControlNet run starting with config: %s", config.as_dict())

    import torch  # local import: only needed on the GPU runtime
    from PIL import Image

    input_image = Image.open(config.input_image_path).convert("RGB")

    # --- Depth map (structural conditioning signal) ---
    feature_extractor, depth_model = load_depth_estimator(config.depth_model_id, device=config.base.device)
    depth_array = predict_depth_array(
        feature_extractor, depth_model, input_image, config.base.resolution, device=config.base.device
    )
    depth_image = depth_array_to_image(depth_array)
    depth_map_path = resolve_output_path(config.base.output_dir, "phase3_depth_map.png")
    save_image(depth_image, depth_map_path)
    print(f"Depth map saved: {depth_map_path}  (eyeball this against the input photo)")

    # --- ControlNet-conditioned generation ---
    pipe = load_sdxl_controlnet_pipeline(
        config.base.model_id,
        config.controlnet_model_id,
        config.vae_model_id,
        device=config.base.device,
        dtype=config.base.dtype,
    )

    if config.base.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    generator_1 = None
    if config.base.seed is not None:
        generator_1 = torch.Generator(device=config.base.device).manual_seed(config.base.seed)
    result_1 = generate_with_controlnet(
        pipe,
        TEST_INSTRUCTION,
        depth_image,
        config,
        depth_map_path=str(depth_map_path),
        generator=generator_1,
        filename="phase3_run1.png",
    )

    peak_vram_gb = None
    if config.base.device == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n=== Phase 3 — Run 1 ===")
    print(f"Input image:              {config.input_image_path}")
    print(f"Depth map:                {depth_map_path}")
    print(f"Output image:             {result_1.image_path}")
    print(f"Metadata:                 {result_1.metadata_path}")
    print(f"ControlNet cond. scale:   {result_1.controlnet_conditioning_scale}")
    print(f"Generation time:          {result_1.generation_time_seconds:.2f}s")
    if peak_vram_gb is not None:
        print(f"Peak VRAM:                {peak_vram_gb:.2f} GB")
        print(f"GPU:                      {torch.cuda.get_device_name(0)}")

    # --- Reproducibility check: same seed, second run ---
    if config.base.seed is not None:
        generator_2 = torch.Generator(device=config.base.device).manual_seed(config.base.seed)
        result_2 = generate_with_controlnet(
            pipe,
            TEST_INSTRUCTION,
            depth_image,
            config,
            depth_map_path=str(depth_map_path),
            generator=generator_2,
            filename="phase3_run2.png",
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

    print("\nPhase 3 ControlNet run complete.")


if __name__ == "__main__":
    main()
