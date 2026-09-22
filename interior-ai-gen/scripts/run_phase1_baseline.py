"""Phase 1 — Minimal Diffusion Baseline.

Run this on a GPU runtime (Kaggle or Google Colab). It is NOT expected to run on
the local dev laptop (AMD Ryzen 5 5500U, 8GB RAM, integrated graphics — no CUDA).

What it does:
    1. Loads the fixed SDXL 1.0 baseline pipeline.
    2. Generates one image from the Phase 1 test prompt.
    3. Re-runs generation with the same fixed seed to demonstrate reproducibility.
    4. Records generation parameters, timing, and (on CUDA) peak VRAM usage.

Usage (see README.md "Running Phase 1 on Kaggle/Colab" for the full notebook cell):

    python scripts/run_phase1_baseline.py

Configuration is entirely environment-variable driven (see .env.example) —
nothing here is hardcoded. Defaults are the Phase 1 baseline settings.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.generation.model_loader import load_sdxl_pipeline  # noqa: E402
from app.generation.text_to_image import generate_image  # noqa: E402

TEST_PROMPT = (
    "Modern luxury bedroom interior, cream-colored walls, warm ambient lighting, "
    "elegant wooden furniture, realistic architectural photography, high-quality "
    "interior design visualization."
)

logger = logging.getLogger(__name__)


def _image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    setup_logging()
    config = load_config()

    logger.info("Phase 1 baseline run starting with config: %s", config.as_dict())

    import torch  # local import: only needed on the GPU runtime

    pipe = load_sdxl_pipeline(config.model_id, device=config.device, dtype=config.dtype)

    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Run 1 ---
    generator_1 = None
    if config.seed is not None:
        generator_1 = torch.Generator(device=config.device).manual_seed(config.seed)
    result_1 = generate_image(pipe, TEST_PROMPT, config, generator=generator_1, filename="phase1_run1.png")

    peak_vram_gb = None
    if config.device == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n=== Phase 1 — Run 1 ===")
    print(f"Image:            {result_1.image_path}")
    print(f"Metadata:         {result_1.metadata_path}")
    print(f"Generation time:  {result_1.generation_time_seconds:.2f}s")
    if peak_vram_gb is not None:
        print(f"Peak VRAM:        {peak_vram_gb:.2f} GB")
        print(f"GPU:              {torch.cuda.get_device_name(0)}")

    # --- Reproducibility check: same seed, second run ---
    if config.seed is not None:
        generator_2 = torch.Generator(device=config.device).manual_seed(config.seed)
        result_2 = generate_image(pipe, TEST_PROMPT, config, generator=generator_2, filename="phase1_run2.png")

        hash_1 = _image_sha256(Path(result_1.image_path))
        hash_2 = _image_sha256(Path(result_2.image_path))
        reproducible = hash_1 == hash_2

        print("\n=== Reproducibility check (fixed seed) ===")
        print(f"Seed:             {config.seed}")
        print(f"Run 1 sha256:     {hash_1}")
        print(f"Run 2 sha256:     {hash_2}")
        print(f"Reproducible:     {reproducible}")

        if not reproducible:
            logger.warning(
                "Same-seed runs produced different images. On CUDA this can happen "
                "if deterministic algorithms aren't forced (cuDNN nondeterminism); "
                "note this in RESEARCH.md if it occurs and consider "
                "torch.use_deterministic_algorithms(True) as a follow-up."
            )
    else:
        print("\nSEED is not set — skipping reproducibility check. Set SEED to a fixed "
              "integer (e.g. SEED=42) to test reproducibility.")

    print("\nPhase 1 baseline run complete.")


if __name__ == "__main__":
    main()
