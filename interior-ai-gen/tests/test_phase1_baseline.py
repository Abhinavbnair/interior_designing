"""Phase 1 tests.

These test the parts of Phase 1 that don't require torch/diffusers/a GPU:
config loading and defaults, and the generate_image()/GenerationResult logic using
a fake pipeline object. Actual SDXL loading and GPU generation
(app.generation.model_loader, scripts/run_phase1_baseline.py) require a CUDA
runtime and are exercised on Kaggle/Colab, not here — see README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.core.config import load_config
from app.generation.text_to_image import generate_image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_match_phase1_baseline():
    config = load_config(env={})
    assert config.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config.resolution == 1024
    assert config.num_inference_steps == 30
    assert config.guidance_scale == 7.5
    assert config.seed is None
    assert config.output_dir == Path("./outputs")
    assert config.device == "cuda"
    assert config.dtype == "float16"


def test_config_reads_overrides_from_env_mapping():
    env = {
        "DIFFUSION_BASE_MODEL": "stabilityai/stable-diffusion-xl-base-1.0",
        "IMAGE_RESOLUTION": "768",
        "NUM_INFERENCE_STEPS": "20",
        "GUIDANCE_SCALE": "5.0",
        "SEED": "42",
        "OUTPUT_DIR": "/tmp/custom-outputs",
        "DEVICE": "cpu",
        "TORCH_DTYPE": "float32",
    }
    config = load_config(env=env)
    assert config.resolution == 768
    assert config.num_inference_steps == 20
    assert config.guidance_scale == 5.0
    assert config.seed == 42
    assert config.output_dir == Path("/tmp/custom-outputs")
    assert config.device == "cpu"
    assert config.dtype == "float32"


def test_config_no_hardcoded_output_path_leaks_into_repo_root(tmp_path):
    """Config should point wherever OUTPUT_DIR says, not a hardcoded location."""
    env = {"OUTPUT_DIR": str(tmp_path / "phase1-out")}
    config = load_config(env=env)
    assert config.output_dir == tmp_path / "phase1-out"
    assert not config.output_dir.exists()  # load_config must not create it as a side effect


# ---------------------------------------------------------------------------
# Generation logic (fake pipe — no torch/diffusers needed)
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, image: Image.Image):
        self.images = [image]


class _FakePipe:
    """Stands in for a diffusers pipeline: same call signature, deterministic output."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, height, width, num_inference_steps, guidance_scale, generator=None):
        self.calls.append(
            dict(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        )
        # A generator (fixed seed) always "generates" the same fixed color;
        # no generator (or a different seed) produces a different color —
        # stands in for real seed-dependent stochasticity.
        color = (10, 20, 30) if generator == "seed-42" else (99, 99, 99)
        return _FakeResult(Image.new("RGB", (width, height), color=color))


def _config(tmp_path, **overrides):
    from app.core.config import GenerationConfig

    base = dict(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        resolution=64,  # small for fast tests
        num_inference_steps=4,
        guidance_scale=7.5,
        seed=None,
        output_dir=tmp_path / "outputs",
        device="cpu",
        dtype="float32",
    )
    base.update(overrides)
    return GenerationConfig(**base)


def test_generate_image_creates_output_dir_and_files(tmp_path):
    pipe = _FakePipe()
    config = _config(tmp_path)

    result = generate_image(pipe, "a test prompt", config, filename="out.png")

    assert config.output_dir.exists()
    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()


def test_generate_image_records_parameters_in_metadata(tmp_path):
    pipe = _FakePipe()
    config = _config(tmp_path, seed=42, num_inference_steps=30, guidance_scale=7.5)

    result = generate_image(pipe, "a test prompt", config, filename="out.png")
    metadata = json.loads(Path(result.metadata_path).read_text())

    assert metadata["prompt"] == "a test prompt"
    assert metadata["model_id"] == config.model_id
    assert metadata["num_inference_steps"] == 30
    assert metadata["guidance_scale"] == 7.5
    assert metadata["seed"] == 42
    assert metadata["generation_time_seconds"] >= 0
    assert Path(result.image_path).exists()


def test_generate_image_passes_generation_params_to_pipe(tmp_path):
    pipe = _FakePipe()
    config = _config(tmp_path, resolution=128, num_inference_steps=10, guidance_scale=6.0)

    generate_image(pipe, "prompt", config, generator="seed-42", filename="out.png")

    assert len(pipe.calls) == 1
    call = pipe.calls[0]
    assert call["prompt"] == "prompt"
    assert call["height"] == 128 and call["width"] == 128
    assert call["num_inference_steps"] == 10
    assert call["guidance_scale"] == 6.0
    assert call["generator"] == "seed-42"


def test_fixed_seed_reproducibility_via_fake_pipe(tmp_path):
    """Simulates the same reproducibility check run in scripts/run_phase1_baseline.py:
    same fixed-seed generator -> identical image; this cannot exercise real SDXL
    stochasticity without a GPU, so it validates the *mechanism* (same generator
    input -> same output, recorded correctly), not SDXL's own determinism."""
    pipe = _FakePipe()
    config = _config(tmp_path, seed=42)

    result_1 = generate_image(pipe, "prompt", config, generator="seed-42", filename="run1.png")
    result_2 = generate_image(pipe, "prompt", config, generator="seed-42", filename="run2.png")

    img_1 = Image.open(result_1.image_path).tobytes()
    img_2 = Image.open(result_2.image_path).tobytes()
    assert img_1 == img_2


def test_different_generator_produces_different_fake_output(tmp_path):
    pipe = _FakePipe()
    config = _config(tmp_path)

    result_1 = generate_image(pipe, "prompt", config, generator="seed-42", filename="a.png")
    result_2 = generate_image(pipe, "prompt", config, generator=None, filename="b.png")

    img_1 = Image.open(result_1.image_path).tobytes()
    img_2 = Image.open(result_2.image_path).tobytes()
    assert img_1 != img_2
