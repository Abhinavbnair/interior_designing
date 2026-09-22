"""Phase 2 tests — same philosophy as Phase 1: test everything that doesn't require
torch/diffusers/a GPU. Real SDXL img2img loading and GPU generation
(app.generation.model_loader.load_sdxl_img2img_pipeline,
scripts/run_phase2_image_to_image.py) require CUDA and are exercised on Kaggle/Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.core.config import load_img2img_config
from app.generation.image_to_image import generate_image_to_image, load_init_image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_img2img_config_defaults():
    config = load_img2img_config(env={})
    assert config.base.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config.strength == 0.6
    assert config.input_image_path == Path("")


def test_img2img_config_reads_overrides():
    env = {
        "INPUT_IMAGE_PATH": "/tmp/room.png",
        "STRENGTH": "0.45",
        "SEED": "7",
        "OUTPUT_DIR": "/tmp/out",
    }
    config = load_img2img_config(env=env)
    assert config.input_image_path == Path("/tmp/room.png")
    assert config.strength == 0.45
    assert config.base.seed == 7
    assert config.base.output_dir == Path("/tmp/out")


# ---------------------------------------------------------------------------
# load_init_image
# ---------------------------------------------------------------------------

def test_load_init_image_resizes_to_square_resolution(tmp_path):
    src = tmp_path / "room.png"
    Image.new("RGB", (400, 300), color=(10, 10, 10)).save(src)

    loaded = load_init_image(src, resolution=128)

    assert loaded.size == (128, 128)
    assert loaded.mode == "RGB"


# ---------------------------------------------------------------------------
# generate_image_to_image (fake pipe)
# ---------------------------------------------------------------------------

class _FakeImg2ImgResult:
    def __init__(self, image: Image.Image):
        self.images = [image]


class _FakeImg2ImgPipe:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, image, strength, num_inference_steps, guidance_scale, generator=None):
        self.calls.append(
            dict(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        )
        color = (5, 5, 5) if generator == "seed-42" else (200, 200, 200)
        return _FakeImg2ImgResult(Image.new("RGB", image.size, color=color))


def _img2img_config(tmp_path, **overrides):
    from app.core.config import GenerationConfig, ImageToImageConfig

    base_keys = {"model_id", "resolution", "num_inference_steps", "guidance_scale", "seed", "output_dir", "device", "dtype"}
    base_defaults = dict(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        resolution=64,
        num_inference_steps=4,
        guidance_scale=7.5,
        seed=None,
        output_dir=tmp_path / "outputs",
        device="cpu",
        dtype="float32",
    )
    for k in list(overrides):
        if k in base_keys:
            base_defaults[k] = overrides.pop(k)
    base = GenerationConfig(**base_defaults)
    return ImageToImageConfig(
        base=base,
        input_image_path=overrides.pop("input_image_path", tmp_path / "room.png"),
        strength=overrides.pop("strength", 0.6),
    )


def _sample_init_image(tmp_path, size=64):
    return Image.new("RGB", (size, size), color=(50, 60, 70))


def test_generate_image_to_image_creates_output_files(tmp_path):
    pipe = _FakeImg2ImgPipe()
    config = _img2img_config(tmp_path)
    init_image = _sample_init_image(tmp_path)

    result = generate_image_to_image(pipe, "make it modern", init_image, config, filename="out.png")

    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()


def test_generate_image_to_image_records_strength_and_input_path(tmp_path):
    pipe = _FakeImg2ImgPipe()
    config = _img2img_config(tmp_path, strength=0.4, seed=42)
    init_image = _sample_init_image(tmp_path)

    result = generate_image_to_image(pipe, "make it modern", init_image, config, filename="out.png")
    metadata = json.loads(Path(result.metadata_path).read_text())

    assert metadata["strength"] == 0.4
    assert metadata["input_image_path"] == str(config.input_image_path)
    assert metadata["seed"] == 42


def test_generate_image_to_image_passes_params_to_pipe(tmp_path):
    pipe = _FakeImg2ImgPipe()
    config = _img2img_config(tmp_path, strength=0.7, num_inference_steps=12, guidance_scale=6.5)
    init_image = _sample_init_image(tmp_path)

    generate_image_to_image(pipe, "prompt", init_image, config, generator="seed-42", filename="out.png")

    call = pipe.calls[0]
    assert call["prompt"] == "prompt"
    assert call["strength"] == 0.7
    assert call["num_inference_steps"] == 12
    assert call["guidance_scale"] == 6.5
    assert call["generator"] == "seed-42"
    assert call["image"] is init_image


def test_fixed_seed_reproducibility_via_fake_pipe(tmp_path):
    pipe = _FakeImg2ImgPipe()
    config = _img2img_config(tmp_path, seed=42)
    init_image = _sample_init_image(tmp_path)

    result_1 = generate_image_to_image(pipe, "prompt", init_image, config, generator="seed-42", filename="r1.png")
    result_2 = generate_image_to_image(pipe, "prompt", init_image, config, generator="seed-42", filename="r2.png")

    img_1 = Image.open(result_1.image_path).tobytes()
    img_2 = Image.open(result_2.image_path).tobytes()
    assert img_1 == img_2
