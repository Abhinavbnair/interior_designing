"""Phase 3 tests — same philosophy as Phase 1/2: test everything that doesn't
require torch/diffusers/transformers/a GPU. Real depth-model loading/inference
(app.vision.depth.load_depth_estimator / predict_depth_array) and real SDXL
ControlNet loading/generation (app.generation.model_loader.load_sdxl_controlnet_pipeline,
scripts/run_phase3_controlnet.py) require CUDA and are exercised on Kaggle/Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import load_controlnet_config
from app.generation.controlnet_generation import generate_with_controlnet
from app.vision.depth import depth_array_to_image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_controlnet_config_defaults():
    config = load_controlnet_config(env={})
    assert config.base.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config.controlnet_model_id == "diffusers/controlnet-depth-sdxl-1.0"
    assert config.depth_model_id == "Intel/dpt-hybrid-midas"
    assert config.vae_model_id == "madebyollin/sdxl-vae-fp16-fix"
    assert config.controlnet_conditioning_scale == 0.8
    assert config.input_image_path == Path("")


def test_controlnet_config_reads_overrides():
    env = {
        "INPUT_IMAGE_PATH": "/tmp/room.png",
        "CONTROLNET_DEPTH_MODEL": "custom/controlnet",
        "DEPTH_MODEL": "custom/depth",
        "VAE_MODEL_ID": "custom/vae",
        "CONTROLNET_CONDITIONING_SCALE": "0.5",
        "SEED": "7",
    }
    config = load_controlnet_config(env=env)
    assert config.input_image_path == Path("/tmp/room.png")
    assert config.controlnet_model_id == "custom/controlnet"
    assert config.depth_model_id == "custom/depth"
    assert config.vae_model_id == "custom/vae"
    assert config.controlnet_conditioning_scale == 0.5
    assert config.base.seed == 7


# ---------------------------------------------------------------------------
# depth_array_to_image (pure numpy — the one Phase 3 piece testable without torch)
# ---------------------------------------------------------------------------

def test_depth_array_to_image_normalizes_to_0_255():
    depth_array = np.array([[0.0, 5.0], [10.0, 2.5]])
    image = depth_array_to_image(depth_array)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    pixels = np.array(image)
    assert pixels.min() == 0
    assert pixels.max() == 255
    # min-depth pixel (0,0) should map to 0, max-depth pixel (1,0) should map to 255,
    # across all 3 (stacked) channels.
    assert tuple(pixels[0, 0]) == (0, 0, 0)
    assert tuple(pixels[1, 0]) == (255, 255, 255)


def test_depth_array_to_image_handles_constant_array_without_nan():
    depth_array = np.full((4, 4), 3.0)
    image = depth_array_to_image(depth_array)

    pixels = np.array(image)
    assert not np.isnan(pixels).any()
    # Degenerate/flat input should fall back to a mid-gray depth map, not crash.
    assert np.all(pixels == pixels[0, 0])


# ---------------------------------------------------------------------------
# generate_with_controlnet (fake pipe)
# ---------------------------------------------------------------------------

class _FakeControlNetResult:
    def __init__(self, image: Image.Image):
        self.images = [image]


class _FakeControlNetPipe:
    def __init__(self):
        self.calls = []

    def __call__(
        self, prompt, image, height, width, num_inference_steps, guidance_scale,
        controlnet_conditioning_scale, generator=None,
    ):
        self.calls.append(
            dict(
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            )
        )
        color = (1, 2, 3) if generator == "seed-42" else (250, 250, 250)
        return _FakeControlNetResult(Image.new("RGB", (width, height), color=color))


def _controlnet_config(tmp_path, **overrides):
    from app.core.config import ControlNetConfig, GenerationConfig

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
    return ControlNetConfig(
        base=base,
        input_image_path=overrides.pop("input_image_path", tmp_path / "room.png"),
        controlnet_model_id=overrides.pop("controlnet_model_id", "diffusers/controlnet-depth-sdxl-1.0"),
        depth_model_id=overrides.pop("depth_model_id", "Intel/dpt-hybrid-midas"),
        vae_model_id=overrides.pop("vae_model_id", "madebyollin/sdxl-vae-fp16-fix"),
        controlnet_conditioning_scale=overrides.pop("controlnet_conditioning_scale", 0.8),
    )


def _sample_depth_image(size=64):
    return Image.new("RGB", (size, size), color=(128, 128, 128))


def test_generate_with_controlnet_creates_output_files(tmp_path):
    pipe = _FakeControlNetPipe()
    config = _controlnet_config(tmp_path)
    depth_image = _sample_depth_image()

    result = generate_with_controlnet(
        pipe, "make it modern", depth_image, config, depth_map_path=str(tmp_path / "depth.png"), filename="out.png"
    )

    assert Path(result.image_path).exists()
    assert Path(result.metadata_path).exists()


def test_generate_with_controlnet_records_metadata(tmp_path):
    pipe = _FakeControlNetPipe()
    config = _controlnet_config(tmp_path, controlnet_conditioning_scale=0.5, seed=42)
    depth_image = _sample_depth_image()
    depth_path = str(tmp_path / "depth.png")

    result = generate_with_controlnet(
        pipe, "make it modern", depth_image, config, depth_map_path=depth_path, filename="out.png"
    )
    metadata = json.loads(Path(result.metadata_path).read_text())

    assert metadata["controlnet_conditioning_scale"] == 0.5
    assert metadata["depth_map_path"] == depth_path
    assert metadata["input_image_path"] == str(config.input_image_path)
    assert metadata["controlnet_model_id"] == config.controlnet_model_id
    assert metadata["depth_model_id"] == config.depth_model_id
    assert metadata["seed"] == 42


def test_generate_with_controlnet_passes_params_to_pipe(tmp_path):
    pipe = _FakeControlNetPipe()
    config = _controlnet_config(tmp_path, controlnet_conditioning_scale=0.7, num_inference_steps=12)
    depth_image = _sample_depth_image()

    generate_with_controlnet(
        pipe, "prompt", depth_image, config, depth_map_path="d.png", generator="seed-42", filename="out.png"
    )

    call = pipe.calls[0]
    assert call["prompt"] == "prompt"
    assert call["controlnet_conditioning_scale"] == 0.7
    assert call["num_inference_steps"] == 12
    assert call["generator"] == "seed-42"
    assert call["image"] is depth_image


def test_fixed_seed_reproducibility_via_fake_pipe(tmp_path):
    pipe = _FakeControlNetPipe()
    config = _controlnet_config(tmp_path, seed=42)
    depth_image = _sample_depth_image()

    result_1 = generate_with_controlnet(
        pipe, "prompt", depth_image, config, depth_map_path="d.png", generator="seed-42", filename="r1.png"
    )
    result_2 = generate_with_controlnet(
        pipe, "prompt", depth_image, config, depth_map_path="d.png", generator="seed-42", filename="r2.png"
    )

    img_1 = Image.open(result_1.image_path).tobytes()
    img_2 = Image.open(result_2.image_path).tobytes()
    assert img_1 == img_2
