# Multimodal AI-Based Controllable Image Generation for Interior Design

Status: **Phases 0-3 and 5-9 implemented. See Status below for what is GPU-verified vs. code-complete-only.**

Read first:
- [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) — architecture, tech stack, phased plan, folder layout
- [`RESEARCH.md`](./RESEARCH.md) — model/dataset choices and the reasoning behind them

## What this is

An interior-redesign pipeline that goes UNDERSTAND → CONSTRAIN → GENERATE → VERIFY → REFINE:
a room photo and a natural-language instruction are turned into structured constraints, used to
condition a ControlNet-guided SDXL generation, and the result is checked against those constraints
by a vision-language model, with automatic refinement on failure.

This is an implementation and experimental evaluation of existing techniques (ControlNet,
IP-Adapter, LLM-based constraint extraction, VLM-based verification/refinement) applied to interior
design — not a claim of new base techniques. See RESEARCH.md §9.

## Status

**Phase 1 — Minimal Diffusion Baseline: complete**, verified on Colab (Tesla T4, 25.97s/run,
9.73GB peak VRAM, reproducibility confirmed). See PROJECT_PLAN.md §10 for full results.

**Phase 2 — Image-to-Image: complete**, verified on Colab (Tesla T4, 17.92s/16.82s, 9.73GB peak
VRAM, reproducibility confirmed). See PROJECT_PLAN.md §11.

**Phase 3 — ControlNet (structural conditioning): implemented, unit-tested, not yet run on a
real GPU.** See "Running Phase 3" below.

**Phases 5-9 — constraint extraction, prompt generation, VLM verification, refinement loop, and
the A-E ablation harness: implemented and unit-tested (61 tests passing), verified end-to-end
offline with the mock provider. Not yet run against real LLM/VLM APIs or on GPU.**
See "Running the full pipeline" and "Running the Phase 9 experiments" below.

**Phase 4 (IP-Adapter) and Phase 10 (FastAPI/Streamlit): not implemented.**

## Running Phase 1 on Kaggle or Google Colab

Phase 1 targets a CUDA GPU runtime. The local dev machine (AMD Ryzen 5 5500U, 8GB RAM,
integrated graphics) is used only for writing/testing code logic — not for running SDXL.

### Colab

1. Runtime → Change runtime type → **GPU** (T4 is enough for Phase 1).
2. In a cell:

```python
!git clone <your-repo-url> interior-ai-gen
%cd interior-ai-gen
!pip install -q -r requirements.txt
```

3. Set the fixed seed for the reproducibility check and run:

```python
import os
os.environ["SEED"] = "42"
os.environ["OUTPUT_DIR"] = "/content/outputs"
!python scripts/run_phase1_baseline.py
```

4. Display the result:

```python
from PIL import Image
Image.open("/content/outputs/phase1_run1.png")
```

### Kaggle

1. New Notebook → Settings → Accelerator → **GPU T4 x2** (or any available NVIDIA GPU).
2. In a cell:

```python
!git clone <your-repo-url> interior-ai-gen
%cd interior-ai-gen
!pip install -q -r requirements.txt
```

3. Run with the same env-var pattern as above:

```python
import os
os.environ["SEED"] = "42"
os.environ["OUTPUT_DIR"] = "/kaggle/working/outputs"
!python scripts/run_phase1_baseline.py
```

### What the script does

- Loads SDXL 1.0 base (fixed baseline — no alternative checkpoints in Phase 1).
- Generates the Phase 1 test prompt once (`phase1_run1.png` + a `_metadata.json` recording
  model id, resolution, steps, guidance scale, seed, device, and generation time).
- If `SEED` is set, re-runs generation with the same seed (`phase1_run2.png`) and compares
  the two images' SHA-256 hashes to verify reproducibility.
- Prints the GPU name and peak VRAM usage (CUDA only).

All parameters are environment-variable driven — see `.env.example`. Nothing is hardcoded.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DIFFUSION_BASE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | Fixed for Phase 1 |
| `IMAGE_RESOLUTION` | `1024` | Square output resolution |
| `NUM_INFERENCE_STEPS` | `30` | Diffusion steps |
| `GUIDANCE_SCALE` | `7.5` | CFG scale |
| `SEED` | unset | Set an integer to enable the reproducibility check |
| `OUTPUT_DIR` | `./outputs` | Where images + metadata are saved |
| `DEVICE` | `cuda` | Use `cpu` only for a slow smoke test |
| `TORCH_DTYPE` | `float16` | Use `float32` on CPU |

## Running Phase 2 on Kaggle or Colab

Phase 2 (image-to-image) needs a real room photo as input. The simplest option is to reuse
Phase 1's own output as the "existing room" — it's already on the same runtime.

```python
import os
os.environ["INPUT_IMAGE_PATH"] = "/content/outputs/phase1_run1.png"  # or your own room photo
os.environ["SEED"] = "42"
os.environ["OUTPUT_DIR"] = "/content/outputs"
!python scripts/run_phase2_image_to_image.py
```

```python
from PIL import Image
Image.open("/content/outputs/phase2_run1.png")
```

Additional Phase 2 config (on top of Phase 1's, see table below):

| Env var | Default | Meaning |
|---|---|---|
| `INPUT_IMAGE_PATH` | *(required, no default)* | Path to the existing room photo |
| `STRENGTH` | `0.6` | Denoising strength: 0 = keep input image, 1 = ignore it (pure text-to-image) |

## Running Phase 3 on Kaggle or Colab

Phase 3 adds structural conditioning: a depth map is computed from the input room photo and
used to guide a fresh SDXL generation (unlike Phase 2, this does **not** start from the room
image's own noisy latents — it starts from noise, guided by the depth map).

```python
import os
os.environ["INPUT_IMAGE_PATH"] = "/content/outputs/phase1_run1.png"  # or your own room photo
os.environ["SEED"] = "42"
os.environ["OUTPUT_DIR"] = "/content/outputs"
!python scripts/run_phase3_controlnet.py
```

```python
from PIL import Image
Image.open("/content/outputs/phase3_depth_map.png")  # eyeball the depth map itself
Image.open("/content/outputs/phase3_run1.png")        # then the generated result
```

Additional Phase 3 config (on top of Phase 1's, see table below):

| Env var | Default | Meaning |
|---|---|---|
| `INPUT_IMAGE_PATH` | *(required, no default)* | Path to the existing room photo |
| `CONTROLNET_DEPTH_MODEL` | `diffusers/controlnet-depth-sdxl-1.0` | SDXL depth ControlNet checkpoint |
| `DEPTH_MODEL` | `Intel/dpt-hybrid-midas` | Depth estimation model |
| `VAE_MODEL_ID` | `madebyollin/sdxl-vae-fp16-fix` | fp16-safe VAE (avoids the casting warning seen in Phases 1-2) |
| `CONTROLNET_CONDITIONING_SCALE` | `0.8` | 0 = ignore the depth map, 1 = strict adherence to it |

## Running the full pipeline (Phases 5-8)

The complete system: room image + instruction -> extracted constraints -> ControlNet-conditioned
generation -> VLM verification -> refinement -> best image + compliance report.

```python
import os
os.environ["INPUT_IMAGE_PATH"] = "/content/outputs/phase1_run1.png"
os.environ["LLM_API_KEY"] = "sk-..."          # required for real extraction/verification
os.environ["SEED"] = "42"
os.environ["OUTPUT_DIR"] = "/content/outputs"
!python scripts/run_full_pipeline.py "Convert this into a modern luxury bedroom. Keep the bed and window positions. Cream walls, wooden furniture, warm lighting, and no television."
```

Set `LLM_PROVIDER=mock` to exercise the control flow with no API key. The images are still real,
but the constraints and verdicts are canned — **never report mock-mode output as system
performance.**

**Important — `mock` vs `fake`:** `LLM_PROVIDER=mock` is a strict test double that only replays
responses scripted in advance (used by the test suite); pointing a real script at it with no
scripted responses will immediately raise `LLMError: MockProvider ran out of scripted responses`
on the very first call. For an actual no-API-key dry run of these scripts, use
`LLM_PROVIDER=fake` instead — it generates plausible-but-heuristic responses (simple keyword
matching, seeded pseudo-random verdicts) so the whole pipeline runs end-to-end. It logs a warning
on first use, and **its output must never be reported as a real result** — it validates that the
plumbing works, not that the system works.

## Running the Phase 9 experiments (A-E ablation)

```python
import os
os.environ["INPUT_IMAGE_PATH"] = "/content/outputs/phase1_run1.png"
os.environ["LLM_API_KEY"] = "sk-..."
os.environ["MAX_CASES"] = "2"                  # start small to validate the harness
!python scripts/run_phase9_experiments.py
```

Writes `experiment_results.csv` and `experiment_results.json` to `OUTPUT_DIR`, plus a summary
table with the primary KPI (mean Constraint Compliance Rate per condition, and the
percentage-point improvement of the full system over the text-only baseline).

**Runtime:** each case runs 5 conditions, and condition E generates up to 3 times. At ~20-26s per
generation on a T4, expect roughly 4-6 minutes per case. Validate with `MAX_CASES=2` before
committing to a long run, and note that Colab/Kaggle sessions time out.

| Env var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` (real) / `fake` (no-key dry run) / `mock` (test suite only) |
| `LLM_API_KEY` | *(required unless fake/mock)* | Never hardcode; read from env only |
| `MAX_REFINEMENT_ITERATIONS` | `3` | Hard cap per success criteria §G |
| `TEST_CASES` | `data/test_cases.json` | Test scenarios |
| `MAX_CASES` | `0` (all) | Limit cases for a quick run |

## Running the tests

```bash
pip install -r requirements.txt   # or just: pip install pytest pillow numpy
pytest tests/ -v
```

The tests cover config loading, generation/metadata-recording logic (Phase 1), img2img
config/loading/generation logic (Phase 2), and depth-postprocessing + ControlNet
config/generation logic (Phase 3), all using fake pipeline objects — they do **not** require
torch, diffusers, transformers, or a GPU, and do not download any model. Real SDXL/depth-model
loading (`app/generation/model_loader.py`, `app/vision/depth.py`) and the end-to-end scripts
(`scripts/run_phase1_baseline.py`, `scripts/run_phase2_image_to_image.py`,
`scripts/run_phase3_controlnet.py`) require an actual CUDA runtime and are exercised on
Kaggle/Colab, not by the test suite.

## Phase 1 execution status

Verified in a sandboxed CPU-only environment with no GPU and no access to huggingface.co:

- ✅ Config loading, defaults, and env-var overrides — unit tested, passing.
- ✅ Generation-result/metadata-recording logic — unit tested against a fake pipeline, passing.
- ✅ All modules import and compile cleanly (`py_compile`), including the torch/diffusers-only
  ones (`model_loader.py`, `run_phase1_baseline.py`).
- ❌ **Not verified:** actual SDXL model loading, GPU generation, real-seed reproducibility,
  GPU name, and VRAM usage. This sandbox has no GPU, no torch/diffusers installed, and no
  network access to huggingface.co — those numbers have to come from an actual run on
  Kaggle or Colab following the instructions above. Do not treat any GPU/VRAM/timing figures
  as measured until that run has happened.
