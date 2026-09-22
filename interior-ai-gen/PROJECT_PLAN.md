# PROJECT_PLAN.md

**Multimodal AI-Based Controllable Image Generation for Interior Design Using LLMs and Diffusion Models**

Status: Phase 0 (research + planning). Repository is currently empty — this plan and `RESEARCH.md`
are the first commits.

## 1. Repository status

- Fresh/empty workspace — no existing code to preserve or work around.
- Folder skeleton has been created (see §6) but contains no implementation yet — every `app/*`
  subpackage is currently just a placeholder directory.
- Nothing has been downloaded; no models fetched yet, per the instruction not to pull large weights
  during Phase 0.

## 2. Architecture (confirmed)

```
USER (room image + text instruction + optional reference image)
        │
        ▼
VISION / MULTIMODAL UNDERSTANDING  (VLM reads the room image → room description)
        │
        ├──────────────► ROOM INFORMATION (objects, layout, room type)
        │
        ▼
USER REQUIREMENTS (free text)
        │
        ▼
LLM: SEMANTIC CONSTRAINT EXTRACTION
        │
        ▼
Structured constraints (Pydantic-validated JSON):
  structural / appearance / object / style / negative
        │
        ▼
GENERATION PROMPT (constraints → text prompt + conditioning config)
        │
        ▼
Diffusion Model + ControlNet (+ optional IP-Adapter)
        │
        ▼
GENERATED IMAGE
        │
        ▼
VLM: CONSTRAINT VERIFICATION → per-constraint pass/fail + compliance score
        │
        ├── PASS ──────────────► OUTPUT (image + compliance report)
        │
        └── FAIL ──► LLM builds a revised instruction ──► REGENERATE (max 3 iterations)
```

This matches the architecture given in the brief; no changes proposed at this stage.

## 3. Technology stack (confirmed, see RESEARCH.md for the reasoning)

- **Language:** Python 3.11
- **Diffusion:** PyTorch + Hugging Face `diffusers`, SDXL 1.0 base (swap to RealVisXL V5.0 once working)
- **Conditioning:** ControlNet (Depth SDXL, +Canny/Union as an ablation), IP-Adapter (`h94/IP-Adapter`, SDXL weights)
- **Vision/CV preprocessing:** OpenCV, PIL, NumPy, Intel DPT-Large or MiDaS for depth maps
- **LLM/VLM:** one API-based multimodal model to start (Claude or GPT-5-class vision), behind a
  provider-agnostic interface in `app/llm/`, with Qwen2.5/3-VL as a self-hosted fallback for
  cost-free reproducibility
- **Backend:** FastAPI
- **Frontend:** Streamlit (initial)
- **Compute:** local laptop for everything except diffusion inference; Kaggle/Colab GPU for SDXL + ControlNet + IP-Adapter runs
- **Testing:** pytest, with mock/stub modes for diffusion and LLM calls so non-GPU logic is testable locally

## 4. Model recommendations (summary — full reasoning in RESEARCH.md)

| Component | Recommendation | Fallback / ablation |
|---|---|---|
| Diffusion base | SDXL 1.0 base → RealVisXL V5.0 | FLUX.1-dev (stretch) |
| Structural conditioning | ControlNet-Depth-SDXL | + ControlNet-Union / Canny (dual-conditioning ablation) |
| Reference conditioning | IP-Adapter SDXL (`ip-adapter_sdxl.bin` or `-plus_sdxl_vit-h`) | — |
| Scene understanding + constraint extraction | Claude or GPT-5-class vision (single provider, swappable interface) | Qwen2.5/3-VL self-hosted |
| Verification | Same model family as above | Separate model, to reduce self-agreement bias — worth testing as an ablation |

## 5. Dataset recommendations

- **Structured3D** (bedroom + living-room subsets) — source of realistic "existing room" test images, some with structural annotations and paired empty/furnished renders.
- **LSUN Bedroom** — supplementary source of room photos for broader test coverage.
- **Hand-built evaluation sets (required, no shortcut):**
  - 50 annotated natural-language instructions → ground-truth structured constraints (for Constraint Extraction Success, §B of the success criteria)
  - 50 room-design scenarios with 5–8 requirements each (for the primary Constraint Compliance KPI)
  - A human-labeled subset for VLM-verification precision/recall/F1 checking

## 6. Project folder structure (created)

```
interior-ai-gen/
├── app/
│   ├── api/           # FastAPI routes
│   ├── core/           # config, settings, shared types
│   ├── generation/     # diffusion + ControlNet + IP-Adapter pipeline
│   ├── vision/         # room-image understanding, preprocessing (depth/edge/segmentation)
│   ├── llm/             # provider-agnostic LLM/VLM interface, constraint extraction, prompt building
│   ├── constraints/     # Pydantic constraint schema + validation
│   ├── evaluation/      # metrics: SSIM/LPIPS, CLIP, FID, compliance scoring
│   └── utils/
├── frontend/            # Streamlit app
├── configs/             # YAML/env-driven config, model IDs, no hardcoded secrets
├── data/                 # datasets / test sets (gitignored beyond samples)
├── models/               # local model cache (gitignored)
├── notebooks/            # Kaggle/Colab experiment notebooks
├── tests/                 # pytest, mock/stub modes for GPU-dependent components
├── scripts/               # experiment runners, reproducibility scripts
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── RESEARCH.md
├── PROJECT_PLAN.md
└── main.py
```

## 7. Experiment / baseline plan

Five conditions, same test cases, per the ablation design in the brief:

- **A.** Prompt → Diffusion (text-only baseline)
- **B.** Room image + prompt → Image-to-image diffusion
- **C.** Room image → ControlNet → Diffusion (structural conditioning, no LLM)
- **D.** ControlNet + IP-Adapter reference conditioning (no LLM)
- **E.** Full proposed system: LLM constraint extraction → ControlNet(+IP-Adapter) → Diffusion → VLM verification → refinement

Each recorded with: input image, requirements, generated image, compliance score, structural score
(SSIM/LPIPS/edge/depth), CLIP alignment, generation time, refinement iteration count — stored as
structured CSV/JSON per the reproducibility requirement.

## 8. Minimum viable prototype

The smallest slice that proves the core idea end-to-end, before any evaluation rigor:

1. Accept one room image + one text instruction (hardcoded test case is fine).
2. LLM extracts constraints into the Pydantic schema.
3. Constraints → generation prompt (simple template, no fancy prompt engineering yet).
4. SDXL + Depth-ControlNet generates one image on Kaggle/Colab.
5. VLM verifies the result against the constraint list, returns pass/fail + score.
6. Print/display the compliance report next to the image.

No refinement loop, no Streamlit UI, no FastAPI yet, no metrics beyond the VLM's own pass/fail — this is Phases 1–3 + a thin slice of 5–7, just enough to validate the pipeline shape before investing in the harder parts (refinement, evaluation, ablation).

## 9. Risks and limitations to track honestly

- **VLM verification reliability** is the biggest unknown — if it can't reliably judge constraints like "warm lighting" or approximate spatial preservation, that has to be reported as a limitation, not hidden (per §H of the success criteria).
- **Free-tier GPU session limits** (Kaggle/Colab) constrain how large the 50-scenario experiment run can be in one sitting — scripts need to checkpoint/resume rather than assume one continuous session.
- **Structured3D and LSUN don't include free-text instructions** — the hand-built 50-instruction and 50-scenario sets are real work, not a formality, and are on the critical path before any KPI can be measured.
- **Model landscape is moving fast** (see RESEARCH.md §4) — whichever LLM/VLM is picked now may not be the best choice by the time Phase 7 is reached; the provider-agnostic interface exists specifically to make that swap cheap.
- **Fake-novelty risk** — per the brief's own instruction, the write-up must frame this as evaluation of existing techniques, not invention of new ones.

## 10. Phase 1 status — Minimal Diffusion Baseline

**Complete.** Implemented, unit-tested, and executed end-to-end on a real GPU (Colab).

Built (fixed SDXL 1.0 baseline only — no RealVisXL/alternative checkpoints yet, per the Phase 1 constraint):

- `app/core/config.py` — env-var-driven `GenerationConfig` (model id, resolution, steps,
  guidance scale, seed, output dir, device, dtype). No hardcoded paths/credentials.
- `app/core/logging_config.py` — basic logging setup.
- `app/generation/model_loader.py` — loads the fixed SDXL 1.0 base pipeline (requires
  torch/diffusers/CUDA; deliberately the only Phase-1 module with that dependency).
- `app/generation/text_to_image.py` — generation + metadata-recording logic, decoupled
  from torch so it's testable without a GPU.
- `scripts/run_phase1_baseline.py` — the real entry point: loads SDXL, runs the test
  prompt, re-runs with a fixed seed to check reproducibility, reports GPU/VRAM/timing.
- `tests/test_phase1_baseline.py` — 8 tests, all passing, covering config and generation
  logic via a fake pipeline (no torch/GPU required to run these).
- README.md — Kaggle/Colab setup + run instructions, test instructions, and an explicit
  "Phase 1 execution status" section.

**Success criteria — all verified:**

| Success criterion | Status |
|---|---|
| Code loads/compiles without errors | ✅ Verified (`py_compile`, all modules) |
| Config/logging/metadata logic correct | ✅ Verified — 8/8 unit tests passing (sandbox, no GPU needed) |
| Model loads without errors (real SDXL) | ✅ Verified on Colab — SDXL 1.0 base loaded successfully |
| Generation completes on target GPU | ✅ Verified on Colab — 30/30 steps completed |
| Valid image file produced (real SDXL) | ✅ Verified — `phase1_run1.png` generated and opened successfully |
| Fixed seed reproducibility (real SDXL) | ✅ Verified — two independent runs with seed=42 produced byte-identical images (matching SHA-256) |
| Runs from a clean environment via docs | ✅ Verified — zip upload → pip install → run, per README, completed successfully |
| GPU/VRAM/generation time recorded | ✅ Verified — see results below |

### Real results (Google Colab, Tesla T4)

| Metric | Value |
|---|---|
| GPU | Tesla T4 |
| Peak VRAM | 9.73 GB |
| Generation time (run 1) | 25.97s (30 steps, ~1.30 it/s) |
| Generation time (run 2) | 25.66s |
| Resolution | 1024×1024 (default) |
| Steps / guidance scale | 30 / 7.5 |
| Seed | 42 |
| Reproducibility | **True** — `e9ab6a9311b5b3307944f6a55abe2cfb6de7174a29e0aff8544d185088ca8784` (identical on both runs) |

**Minor non-blocking note:** the run logged a diffusers warning about `AutoencoderKL` modules
being cast via `.to()` rather than `torch_dtype` in `from_pretrained()` — cosmetic in Phase 1
(didn't affect reproducibility or output validity), but worth passing `torch_dtype` consistently
through VAE loading in a later cleanup pass so the warning doesn't get louder once ControlNet/
IP-Adapter add more submodules in Phase 3–4.

Phase 1 is now fully complete.

## 11. Phase 2 status — Image-to-Image

**Complete.** Implemented, unit-tested, and executed end-to-end on a real GPU (Colab).

Built (SDXL 1.0 img2img, still no ControlNet/IP-Adapter/LLM/VLM/FastAPI/Streamlit — those are
Phase 3+):

- `app/core/config.py` — added `ImageToImageConfig` (wraps the Phase 1 `GenerationConfig` plus
  `input_image_path` and `strength`), loaded via `load_img2img_config()`.
- `app/generation/model_loader.py` — added `load_sdxl_img2img_pipeline()`. Loaded independently
  from the Phase 1 text2img pipeline rather than sharing weights, to keep Phase 2's GPU-memory
  footprint simple and separately measurable (see RESEARCH.md addendum).
- `app/generation/image_to_image.py` — `load_init_image()` (loads + resizes the input photo) and
  `generate_image_to_image()`, same torch-free design as Phase 1's generation module.
- `app/generation/result_io.py` — new shared helper (`resolve_output_path`/`save_image`/
  `write_metadata`), extracted from Phase 1's `text_to_image.py` to avoid duplicating file-writing
  logic between Phase 1 and Phase 2 (Phase 1 was refactored to use it too — all 8 original tests
  still pass unchanged).
- `scripts/run_phase2_image_to_image.py` — real entry point: loads the input photo + SDXL img2img
  pipeline, runs the test instruction, re-runs with the same seed for reproducibility, reports
  GPU/VRAM/timing — mirrors Phase 1's script structure.
- `tests/test_phase2_image_to_image.py` — 7 tests, all passing: config defaults/overrides,
  `load_init_image` resizing, output-file creation, metadata correctness (including `strength`
  and `input_image_path`), parameter pass-through, and the seed-reproducibility mechanism —
  all via a fake pipe, no GPU required.

**Status table — all verified:**

| Success criterion (from the Phase 1 template, applied to Phase 2) | Status |
|---|---|
| Code loads/compiles without errors | ✅ Verified (`py_compile`, all modules) |
| Config/loading/metadata logic correct | ✅ Verified — 7/7 new unit tests passing, 15/15 total, no regressions |
| Model loads without errors (real SDXL img2img) | ✅ Verified on Colab |
| Generation completes on target GPU | ✅ Verified — 18/18 steps completed |
| Valid transformed image produced | ✅ Verified — `phase2_run1.png` generated successfully |
| Fixed seed reproducibility (real SDXL) | ✅ Verified — two runs with seed=42 produced byte-identical images (matching SHA-256) |
| GPU/VRAM/generation time recorded | ✅ Verified — see results below |

### Real results (Google Colab, Tesla T4)

| Metric | Value |
|---|---|
| GPU | Tesla T4 |
| Peak VRAM | 9.73 GB (same as Phase 1 — expected, same base model + resolution) |
| Generation time (run 1) | 17.92s |
| Generation time (run 2) | 16.82s |
| Effective diffusion steps | 18/18 (not 30) — see note below |
| Strength | 0.6 |
| Seed | 42 |
| Reproducibility | **True** — `c3bddc1963267a11cc4bfa53a5bb0ee3e747bd3d788173186196890cc3bce14a` (identical on both runs) |

**Note on step count:** `num_inference_steps=30` was requested, but the pipeline log shows
18/18 steps — this is expected diffusers img2img behavior, not a bug: effective steps ≈
`strength × num_inference_steps` (0.6 × 30 = 18), since a partial-strength img2img run only
denoises from partway through the schedule rather than from pure noise. Worth stating explicitly
in any later write-up so "steps" isn't misreported as 30 when comparing Phase 1 vs. Phase 2 timing.

**Minor non-blocking notes**, same `AutoencoderKL` dtype-casting warning as Phase 1, plus one new
one: a `FutureWarning` that `torch_dtype` is deprecated in favor of `dtype` in `from_pretrained()`,
and a related `upcast_vae` deprecation warning inside the img2img pipeline call. None of these
affected correctness or reproducibility here; tracked as a forward-compatibility item in
RESEARCH.md since a future diffusers release will presumably require the newer argument names.

Phase 2 is now fully complete.

## 13. Phase 3 status — ControlNet (structural conditioning)

**Implemented, unit-tested (8 new tests, 23/23 passing overall), not yet executed on a real GPU.**

Built (SDXL 1.0 + depth ControlNet only — still no IP-Adapter/LLM/VLM/FastAPI/Streamlit; those
remain Phase 4+):

- `app/vision/depth.py` — `load_depth_estimator()` + `predict_depth_array()` (torch/transformers,
  GPU-only, not tested) and `depth_array_to_image()` (pure numpy postprocessing — unit-tested,
  including a degenerate/flat-input edge case that would otherwise divide by zero).
- `app/core/config.py` — added `ControlNetConfig` (input image, ControlNet model id, depth model
  id, a dedicated fp16-safe VAE id, and `controlnet_conditioning_scale`), loaded via
  `load_controlnet_config()`.
- `app/generation/model_loader.py` — added `load_sdxl_controlnet_pipeline()`, using
  `madebyollin/sdxl-vae-fp16-fix` instead of the base model's own VAE — this directly addresses
  the `AutoencoderKL` fp16 casting warning that showed up in both Phase 1 and Phase 2's real
  runs (see RESEARCH.md Phase 3 addendum for why it wasn't backported to those phases).
- `app/generation/controlnet_generation.py` — `generate_with_controlnet()`, same torch-free
  design as Phases 1-2, records `controlnet_conditioning_scale`, `depth_map_path`, and both
  model ids in metadata.
- `scripts/run_phase3_controlnet.py` — real entry point: computes + saves the depth map,
  generates via ControlNet, re-runs with the same seed for reproducibility, reports
  GPU/VRAM/timing. Uses the same `TEST_INSTRUCTION` text as Phase 2 for a fair eyeball
  comparison between the two mechanisms.
- `tests/test_phase3_controlnet.py` — 8 tests, all passing: config defaults/overrides, depth
  array normalization (including the constant-array edge case), output-file creation, metadata
  correctness, parameter pass-through, and the seed-reproducibility mechanism.

**Status table**, same caveat as before — built and tested in a CPU-only sandbox with no GPU
and no huggingface.co access:

| Success criterion (same template as Phases 1-2) | Status |
|---|---|
| Code loads/compiles without errors | ✅ Verified (`py_compile`, all modules) |
| Config/depth-postprocessing/metadata logic correct | ✅ Verified — 8/8 new unit tests passing, 23/23 total, no regressions |
| Depth model + ControlNet pipeline load without errors | ⬜ **Not verified** — needs Kaggle/Colab |
| Generation completes on target GPU | ⬜ **Not verified** — needs Kaggle/Colab |
| Valid conditioned image produced | ⬜ **Not verified** — needs Kaggle/Colab |
| Fixed seed reproducibility (real SDXL+ControlNet) | ⬜ **Not verified** — mechanism unit-tested via fake pipe |
| GPU/VRAM/generation time recorded | ⬜ **Not verified** — script reports these, no real numbers yet |
| Structural preservation vs. Phase 2 | ⬜ **Not measured** — Phase 9 (Evaluation) has the real SSIM/LPIPS/edge/depth metrics; for now the plan is another eyeball comparison against Phase 2's `phase2_run1.png`, same as we did for Phase 1 vs. Phase 2 |

**Next action:** run `scripts/run_phase3_controlnet.py` on Colab/Kaggle (pointing
`INPUT_IMAGE_PATH` at the same room photo used for Phase 2, so the three-way comparison is
apples-to-apples) and report back the console output.

## 14. Phases 5-9 status — the research core

**Implemented and unit-tested (38 new tests; 61/61 passing overall). Verified end-to-end offline
with the mock provider. NOT yet run against real LLM/VLM APIs or on GPU.**

Built in one batch due to time constraints, skipping Phase 4 (IP-Adapter), which the original
plan already flagged as optional.

### Phase 5 — Constraint extraction
- `app/constraints/schema.py` — the Pydantic constraint schema. Constraints are a flat list of
  atomically verifiable items with stable ids rather than the nested dict in the original brief,
  because the compliance KPI and the refinement loop both need to reference individual
  constraints — impossible if they are collapsed into nested free text. Includes a `verifiable`
  flag so constraints a VLM cannot reliably judge are marked rather than silently guessed (§H).
- `app/llm/provider.py` — swappable provider interface (`LLMProvider` ABC), with
  `AnthropicProvider` (text+vision) and a deterministic `MockProvider` for offline testing.
  API keys are read from env only.
- `app/llm/constraint_extraction.py` — instruction → validated `ConstraintSet`. Raises on empty
  or malformed extraction rather than returning an empty list, which would score 0/0 compliance
  and silently corrupt the primary KPI.

### Phase 6 — Prompt generation
- `app/llm/prompt_builder.py` — deterministic, template-based constraint → prompt mapping.
  Deliberately NOT another LLM call: condition D in the ablation is supposed to isolate "LLM
  prompt enhancement", so the constrained pipeline must not secretly also do LLM prompt
  rewriting. Negative constraints route to the diffusion `negative_prompt`, which is the
  mechanically correct place — appending "no television" to a positive prompt tends to summon
  televisions, since CLIP text encoders do not handle negation.
- Structural constraints are intentionally NOT restated in the text prompt; ControlNet enforces
  them more reliably via the depth map.

### Phase 7 — VLM verification
- `app/llm/verification.py` — image + constraints → per-constraint verdicts. The system prompt
  explicitly instructs the verifier to act as an independent checker rather than the generator's
  advocate. Unknown constraint ids are dropped and omissions logged, so partial verification is
  visible rather than scored as complete.
- `app/evaluation/compliance.py` — `VerificationReport` and the CCR definition, defined once and
  shared by the live loop and the offline experiment runner. Unverifiable constraints are
  excluded from the denominator rather than counted as passes or failures.

### Phase 8 — Refinement loop
- `app/core/pipeline.py` — the orchestrator. Hard iteration cap (default 3, §G). Keeps the
  BEST-scoring image rather than the last, and exposes `regressed_constraint_ids` so cases where
  refinement breaks previously-satisfied constraints are measurable rather than hidden. The
  generation backend is injected as a callable, which is what lets the same loop drive every
  ablation condition and keeps the module free of torch.
- Seeds vary per iteration: regenerating with an identical seed and a barely-changed prompt tends
  to reproduce the same failure.

### Phase 9 — Evaluation harness
- `app/evaluation/experiment.py` — runs every case through every condition, writes CSV+JSON,
  computes mean CCR, stdev, negative-constraint accuracy (§F), refinement improvement rate (§G),
  and the headline baseline-vs-proposed improvement. Generation failures are recorded as error
  rows rather than aborting the run (§C). The summary explicitly flags when the sample is below
  the specified 50 cases.
- `app/evaluation/structural.py` — SSIM and edge-similarity (IoU of Sobel edge maps) in pure
  numpy, so structural metrics run without a GPU. Two metrics, not one, per the brief's
  instruction; LPIPS needs torch and belongs in the GPU notebook.
- `data/test_cases.json` — 10 starter scenarios, each with 5-8 explicit requirements including
  negatives. **Well short of the 50 the success criteria require** — see limitations below.
- `scripts/run_full_pipeline.py`, `scripts/run_phase9_experiments.py` — GPU entry points.

### Status table

| Item | Status |
|---|---|
| All modules compile | ✅ Verified |
| Constraint schema, extraction, prompt building, verification, compliance, refinement logic | ✅ Verified — 61/61 unit tests passing |
| Full pipeline end-to-end (mock LLM, fake generator) | ✅ Verified offline — 80% → 100% compliance across one refinement, negative constraint correctly re-weighted, unverifiable constraint correctly excluded |
| Experiment harness aggregation, CSV/JSON output, failure handling | ✅ Verified offline with synthetic data |
| Real LLM constraint extraction quality | ⬜ **Not verified** — needs an API key |
| Real VLM verification accuracy | ⬜ **Not verified** — needs an API key; §H (F1 vs. human labels) not measured |
| Full pipeline on GPU with real models | ⬜ **Not verified** |
| Actual KPI numbers (baseline vs. proposed) | ⬜ **Not measured** — the harness computes them, but no real run has happened |
| Phase 3 ControlNet on GPU | ⬜ **Still not verified** from the earlier phase |

### Known gaps against the success criteria

Stated plainly rather than papered over:

- **§B (50 annotated instructions, ≥90% extraction accuracy):** not done. No annotated
  ground-truth set exists, so extraction accuracy is unmeasured.
- **§D (50 scenarios):** only 10 starter scenarios exist. Any KPI computed now is preliminary.
- **§E (structural preservation):** metrics are implemented but have not been wired into the
  experiment rows, and no baseline has been established.
- **§H (VLM verification F1 vs. human labels):** not done. This is the gap that most undermines
  the headline compliance number — an unvalidated verifier could be systematically wrong, and
  the compliance KPI inherits that error directly.
- **Phase 4 (IP-Adapter):** skipped. The original plan called it optional.
- **Phase 10 (FastAPI/Streamlit):** not implemented.

## 15. Next step

Run `scripts/run_full_pipeline.py` on Colab with a real API key. That is the single highest-value
next action: it converts the largest block of unverified work into verified work. Then
`scripts/run_phase9_experiments.py` with `MAX_CASES=2` to validate the harness before a longer run.

Before the project can be written up as academically complete, §B, §D and §H above need real
work — particularly §H, since every compliance number depends on the verifier being trustworthy.
