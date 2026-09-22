# RESEARCH.md

Phase 0 research notes for **Multimodal AI-Based Controllable Image Generation for Interior Design**.
This is a living document — model/dataset choices should be revisited before Phase 3+ since this
space moves fast. Sources checked as of Sept 2026.

---

## 1. Diffusion base model

| Option | Notes | Verdict |
|---|---|---|
| **SDXL 1.0 base** (`stabilityai/stable-diffusion-xl-base-1.0`) | Mature, huge ecosystem, works with every ControlNet/IP-Adapter variant on the market, well documented in `diffusers`. | **Use as default.** |
| **RealVisXL V5.0** (SDXL fine-tune) | Photoreal fine-tune specifically popular for interior/architectural visualization; several public interior-design pipelines (e.g. `rocketdigitalai/interior-design-sdxl` on Replicate) use RealVisXL + ControlNet-Depth + ControlNet-Union as their backbone. | **Good drop-in swap once the pipeline works on base SDXL** — same API, better photorealism. |
| SD 1.5 | Lighter, but weaker structure/prompt adherence than SDXL, and most modern ControlNet/IP-Adapter research has moved to SDXL/FLUX. | Skip. |
| FLUX.1-dev | Newer, higher quality, has IP-Adapter variants now, but heavier and the ControlNet/tooling ecosystem for structural conditioning is thinner and shifting fast. | Reasonable "if we have spare Colab time" experiment, not the primary track. |

**Recommendation:** build the whole pipeline against SDXL 1.0 + RealVisXL, since ControlNet/IP-Adapter support is most mature there and it directly matches what published interior-redesign pipelines use.

## 2. Structural conditioning (ControlNet)

Two conditioning signals are consistently used for interior redesign, and they're complementary rather than competing:

- **Depth ControlNet** (`diffusers/controlnet-depth-sdxl-1.0`, preprocessed with Intel DPT-Large or MiDaS) — preserves spatial layout/volume, good for "keep the room's 3D shape."
- **Canny / line-based ControlNet** (or **ControlNet-Union-SDXL-1.0**, which folds depth+canny+other signals into one model) — preserves architectural edges (window frames, wall lines), good for "keep the window/door where it is."

A specialized option worth noting from Astria's room-redesign docs: `controlnet-segroom`, trained specifically to preserve wall/window/floor semantics, and `controlnet-mlsd` for straight architectural lines. These are narrower/less-maintained than the SDXL-official ones, so treat as a possible ablation variant rather than the default.

**Recommendation:** start with **Depth ControlNet alone** (Phase 3, simplest), then add a second conditioning signal — either Canny or ControlNet-Union — as Phase 3b/ablation, since dual-conditioning (depth @ moderate strength + edges @ lower strength) is the pattern used in production interior-staging pipelines.

## 3. Reference-style conditioning (IP-Adapter)

- **IP-Adapter** (`h94/IP-Adapter`, SDXL weights: `ip-adapter_sdxl.bin`, or `ip-adapter-plus_sdxl_vit-h` for patch-level embeddings with better fidelity) is the standard choice, ships in `diffusers`, and combines cleanly with ControlNet in the same pipeline (`set_ip_adapter_scale`, `ip_adapter_image=`, alongside `control_image=`).
- Face-specific variants (FaceID, Plus-Face) are irrelevant here — no people in scope.
- SD3.5/FLUX IP-Adapter variants exist but would mean leaving the SDXL ecosystem — not worth it for an MVP.

**Recommendation:** `h94/IP-Adapter` SDXL weights, combined with ControlNet in the same `diffusers` pipeline call. This is Phase 4, and should be treated as optional/ablation — the core contribution is the constraint-extraction + verification loop, not IP-Adapter itself.

## 4. Scene understanding / LLM constraint extraction / VLM verification

This is the one part of the stack most likely to be reshaped by whatever's newest — verify pricing/availability before committing.

- Frontier vision-capable LLMs in this class (any works via a single swappable interface): **Claude** (Sonnet/Opus vision), **GPT-5 vision**, **Gemini 3 Pro**. Current write-ups characterize Gemini as strongest on document/long-context understanding, GPT-5 vision as strongest on spatial reasoning, and Claude as strongest on instruction-following over charts/diagrams — none of these differences are large enough to matter for a first pass; pick one, keep the interface swappable, and don't over-index on leaderboard deltas.
- **Open-weight alternative** for later cost-free experimentation: **Qwen3-VL** (or Qwen2.5-VL-72B if 3-VL isn't accessible on Colab-class hardware) — reported to rival GPT-5/Gemini-2.5-Pro on general multimodal benchmarks and is realistically self-hostable on a single Kaggle/Colab GPU at smaller parameter counts (7B/32B).

**Recommendation:**
- Use **one** API-based multimodal LLM (Claude or GPT-5-class) for both (a) room-image understanding + constraint extraction from text, and (b) post-generation verification — same model can plausibly do both jobs, simplifying the pipeline.
- Design the LLM/VLM call behind a single interface (`app/llm/`) with a provider-agnostic function signature so swapping to Gemini or a self-hosted Qwen3-VL later is a config change, not a rewrite — this was already a stated requirement and the research confirms it's worth taking seriously, since the "best" model here will keep changing.
- Keep a lightweight open-weight fallback (Qwen2.5-VL, run on Kaggle) in mind for the reproducibility/cost story — reviewers/employers will likely ask "what if I don't have an API key."

## 5. Datasets

| Dataset | Size / content | Use for |
|---|---|---|
| **Structured3D** | 21,835 rooms / ~196k photorealistic 2D renders, some paired empty↔furnished, rich structural (wall/window/door) annotations | Best fit for structural-preservation ground truth and paired before/after evaluation; also the only dataset with a meaningful paired empty/furnished subset. |
| **LSUN Bedroom** | ~3M bedroom photos, no structural annotation | Good for pretraining/sanity-checking generation quality and for sourcing "existing room" test images; not annotated enough for constraint ground truth. |
| **ADE20K** (indoor subset) | Scene-parsing segmentation labels | Useful if we need object-level segmentation for structural-preservation metrics (e.g., confirming a "sofa" region persists). |
| Manually curated test set | 50 own room photos/instructions per the KPI section | Required regardless of the above — the constraint-extraction and compliance benchmarks specified in the success criteria (50 annotated instructions, 50 design scenarios) have to be hand-built; no public dataset already has "room + free-text instruction + ground-truth constraint list." |

**Recommendation:** use Structured3D (bedroom/living-room subset) as the source of "existing room" input images for experiments, and build the required 50-instruction / 50-scenario evaluation set by hand on top of it (or on a small set of personally-sourced photos) — there's no shortcut around this since it's evaluating instruction-following, not just image quality.

## 6. Evaluation metrics — what's actually appropriate

- **Structural preservation:** SSIM and LPIPS are standard and easy to compute in `torch`/`lpips` package; depth-consistency (compare depth maps of input vs. output) and edge-similarity (compare Canny maps) are cheap add-ons using the same preprocessors already needed for ControlNet. Don't rely on SSIM alone — it's known to be insensitive to structural rearrangement that a human would flag.
- **Text/image alignment:** CLIPScore (openai/clip or open_clip) is the standard cheap metric; VLM-based scoring (ask the verification model to rate 1-5) is a useful secondary signal but is not a substitute for CLIP since it's the same model family doing generation-adjacent judging.
- **Image quality:** FID needs a reference distribution (Structured3D or LSUN renders work) and enough generated samples (typically 1k+) to be stable — likely only worth computing once, at the end, across the full test set rather than per-iteration.
- **VLM verification accuracy:** the project's own KPI (≥85% F1 against human labels) already includes the right check — don't skip the human-labeled subset, since an unvalidated VLM verifier can silently rubber-stamp bad generations.

## 7. Hardware / where things run

- Laptop (Ryzen 5 5500U, 8GB RAM, integrated graphics): **no local CUDA.** Fine for FastAPI/Streamlit, Pydantic schema work, JSON handling, prompt engineering, unit tests on mocked components, and orchestration logic.
- SDXL + ControlNet + IP-Adapter inference needs a CUDA GPU with **≥12–16GB VRAM** for comfortable SDXL generation (less with fp16/LCM tricks, but budget for the higher number). Kaggle (free T4/P100, ~16GB) and Colab (free/Pro, T4/A100 depending on tier) both work; Kaggle's free 30hr/week GPU quota is likely the more reliable free option for repeated experiment runs.
- Plan generation-heavy phases (3 onward) around batched Kaggle/Colab sessions rather than interactive iteration — session time limits mean experiment scripts (Phase 9) should be written to run unattended and log results to CSV/JSON rather than assuming a live notebook session.

## 8. Licensing notes

- SDXL base: CreativeML Open RAIL++-M — permissive for research/portfolio use, has behavioral-use restrictions (no illegal content etc.), fine for this project.
- ControlNet/IP-Adapter checkpoints (`diffusers`/`h94`, `lllyasviel`, `xinsir`, `TencentARC`): Apache-2.0 / OpenRAIL, generally research/portfolio-safe — check the specific card before using any fine-tuned checkpoint (e.g., RealVisXL) since community fine-tunes sometimes carry stricter or different licenses than the base model.
- Structured3D: research/non-commercial license (from Kujiale.com data) — fine for an academic/portfolio project, not for a commercial product without separate permission.
- LSUN: unrestricted for research use.

## 9. Novelty framing — literature check

Every individual component here (ControlNet, IP-Adapter, LLM-based prompt/constraint extraction, VLM-based verification and refinement loops for image generation) is already published and, in the case of ControlNet/IP-Adapter, productized. Papers already exist on "personalized LLM interior designer" style pipelines (e.g. I-Design, ECCV 2024) that combine LLMs with layout/generation systems, and agentic generate→verify→refine loops are an active research area generally, not unique to interiors.

**Honest framing for this project:** it's an **implementation and experimental evaluation** of an existing class of techniques (multimodal constraint extraction + controlled diffusion + VLM-based verification/refinement) applied specifically to interior redesign, with its own hand-built evaluation benchmark and ablation study — not a claim of a new architecture or new base technique. That framing is defensible and doesn't require overclaiming.

## 10. Phase 1 addendum — no assumption changes, one caveat added

Phase 1 (minimal SDXL 1.0 baseline) did not surface any need to revise the model/library choices
above. One caveat worth recording:

- **Same-seed reproducibility on CUDA is not absolutely guaranteed by default.** `torch.Generator(...).manual_seed()` fixes the RNG, but cuDNN/attention-kernel nondeterminism can, in rare cases, still cause tiny pixel-level differences across runs (and reliably differs across different GPU models/driver versions). `scripts/run_phase1_baseline.py` checks this empirically (SHA-256 of both images) rather than assuming it. If Phase 1's real Kaggle/Colab run shows non-reproducible output, the fix is `torch.use_deterministic_algorithms(True)` (at some throughput cost) — not yet applied, since it's premature to add before confirming it's actually needed.

## 11. Phase 2 addendum

- **Img2img pipeline is loaded independently, not converted from the Phase 1 text2img pipe.**
  Diffusers supports `StableDiffusionXLImg2ImgPipeline(**text2img_pipe.components)` to reuse
  already-loaded weights and save VRAM/load-time when both pipelines are needed in the same
  process. Phase 2 deliberately doesn't do this yet, so its GPU-memory footprint and load time
  can be measured on their own as a clean baseline. Worth revisiting as an optimization once
  Phase 10 (FastAPI app) needs multiple pipelines resident at once.
- No other model/library assumptions changed by Phase 2.
- **Forward-compatibility note:** the real Colab run surfaced a `FutureWarning` that
  diffusers' `torch_dtype` argument to `from_pretrained()` is deprecated in favor of `dtype`,
  plus a related `upcast_vae` deprecation warning in the img2img call path. Both are currently
  harmless (correctness/reproducibility unaffected), but the codebase should switch to `dtype=`
  before a diffusers major version removes the old argument — worth a small cleanup pass
  alongside the `AutoencoderKL` casting fix already noted above, rather than two separate patches.

## 12. Phase 3 addendum

- **Depth model:** `Intel/dpt-hybrid-midas` (via `transformers`' `DPTForDepthEstimation` +
  `DPTImageProcessor`) — the standard, well-documented pairing for SDXL depth ControlNet, matching
  the depth-computation approach shown on the `diffusers/controlnet-depth-sdxl-1.0` model card.
- **VAE fix applied in Phase 3, not backported to Phase 1/2:** `madebyollin/sdxl-vae-fp16-fix`
  replaces the base SDXL VAE to resolve the `AutoencoderKL` fp16 casting warning both earlier
  phases logged. It's introduced here rather than retrofitted into Phase 1/2 because (a) Phase
  1/2 are already GPU-verified and it's not worth re-running/re-validating already-accepted
  baselines for a cosmetic warning, and (b) Phase 3 is a natural point to pick it up since
  ControlNet pipelines are somewhat more VAE-sensitive in fp16 in general. If a later phase
  needs Phase 1/2's pipelines cleaned up too (e.g. for a consistent ablation study in Phase 9),
  apply the same `vae_model_id` pattern there at that point.
- **ControlNet conditioning scale default (0.8):** a reasonable starting point balancing
  structural adherence against the model's freedom to actually restyle the room; no experiment
  has been run yet to justify this specific number over e.g. 0.6 or 1.0 — Phase 9's evaluation
  is where that should actually get tuned/justified with data, not asserted here.
- No other model/library assumptions changed by Phase 3.
