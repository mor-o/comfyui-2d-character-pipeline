# Workflow 2 — Video Generation

Parent: [comfyui-animation-generation](./pipeline-overview.md). Related: [Workflow 3 — Spritesheet](./workflow-3-spritesheet.md) (consumes the mp4). Helper: [Workflow 1 — Pose Edit](./workflow-1-pose-edit.md) (can produce a specifically-posed keyframe if the default base character image is not suitable).

Takes the base character image and generates a short mp4 of the character performing the target animation (idle breathing, walking, running, jumping, …). Uses WAN 2.2 14B i2v with two-stage MoE eviction so the high-noise and low-noise experts never coexist in VRAM on a 24 GB card.

## Goal

Extrapolate a short motion clip from a single character image. The image anchors the character identity and starting frame; the positive prompt describes the motion; WAN 2.2 i2v hallucinates the rest of the frames from there.

This workflow takes a **single** keyframe and lets WAN invent the full motion, rather than requiring a start + end keyframe pair and interpolating between them. That is both simpler (no end-keyframe authoring) and more forgiving (no "end pose must match start in facing direction and envelope" failure mode) — in exchange for slightly less control over where the motion ends up.

## Input / output

**Input:**

- `F:/ComfyUI/input/keyframe_<animation>.png` — the starting keyframe (512×512). Normally this is the shared base character image. If a specific starting pose is required, use [Workflow 1 — Pose Edit](./workflow-1-pose-edit.md) to produce a posed variant first, then copy its output here:

  ```bash
  cp F:/ComfyUI/output/character_pipeline/base_animations/<animation>/poses/posed_<animation>_NNNNN_.png \
     F:/ComfyUI/input/keyframe_<animation>.png
  ```

**Output:**

- `F:/ComfyUI/output/character_pipeline/base_animations/<animation>/<animation>_NNNNN_.mp4` — the rendered video clip at 512×512, 16 fps.

## Implementation

**Workflow file:** `F:/ComfyUI/user/default/workflows/w2-video-gen/workflow_api.json`.

**Driver:** `F:/ComfyUI/user/run_w2_video_gen.py` — edit `ANIMATION_NAME`, `KEYFRAME_IMAGE`, `SEED`, `NUM_FRAMES`, `POS_PROMPT`, `NEG_PROMPT` at the top, then run with `F:/ComfyUI/.venv/Scripts/python.exe F:/ComfyUI/user/run_w2_video_gen.py`.

### Node graph

1. `UnetLoaderGGUF` (high-noise expert) → `ModelSamplingSD3` (shift 8) → `KSamplerAdvanced` **stage 1** (`add_noise=enable`, steps 4, `start_at_step=0`, `end_at_step=2`, `return_with_leftover_noise=enable`).
2. Stage 1 latent → `VRAMUnloadModel` (evicts the high-noise patcher from VRAM) → latent passthrough.
3. `UnetLoaderGGUF` (low-noise expert) → `ModelSamplingSD3` (shift 8) → `KSamplerAdvanced` **stage 2** (`add_noise=disable`, steps 4, `start_at_step=2`, `end_at_step=10000`, `return_with_leftover_noise=disable`).
4. Stage 2 latent → `VRAMUnloadModel` (evicts the low-noise patcher) → `VAEDecode`.
5. `VAEDecode` → `CreateVideo` (fps 16) → `SaveVideo` into `character_pipeline/base_animations/<animation>/<animation>` (`_NNNNN_.mp4` counter + extension added automatically).

The single-image conditioning node is `WanImageToVideo` (not `WanFirstLastFrameToVideo`) — it takes just `start_image`, which is how single-keyframe i2v works in ComfyUI master.

### Frame count constraint

`length` must satisfy WAN's `4n + 1` rule: `9`, `13`, `17`, `21`, `25`, `33`, `49`, `81`. A value like `10` or `50` will fail. At 16 fps:

| Frames | Seconds |
|---|---|
| 13 | ~0.8 s |
| 17 | ~1.0 s |
| 25 | ~1.6 s |
| 33 | ~2.0 s |
| 49 | ~3.0 s |

Default: **33** (2 s). Idle / breathing / subtle motion plays well at 25-33; walking / running cycles typically want 33-49.

### Two-stage VRAM eviction (the critical part)

The WAN 2.2 i2v MoE architecture splits sampling across two expert UNETs (high-noise → low-noise), each ~10.1 GB on disk at Q5_K_M. Without eviction, both would coexist in VRAM plus UMT5 (~3.5 GB) + VAE + activations, which overruns a 24 GB card.

The workflow wires **two** `VRAMUnloadModel` instances:

- **After stage 1**, before stage 2 loads: evicts the high-noise patcher so the low-noise `UnetLoaderGGUF` has room to allocate.
- **After stage 2**, before `VAEDecode` and before the workflow ends: evicts the low-noise patcher so the next run's high-noise load doesn't stack on top.

`VRAMUnloadModel` is a custom node at `F:/ComfyUI/custom_nodes/comfyui_vram_helpers/__init__.py`. It meta-tensors the target `nn.Module` in place (`inner.to('meta')`) — freeing all CUDA allocations — and surgically pops the upstream loader chain from ComfyUI's `PromptExecutor.caches.outputs` so the **next** enqueue re-loads the expert from disk instead of reusing the now-meta-tensored ModelPatcher (which would crash with `Tensor.item() cannot be called on meta tensors`).

**Validation:** the ComfyUI log should show two `[VRAMUnloadModel] dropping … dropped …` pairs per run, each freeing ~10.8 GB:

```
[VRAMUnloadModel] dropping: GGUFModelPatcher<WAN21> […, weights=10.80 GB]
[VRAMUnloadModel] dropped:  GGUFModelPatcher<WAN21> […, weights=10.80 GB] | VRAM freed: +10.80 GB (free 7.91 -> 18.71 GB)
[VRAMUnloadModel] invalidated output cache for nodes: 1, 3
… then stage 2 …
[VRAMUnloadModel] dropping: GGUFModelPatcher<WAN21> […, weights=10.80 GB]
[VRAMUnloadModel] dropped:  GGUFModelPatcher<WAN21> […, weights=10.80 GB] | VRAM freed: +10.80 GB (free 7.92 -> 18.72 GB)
[VRAMUnloadModel] invalidated output cache for nodes: 4, 6
```

If either pair is missing, the next run will either OOM on dual-resident experts or crash with the meta-tensor error. Fetch logs from `F:/ComfyUI/user/comfyui_8000.log` or via `GET /internal/logs/raw`.

## Models used

All present at `F:/ComfyUI/models/`.

| File | Location | Size (disk) | Purpose |
|---|---|---|---|
| `wan2.2_i2v_high_noise_lightx2v_Q5_K_M.gguf` | `models/unet/` | ~10.8 GB | WAN 2.2 high-noise expert (stage 1), distilled lightx2v 4-step baked in. |
| `wan2.2_i2v_low_noise_lightx2v_Q5_K_M.gguf` | `models/unet/` | ~10.8 GB | WAN 2.2 low-noise expert (stage 2), distilled lightx2v 4-step baked in. |
| `umt5-xxl-encoder-Q5_K_M.gguf` | `models/text_encoders/` | — | UMT5-XXL text encoder, GGUF Q5_K_M variant. Loaded via `CLIPLoaderGGUF` with `type=wan`. |
| `wan_2.1_vae.safetensors` | `models/vae/` | — | WAN 2.1 VAE (**not** 2.2 — WAN 2.2 14B i2v inherited the 2.1 VAE; using `wan2.2_vae.safetensors` gives a `36 vs 64 channels` error). |

Pulled from [jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF). The 4-step lightx2v distillation is baked into the GGUF — no separate LoRA needed (unlike the fp8 variant).

### Required custom nodes

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — `UnetLoaderGGUF`, `CLIPLoaderGGUF`.
- `comfyui_vram_helpers` — in-tree at `F:/ComfyUI/custom_nodes/comfyui_vram_helpers/`. Provides `VRAMUnloadModel` (MODEL passthrough via LATENT) and `VRAMUnloadClip` (CLIP eviction via CONDITIONING passthroughs, used by Workflow 1).

## VRAM usage

Measured on a 24 GB RTX 3090 at 512×512:

| Moment in the run | VRAM used (GPU) |
|---|---|
| Idle baseline (no models loaded) | ~0 GB |
| After stage-1 high-noise load + UMT5 + VAE | ~16 GB |
| Peak during stage 1 sampling | ~18 GB |
| After stage-1 eviction (before stage 2 loads) | ~6 GB |
| Peak during stage 2 sampling | ~18 GB |
| After stage-2 eviction (before VAE decode) | ~6 GB |
| VAE decode at 33 frames × 512² | ~8 GB |

Peak ~18 GB on a 24 GB card, with enough headroom for the OS. The two-stage eviction is what keeps the peak below 20 GB — without it, stage 2 tries to load a fresh 10.8 GB expert on top of the resident stage 1 and overruns.

## Configuration knobs

Edit at the top of `run_w2_video_gen.py`:

| Knob | Default | Effect |
|---|---|---|
| `ANIMATION_NAME` | `"idle_breathing"` | Used in input filename lookup and output directory. |
| `KEYFRAME_IMAGE` | `"keyframe_idle_breathing.png"` | Filename in `F:/ComfyUI/input/`. |
| `SEED` | `12345` | Applied to both sampler stages. Re-roll if the motion has ghost limbs or jitter. |
| `NUM_FRAMES` | `33` | Must satisfy `4n + 1`. Propagate into Workflow 3 so the sprite sheet cell count matches. |
| `FPS` | `16` | `CreateVideo` frame rate. |
| `POS_PROMPT` | (see file) | Describes the motion explicitly ("breathing gently, chest rising and falling…"). Must agree with the keyframe — if the keyframe is standing and the prompt says "running", WAN produces nonsense. |
| `NEG_PROMPT` | (see file) | Prohibits facial features, clothing, ground shadow, jitter, warping — copied from the recommended faceless default for keyframe generation. |

## Wall-time expectations

On a 3090, 33 frames at 512²:

- **Cold first run** (models load from disk): ~4-5 min.
- **Warm subsequent runs** (UMT5 + VAE stay hot, only the two experts re-load): ~2.5-3 min.

Per-stage breakdown (warm):

| Stage | Wall time |
|---|---|
| Stage 1 high-noise sampling (4 steps) | ~40 s |
| `VRAMUnloadModel` drop #1 | <1 s |
| Stage 2 low-noise load + sampling | ~60 s |
| `VRAMUnloadModel` drop #2 | <1 s |
| VAE decode (33 frames, non-tiled) | ~15 s |
| Video encode + save | ~5 s |

## Troubleshooting

### `36 vs 64 channels` error at VAE decode

Wrong VAE. Must be `wan_2.1_vae.safetensors`, **not** `wan2.2_vae.safetensors`. The WAN 2.2 VAE is for the 5B ti2v model.

### OOM during stage 2 / long hang at 99 % VRAM

`VRAMUnloadModel` between stages didn't fire (or its cache-invalidation hook broke after a ComfyUI upgrade). Check the log for the `[VRAMUnloadModel] dropping / dropped` lines after stage 1.

- If the lines are missing: the node isn't wired correctly. Verify node IDs 19 and 20 in the workflow are present and that 19's `latent` input comes from stage 1's latent output, and 20's `latent` input comes from stage 2.
- If the lines are present but OOM still happens: a ComfyUI upgrade may have broken the stack-walking cache hook. Fallback: `POST http://127.0.0.1:8000/free` with `{"unload_models": true, "free_memory": true}` before every enqueue. Every run then starts cold (~+40 s of text-encode + VAE load) but is correct.

### `RuntimeError: Tensor.item() cannot be called on meta tensors` on a re-run

`VRAMUnloadModel`'s cache invalidation didn't fire. ComfyUI cached the `UnetLoaderGGUF` output by reference, and on the second enqueue it tried to reuse the now-meta-tensored patcher.

- Check for the `[VRAMUnloadModel] invalidated output cache for nodes: …` line in the log. It should name nodes `1, 3` (ancestors of stage 1's VRAMUnloadModel) and `4, 6` (ancestors of stage 2's VRAMUnloadModel).
- If the line is missing, ComfyUI's `execution.execute()` may have been reshaped by an upgrade and the stack-walking hook can no longer locate `caches`. Restart ComfyUI or fall back to `POST /free` between runs.

## Current status

**Verified 2026-04-18 on `idle_breathing`.** Two-stage MoE eviction validated via log output (`[VRAMUnloadModel] dropped … +10.80 GB` for both stage 1 and stage 2). Cache invalidation also validated (`invalidated output cache for nodes: 1, 3` / `4, 6`). Peak VRAM stayed under 20 GB. Output: `F:/ComfyUI/output/character_pipeline/base_animations/idle_breathing/idle_breathing_00001_.mp4` (33 frames, 2 s at 16 fps).
