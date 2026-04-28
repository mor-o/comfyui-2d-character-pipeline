# Workflow 1 — Pose Edit

Parent: [comfyui-animation-generation](./pipeline-overview.md).

Takes a character PNG and a pre-rendered OpenPose skeleton PNG, and produces a single posed keyframe of the same character in the target pose. The character identity (proportions, colors, line art style) comes from the source image; the pose comes from the skeleton through the InstantX Qwen-Image ControlNet-Union on the `openpose` sub-type.

This is the first workflow in the base-animation pipeline. Its output feeds into [Workflow 2](./workflow-2-video-gen.md) as the single starting keyframe for video generation.

## Goal

Pose a character image according to an OpenPose skeleton while preserving every pixel of character identity that the skeleton does not specify. The skeleton constrains joint positions (nose, neck, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears); the source image carries everything else (head silhouette, skin tone, line art style, body proportions).

This workflow produces **one** keyframe — Workflow 2 extrapolates motion from that single frame, rather than interpolating between a start + end keyframe pair.

## Input / output

**Input (two LoadImage nodes):**

- `F:/ComfyUI/input/source_<animation>.png` — the character to pose. Any resolution; workflow scales to 512×512 internally.
- `F:/ComfyUI/input/pose_<animation>.png` — OpenPose skeleton PNG at 512×512, rendered via `tools/render_openpose.py` from a canonical COCO-18 JSON.

**Output (one SaveImage):**

- `F:/ComfyUI/output/character_pipeline/base_animations/<animation>/poses/posed_<animation>_NNNNN_.png`

Alongside the pose artifacts:

- `…/poses/<animation>_pose.json` — COCO-18 OpenPose JSON (source of truth; hand-authored or produced by any OpenPose editor).
- `…/poses/<animation>_pose.png` — rendered skeleton PNG (derived from the JSON).

Only the JSON is canonical — both PNGs are reproducible from it.

## Implementation

**Workflow file:** `F:/ComfyUI/user/default/workflows/w1-pose-edit/workflow_api.json`.

**Driver:** `F:/ComfyUI/user/run_w1_pose_edit.py` — edit `ANIMATION_NAME`, `SOURCE_IMAGE`, `POSE_SKELETON`, `SEED`, `CONTROLNET_STRENGTH`, `POS_PROMPT` at the top, then run with `F:/ComfyUI/.venv/Scripts/python.exe F:/ComfyUI/user/run_w1_pose_edit.py`.

### Node graph

1. `UnetLoaderGGUF` (Qwen-Image-Edit-2509 Q5_K_M) → `LoraLoaderModelOnly` (2509 Lightning 4-step) → `ModelSamplingAuraFlow` (shift 3) → `CFGNorm` (strength 1) → `KSampler.model`.
2. `LoadImage` (source character) → `ImageScale` to 512×512 → `VAEEncode` → `KSampler.latent_image`. The scaled image also feeds into both `TextEncodeQwenImageEditPlus` nodes as `image1` (character identity).
3. Positive `TextEncodeQwenImageEditPlus` carries the pose prompt; negative `TextEncodeQwenImageEditPlus` is empty (**not** `ConditioningZeroOut` — that strips image conditioning and the model collapses).
4. `VRAMUnloadClip` sits between the text encoders and `ControlNetApplyAdvanced`: both conditionings and the CLIP pass in, the VL encoder's weights are dropped to meta tensors, and the conditionings pass through unchanged. This runs **before** the UNET loads for sampling.
5. `ControlNetLoader` (InstantX Union) → `SetUnionControlNetType` (type=`openpose`) → `ControlNetApplyAdvanced` with the skeleton PNG as `image` and the unloaded-clip conditionings as `positive` / `negative`.
6. `KSampler`: 4 steps, cfg 1, euler / simple, denoise 1. ControlNet `strength` defaults to 1.3 (asymmetric poses may want 1.6).
7. `VRAMUnloadModel` between `KSampler` and `VAEDecode`: drops the Qwen-Edit UNET weights, passes the latent through.
8. `VAEDecode` → `SaveImage` into the per-animation `poses/` folder.

### Two-phase VRAM eviction

The VL text encoder and the Qwen-Edit UNET are the two biggest residents (~7 GB and ~15 GB respectively). They never sit in VRAM at the same time:

- **Phase 1 — text encode.** CLIP loader materializes the VL encoder on GPU, both `TextEncodeQwenImageEditPlus` nodes produce conditionings, then `VRAMUnloadClip` meta-tensors the CLIP's `cond_stage_model` + `patcher` and pops it from `current_loaded_models`. Measured free VRAM jump: **+7.0 GB** back to the pool before any UNET weights allocate.
- **Phase 2 — sampling.** UNET loads, ControlNet loads, `KSampler` runs its 4 steps, then `VRAMUnloadModel` meta-tensors the Qwen-Edit UNET (the LoRA-merged, CFGNorm-wrapped `MODEL` output). Measured free VRAM jump: **+15.5 GB** before VAE decode runs.

Both unload nodes invalidate the cache entries of their upstream loader chain in ComfyUI's `PromptExecutor` (see `custom_nodes/comfyui_vram_helpers/__init__.py`). The consequences on re-enqueue:

- **Seed-only change.** Conditionings from nodes 9/10 are still cached (their prompts/CLIP outputs didn't change), so the CLIP chain does **not** re-run and `VRAMUnloadClip` does not fire. `KSampler` re-executes with the new seed; the UNET loader chain is cache-invalidated, so it reloads fresh from disk and `VRAMUnloadModel` fires again at the end.
- **Prompt change.** The CLIP loader's invalidated cache forces a full re-execution: fresh CLIP → new conditionings → `VRAMUnloadClip` fires → UNET loads → sampling → `VRAMUnloadModel`.

Without the cache invalidation, a second run would hit the meta-tensored `ModelPatcher` instances and crash with `RuntimeError: Tensor.item() cannot be called on meta tensors`.

### Why 512×512 and not Qwen-Edit's default 1 MP canvas

Joint attention in Qwen-Image plus the ControlNet overflows a 24 GB GPU at 1024². The workflow uses `ImageScale` to 512 instead of `FluxKontextImageScale`'s default ~1 MP canvas. Workflow 2 runs at 512² anyway, so no quality lost vs passing through a larger intermediate.

### Why `--gpu-only` is mandatory

Without it, GGUF dequantization to fp16 in system RAM saturates a 32 GB box during sampling and hard-crashes with `DefaultCPUAllocator: not enough memory`. ComfyUI must be launched with `--gpu-only` (and `--reserve-vram 1` for OS headroom) — this is baked into the current startup args.

## Models used

All models already present under `F:/ComfyUI/models/`.

| File | Location | Size | Purpose |
|---|---|---|---|
| `Qwen-Image-Edit-2509-Q5_K_M.gguf` | `models/unet/` | ~15 GB | Qwen-Image-Edit 2509 UNET, Q5_K_M GGUF. Loaded via `UnetLoaderGGUF`. Fits on a 24 GB card because the VL encoder is evicted before this loads. |
| `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | `models/loras/` | — | 4-step distillation LoRA (2509-specific — **not** the older V1.0 variant). Applied at strength 1.0. |
| `Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf` | `models/text_encoders/` | 5.44 GB | VL text encoder, Q5_K_M GGUF. Loaded via `CLIPLoaderGGUF` with `type=qwen_image`. |
| `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf` | `models/text_encoders/` | 1.35 GB | **Required companion** for the VL encoder — auto-discovered by `CLIPLoaderGGUF` in the same folder. Without it, `image1` conditioning fails. |
| `Qwen-Image-InstantX-ControlNet-Union.safetensors` | `models/controlnet/` | 3.54 GB | InstantX Union ControlNet. Supports canny / soft-edge / depth / openpose; `SetUnionControlNetType` selects `openpose`. |
| `qwen_image_vae.safetensors` | `models/vae/` | — | VAE for Qwen-Image-Edit. |

### Why the 2509 Lightning LoRA and not the older V1.0

The Lightning LoRA at `models/loras/` has two variants — the root V1.0 is for the older Qwen-Image base model and produces noise on top of 2509. Always use `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`.

### Why the InstantX Union ControlNet on the `openpose` sub-type

The Union ControlNet supports four sub-types (canny / soft-edge / depth / openpose); we pick `openpose` via `SetUnionControlNetType`. No separate OpenPose-only ControlNet is installed — the Union model covers every ControlNet sub-type the pipeline needs with one 3.54 GB file.

### Required custom nodes

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — `UnetLoaderGGUF`, `CLIPLoaderGGUF`.
- `comfyui_vram_helpers` — in-tree at `F:/ComfyUI/custom_nodes/comfyui_vram_helpers/`. Provides `VRAMUnloadClip` (drops a `CLIP` after text encoding, returns its CONDITIONING outputs unchanged) and `VRAMUnloadModel` (drops a `MODEL` after sampling, returns its LATENT output unchanged). Both meta-tensor the inner `nn.Module` in place and surgically pop the upstream loader chain from the `PromptExecutor`'s output cache so the next enqueue loads fresh weights.

## VRAM usage

Measured at 512×512 canvas with `--gpu-only` on a 3090. Because of the two-phase eviction, the VL encoder and the Qwen-Edit UNET are **never** resident at the same time:

| Phase | Resident | Approx. VRAM in use |
|---|---|---|
| Phase 1 — text encode | VL encoder (~7 GB) + VAE (~0.3 GB) + activations | ~11 GB |
| Phase 2 — sampling | Qwen-Edit UNET (~15 GB) + ControlNet (~3.5 GB) + VAE + activations | ~21 GB |
| **Peak** | — | **~21 GB** on a 24 GB card |

Sample log output confirms the eviction magnitudes:

```
[VRAMUnloadClip] dropped: CLIP<QwenImageTEModel_> via GGUFModelPatcher [weights=7.06 GB] | VRAM freed: +7.03 GB
[VRAMUnloadModel] dropped: GGUFModelPatcher<QwenImage> [config=QwenImage, type=ModelType.FLUX, weights=14.93 GB] | VRAM freed: +15.74 GB
```

Cold run (first enqueue of the day) takes ~3 min due to GGUF dequantization. A hot re-roll with the same prompt but a new seed finishes in ~10 s — the cached text conditionings skip the CLIP chain entirely, and only the UNET + KSampler + VAEDecode re-execute.

At 1024² the stack overflows a 24 GB card — keep the `ImageScale` to 512.

## Configuration knobs

Edit at the top of `run_w1_pose_edit.py`:

| Knob | Default | Effect |
|---|---|---|
| `ANIMATION_NAME` | `"idle_breathing"` | Identifier used in output paths. |
| `SOURCE_IMAGE` | `"source_idle_breathing.png"` | Filename in `F:/ComfyUI/input/`. |
| `POSE_SKELETON` | `"pose_idle_breathing.png"` | Filename in `F:/ComfyUI/input/`. |
| `SEED` | `20260418` | KSampler seed. Re-roll if the checklist fails (see below). |
| `CONTROLNET_STRENGTH` | `1.3` | 1.0 works for near-symmetric poses close to the source. Bump to 1.6 for asymmetric / wide-stride poses where the skeleton needs to dominate. |
| `POS_PROMPT` | (see file) | Describes the pose explicitly and forbids facial features / clothing / ground shadow. |
| `NEG_PROMPT` | `""` | Must be empty string, not `ConditioningZeroOut`. |

## Authoring a new pose

Canonical workflow: **author JSON → render PNG → stage for ComfyUI**.

1. Write a COCO-18 pose JSON at `F:/ComfyUI/output/character_pipeline/base_animations/<animation>/poses/<animation>_pose.json`. The JSON format is the OpenPose standard emitted by any editor (e.g. [openposeai.com](https://openposeai.com)):

   ```json
   {
     "canvas_width": 512,
     "canvas_height": 512,
     "people": [{"pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ..., x17,y17,c17]}]
   }
   ```

   Keypoint order is COCO-18 (nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear). Confidence ≤ 0 marks a missing joint. The skeleton should use **human proportions** (~5 heads tall) — chibi head-size is inferred from the source image, not the skeleton.

2. Render the skeleton PNG:

   ```bash
   F:/ComfyUI/.venv/Scripts/python.exe tools/render_openpose.py F:/ComfyUI/output/character_pipeline/base_animations/<animation>/poses/<animation>_pose.json
   ```

   Output matches `controlnet_aux.util.draw_bodypose()` byte-for-byte — same drawing algorithm ControlNet was trained on.

3. Stage for ComfyUI:

   ```bash
   cp F:/ComfyUI/output/character_pipeline/base_animations/<animation>/poses/<animation>_pose.png F:/ComfyUI/input/pose_<animation>.png
   ```

4. Stage the source character image:

   ```bash
   cp <character>.png F:/ComfyUI/input/source_<animation>.png
   ```

5. Edit `ANIMATION_NAME`, `SOURCE_IMAGE`, `POSE_SKELETON` in `run_w1_pose_edit.py` and run it.

## Keyframe acceptance checklist

Reject the output and re-roll the seed if any of the following fail:

1. **Head is perfectly smooth and blank** — no eyes, pupils, mouth, nose, ears, blush, or facial curves.
2. **Body is naked** — no shirt, shorts, belt, collar, hair, hat.
3. **No ground shadow under the character** — or a mild one at most (BiRefNet strips these downstream, but fuzzy-edged shadows can get clipped into the alpha mask).
4. **Full body inside the frame** — feet not cropped at the bottom, head not cropped at the top.
5. **Body upright, not rotated.**
6. **Facing direction is unambiguous and matches the target pose.** Check by the feet, not the head.

Typical failure: 1-5 seed re-rolls land a clean keyframe. If re-rolls keep failing the same row, see the [troubleshooting](#troubleshooting) section.

## Troubleshooting

### Output looks nothing like the source character

- Verify the Lightning LoRA is the 2509 variant (filename contains `2509`), not the older V1.0.
- Verify `CFGNorm` is in the model chain at strength 1. Removing it collapses the distilled guidance.
- Verify `TextEncodeQwenImageEditPlus` for both positive and negative has `image1` wired to the scaled source image. Missing it means no character identity.

### Output is noise / static

- VL text encoder mmproj is missing from `models/text_encoders/`. Without `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf`, `image1` conditioning silently fails.
- `ConditioningZeroOut` on the negative branch instead of an empty-prompt `TextEncodeQwenImageEditPlus`. Never zero-out.
- `ModelSamplingSD3` in place of `ModelSamplingAuraFlow`. Qwen uses AuraFlow-style noise scheduling.

### Character is rendered at wrong scale

- Resolution in `ImageScale` is not 512×512. The whole pipeline assumes 512×512; downstream Workflow 2 takes 512² inputs.
- Source image is cropped too tight — padding around the character helps Qwen land it at the right scale.

### Pose is ignored, character stays in source pose

- ControlNet strength too low for asymmetric / wide-stride poses. Start at 1.3 (works for near-symmetric idle-style poses); bump to 1.6 if Qwen reproduces the source pose instead of the skeleton pose.
- Skeleton is outside the canvas (joints at negative coords or beyond 512). Redraw the JSON so every keypoint sits at `[0, canvas_width]` × `[0, canvas_height]`.
- Positive prompt contradicts the skeleton. The text must agree with the pose — if the skeleton says mid-stride, the prompt must say "mid-stride" or "walking" too.

### Facial features leak despite empty-faceless-head negative

- Rewrite the positive prompt to concretely describe the head as blank ("smooth perfectly blank bald scalp with NO face whatsoever — no eyes, no mouth…"). Qwen responds better to a positive than a long negative alone.
- Re-roll seeds. 3-10 seeds per pose is typical.

### "paging file too small" / OOM on model load

- ComfyUI not launched with `--gpu-only`. Fix the startup args. The current process is fine — it was started with `--gpu-only --reserve-vram 1`.

### "Tensor.item() cannot be called on meta tensors" on a re-run

- `VRAMUnloadClip` / `VRAMUnloadModel` meta-tensored the previous run's weights but the executor's output cache still holds the stale `ModelPatcher`. This indicates the cache invalidation in `custom_nodes/comfyui_vram_helpers/__init__.py` did not find the `PromptExecutor`'s `caches` on the Python call stack — typically because ComfyUI's `execution.py` internals changed in a version bump. Check that `_find_executor_caches()` still returns non-None (the node logs `cache invalidation skipped (executor caches not on stack)` when it fails) and update the stack-walk heuristic if needed.

## Wall-time expectations

On a 3090 + `--gpu-only`:

- First run (cold model load): ~3 min.
- Subsequent runs (hot models, only the KSampler seed changing): ~10 s.

## Current status

**Verified 2026-04-18 on `idle_breathing`.** Three consecutive runs with different seeds (30123 / 30124 / 30125, ControlNet strength 1.6) all landed clean keyframes on the first try. Peak VRAM stayed at ~21 GB on a 24 GB 3090; `VRAMUnloadClip` and `VRAMUnloadModel` fired on every run, and the seed-only re-enqueues (runs 2 and 3) correctly reused the cached conditionings without reloading the VL encoder. Output passed every row of the keyframe acceptance checklist except a mild ground shadow that BiRefNet will strip downstream.
