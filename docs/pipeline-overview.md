# ComfyUI Animation Generation

Status: **Work in progress.**

The goal is to create and use pipelines to generate the assets for the character animation system.

Pipeline 1: generates base animations for the runtime (work in progress).
Pipeline 2: generates cosmetic layers for each animation we can render on top of the base animation as a layer to allow players to customize characters (**in progress** — first workflow being built) — the cosmetics layers must align perfectly with the base animations so we can render them on top.

Read the plan for the character animation system at the runtime that consumes the output (not implemented yet).

## Pipeline 1 — base-animation generation

**Input:** a single base character image PNG. Every base animation in the runtime is generated from this one image — the character's identity, proportions, colors, and line art style are anchored by it.

Two sequential workflows convert the base character image into a base animation asset (mp4 + greyscale RGBA sprite sheet), plus a final install step that moves the sprite sheet into the runtime repo:

1. **[Workflow 2 — Video Generation](./workflow-2-video-gen.md).** Takes the base character image and generates a short mp4 of the character performing the animation (idle breathing, walk, run, jump …). Uses WAN 2.2 i2v 14B (Q5_K_M GGUF, distilled lightx2v 4-step) with two-stage MoE eviction (`VRAMUnloadModel`) so the high-noise and low-noise experts never coexist in VRAM on a 24 GB card.
2. **[Workflow 3 — Spritesheet](./workflow-3-spritesheet.md).** Takes the mp4 from Workflow 2 and produces a horizontal greyscale RGBA sprite sheet. BiRefNet semantic segmentation strips the background; a YUV-based conversion produces greyscale frames that the runtime tints per-player. Stitch chain sized dynamically to `NUM_FRAMES` — no ping-pong loop in the asset.
3. **Install into the repo.** The W3 output path under `F:/ComfyUI/output/…` is a staging location — downstream consumers both read base animations from your asset directory. After each successful W3 run, copy the sprite sheet to `<your-asset-dir>/base_animations/<animation_slug>/<animation_slug>.png` and register it in `<your-asset-dir>/animations.json` under `baseAnimations`. See [Workflow 3 — Install into the runtime repo](./workflow-3-spritesheet.md#install-into-the-runtime-repo) for the full step list and loop-mode rules.

End-to-end verified 2026-04-18 on `idle_breathing`, producing `F:/ComfyUI/output/character_pipeline/base_animations/idle_breathing/idle_breathing_spritesheet_00001_.png` from the base character image.

### Output layout

Every base animation has a self-contained folder at `F:/ComfyUI/output/character_pipeline/base_animations/<animation_name>/`:

```
<animation_name>/
  <animation_name>.mp4                 — Workflow 2 output
  <animation_name>_spritesheet.png     — Workflow 3 output
```

## Pipeline 2 — cosmetic layer generation

**Input:** a base-animation mp4 (output of Pipeline 1) + a still image of the character wearing the target cosmetic (pink hair, glasses, hat, jacket, …) + *(optional but recommended)* a still image of the same base character **without** the cosmetic. The cosmetic reference image is a PNG of the **same base character** but with the cosmetic added; the no-cosmetic reference (if provided) is used to auto-derive the cosmetic region mask.

Two sequential workflows convert the base-animation mp4 into a cosmetic layer asset:

1. **[Workflow 4 — Cosmetic Inpaint](./workflow-4-cosmetic-inpaint.md).** Takes the base-animation mp4, the character-with-cosmetic reference image, and optionally the no-cosmetic base character image. Auto-derives the cosmetic region mask via pixel-diff between the two references, propagates the mask across all frames, and runs **Wan 2.1 VACE 14B (Q5_K_M GGUF) + CausVid distill LoRA** inpainting to regenerate **only** the cosmetic region. `ImageCompositeMasked` then pastes the inpainted region back onto the original source frames, so every pixel outside the mask is **bit-exact preserved** from the base animation. Uses per-stage VRAM eviction (`VRAMUnloadClip` for UMT5 after text encode, `VRAMUnloadModel` for the VACE DiT after sampling).
2. **[Workflow 5 — Cosmetic Spritesheet](./workflow-5-cosmetic-spritesheet.md).** Takes the cosmetic mp4 from Workflow 4 and runs per-frame SAM3 text-prompted segmentation to isolate the cosmetic region. Produces a horizontal RGBA sprite sheet (optionally greyscale) containing **only** the cosmetic pixels — transparent everywhere the base animation is unchanged. Layout is pixel-identical to the base spritesheet from Workflow 3, so the runtime stacks cosmetic layers over the base without re-alignment. **Eye cosmetics** run W5 normally to produce the eye sheet, then run a small post-processing script — [Iris Green-Key Extraction](./iris-greenkey.md) — to produce the paired iris sheet from the same W4 pass 1 mp4. The iris step is separate because SAM3 proved unreliable at isolating just the iris disc within the eye; a global green-colour-key on the pass 1 mp4 works if the reference iris is authored bright green.
3. **Install into the repo.** The W5 output path under `F:/ComfyUI/output/…` is a staging location — the runtime reads cosmetics from your asset directory. After each successful W5 run, copy the sprite sheet to `<your-asset-dir>/layers/<cosmetic_slug>/<animation_slug>/<animation_slug>.png` and register it in `<your-asset-dir>/animations.json`. See [Workflow 5 — Install into the runtime repo](./workflow-5-cosmetic-spritesheet.md#install-into-the-runtime-repo) for the full step list and slug-mapping rules.

**Why inpaint over character replacement?** A first pass with Wan 2.2 Animate (character replacement) was tried on 2026-04-19 and rejected: it could not preserve the base animation's motion exactly (DWPose can't represent sub-pixel torso breathing; the model hallucinates legs). Pipeline 2 requires pixel-exact alignment between the cosmetic video and the base video so the Workflow 5 diff produces clean cosmetic-only layers. Inpainting just the cosmetic region (VACE) guarantees that alignment; character replacement (Animate) doesn't.

The same character-with-cosmetic reference image is fed into Workflow 4 against every base animation mp4 to produce a consistent cosmetic across `idle_breathing`, `walking`, `jumping`, etc. The mask is re-derived per-run from the two reference images.

## Helper workflows

Helper workflows are not part of the base-animation pipeline. They exist to prepare or transform inputs that feed into it (or to produce the base character image itself).

- **[Workflow 1 — Pose Edit](./workflow-1-pose-edit.md).** Takes a character PNG and a pre-rendered OpenPose skeleton PNG, produces one posed keyframe using Qwen-Image-Edit 2509 (Q5_K_M GGUF) + InstantX Qwen-Image ControlNet-Union on the `openpose` sub-type. Useful for generating the base character image itself, or for producing a specifically-posed variant of a character when the default base image is not suitable as a starting frame for an animation. Two-phase VRAM eviction: `VRAMUnloadClip` drops the VL text encoder after text encoding, `VRAMUnloadModel` drops the UNET after sampling, so the two big residents never coexist on a 24 GB card. The skeleton constrains joint positions; the source image carries character identity, proportions, and line art style. Outputs and pose artifacts (JSON, rendered skeleton PNG, posed keyframe PNG) live under the per-animation folder's `poses/` subdirectory — see the workflow doc for details.

## Technical details

- **Target hardware:** NVIDIA RTX 3090, 24 GB VRAM, 32 GB system RAM. All workflows are tuned for this card — peaks aimed at ~22.5 GB VRAM to leave headroom for the OS and ComfyUI overhead. `--reserve-vram 1` is set at launch so ComfyUI keeps ~1 GB of VRAM unallocated.
- use the comfyui mcp to create and update the workflows we need.
- we should use only models that fit my GPU's VRAM which is 24GB, but we should aim for a total usage of 22.5GB so there is some room left for system stuff
- choose variants of models that fit the VRAM requirements
- Every workflow should have all of its models fit VRAM together. if a workflow has 3 models - we need all of them to fit VRAM properly
- Always search the internet for the best models to do the job we need, gather the most relevant and up to date information from forums, hugging face, reddit, civitai and other forums. things are changing fast in this world. stable diffusion is considered old already and we have other model such as flux and wan2.2
- we should make sure the we have enough space on disk as well to make download the required models (we are currently installing on disk F: on this machine) - if there isnt enough space try to delete models that are not in use in any workflow. if still we don't have room for all the models we need ask me to clear some space.
- we might want to consider using LORAs that help create what we need. but make sure the LORA fits the model we are using. not every LORA fits every model.
- most probably each one of the workflows we need here was already created by someone on the web and we might be able to download and reuse their workflow. before implementing a workflow check if there are any good candidate workflows we can try to use. make sure they get good reviews before downloading any workflow.
- make sure models or LORAs have good reviews before downloading.
- only touch files inside the comfyui directory `F:\ComfyUI`
- if you need to troubleshoot the output of the workflow, a good starting point is to check compatibility of the model with the LORAs used and recommended settings used by others. you should verify people on the internet are using the same combination of models + LORAs and report it to be working. you should verify the recommended settings as well.
- when you download a model, its file size usually says how much space it will take on the GPU +-, if you are about to download a model larger than 21GB it will likely not fit in my VRAM.
