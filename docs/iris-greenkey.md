# Iris Green-Key Extraction (eye-cosmetic post-processing)

Parent: [comfyui-animation-generation](./pipeline-overview.md). **A small post-processing step that runs after [Workflow 4](./workflow-4-cosmetic-inpaint.md) to produce the iris sprite sheet for an eye cosmetic.** Does not touch Workflow 5 — the full eye sprite sheet still comes from a normal W5 SAM3 run.

## What it does

Eye cosmetics render as two paired layers: a full eye sheet (sclera, lashes, specular highlights, baked default iris) produced by Workflow 5, plus a near-white greyscale iris sheet on top whose pixels are tinted at runtime to the player's chosen eye colour. The iris sheet must contain **only** the iris disc — no sclera, no lashes, no highlights — because `Sprite.tint` multiplies every opaque pixel, so a single stray sclera pixel would appear as a bright tinted speckle.

SAM3 with prompts like `"iris"` / `"pupil"` proved unreliable at isolating just the iris sub-region across frames. This script avoids segmentation entirely: the cosmetic reference is authored with a **bright green iris**, and a global colour-key on the pass 1 mp4 (`G > R + margin` AND `G > B + margin`) keeps those iris pixels and rejects everything else. The kept luminance is lifted toward white so runtime tint produces saturated iris colours.

## Required: the reference iris must be bright green

Green is the convention because it's absent from sclera white, lash black, and specular highlights. When generating an eye cosmetic reference (via Qwen-Image-Edit or any other source), prompt explicitly for `"large anime eyes with bright green irises"` and verify the 512×512 PNG has visibly-green iris pixels before running the pipeline. If the iris reads olive, amber, or yellow-green, tweak the prompt and regenerate — the green-key test will miss a too-warm or desaturated iris.

Future variants (blue / red iris) could swap the green-key for a blue-key / red-key by changing the colour-dominance test; every currently-supported eye cosmetic is authored green for consistency.

## Driver

`F:/ComfyUI/user/run_iris_greenkey.py` — pure Python, no ComfyUI graph. Edit the per-run knobs at the top, then run with `F:/ComfyUI/.venv/Scripts/python.exe F:/ComfyUI/user/run_iris_greenkey.py`.

## Knobs

| Knob | Default | Effect |
|---|---|---|
| `COSMETIC_NAME` | `"green_eyes"` | Staging slug. Must match the W4 run. |
| `ANIMATION_NAME` | `"idle_breathing"` | Staging slug. Must match the W4 run (`idle_breathing`, `walking_right`, `in-air`). |
| `PASS1_MP4` | `None` | Explicit pass 1 mp4 path. When `None`, auto-picks the latest under `cosmetics/<cos>/<anim>/pass1/`. |
| `NUM_FRAMES` | `33` | Must match W4. 9 for `in-air`, 33 for idle / walking. |
| `CELL_SIZE` | `128` | Must match the base sheet. |
| `GREEN_MARGIN` | `20` | Min green dominance over red and blue. Lower catches darker iris edges at the risk of sclera anti-alias leakage; raise if non-iris pixels appear. |
| `IRIS_LIFT_MIN` | `0.55` | Lifts iris luminance into `[LIFT_MIN, 1.0]` so runtime `Sprite.tint` yields saturated colours. Too high = loses iris shading; too low = tinted iris looks muddy. |

## Flow

1. Decode `NUM_FRAMES` frames from the pass 1 mp4 at source resolution (512×512 RGB).
2. Per-pixel colour key: `G > R + GREEN_MARGIN` AND `G > B + GREEN_MARGIN` AND `G > 40` → iris mask.
3. Iris luminance from the green channel, lifted to `[IRIS_LIFT_MIN, 1.0]` and written into R = G = B. Non-iris pixels set to fully transparent.
4. Downscale each frame to `CELL_SIZE` with `cv2.INTER_AREA` (clean alpha edges at chibi resolution).
5. Horizontal stitch → single RGBA strip at `NUM_FRAMES × CELL_SIZE`.

Wall time: seconds, not minutes — pure CPU. No SAM3, no VRAM, no models.

## Install

Copy the output into the repo alongside the W5 eye sheet:

```
F:/ComfyUI/output/character_pipeline/cosmetics/<cosmetic>/<anim>/<cosmetic>_<anim>_iris_spritesheet.png
  → <your-asset-dir>/layers/<iris_slug>/<repo_anim>/<repo_anim>.png
```

…where `<iris_slug>` is the paired cosmetic that the eye layer declares via `pairedCosmetic` (see your runtime's paired-cosmetic mechanism for the full paired-cosmetic schema and runtime behaviour). Register the paired cosmetic in `<your-asset-dir>/animations.json` with `selectable: false` on the iris so the player's eye-row dropdown only shows the eye cosmetic.

## End-to-end recipe for a new eye cosmetic

1. **Author reference** — 512×512 PNG of the base character with bright-green anime eyes.
2. **Workflow 4** for each animation — `CG_PASS1_MASK_DILATE=50` for idle / walking, `70` for in-air (eyes clip from below at 50 during the peak pose). `COSMETIC_DESCRIPTION="eyes, eyebrows"` works for W4's SAM3 (it only needs to locate the region, not isolate it tightly).
3. **Workflow 5** for each animation — standard driver, no eye-specific changes. Known-good settings for eyes:
   - `COSMETIC_DESCRIPTION = "eyes"`
   - `SAM_CONFIDENCE = 0.15`
   - `MASK_OFFSET = 1`
   - `MASK_SMOOTH = 1`
   - `GREYSCALE = True`
4. **Iris green-key** (this script) for each animation — produces the paired iris sheet from the same W4 pass 1 mp4 W5 consumed.
5. **Install** — copy both sheets into `<your-asset-dir>/layers/` and register the pair in `<your-asset-dir>/animations.json`.
6. **Verify** in the animation tester.

## Current status

**Verified 2026-04-22 on `green_eyes` × `idle_breathing` / `walking_right` / `in-air`** — shipped as `eye_style_3` + `iris_style_3`. W4 pass 1 at `CG_PASS1_MASK_DILATE=50` for idle / walking and `70` for in-air. W5 at `COSMETIC_DESCRIPTION="eyes"`, `SAM_CONFIDENCE=0.15`, `MASK_OFFSET=1`, `GREYSCALE=True` — produced tight eye sheets (~11k opaque px for 33-frame runs, ~2.4k for in-air) with no forehead / ear / chin leakage. Iris green-key at `GREEN_MARGIN=20`, `IRIS_LIFT_MIN=0.55` — produced iris sheets (~1.5k opaque px for 33-frame runs, ~290 for in-air). Tinted composites over the greyscale base render cleanly on all three animations; sclera / lashes / highlights stay un-tinted while the iris takes the player-picked colour.
