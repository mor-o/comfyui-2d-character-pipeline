# Workflow 3 — Spritesheet

Parent: [comfyui-animation-generation](./pipeline-overview.md). Related: [Workflow 2 — Video Generation](./workflow-2-video-gen.md) (produces the mp4 input).

Takes the mp4 from Workflow 2 and produces a horizontal greyscale RGBA sprite sheet suitable for the character animation system. BiRefNet semantic segmentation strips the background per frame; a YUV-based conversion produces the greyscale channel that the runtime tints per-player.

The sheet contains exactly `NUM_FRAMES` cells, one per source frame. The runtime handles any looping by replaying the sheet in order.

## Goal

Turn a video clip into a character-animation sprite sheet:

- **Background removed**: BiRefNet identifies the character silhouette and writes transparent alpha everywhere else. Keeps interior highlights (forehead, shoulders) opaque because it segments semantically rather than by RGB threshold.
- **Greyscale**: the runtime will tint the sheet per-player (skin tone, team color, damage flash) — one base sheet covers every variant. Greyscale is therefore mandatory for production sheets.
- **Cell-aligned horizontal strip**: 128×128 cells concatenated left-to-right, final strip is `NUM_FRAMES × 128 px` wide by 128 px tall.

## Input / output

**Input:** the latest mp4 produced by Workflow 2 at
`F:/ComfyUI/output/character_pipeline/base_animations/<animation>/<animation>_NNNNN_.mp4`.
The driver auto-discovers the newest mp4 in that folder and stages it to
`F:/ComfyUI/input/video_<animation>.mp4`.

**Output:**
`F:/ComfyUI/output/character_pipeline/base_animations/<animation>/<animation>_spritesheet_NNNNN_.png` — RGBA, `NUM_FRAMES × 128 px` wide × 128 px tall. Example for 33 frames: 4224 × 128.

**Install into the repo after generation.** The ComfyUI output path above is the staging location — the runtime reads base animations from your asset directory. After a successful run, copy the sprite sheet into the repo and register it in `<your-asset-dir>/animations.json`. See [Install into the runtime repo](#install-into-the-runtime-repo) below.

## Implementation

**Workflow file:** `F:/ComfyUI/user/default/workflows/w3-spritesheet/workflow_api.json` — holds the fixed nodes up through the per-frame RGBA batch (node 25). The driver appends the per-frame slice + stitch chain dynamically based on `NUM_FRAMES`.

**Driver:** `F:/ComfyUI/user/run_w3_spritesheet.py` — edit `ANIMATION_NAME`, `NUM_FRAMES`, `CELL_SIZE`, `RMBG_MODEL` at the top, then run with `F:/ComfyUI/.venv/Scripts/python.exe F:/ComfyUI/user/run_w3_spritesheet.py`.

### Node graph

**Fixed nodes (loaded from JSON):**

1. `LoadVideo` (`video_<animation>.mp4`) → `GetVideoComponents` → IMAGE batch.
2. `ImageFromBatch` (`batch_index=0, length=NUM_FRAMES`) — slice the first N frames.
3. Two parallel branches:
   - **Mask branch:** `BiRefNetRMBG` (`model=BiRefNet_toonout, refine_foreground=true`) → mask (`1 = character`). Convert to image for resizing: `MaskToImage` → `ImageScale` to 128 → `ImageToMask` → `InvertMask`. The invert is required because `JoinImageWithAlpha.alpha` treats `1 = transparent` — see the detailed explanation below.
   - **Colour branch:** `ImageScale` the original RGB batch to 128 → `ImageRGBToYUV` (Y channel goes into R slot of the YUV image) → `ImageToMask(channel=red)` → `MaskToImage` → greyscale RGB image (R=G=B=Y).
4. `JoinImageWithAlpha(greyscale_image, inverted_mask)` → 128×128 RGBA batch.

**Dynamic nodes (appended in the driver based on `NUM_FRAMES`):**

- `100 .. 100+N-1`: `ImageFromBatch(batch_index=i, length=1)` — slice each individual frame out of the RGBA batch.
- `200 .. 200+N-2`: `ImageStitch(image1=prev, image2=frame_i, direction="right", match_image_size=true, spacing_width=0)` — concatenate left-to-right.
- `300`: `SaveImage` with `filename_prefix = character_pipeline/base_animations/<animation>/<animation>_spritesheet`.

Frame 0 is `ImageFromBatch` id 100 without an ImageStitch predecessor (it's the starting point); frame 1 is stitched to it via id 200; frame 2 to the result via 201; and so on. The last ImageStitch (id `200 + N - 2`) feeds the `SaveImage`.

### Why the mask is inverted before `JoinImageWithAlpha`

ComfyUI has an inconsistent mask convention. `BiRefNetRMBG` returns a mask where **1 = character, 0 = background** (most semantic-segmentation models). `JoinImageWithAlpha` treats its `alpha` input as "this region is **transparent**" where 1 = transparent. Without inverting, the character becomes transparent and the background becomes opaque. Always `InvertMask` between the two.

### Why greyscale, and why via YUV

The runtime tints the sheet per-player (skin tone, team color, damage flash). A coloured sheet locks in one specific tint. Greyscale keeps the luminance (shading, line art contrast) intact while throwing out chroma — and tinting simply multiplies a colour over R=G=B.

`ImageRGBToYUV` is a native ComfyUI node. The Y channel is stored in the R slot of the YUV image, so extracting via `ImageToMask(channel="red")` then promoting back to a 3-channel image gives a perceptually correct greyscale (Rec. 601 weights — 0.299 R + 0.587 G + 0.114 B) without any custom math. No ML / no model. Runs in milliseconds.

## Models used

| File | Location | Size | Purpose |
|---|---|---|---|
| `BiRefNet_toonout` | auto-downloaded by ComfyUI-RMBG on first use into `models/RMBG/` | ~300 MB | Semantic segmentation model tuned for anime / cartoon subjects. Keeps interior character highlights opaque by identifying the silhouette semantically. |

No other models. The YUV conversion and all scaling / stitching use native ComfyUI nodes.

### Required custom nodes

- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) — `BiRefNetRMBG`.

### Alternate BiRefNet models

Swap `RMBG_MODEL` in the driver for different subject types:

| Character style | Model |
|---|---|
| Default chibi / anime | `BiRefNet_toonout` |
| Semi-realistic | `BiRefNet-general` |
| Semi-realistic, high-res reference | `BiRefNet-HR` |

All are auto-downloaded on first reference.

## VRAM usage

Measured on a 24 GB RTX 3090 with a 33-frame 512² source video:

| Component | Approx. VRAM |
|---|---|
| BiRefNet_toonout | ~1 GB |
| Frame batch + intermediate images | <0.5 GB |
| **Peak** | **~1.5 GB** |

Trivial compared to Workflows 1 and 2. Runs fine even when Workflow 2's WAN 2.2 experts are still resident — though the eviction chain in Workflow 2 clears them before this workflow runs anyway.

## Configuration knobs

Edit at the top of `run_w3_spritesheet.py`:

| Knob | Default | Effect |
|---|---|---|
| `ANIMATION_NAME` | `"idle_breathing"` | Used for input mp4 discovery and output filename. |
| `NUM_FRAMES` | `33` | Must match Workflow 2's `NUM_FRAMES`. Must satisfy the `4n+1` rule from Workflow 2 (because the mp4 has that many frames). |
| `CELL_SIZE` | `128` | Output cell width / height in pixels. |
| `RMBG_MODEL` | `"BiRefNet_toonout"` | Segmentation model — swap for semi-realistic subjects (see alternates table above). |

## Wall-time expectations

On a 3090 with BiRefNet hot:

| Frames | Wall time |
|---|---|
| 13 | ~12 s |
| 25 | ~18 s |
| 33 | ~25 s |
| 49 | ~40 s |

Most time is BiRefNet inference; the stitch chain is near-instant.

## Troubleshooting

### Sprite sheet has holes in the character body

- `BiRefNet_toonout` over-segmented a highlight as background. Try `BiRefNet-general` or `BiRefNet-HR` for less aggressive silhouette carving.
- `mask_blur` too high. Leave it at 2 (default).

### Character is transparent and background is opaque (inverted result)

- `InvertMask` is missing or disconnected from the `JoinImageWithAlpha.alpha` input. Verify the graph: BiRefNet mask → `MaskToImage` → scale → `ImageToMask` → **`InvertMask`** → `JoinImageWithAlpha.alpha`.

### Sprite sheet has more / fewer cells than expected

- `NUM_FRAMES` in the driver doesn't match the actual frame count of the mp4. Workflow 2 produces `NUM_FRAMES` frames (from its `WanImageToVideo.length` input); keep them in sync.
- The `ImageFromBatch` at node id 3 (`length` field) is still hardcoded to the old count in the JSON. The driver rewrites this at build time — but if editing the workflow by hand, remember to update it.

### Output is coloured, not greyscale

- The YUV conversion node ids (26 → 27 → 28) must feed the `JoinImageWithAlpha.image` input. Verify the graph — if `JoinImageWithAlpha` takes the raw scaled RGB (node 20) instead, the sheet is coloured.

### `ImageStitch` fails with a shape-mismatch error

- The mask branch and RGB branch resized to different dimensions. Keep both `ImageScale` nodes at the same `width` / `height` (the driver enforces this via `CELL_SIZE`).

### `BiRefNetRMBG` unknown node

- ComfyUI-RMBG isn't installed. Restart ComfyUI with ComfyUI-Manager enabled and install [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) via the registry.

## Install into the runtime repo

After the driver prints `saved -> …_spritesheet_NNNNN_.png`, install the sprite sheet into the repo so the runtime and [Animation Tester](../animation-tester.md) can pick it up. The repo's `animations/` directory (consumed by both the client and the tester via `animations.json`) is the runtime source of truth — ComfyUI output is a staging area. **Do this every time you produce a new base animation** — leaving the sheet in the ComfyUI output dir means neither the runtime nor the tester can see it.

1. **Copy the sprite sheet** to `<your-asset-dir>/base_animations/<animation_slug>/<animation_slug>.png` in the repo. The file must be named after the animation slug (e.g. `jump.png`), not the ComfyUI output filename.

   Example:
   ```
   F:/ComfyUI/output/character_pipeline/base_animations/jump/jump_spritesheet_00001_.png
     → <your-asset-dir>/base_animations/jump/jump.png
   ```

   Note the slug mapping: ComfyUI directory names can be iteration-local (`idle_breathing_5`, `walking_right`) — the repo uses stable runtime slugs (`idle`, `walking`, `jump`). Re-map the output into the slug rather than renaming the repo slug to match a ComfyUI directory.

2. **Register the base animation** in `<your-asset-dir>/animations.json` under `baseAnimations`:

   ```json
   "baseAnimations": {
     "jump": {
       "layer": "base",
       "spriteSheet": "base_animations/jump/jump.png",
       "frames": 17,
       "fps": 12,
       "loop": "once"
     }
   }
   ```

   `frames` must equal the W2 / W3 `NUM_FRAMES`. `loop` is one of `loop` (idle, walking cycles), `pingpong`, or `once` (one-shot actions like `jump` that end mid-pose and don't return to the starting frame). `layer` is almost always `base`. See your runtime's catalog schema for the full schema constraints.

3. **Verify in the animation tester.** Run `npm run dev:animation-tester` and confirm the new base animation plays cleanly (`once` animations will stop at the final frame). Any existing cosmetics registered for this animation slug will automatically layer on top; cosmetics missing an entry for the new animation will log a benign `console.warn` when the tester switches to it (the catalog loader does not require cosmetics to cover every base animation).

## Current status

**Verified 2026-04-18 on `idle_breathing`.** The 33-frame mp4 from Workflow 2 produced a 4224 × 128 RGBA sprite sheet with 33 greyscale chibi cells and clean alpha on the character silhouette. Ground-shadow artifacts that leaked through Workflow 1 were correctly stripped by BiRefNet. Output: `F:/ComfyUI/output/character_pipeline/base_animations/idle_breathing/idle_breathing_spritesheet_00001_.png`.
