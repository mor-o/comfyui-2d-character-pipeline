# Workflow 5 — Cosmetic Spritesheet

Parent: [comfyui-animation-generation](./pipeline-overview.md). Part of **Pipeline 2 — cosmetic layer generation**. Consumes the cosmetic mp4 produced by [Workflow 4](./workflow-4-cosmetic-inpaint.md) and produces the final deliverable for the runtime.

Takes the cosmetic mp4 (base animation with the cosmetic painted on top) and produces a horizontal **cosmetic-only RGBA sprite sheet** — transparent everywhere the base animation is unchanged, opaque (optionally greyscale) on the cosmetic pixels. SAM3 does per-frame semantic segmentation with a text prompt describing the cosmetic region.

The sheet layout is pixel-identical to the base-animation spritesheet produced by [Workflow 3](./workflow-3-spritesheet.md), so the runtime can stack it as a layer over the base sheet without per-cell re-alignment.

## Goal

- **Isolate the cosmetic:** SAM3 identifies the cosmetic region per frame using a text prompt (e.g. `"hair"`). Every pixel outside that region becomes transparent.
- **Aligned to the base sheet:** same `NUM_FRAMES`, same `CELL_SIZE` (128), same horizontal-strip layout as [Workflow 3](./workflow-3-spritesheet.md). The cosmetic mp4 was inpainted over the base mp4 with bit-exact preservation outside the cosmetic region, so mask coordinates and cell positions match automatically.
- **Greyscale by default:** matches the base-animation sheet's tint-able format. Togglable — some cosmetics (hair colors the player picked deliberately) are shipped in color.

## Why per-frame segmentation, not a single pass over the video

Cosmetic alignment is already guaranteed pixel-wise by W4 — the base frames under the cosmetic are identical to the base-animation frames used for W3. That removes the usual reason to care about temporal mask smoothness. Per-frame SAM3 is simpler, doesn't require a video-capable SAM3 node (which ComfyUI-RMBG does not expose), and any residual per-frame edge jitter is hidden at runtime because the base layer underneath supplies the exact same pixels the mask would otherwise mis-include.

## Input / output

**Input:** cosmetic mp4 from Workflow 4 (pass1 or pass2; pass2 preferred when available).
Default resolution: `F:/ComfyUI/output/character_pipeline/cosmetics/<cosmetic>/<animation>/pass2/<cosmetic>_<animation>_pass2_NNNNN_.mp4` — or any explicit path via the `SOURCE_VIDEO` knob. The driver stages it to `F:/ComfyUI/input/cosmetic_<cosmetic>_<animation>.mp4`.

**Output:** `F:/ComfyUI/output/character_pipeline/cosmetics/<cosmetic>/<animation>/<cosmetic>_<animation>_spritesheet_NNNNN_.png` — RGBA, `NUM_FRAMES × 128 px` wide × 128 px tall. Same dimensions as the paired base-animation spritesheet.

**Install into the repo after generation.** The ComfyUI output path above is the staging location — the runtime reads cosmetics from your asset directory. After a successful run, copy the sprite sheet into the repo and register it in `<your-asset-dir>/animations.json`. See [Install into the runtime repo](#install-into-the-runtime-repo) below.

**Eye cosmetics need a second sheet.** W5 produces the full eye sheet (sclera, lashes, highlights, baked default iris). The paired iris-only sheet — required for runtime eye-colour tinting — is produced by a separate post-processing script, not by W5. After running W5, run [Iris Green-Key Extraction](./iris-greenkey.md) on the same W4 pass 1 mp4. Both sheets get installed side-by-side and the pair is registered in `animations.json` via `pairedCosmetic` + `tintTarget: "paired"`. See your runtime's paired-cosmetic mechanism.

## Implementation

**Workflow file:** `F:/ComfyUI/user/default/workflows/w5-cosmetic-spritesheet/workflow_api.json` — fixed nodes up through the per-frame RGBA batch. The driver appends the per-frame slice + stitch chain dynamically based on `NUM_FRAMES`.

**Driver:** `F:/ComfyUI/user/run_w5_cosmetic_spritesheet.py` — edit `COSMETIC_NAME`, `ANIMATION_NAME`, `COSMETIC_DESCRIPTION`, `NUM_FRAMES`, `CELL_SIZE`, `GREYSCALE`, `SOURCE_VIDEO` at the top, then run with `F:/ComfyUI/.venv/Scripts/python.exe F:/ComfyUI/user/run_w5_cosmetic_spritesheet.py`.

### Node graph

**Fixed nodes (loaded from JSON):**

1. `LoadVideo` (cosmetic mp4) → `GetVideoComponents` → IMAGE batch at source resolution (512×512).
2. `ImageFromBatch` (`batch_index=0, length=NUM_FRAMES`) — slice first N frames.
3. **Mask branch:** `SAM3Segment` (`prompt=COSMETIC_DESCRIPTION`, `confidence_threshold=SAM_CONFIDENCE`, `unload_model=true`) → per-frame mask batch (`1 = cosmetic`). `AILab_MaskEnhancer` (`mask_offset=MASK_OFFSET`, `smooth=MASK_SMOOTH`, `fill_holes=true`) for edge cleanup. `MaskToImage` → `ImageScale` to 128 → `ImageToMask(channel=red)` → `InvertMask`. The invert is required: ComfyUI's `JoinImageWithAlpha` treats its `alpha` input as "1 = transparent", so the cosmetic region (which we want opaque) must be inverted to 0 before being joined.
4. **Colour branch:** `ImageScale` the original RGB batch to 128. When `GREYSCALE=True`, convert via `ImageRGBToYUV` → `ImageToMask(channel=red)` → `MaskToImage` to get a Rec. 601 greyscale image; when `GREYSCALE=False`, feed the scaled RGB directly.
5. `JoinImageWithAlpha(colour_image, cosmetic_mask)` → 128×128 RGBA batch where cosmetic is opaque (optionally greyscale), everything else fully transparent.

**Dynamic nodes (appended in the driver based on `NUM_FRAMES`) — identical to Workflow 3:**

- `100 .. 100+N-1`: `ImageFromBatch` per frame.
- `200 .. 200+N-2`: `ImageStitch(direction="right", match_image_size=true, spacing_width=0)` — concatenate left-to-right.
- `300`: `SaveImage` with `filename_prefix = character_pipeline/cosmetics/<cosmetic>/<animation>/<cosmetic>_<animation>_spritesheet`.

### Why the mask is inverted before `JoinImageWithAlpha`

ComfyUI's `JoinImageWithAlpha` interprets its `alpha` input as "1 = transparent" (the opposite of the PNG alpha convention). Both BiRefNet and SAM3 return 1 on the segmented region, so both pipelines need an `InvertMask` between the scaled mask and the `alpha` input. See [Workflow 3 — Why the mask is inverted](./workflow-3-spritesheet.md#why-the-mask-is-inverted-before-joinimagewithalpha) for the same note on the base-animation side.

### Alignment with the base spritesheet

Alignment is guaranteed by three invariants inherited from Workflow 4:
- The cosmetic mp4 is the base mp4 with only the cosmetic region inpainted — bit-exact preservation elsewhere (see W4's `ImageCompositeMasked` step).
- Workflow 5 uses the same `CELL_SIZE` downscale (`lanczos`) and the same `NUM_FRAMES` slice as Workflow 3.
- Horizontal stitching is identical to W3 (same `ImageStitch` parameters, same ordering).

Under those three, each cell in the cosmetic sheet lines up 1:1 with the matching cell in the base sheet. No landmark detection or re-alignment needed at runtime.

## Models used

| File | Location | Size | Purpose |
|---|---|---|---|
| `sam3.pt` | `models/sam3/` | ~3.45 GB | Already present from Workflow 4. Image-level segmentation driven by a text prompt. Evicted via `unload_model=true` after the mask batch is produced. |

No other models. YUV conversion, scaling, stitching use native ComfyUI nodes.

### Required custom nodes

- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) — `SAM3Segment`, `AILab_MaskEnhancer` (already required by W4).

## VRAM usage

Peak is SAM3 inference on the frame batch: ~5–6 GB during segmentation, then drops to baseline once `unload_model=true` evicts the model. The stitch / scale chain is negligible. Well under the 22.5 GB budget; runs fine from a cold VRAM state.

## Configuration knobs

Edit at the top of `run_w5_cosmetic_spritesheet.py`:

| Knob | Default | Effect |
|---|---|---|
| `COSMETIC_NAME` | `"yellow_hair"` | Directory name and filename prefix. |
| `ANIMATION_NAME` | `"walking_right"` | Directory name and filename prefix. |
| `COSMETIC_DESCRIPTION` | `"hair"` | SAM3 text prompt. Describe the **body region** (e.g. `"hat, head"` for a hat, `"jacket"` for a coat). Colors hurt — SAM3 segments concepts, not colors. |
| `NUM_FRAMES` | `33` | Must match the W2/W3 run that produced the paired base sheet. |
| `CELL_SIZE` | `128` | Must match the paired base sheet. |
| `GREYSCALE` | `True` | `True` produces a Rec. 601 greyscale cosmetic sheet (matches base sheet format). `False` keeps the cosmetic's original colors — use when the cosmetic's color identity matters (dyed hair, patterned fabrics). |
| `SAM_CONFIDENCE` | `0.2` | SAM3 confidence threshold (0.05–0.95). Lower = more generous. `0.2` is the W4 default for stylized chibi hair; drop to `0.15` for very thin strands. |
| `MASK_OFFSET` | `0` | Post-SAM3 mask dilation (px). Keep at 0 for tight masks; raise slightly (1–3) if the sheet shows a hard edge against the base. |
| `MASK_SMOOTH` | `1` | Edge smoothing passes. Keep low — over-smoothing bleeds cosmetic pixels onto adjacent body pixels. |
| `SOURCE_VIDEO` | *(auto)* | Explicit absolute path to a cosmetic mp4. When empty, driver picks the latest pass2 mp4 under the standard `<cosmetic>/<animation>/` tree, falling back to pass1. |

## Wall-time expectations

On a 3090 with SAM3 hot:

| Frames | Wall time |
|---|---|
| 33 | ~20–30 s |

Most time is SAM3 inference; the stitch chain is near-instant.

## Troubleshooting

### Sheet is fully transparent

- `COSMETIC_DESCRIPTION` didn't match the cosmetic's body region. Use the region word (`"hair"`, `"hat"`, `"jacket"`) — not the color (`"pink hair"` often fails).
- `SAM_CONFIDENCE` too high. Drop to 0.15–0.2 for stylized art.
- Wrong input mp4 — a base-animation mp4 (no cosmetic) will obviously return empty masks. Verify `SOURCE_VIDEO` points at the W4 cosmetic output, not the W2 base output.
- `InvertMask` (node 15) removed or bypassed in the JSON. Without it, SAM3's `1 = cosmetic` mask is fed to `JoinImageWithAlpha.alpha` which interprets `1 = transparent`, producing a fully-transparent cosmetic and opaque background (and the *inverse* sheet — body pixels visible, cosmetic invisible). Restore the wiring: node 14 → node 15 → node 25's `alpha` input.

### Sheet has visible seam around the cosmetic

- `MASK_OFFSET` too small — raise to 1–2. Beyond that, the extra opaque halo starts covering non-cosmetic pixels, which at composition time paints over base-animation pixels incorrectly.
- Alternatively raise `MASK_SMOOTH` to 2 for a soft edge without expanding the opaque region.

### Sheet has colored fringe / bleed onto adjacent body

- `MASK_OFFSET` too large — drop to 0. `MASK_SMOOTH` too high — drop to 1.
- If the cosmetic mp4 itself has visible color bleed (halo around the cosmetic), fix it upstream by running Workflow 4 pass 2 or shrinking `PASS2_MASK_DILATE`. Workflow 5 can't remove bleed that's present in its input.

### Sheet cells don't align with the base sheet cells

- `CELL_SIZE` or `NUM_FRAMES` differ between this run and the W3 run that produced the base sheet. Both must match exactly.
- The input mp4 was not produced by W4 from the same base mp4 as W3 consumed. Use the W4 output paired with that specific base animation.

### `SAM3Segment` unknown node

- ComfyUI-RMBG isn't installed. See [Workflow 4 — Required custom nodes](./workflow-4-cosmetic-inpaint.md#required-custom-nodes).

## Install into the runtime repo

After the driver prints `saved -> …_spritesheet_NNNNN_.png`, install the sprite sheet into the repo so the runtime and [Animation Tester](../animation-tester.md) can pick it up. The repo's `animations/` directory (consumed by both the client and the tester via `animations.json`) is the runtime source of truth — ComfyUI output is a staging area.

1. **Copy the sprite sheet** to `<your-asset-dir>/layers/<cosmetic_slug>/<animation_slug>/<animation_slug>.png` in the repo. The file must be named after the animation (e.g. `idle.png`), not the ComfyUI output filename.

   Example:
   ```
   F:/ComfyUI/output/character_pipeline/cosmetics/yellow_hair/idle_breathing_5/yellow_hair_idle_breathing_5_spritesheet_00001_.png
     → <your-asset-dir>/layers/hair_style_1/idle/idle.png
   ```

   Note the slug mapping: ComfyUI uses the raw cosmetic / animation names (`yellow_hair`, `idle_breathing_5`); the repo uses the stable display slugs registered in `animations.json` (`hair_style_1`, `idle`). The ComfyUI names are iteration-local (seed sweeps, prompt variants); the repo names are the contract the runtime depends on. Never rename the repo slug to match a ComfyUI output — re-map the output into the slug instead.

2. **(New cosmetic only.)** If this is the first animation for a cosmetic that doesn't exist in `<your-asset-dir>/animations.json` yet, add the outer `cosmetics.<cosmetic_slug>` block before step 3. It needs `displayName`, `layer` (must match a key under the top-level `layers` object — `hair`, `hat`, `shirt`, etc.), and an empty `animations: {}` object. If the cosmetic slug is already registered, skip this step.

3. **Register the cosmetic animation** in `<your-asset-dir>/animations.json` under `cosmetics.<cosmetic_slug>.animations`:

   ```json
   "cosmetics": {
     "hair_style_1": {
       "displayName": "Hair Style 1",
       "layer": "hair",
       "animations": {
         "idle": "layers/hair_style_1/idle/idle.png",
         "walking": "layers/hair_style_1/walking/walking.png"
       }
     }
   }
   ```

   Keys under `animations` must match the keys in the top-level `baseAnimations` object — frame count and cell size are inherited from the corresponding base animation, so any mismatch will misalign cells at runtime. See your runtime's catalog schema for the full schema constraints.

4. **Verify in the animation tester.** Run `npm run dev:animation-tester` and confirm the cosmetic composites cleanly over the base animation. Misalignment here almost always means the wrong source mp4 was segmented in W5 — confirm W4 and W5 used the same base animation.

## Current status

**Verified 2026-04-20 on `yellow_hair` + `walking_right`.** Input: `F:/ComfyUI/output/character_pipeline/cosmetics/yellow_hair/walking_right/pass1/yellow_hair_walking_right_d35_pass1_00001_.mp4` (33 frames, 512×512, yellow spiky chibi hair on bare base character). Output: `F:/ComfyUI/output/character_pipeline/cosmetics/yellow_hair/walking_right/yellow_hair_walking_right_spritesheet_00002_.png` (4224 × 128 RGBA). 33 cells, each containing only the hair silhouette in greyscale; ~14% opaque / ~83% fully transparent — matching the hair-only coverage of the raw SAM3 masks. No body pixels, no background pixels.

Design note surfaced during first-run debugging: `JoinImageWithAlpha.alpha` treats `1 = transparent`, not `1 = opaque`. The initial workflow was missing `InvertMask` between the scaled SAM3 mask and the join, producing the inverse (body opaque / hair transparent). Fixed by adding node `15` `InvertMask`. This matches Workflow 3's identical pattern.
