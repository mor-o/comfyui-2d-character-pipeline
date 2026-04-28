# W5 — Cosmetic Sprite Sheet

Pipeline 2, step 2. Takes the cosmetic mp4 from [W4](../w4-cosmetic-inpaint/) and produces an RGBA sprite sheet containing **only** the cosmetic pixels — transparent everywhere the base animation is unchanged.

- **Segmentation:** per-frame SAM3 text-prompted (e.g. `"hair"`, `"shirt"`) via `ComfyUI-RMBG`
- **Layout:** pixel-identical to the base sprite sheet from [W3](../w3-spritesheet/), so the runtime stacks layers without re-alignment
- **Optional greyscale** for runtime tinting

**Eye cosmetics:** run W5 normally to produce the eye sheet, then run the [iris green-key](../../docs/iris-greenkey.md) post-processing script to produce the paired iris sheet. SAM3 isn't reliable at isolating just the iris disc, so the iris uses a global colour-key on a green-authored reference.

**Driver:** [`drivers/run_w5_cosmetic_spritesheet.py`](../../drivers/run_w5_cosmetic_spritesheet.py)

**Documentation:** [`docs/workflow-5-cosmetic-spritesheet.md`](../../docs/workflow-5-cosmetic-spritesheet.md)

**Input:** `cosmetic_<cosmetic>_<animation>.mp4` (output of [W4](../w4-cosmetic-inpaint/), pass 2 if run, else pass 1)

**Output:** `<ComfyUI>/output/character_pipeline/cosmetics/<cosmetic>/<animation>/<animation>_spritesheet.png`
