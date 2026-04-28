# W3 — Sprite Sheet

Pipeline 1, step 2. Takes the mp4 from [W2](../w2-video-gen/) and produces a horizontal greyscale RGBA sprite sheet.

- **Background removal:** BiRefNet (via `ComfyUI-RMBG`)
- **Greyscale:** YUV-based luminance extraction so the runtime can tint per-pixel
- **Layout:** dynamic stitch sized to `NUM_FRAMES` — no ping-pong loop in the asset

**Driver:** [`drivers/run_w3_spritesheet.py`](../../drivers/run_w3_spritesheet.py)

**Documentation:** [`docs/workflow-3-spritesheet.md`](../../docs/workflow-3-spritesheet.md)

**Input:** `video_<animation>.mp4` (output of [W2](../w2-video-gen/))

**Output:** `<ComfyUI>/output/character_pipeline/base_animations/<animation>/<animation>_spritesheet.png`
