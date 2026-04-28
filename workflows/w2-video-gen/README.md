# W2 — Video Generation

Pipeline 1, step 1. Takes a single keyframe PNG and generates a short mp4 of the character performing the animation (idle breathing, walk, run, jump, …).

- **Model:** WAN 2.2 i2v 14B (Q5_K_M GGUF), distilled with Lightx2v 4-step
- **VRAM strategy:** two-stage MoE eviction — the high-noise expert is unloaded via `VRAMUnloadModel` after stage 1 sampling, before the low-noise expert loads for stage 2. Both never coexist on a 24 GB card.
- **Output:** 33 frames (or 9 for in-air), 512×512, 16 fps

**Driver:** [`drivers/run_w2_video_gen.py`](../../drivers/run_w2_video_gen.py)

**Documentation:** [`docs/workflow-2-video-gen.md`](../../docs/workflow-2-video-gen.md)

**Input:** `keyframe_<animation>.png` (output of [W1](../w1-pose-edit/) or any 512×512 PNG)

**Output:** `<ComfyUI>/output/character_pipeline/base_animations/<animation>/<animation>.mp4`
