# W1 — Pose Edit

Helper workflow. Takes a character PNG + a pre-rendered OpenPose skeleton PNG, produces a single posed keyframe of the same character. The skeleton constrains joint positions; the source image carries character identity, proportions, and line art style.

Used either to generate the **base character image** for the pipeline, or to produce a posed starting frame for [W2](../w2-video-gen/) when the default base image isn't suitable.

- **Model:** Qwen-Image-Edit 2509 (Q5_K_M GGUF)
- **ControlNet:** InstantX Qwen-Image ControlNet-Union (`openpose` sub-type)
- **VRAM peak:** ~22.5 GB on a 24 GB card (two-phase eviction via `VRAMUnloadClip` + `VRAMUnloadModel`)
- **Steps:** 4 (Lightning 4-step LoRA), CFG 1, euler / simple

**Driver:** [`drivers/run_w1_pose_edit.py`](../../drivers/run_w1_pose_edit.py)

**Documentation:** [`docs/workflow-1-pose-edit.md`](../../docs/workflow-1-pose-edit.md)

**Inputs (drop into ComfyUI's `input/` folder):**
- `source_<animation>.png` — the character to pose
- `pose_<animation>.png` — OpenPose skeleton, 512×512

**Output:** `<ComfyUI>/output/character_pipeline/base_animations/<animation>/poses/posed_<animation>_NNNNN_.png`
