# W4 — Cosmetic Inpaint (two-pass bootstrap)

Pipeline 2, step 1. Takes a base-animation mp4 + a still image of the character with a cosmetic added, produces an mp4 where the cosmetic has been applied while preserving base motion bit-exact outside the cosmetic region.

This folder contains **two API workflows** that the driver runs in sequence:

| File | Pass | Purpose |
|---|---|---|
| `pass1_rough_api.json` | 1 — rough | Static cosmetic-ref mask + dilation, broadcast across frames. Produces a rough cosmetic video. |
| `pass2_clean_api.json` | 2 — clean (optional) | Per-frame SAM3 segmentation on the pass-1 video → tight per-frame masks. VACE inpaints the **original** base video using these. Pixels outside the mask are bit-exact preserved via `ImageCompositeMasked`. |

Pass 2 is a polish step. Skip it if pass 1 looks clean (no color bleed into adjacent skin/body); run it when you see haloing or blending.

- **Model:** Wan 2.1 VACE 14B (Q5_K_M GGUF)
- **LoRA:** CausVid V2 distill (`Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors`)
- **Steps:** 8, LoRA strength 1.0
- **VRAM strategy:** `VRAMUnloadClip` after UMT5 text encode, `VRAMUnloadModel` after VACE sampling

**Driver:** [`drivers/run_w4_cosmetic_inpaint.py`](../../drivers/run_w4_cosmetic_inpaint.py)

**Documentation:** [`docs/workflow-4-cosmetic-inpaint.md`](../../docs/workflow-4-cosmetic-inpaint.md)

**Inputs:**
- `base_<animation>.mp4` — base-animation clip from [W2](../w2-video-gen/)
- `cosmetic_ref_<cosmetic>.png` — **must be 512×512 PNG**, character with cosmetic added

**Outputs:**
- `<ComfyUI>/output/character_pipeline/cosmetics/<cosmetic>/<animation>/pass1/...mp4`
- `<ComfyUI>/output/character_pipeline/cosmetics/<cosmetic>/<animation>/pass2/...mp4` (optional)
