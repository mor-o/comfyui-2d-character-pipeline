# Models

Full download list for the pipeline. Total disk: **≈ 80 GB**. Sized for a single 24 GB VRAM card — every workflow's models fit together with ~1.5 GB headroom for ComfyUI overhead.

> **Picking quants:** every UNET / DiT in the pipeline uses **Q5_K_M GGUF** — the largest GGUF that still leaves headroom on a 24 GB card. If you have less VRAM, drop to Q4_K_M (slightly worse fidelity, ~15% smaller). If you have more (40+ GB), use the original `.safetensors` for marginal quality gains.

## ComfyUI directory layout

```
ComfyUI/
└── models/
    ├── unet/              ← UNETs and DiTs (GGUFs go here)
    ├── clip/              ← Text encoders (GGUFs go here)
    ├── vae/               ← VAEs
    ├── loras/             ← LoRAs
    ├── controlnet/        ← ControlNet weights
    ├── BiRefNet/          ← BiRefNet (auto-downloaded by ComfyUI-RMBG on first use)
    └── sams/              ← SAM3 (auto-downloaded by ComfyUI-RMBG on first use)
```

## Model list

### W1 — Pose Edit

| File | Folder | Size (approx) | Source |
|---|---|---|---|
| `Qwen-Image-Edit-2509-Q5_K_M.gguf` | `unet/` | ~10 GB | <https://huggingface.co/city96/Qwen-Image-Edit-2509-gguf> |
| `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | `loras/` | ~600 MB | <https://huggingface.co/lightx2v/Qwen-Image-Lightning> |
| `Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf` | `clip/` | ~5 GB | <https://huggingface.co/city96/Qwen2.5-VL-7B-Instruct-gguf> |
| `qwen_image_vae.safetensors` | `vae/` | ~250 MB | <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI> |
| `Qwen-Image-InstantX-ControlNet-Union.safetensors` | `controlnet/` | ~2.5 GB | <https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union> |

### W2 — Video Generation

| File | Folder | Size (approx) | Source |
|---|---|---|---|
| `wan2.2_i2v_high_noise_lightx2v_Q5_K_M.gguf` | `unet/` | ~10 GB | <https://huggingface.co/city96/Wan2.2-I2V-14B-gguf> (high-noise expert) |
| `wan2.2_i2v_low_noise_lightx2v_Q5_K_M.gguf` | `unet/` | ~10 GB | <https://huggingface.co/city96/Wan2.2-I2V-14B-gguf> (low-noise expert) |
| `umt5-xxl-encoder-Q5_K_M.gguf` | `clip/` | ~5 GB | <https://huggingface.co/city96/umt5-xxl-encoder-gguf> |
| `wan_2.1_vae.safetensors` | `vae/` | ~250 MB | <https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged> |

### W3 — Sprite Sheet

| File | Folder | Size (approx) | Source |
|---|---|---|---|
| `BiRefNet_toonout` | `BiRefNet/` (auto) | ~1 GB | Auto-downloaded by `ComfyUI-RMBG` on first run |

### W4 — Cosmetic Inpaint

| File | Folder | Size (approx) | Source |
|---|---|---|---|
| `Wan2.1_14B_VACE-Q5_K_M.gguf` | `unet/` | ~10 GB | <https://huggingface.co/city96/Wan2.1-VACE-14B-gguf> |
| `Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors` | `loras/` | ~600 MB | <https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan21_CausVid> |
| `umt5-xxl-encoder-Q5_K_M.gguf` | `clip/` | (shared with W2) | (already downloaded) |
| `wan_2.1_vae.safetensors` | `vae/` | (shared with W2) | (already downloaded) |
| SAM3 weights | `sams/` (auto) | ~2.5 GB | Auto-downloaded by `ComfyUI-RMBG` on first segmentation run |

### W5 — Cosmetic Sprite Sheet

| File | Folder | Size (approx) | Source |
|---|---|---|---|
| SAM3 weights | `sams/` (auto) | (shared with W4) | (already downloaded) |

## Disk-space sanity check

Q5_K_M variants of the three big DiTs (Qwen-Image-Edit, WAN 2.2 ×2 experts, WAN 2.1 VACE) total ~40 GB. Add the text encoders (~10 GB), VAEs / ControlNets / LoRAs (~6 GB), and the auto-downloaded segmentation models (~3.5 GB), and you land around 60 GB. Budget 80 GB to leave room for cache and the occasional duplicate filename.

## When a model won't fit

- **Drop to Q4_K_M.** Same loader, smaller weights, marginal fidelity hit.
- **Use a smaller variant.** WAN 2.2 has a 5B "TI2V" variant; Qwen-Image has a non-Edit variant. Performance varies.
- **Don't try Q3 or Q2.** Quality cliff hits before VRAM pressure becomes painful.

## Where to find newer / better quants

The GGUF community (city96, second-state, etc.) re-quantizes popular models within days of release. Check the model authors' HuggingFace pages directly:

- **GGUF re-quants:** <https://huggingface.co/city96>
- **WAN community LoRAs:** <https://huggingface.co/Kijai/WanVideo_comfy>
- **Lightning / distill LoRAs:** <https://huggingface.co/lightx2v>

If you find a Q5_K_M of a newer base model that fits the VRAM budget, the workflows are mostly drop-in: replace the filename in the `UnetLoaderGGUF` / `CLIPLoaderGGUF` node and verify the recommended sampler settings (steps, CFG, scheduler) for the new model.
