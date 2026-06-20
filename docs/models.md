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

Links go directly to the file — click to download or copy the URL into `huggingface-cli download`.

### W1 — Pose Edit (~19 GB)

| File | Folder | Size | Direct download |
|---|---|---|---|
| `Qwen-Image-Edit-2509-Q5_K_M.gguf` | `unet/` | ~10 GB | [download](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main/Qwen-Image-Edit-2509-Q5_K_M.gguf) |
| `Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf` | `clip/` | ~5 GB | [download](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf) |
| `Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf` | `clip/` | ~1.3 GB | [download](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-F16.gguf) — save as `Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf` |
| `qwen_image_vae.safetensors` | `vae/` | ~250 MB | [download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | `loras/` | ~600 MB | [download](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors) |
| `Qwen-Image-InstantX-ControlNet-Union.safetensors` | `controlnet/` | ~2.5 GB | [download](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/diffusion_pytorch_model.safetensors) — save as `Qwen-Image-InstantX-ControlNet-Union.safetensors` |

> **Note on the mmproj file:** ComfyUI-GGUF auto-discovers the vision encoder by matching the model name stem in the filename. Both CLIP GGUFs must live in `models/clip/` and the mmproj filename must contain both `mmproj` and `Qwen2.5-VL-7B-Instruct`.

### W2 — Video Generation (~25 GB)

| File | Folder | Size | Direct download |
|---|---|---|---|
| `wan2.2_i2v_high_noise_lightx2v_Q5_K_M.gguf` | `unet/` | ~10 GB | [download](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF/resolve/main/high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q5_K_M.gguf) |
| `wan2.2_i2v_low_noise_lightx2v_Q5_K_M.gguf` | `unet/` | ~10 GB | [download](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF/resolve/main/low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q5_K_M.gguf) |
| `umt5-xxl-encoder-Q5_K_M.gguf` | `clip/` | ~5 GB | [download](https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf) |
| `wan_2.1_vae.safetensors` | `vae/` | ~250 MB | [download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |

### W3 — Sprite Sheet

| File | Folder | Size | Source |
|---|---|---|---|
| `BiRefNet_toonout` | `BiRefNet/` (auto) | ~1 GB | Auto-downloaded by `ComfyUI-RMBG` on first run |

### W4 — Cosmetic Inpaint (~11 GB extra; reuses W2's encoder and VAE)

| File | Folder | Size | Direct download |
|---|---|---|---|
| `Wan2.1_14B_VACE-Q5_K_M.gguf` | `unet/` | ~10 GB | [download](https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/Wan2.1_14B_VACE-Q5_K_M.gguf) |
| `Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors` | `loras/` | ~600 MB | [download](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors) |
| `umt5-xxl-encoder-Q5_K_M.gguf` | `clip/` | (shared with W2) | (already downloaded) |
| `wan_2.1_vae.safetensors` | `vae/` | (shared with W2) | (already downloaded) |
| SAM3 weights | `sams/` (auto) | ~2.5 GB | Auto-downloaded by `ComfyUI-RMBG` on first segmentation run |

### W5 — Cosmetic Sprite Sheet

| File | Folder | Size | Source |
|---|---|---|---|
| SAM3 weights | `sams/` (auto) | (shared with W4) | (already downloaded) |

## Automated download script

The repo ships a download script that handles all renaming and correct placement automatically:

```bash
python drivers/download_models.py --comfyui-root /path/to/ComfyUI --all
python drivers/download_models.py --comfyui-root /path/to/ComfyUI --w2   # video gen only
python drivers/download_models.py --comfyui-root /path/to/ComfyUI --w1   # pose edit only
python drivers/download_models.py --comfyui-root /path/to/ComfyUI --w4   # cosmetic inpaint only

# HuggingFace token (required for gated repos):
python drivers/download_models.py --comfyui-root /path/to/ComfyUI --all --token hf_xxxxxxxxxxxx
# or: export HF_TOKEN="hf_xxxxxxxxxxxx"       (bash)
# or: $env:HF_TOKEN = "hf_xxxxxxxxxxxx"       (PowerShell)
```

Get a free token (Read access is enough) at <https://huggingface.co/settings/tokens>.

## Disk-space sanity check

Q5_K_M variants of the three big DiTs (Qwen-Image-Edit, WAN 2.2 ×2 experts, WAN 2.1 VACE) total ~40 GB. Add the text encoders (~10 GB), VAEs / ControlNets / LoRAs (~6 GB), and the auto-downloaded segmentation models (~3.5 GB), and you land around 60 GB. Budget 80 GB to leave room for cache and the occasional duplicate filename.

## When a model won't fit

- **Drop to Q4_K_M.** Same loader, smaller weights, marginal fidelity hit.
- **Use a smaller variant.** WAN 2.2 has a 5B "TI2V" variant; Qwen-Image has a non-Edit variant. Performance varies.
- **Don't try Q3 or Q2.** Quality cliff hits before VRAM pressure becomes painful.

## Where to find newer / better quants

The GGUF community re-quantizes popular models within days of release. Check the model authors' HuggingFace pages directly:

- **GGUF re-quants (WAN, UMT5):** <https://huggingface.co/city96>
- **WAN community LoRAs:** <https://huggingface.co/Kijai/WanVideo_comfy>
- **Lightning / distill LoRAs:** <https://huggingface.co/lightx2v>
- **Qwen-Image GGUFs:** <https://huggingface.co/QuantStack>

If you find a Q5_K_M of a newer base model that fits the VRAM budget, the workflows are mostly drop-in: replace the filename in the `UnetLoaderGGUF` / `CLIPLoaderGGUF` node and verify the recommended sampler settings (steps, CFG, scheduler) for the new model.
