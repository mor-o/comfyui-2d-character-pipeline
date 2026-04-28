# comfyui_vram_helpers

Two ComfyUI nodes for targeted VRAM eviction in multi-stage workflows. Drop a model's weights entirely (instead of offloading to system RAM) so the next stage's loader has the full VRAM budget.

## Nodes

| Class | Inputs | Outputs | Purpose |
|---|---|---|---|
| `VRAMUnloadModel` | `latent`, `model` | `latent` | Drop a `MODEL`'s weights after a `KSampler` runs. Wire the latent through this node so it executes between sampling and the next stage. |
| `VRAMUnloadClip` | `positive`, `negative`, `clip` | `positive`, `negative` | Drop a `CLIP`'s weights after text encoding. Both conditionings pass through unchanged so the executor runs this after the encode and before any downstream consumer. |

Both nodes drop weights to PyTorch meta tensors (zero storage) and pop the upstream loader chain from the executor's output cache. On the next enqueue, the loader chain re-runs from disk while every other cache entry stays hot — a seed-only re-run only re-runs the affected sampler.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this-repo>/custom_nodes/comfyui_vram_helpers
# or copy this folder into ComfyUI/custom_nodes/
```

Restart ComfyUI. Nodes appear under the `vram` category.

## When to use

- **Two-stage MoE samplers** (e.g. WAN 2.2 high-noise → low-noise expert): unload the high-noise expert between stages so the low-noise expert has the full VRAM budget.
- **Large text encoder + large UNET on a 24 GB card** (e.g. Qwen-Image-Edit 2509): drop the VL text encoder after text-encode so the image-edit UNET fits.
- **Anywhere you have a model you'll never use again in this run**, and offloading to system RAM would crowd the next allocation.

## When not to use

- If you'll re-use the same model later in the same prompt graph, don't unload it — it'll just have to reload from disk.
- If you have plenty of VRAM headroom, the standard ComfyUI offload behaviour is fine.

## How it works

`inner.to('meta')` replaces every parameter and buffer with a meta tensor. The CUDA allocations lose their last reference and the cache allocator reclaims them on the next `torch.cuda.empty_cache()`. The model is unusable after this — any forward pass raises — which is fine because the workflow never touches it again.

The node also walks the Python call stack to find the running `PromptExecutor`'s `CacheSet` and pops every ancestor of its `model` (or `clip`) input from the output cache. This forces a fresh loader run on the next enqueue — without it, ComfyUI would reuse the meta-tensored `ModelPatcher` and crash with `Tensor.item() cannot be called on meta tensors`.

## License

MIT — see [../../LICENSE](../../LICENSE).
