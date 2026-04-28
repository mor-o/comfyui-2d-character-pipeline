# Custom Nodes

Four ComfyUI custom node packs are required. The first is shipped in this repo; the other three are public.

## Required

| Pack | Source | Provides |
|---|---|---|
| **`comfyui_vram_helpers`** | [`custom_nodes/comfyui_vram_helpers/`](../custom_nodes/comfyui_vram_helpers/) (this repo) | `VRAMUnloadModel`, `VRAMUnloadClip` — drop weights between stages. |
| **`ComfyUI-GGUF`** | <https://github.com/city96/ComfyUI-GGUF> | `UnetLoaderGGUF`, `CLIPLoaderGGUF` — quantized model loaders. |
| **`comfyui_controlnet_aux`** | <https://github.com/Fannovel16/comfyui_controlnet_aux> | `SetUnionControlNetType` and ControlNet preprocessors. |
| **`ComfyUI-RMBG`** | <https://github.com/1038lab/ComfyUI-RMBG> | `BiRefNetRMBG` (W3 background removal), `SAM3Segment` (W4/W5 segmentation). |

## Recommended

| Pack | Source | Why |
|---|---|---|
| **`ComfyUI-Manager`** | <https://github.com/ltdrdata/ComfyUI-Manager> | Detects missing custom nodes when loading a workflow, offers one-click install. Saves you the install dance below. |

## Install (manual)

```bash
cd ComfyUI/custom_nodes

# This repo's node pack — copy or symlink:
cp -r <path-to-this-repo>/custom_nodes/comfyui_vram_helpers .

# Public node packs:
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
git clone https://github.com/1038lab/ComfyUI-RMBG

# Optional but recommended:
git clone https://github.com/ltdrdata/ComfyUI-Manager
```

Then for each pack that has a `requirements.txt`, install its Python deps into ComfyUI's environment:

```bash
cd ComfyUI
.venv/Scripts/pip.exe install -r custom_nodes/comfyui_controlnet_aux/requirements.txt
.venv/Scripts/pip.exe install -r custom_nodes/ComfyUI-RMBG/requirements.txt
# (city96/ComfyUI-GGUF has no requirements.txt — relies on ComfyUI's existing torch)
```

Restart ComfyUI. Verify by opening the node menu — you should see categories for `vram` (this repo), `gguf`, `image/segmentation`, `image/preprocessing`, etc.

## Install (via ComfyUI-Manager)

If you installed Manager, load any workflow JSON from this repo (`workflows/w1-pose-edit/workflow_api.json`, etc.) into ComfyUI's UI. Manager will detect the missing custom nodes and offer to install them. **`comfyui_vram_helpers` is not on the Manager registry** — install that one manually as above.

## Verifying the install

Quick smoke test, in ComfyUI's Python venv:

```python
import importlib
for pack in ("comfy", "custom_nodes.comfyui_vram_helpers"):
    print(pack, "ok" if importlib.util.find_spec(pack) else "MISSING")
```

You should see `ok` for both. If `comfyui_vram_helpers` is `MISSING`, double-check the folder is at `ComfyUI/custom_nodes/comfyui_vram_helpers/` (not nested another level deep).

## Troubleshooting

- **`Cannot import name 'VRAMUnloadModel'`** — the node pack failed to load. Check ComfyUI's startup log for an exception in the `comfyui_vram_helpers` import. Most common cause: ComfyUI was started with a different Python interpreter than the one the venv is set up for.
- **`UnetLoaderGGUF` not found** — `ComfyUI-GGUF` not installed or failed to load. Check that the folder is at `ComfyUI/custom_nodes/ComfyUI-GGUF/` and the GGUF dependency (`gguf` Python package) is installed.
- **`SAM3Segment` not found** — `ComfyUI-RMBG` not installed, or the SAM3 weights haven't downloaded yet. The first run will auto-download from HuggingFace; if your machine has no internet, pre-download the SAM3 weights into `ComfyUI/models/sams/`.
