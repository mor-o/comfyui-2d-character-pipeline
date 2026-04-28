# ComfyUI 2D Character Animation Pipeline

A set of ComfyUI workflows that turn a **single character image** into a full library of layered 2D animation sprite sheets — base motion plus separately-tintable cosmetic layers (hair, eyes, clothing, …) — pixel-aligned for runtime composition.

Designed for **24 GB VRAM** (e.g. RTX 3090). All workflows are tuned to peak at ~22.5 GB so the OS / browser / display compositor have headroom.

> **This repo is harness-driven** It's design to be run fully automatically by an AI agent such as Claude code or Cursor. The workflows are shipped in **API format** (the JSON ComfyUI's `/prompt` endpoint accepts) — they are **not** visual / UI workflows and will not appear as a laid-out node graph if you drag them into ComfyUI's editor.
>
> If you want to inspect or edit the graph visually, modern ComfyUI versions can auto-arrange nodes from API JSON, but layouts will be ad-hoc — these workflows were authored and maintained as API JSON, not in the UI.
>
> See [`docs/harness-integration.md`](docs/harness-integration.md) for the drive protocol and minimal Python / Node examples.

![pipeline overview](samples/base_animations/idle/idle.png)

 Idle base animation produced by W2 → W3 from the seed input at [`samples/inputs/source_idle_breathing.png`](samples/inputs/source_idle_breathing.png). Greyscale RGBA so the runtime can tint pixels at draw time. See [`samples/`](samples/) for the full sprite-sheet set this pipeline produced.

## What this pipeline produces

- **Base animations** — greyscale RGBA sprite sheets for idle, walking, jumping, …. Each sheet is a horizontal strip of equal-sized cells. Frame count and cell size are configurable per animation.
- **Cosmetic layers** — RGBA sprite sheets for hair, eyes, clothing, etc. Each cosmetic layer is pixel-identical in layout to the base animation it pairs with, so a runtime stacks `cosmetic.png` over `base.png` without re-alignment.
- **Eye / iris pairs** — the iris ships as a separate near-white greyscale sheet so the runtime can tint it to the player's chosen eye colour without affecting sclera, lashes, or specular highlights.

## Two pipelines, five workflows

```
                                 ┌──────────────────────────────┐
                                 │ Pipeline 1: BASE ANIMATIONS  │
                                 └──────────────────────────────┘
   character image  ─►  [W1 helper: pose edit]  ─►  posed keyframe
        │
        └──────────────────────────────────────────►  [W2: video gen]  ─►  base.mp4
                                                            │
                                                            └─►  [W3: spritesheet]  ─►  base sprite sheet  ✓


                                 ┌──────────────────────────────┐
                                 │ Pipeline 2: COSMETIC LAYERS  │
                                 └──────────────────────────────┘
   base.mp4
   cosmetic reference image  ─►  [W4: cosmetic inpaint, two-pass]  ─►  cosmetic.mp4
                                                                              │
                                                                              └─►  [W5: cosmetic spritesheet]  ─►  cosmetic sprite sheet  ✓
                                                                                                                  (+ iris green-key script for eye cosmetics)
```

| Workflow | Model | Purpose |
|---|---|---|
| [W1 — Pose Edit](docs/workflow-1-pose-edit.md) (helper) | Qwen-Image-Edit 2509 + InstantX ControlNet-Union | Pose a character from an OpenPose skeleton. Used to produce the seed image for W2 or a one-off posed keyframe. |
| [W2 — Video Generation](docs/workflow-2-video-gen.md) | WAN 2.2 i2v 14B (Q5_K_M GGUF) + Lightx2v 4-step distill | Image-to-video: extrapolate motion from a single frame. |
| [W3 — Sprite Sheet](docs/workflow-3-spritesheet.md) | BiRefNet (background removal) | Horizontal greyscale RGBA sprite sheet from the W2 mp4. |
| [W4 — Cosmetic Inpaint](docs/workflow-4-cosmetic-inpaint.md) | Wan 2.1 VACE 14B + CausVid V2 LoRA + SAM3 (mask) | Two-pass inpainting: paint the cosmetic onto the base motion while preserving non-cosmetic pixels bit-exact. |
| [W5 — Cosmetic Sprite Sheet](docs/workflow-5-cosmetic-spritesheet.md) | SAM3 (segmentation) | Per-frame text-prompted segmentation → cosmetic-only RGBA sprite sheet, pixel-aligned with the base sheet. |
| [Iris Green-Key](docs/iris-greenkey.md) (script) | — | Post-processing script for eye cosmetics: extracts a tintable iris-only sheet from the W4 pass-1 mp4 via colour-keying. |

For the full pipeline overview and design rationale, see **[`docs/pipeline-overview.md`](docs/pipeline-overview.md)**.

## Prerequisites

### Hardware

- **NVIDIA GPU with ≥ 24 GB VRAM** (tuned on RTX 3090). All workflows aim for ~22.5 GB peak.
- **≥ 32 GB system RAM**.
- **~120 GB free disk** for model weights (most are GGUF Q5_K_M variants).

### Software

1. **ComfyUI** — install per the [official ComfyUI README](https://github.com/comfyanonymous/ComfyUI). Launch with `--reserve-vram 1` so ~1 GB stays unallocated for ComfyUI overhead.

2. **Custom node packs** — install per [`docs/custom-nodes.md`](docs/custom-nodes.md):
   - `comfyui_vram_helpers` (this repo, [`custom_nodes/`](custom_nodes/comfyui_vram_helpers/)) — two-stage VRAM eviction nodes
   - `ComfyUI-GGUF` — GGUF model loaders
   - `comfyui_controlnet_aux` — ControlNet utilities
   - `ComfyUI-RMBG` — BiRefNet + SAM3 segmentation
   - `ComfyUI-Manager` (optional but recommended) — auto-installs missing nodes

3. **Models** — download per [`docs/models.md`](docs/models.md). Total ≈ 80 GB for the full pipeline (Q5_K_M variants of WAN 2.2, WAN 2.1 VACE, Qwen-Image-Edit; SAM3, BiRefNet, several VAEs and ControlNets).

4. **ComfyUI MCP** — required if you're driving the pipeline from an LLM coding agent (Claude Code, Cursor, etc.). This repo was developed and validated against **[`comfyui-mcp` by artokun](https://github.com/artokun/comfyui-mcp)** — install with:

   ```bash
   # Add to your MCP config (Claude Code: ~/.claude.json under "mcpServers")
   {
     "comfyui": {
       "type": "stdio",
       "command": "npx",
       "args": ["-y", "comfyui-mcp"],
       "env": {}
     }
   }
   ```

   That MCP exposes `enqueue_workflow`, `get_job_status`, `get_history`, `modify_workflow`, `start_comfyui` / `stop_comfyui`, `clear_vram`, `download_model`, and ~30 other tools that map directly to the operations these drivers perform. **Use the same MCP we use** — different ComfyUI MCPs expose different tool surfaces, and the prompt examples in [`docs/harness-integration.md`](docs/harness-integration.md) assume artokun's tool names.

## Quick start

1. **Install** ComfyUI, ComfyUI MCP, the custom nodes, and the models per the prerequisites above.

2. **Drop a character image** into ComfyUI's `input/` folder as `source_<animation>.png`. A working sample is at [`samples/inputs/source_idle_breathing.png`](samples/inputs/source_idle_breathing.png) — copy that to `<ComfyUI>/input/` to validate your install end-to-end.

3. **Launch ComfyUI** (`python main.py --reserve-vram 1`).

4. **Run W2 → W3** for a base animation, then **W4 → W5** for each cosmetic layer:

   ```bash
   python drivers/run_w2_video_gen.py        # produces base.mp4
   python drivers/run_w3_spritesheet.py      # produces base sprite sheet

   python drivers/run_w4_cosmetic_inpaint.py # produces cosmetic.mp4
   python drivers/run_w5_cosmetic_spritesheet.py  # produces cosmetic sprite sheet
   ```

   Each driver has a small block of per-run knobs at the top (animation name, seed, etc.) — edit and re-run.

5. **Compare against the samples** at [`samples/base_animations/`](samples/base_animations/) and [`samples/layers/`](samples/layers/). If your output looks close, your install is healthy.

## Repo layout

| Directory | Contents |
|---|---|
| [`workflows/`](workflows/) | One folder per workflow. Each contains the API-format JSON (what ComfyUI's `/prompt` endpoint accepts) plus a short README pointing at the full doc. |
| [`drivers/`](drivers/) | Python scripts that drive the workflows: edit knobs at the top, run the script, it POSTs the workflow to `http://127.0.0.1:8000/prompt` and waits for completion. Single-file, no dependencies beyond the standard library. |
| [`custom_nodes/comfyui_vram_helpers/`](custom_nodes/comfyui_vram_helpers/) | Two ComfyUI nodes (`VRAMUnloadModel`, `VRAMUnloadClip`) for targeted VRAM eviction in multi-stage workflows. Drop into `ComfyUI/custom_nodes/`. |
| [`docs/`](docs/) | Pipeline overview, per-workflow design docs, model and custom-node install guides, harness-integration guide. |
| [`samples/`](samples/) | Working seed input + the full sprite-sheet output the pipeline produced — useful for sanity-checking your install. |
| [`tools/`](tools/) | Standalone helpers (OpenPose skeleton renderer, depth renderer, frame inspector). |

## Workflow format — API only, harness-driven

Every JSON in [`workflows/`](workflows/) is **API format** — a flat `{node_id: {class_type, inputs, _meta}}` dict. This is the format ComfyUI's `/prompt` endpoint accepts and the format any automated harness (Python script, MCP-enabled coding agent, Node script, curl one-liner) consumes.

These workflows are **not** UI workflows. They were authored and maintained as API JSON — there is no saved node layout, no manually-arranged graph, no widget-position metadata. If you drag one into ComfyUI's editor, modern versions will auto-arrange the nodes so you can inspect the structure, but the result is ad-hoc and not optimised for human editing. **The intended way to use this repo is to POST workflows to ComfyUI from a harness, not to click "Queue" in the UI.**

The Python drivers in [`drivers/`](drivers/) are reference harnesses — they read the workflow JSON, patch a few inputs (animation name, seed, …), POST to `/prompt`, poll `/history`, and return the output paths. Each driver also handles cross-workflow gluing the workflows themselves can't express: passing W2's output mp4 path into W3's `LoadVideo` node, version-bumping output directories, sweeping seeds across re-runs, picking the latest mp4 from a versioned folder. **If you re-implement in another language, you re-implement that orchestration too.** See [`docs/harness-integration.md`](docs/harness-integration.md) for a minimal language-agnostic drive loop and Python / Node examples.

## Documentation

- **[`docs/pipeline-overview.md`](docs/pipeline-overview.md)** — full pipeline design, rationale, hardware tuning rules.
- **[`docs/workflow-1-pose-edit.md`](docs/workflow-1-pose-edit.md)** — W1 helper (Qwen-Image-Edit + ControlNet).
- **[`docs/workflow-2-video-gen.md`](docs/workflow-2-video-gen.md)** — W2 (WAN 2.2 i2v).
- **[`docs/workflow-3-spritesheet.md`](docs/workflow-3-spritesheet.md)** — W3 (BiRefNet → greyscale strip).
- **[`docs/workflow-4-cosmetic-inpaint.md`](docs/workflow-4-cosmetic-inpaint.md)** — W4 (VACE inpaint, two-pass bootstrap).
- **[`docs/workflow-5-cosmetic-spritesheet.md`](docs/workflow-5-cosmetic-spritesheet.md)** — W5 (SAM3 → cosmetic-only strip).
- **[`docs/iris-greenkey.md`](docs/iris-greenkey.md)** — eye cosmetic post-processing.
- **[`docs/models.md`](docs/models.md)** — full model list with download links and disk-path conventions.
- **[`docs/custom-nodes.md`](docs/custom-nodes.md)** — custom-node install instructions.
- **[`docs/harness-integration.md`](docs/harness-integration.md)** — driving these workflows from MCP / Python / Node / any harness.

## Contributing

Issues and PRs welcome. The pipeline has been validated on idle, walking, and in-air animations on a single base character — extending to new animations or new cosmetic categories should "just work" via the configuration knobs at the top of each driver, but feedback on edge cases is appreciated.

## License

[MIT](LICENSE) — use, modify, and redistribute freely. If this saves you time, a credit link back is appreciated but not required.
