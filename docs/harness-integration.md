# Harness Integration

How to drive these workflows from any harness — Python, Node, an MCP-enabled coding agent, a Bash one-liner, whatever.

## What "API format" means and why it matters

Every JSON in [`workflows/`](../workflows/) is **API format** — a flat `{node_id: {class_type, inputs, _meta}}` dict. This is the only format ComfyUI's HTTP `/prompt` endpoint accepts. ComfyUI's web UI **also loads API format** (modern versions auto-arrange the graph), so a human can drag-drop the JSON into the editor for inspection / one-off runs.

Any harness that can POST JSON over HTTP can run these workflows.

## The minimal drive loop

```
1. Read workflow JSON
2. Patch the inputs you want to change (file names, prompts, seeds, …)
3. POST to http://127.0.0.1:8000/prompt with body {"prompt": <workflow>}
4. Receive {"prompt_id": "..."} back
5. Poll http://127.0.0.1:8000/history/<prompt_id> until the entry exists
6. Read the output filenames from history[prompt_id].outputs
```

That's the whole protocol. The Python drivers in [`drivers/`](../drivers/) implement exactly this — see [`drivers/run_w1_pose_edit.py`](../drivers/run_w1_pose_edit.py) for a reference implementation in ~80 lines of stdlib.

## Reference: minimal Python driver

```python
import json
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8000"

def run_workflow(workflow_path: Path, patches: dict) -> dict:
    """Load, patch, enqueue, wait, return history entry."""
    wf = json.loads(workflow_path.read_text())

    # `patches` is {node_id: {input_name: new_value}}
    for nid, fields in patches.items():
        for k, v in fields.items():
            wf[nid]["inputs"][k] = v

    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    pid = json.loads(urllib.request.urlopen(req).read().decode())["prompt_id"]

    while True:
        h = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}").read().decode())
        if pid in h:
            return h[pid]
        time.sleep(2)


# Example: run W1 with a custom seed and source image
result = run_workflow(
    Path("workflows/w1-pose-edit/workflow_api.json"),
    patches={
        "7":  {"image": "my_character.png"},   # LoadImage (source)
        "12": {"seed": 42},                    # KSampler
    },
)
print(result["outputs"])
```

To find the right node IDs to patch, open the workflow JSON and look for the `_meta.title` field on each node — those are descriptive labels (`"Source character"`, `"KSampler (4-step Lightning)"`, …).

## Reference: minimal Node.js driver

```javascript
import fs from "node:fs";

const COMFY = "http://127.0.0.1:8000";

async function runWorkflow(workflowPath, patches) {
  const wf = JSON.parse(fs.readFileSync(workflowPath, "utf8"));
  for (const [nid, fields] of Object.entries(patches)) {
    Object.assign(wf[nid].inputs, fields);
  }

  const enq = await fetch(`${COMFY}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: wf }),
  });
  const { prompt_id } = await enq.json();

  while (true) {
    const h = await (await fetch(`${COMFY}/history/${prompt_id}`)).json();
    if (prompt_id in h) return h[prompt_id];
    await new Promise((r) => setTimeout(r, 2000));
  }
}
```

## Driving from an LLM coding agent (MCP)

If you use Claude Code, Cursor, or another MCP-enabled coding agent, install a ComfyUI MCP server. The MCP wraps ComfyUI's HTTP endpoints as MCP tools so the agent can drive the pipeline conversationally instead of shelling out to the Python drivers.

### Use the same MCP we use

This repo was developed and validated against **[`comfyui-mcp` by artokun](https://github.com/artokun/comfyui-mcp)** (npm: [`comfyui-mcp`](https://www.npmjs.com/package/comfyui-mcp)). Different ComfyUI MCPs expose different tool surfaces — sticking with the same one means the agent prompts in this repo work as written.

**Install:** add the following to your MCP config (for Claude Code, that's `~/.claude.json` under `"mcpServers"`):

```json
{
  "comfyui": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "comfyui-mcp"],
    "env": {}
  }
}
```

Restart your agent. The MCP picks up `COMFYUI_URL` from env (defaults to `http://127.0.0.1:8000`).

### Tool surface

The artokun MCP exposes (non-exhaustive):

| Tool | What it does |
|---|---|
| `enqueue_workflow` | POST a workflow JSON to `/prompt`, return the prompt ID. |
| `get_job_status` / `get_history` | Poll a running prompt; fetch outputs once it finishes. |
| `get_queue` / `cancel_job` / `clear_queue` | Queue management. |
| `modify_workflow` / `validate_workflow` / `analyze_workflow` | Edit a workflow JSON in place; verify it's well-formed. |
| `start_comfyui` / `stop_comfyui` / `restart_comfyui` | Manage the ComfyUI process. |
| `clear_vram` | Force-free VRAM between runs. |
| `download_model` / `search_models` / `list_local_models` | Model management for first-time setup. |
| `list_output_images` / `workflow_from_image` | Discover and inspect outputs. |
| `get_logs` / `get_system_stats` / `generation_stats` | Debugging and observability. |

### Typical agent prompts

> "Start ComfyUI, then run W2 for the `walking` animation with the keyframe at `keyframe_walking.png` and seed 12345. When it finishes, run W3 on the resulting mp4 with `NUM_FRAMES=33` and `LOOP_MODE=loop`. Report the path to the final sprite sheet."

> "Generate the `pink_hair` cosmetic for `idle` and `walking`. The cosmetic reference is at `cosmetic_ref_pink_hair.png`. Run W4 pass 1 only — skip pass 2 unless you see colour bleed in the output. Then run W5 with prompt `hair`."

The agent loads the workflow JSON via `get_workflow` (or by reading the file directly), patches the inputs it needs to change via `modify_workflow`, calls `enqueue_workflow`, polls until done, and chains the next step. The Python drivers in [`drivers/`](../drivers/) are the reference implementation of that loop — read them when you want to know what an agent should do at each step.

## Cross-workflow orchestration is the hard part

Each workflow runs in isolation. The pipeline is **W1 → W2 → W3 → (W4 → W5 per cosmetic)**, and the gluing logic — passing W2's output filename into W3's `LoadVideo` node, version-bumping output directories, sweeping seeds across re-runs, picking the latest mp4 from a versioned folder — is what the [`drivers/`](../drivers/) do.

If you're rewriting the orchestration in a non-Python harness, the patterns to port:

- **Versioned output dirs.** W1 writes to `<output>/base_animations/<animation>/`. If that folder already exists, fall back to `<animation>_2`, `<animation>_3`, …. W2 and W3 auto-pick the highest-numbered existing version.
- **Filename hand-off.** W1 → W2 hand-off: copy `posed_<animation>_NNNNN_.png` from W1's output to ComfyUI's `input/` as `keyframe_<animation>.png`. W2 → W3 hand-off: copy `<animation>.mp4` from W2's output as `video_<animation>.mp4`. W4 → W5 hand-off: similar for cosmetic mp4s.
- **Seed sweeping.** Some W1 / W2 prompts need ~3–10 seed re-rolls before producing an acceptable frame. The driver pattern: re-enqueue with a new seed, manually inspect, accept or re-roll. Don't bake auto-acceptance — each animation has subjective quality criteria.

Read the Python drivers as the reference: each one is < 300 lines and the comments call out every place a fixed convention exists.

## Talking to a remote ComfyUI

The drivers default to `http://127.0.0.1:8000` (ComfyUI's default). To target a remote server, change `COMFY_URL` at the top of the driver. ComfyUI has no auth out of the box — if you're exposing it over the network, put it behind a VPN, an ssh tunnel, or a reverse proxy with auth.
