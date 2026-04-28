"""Enqueue w3-spritesheet/workflow_api.json for a single animation.

Takes the mp4 produced by Workflow 2 and produces a horizontal greyscale
RGBA sprite sheet. BiRefNet strips the background, a YUV conversion
produces the greyscale channel that the runtime tints per-player, and a
dynamic chain of ImageFromBatch + ImageStitch nodes (generated here at
build time — one per frame) concatenates the frames left-to-right into a
single PNG. The sheet contains exactly NUM_FRAMES cells.

Resolves ANIMATION_NAME automatically: finds the LATEST versioned dir
matching ANIMATION_NAME_BASE under F:/ComfyUI/output/character_pipeline/
base_animations/ (e.g. idle_breathing, idle_breathing_2, idle_breathing_3
— picks the highest numbered one). This pairs with W1's auto-versioning
and W2's same-dir output.

Input:  latest mp4 in the resolved animation dir.
Output: <resolved_name>_spritesheet_NNNNN_.png in the same dir.

Run with (from the repo root):
    python drivers/run_w3_spritesheet.py
(or whatever ComfyUI-compatible Python interpreter you use; the script
only needs `urllib`, no extra dependencies).
"""
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8000"
INPUT_DIR = Path(r"F:/ComfyUI/input")
OUTPUT_DIR = Path(r"F:/ComfyUI/output")
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "w3-spritesheet" / "workflow_api.json"
ANIM_ROOT = OUTPUT_DIR / "character_pipeline" / "base_animations"

# --- per-run knobs ---
ANIMATION_NAME_BASE = "idle_breathing"
NUM_FRAMES = 33
CELL_SIZE = 128
RMBG_MODEL = "BiRefNet_toonout"


def resolve_latest_animation_name(base: str) -> str:
    """Return the highest-numbered versioned dir for `base`.

    Checks <base>, <base>_2, <base>_3, ... and returns the highest numbered
    one that exists on disk. Raises if nothing matches.
    """
    candidates: list[tuple[int, str]] = []
    if (ANIM_ROOT / base).exists():
        candidates.append((1, base))
    pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
    if ANIM_ROOT.exists():
        for child in ANIM_ROOT.iterdir():
            if not child.is_dir():
                continue
            m = pat.match(child.name)
            if m:
                candidates.append((int(m.group(1)), child.name))
    if not candidates:
        raise RuntimeError(
            f"no animation dir matching base '{base}' under {ANIM_ROOT}. "
            "Run W1 and W2 first."
        )
    candidates.sort(key=lambda p: p[0])
    return candidates[-1][1]


def post_prompt(workflow: dict) -> str:
    body = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    if "prompt_id" not in resp:
        raise RuntimeError(f"comfyui rejected workflow: {resp}")
    return resp["prompt_id"]


def wait_for(prompt_id: str, timeout_s: int = 600) -> dict:
    start = time.time()
    last_log = 0.0
    while True:
        try:
            history = json.loads(
                urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}").read().decode()
            )
        except Exception as e:
            history = {}
            if time.time() - last_log > 10:
                print(f"  ...waiting for {prompt_id} ({e})")
                last_log = time.time()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"job {prompt_id} errored: {status.get('messages')}")
            return entry
        if time.time() - start > timeout_s:
            raise TimeoutError(f"job {prompt_id} did not finish within {timeout_s}s")
        if time.time() - last_log > 15:
            print(f"  ...waiting for {prompt_id}")
            last_log = time.time()
        time.sleep(2)


def output_files(history_entry: dict) -> list[Path]:
    files: list[Path] = []
    for node_outputs in history_entry.get("outputs", {}).values():
        for spec in node_outputs.get("images", []) or []:
            sub = spec.get("subfolder", "")
            fn = spec.get("filename")
            if fn:
                files.append(OUTPUT_DIR / sub / fn)
    return files


def latest_source_video(animation_name: str) -> Path:
    """Pick the most recent mp4 in the per-animation output dir."""
    d = ANIM_ROOT / animation_name
    candidates = sorted(
        d.glob(f"{animation_name}*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Filter out spritesheet .pngs if any glob collision; keep mp4s only.
    candidates = [c for c in candidates if c.suffix.lower() == ".mp4"]
    if not candidates:
        raise RuntimeError(f"no mp4 found in {d} for {animation_name}")
    return candidates[0]


def build_workflow(animation_name: str, input_video_name: str) -> dict:
    """Load the base workflow and append the per-frame slice + stitch chain.

    The base JSON ends at node id 25 (JoinImageWithAlpha → RGBA batch). We
    append:
      100..100+N-1  ImageFromBatch per frame
      200..200+N-2  ImageStitch left-to-right
      300           SaveImage of the final strip
    """
    wf = json.loads(WORKFLOW_PATH.read_text())
    wf["1"]["inputs"]["file"] = input_video_name
    wf["3"]["inputs"]["length"] = NUM_FRAMES
    wf["10"]["inputs"]["model"] = RMBG_MODEL
    for resize_nid in ("20", "22"):
        wf[resize_nid]["inputs"]["width"] = CELL_SIZE
        wf[resize_nid]["inputs"]["height"] = CELL_SIZE

    rgba_batch_nid = "25"

    for i in range(NUM_FRAMES):
        wf[str(100 + i)] = {
            "class_type": "ImageFromBatch",
            "inputs": {
                "image": [rgba_batch_nid, 0],
                "batch_index": i,
                "length": 1,
            },
            "_meta": {"title": f"Slice frame {i}"},
        }

    prev_nid = str(100)
    for i in range(1, NUM_FRAMES):
        stitch_nid = str(200 + i - 1)
        wf[stitch_nid] = {
            "class_type": "ImageStitch",
            "inputs": {
                "image1": [prev_nid, 0],
                "image2": [str(100 + i), 0],
                "direction": "right",
                "match_image_size": True,
                "spacing_width": 0,
                "spacing_color": "white",
            },
            "_meta": {"title": f"Stitch frame {i}"},
        }
        prev_nid = stitch_nid

    wf["300"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": [prev_nid, 0],
            "filename_prefix": (
                f"character_pipeline/base_animations/{animation_name}/"
                f"{animation_name}_spritesheet"
            ),
        },
        "_meta": {"title": "Save sprite sheet"},
    }
    return wf


def main() -> None:
    animation_name = resolve_latest_animation_name(ANIMATION_NAME_BASE)
    src = latest_source_video(animation_name)
    staged = INPUT_DIR / f"video_{animation_name}.mp4"
    shutil.copy2(src, staged)
    print(f"[W3] staged {src.name} -> {staged}")
    print(
        f"[W3] {animation_name} | frames={NUM_FRAMES} | cell={CELL_SIZE}px | "
        f"rmbg={RMBG_MODEL}"
    )
    wf = build_workflow(animation_name, staged.name)
    pid = post_prompt(wf)
    print(f"  enqueued: {pid}")
    history = wait_for(pid)
    saved = output_files(history)
    if not saved:
        raise RuntimeError("no output produced")
    for p in saved:
        print(f"  saved -> {p}")


if __name__ == "__main__":
    sys.exit(main() or 0)
