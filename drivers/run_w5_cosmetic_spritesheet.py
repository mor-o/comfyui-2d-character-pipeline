"""Enqueue w5-cosmetic-spritesheet/workflow_api.json for a single cosmetic/animation pair.

Takes a cosmetic mp4 produced by Workflow 4 (base animation with cosmetic
painted on top — bit-exact outside the cosmetic region in pass2; pass1
also works but has halo artifacts from the dilated mask) and produces a
horizontal cosmetic-only RGBA sprite sheet. SAM3 does per-frame text-
prompted segmentation to isolate the cosmetic region; everything outside
becomes transparent.

The sheet layout matches Workflow 3's base-animation spritesheet
(NUM_FRAMES x CELL_SIZE cells, horizontal strip), so the runtime can
stack cosmetic layers over the base sheet without re-alignment.

Input discovery:
  - If SOURCE_VIDEO is set, use it directly.
  - Otherwise find the latest pass2 mp4 under
    F:/ComfyUI/output/character_pipeline/cosmetics/<cosmetic>/<animation>/pass2/,
    falling back to pass1/ if pass2 is missing.

Output:
  F:/ComfyUI/output/character_pipeline/cosmetics/<cosmetic>/<animation>/
    <cosmetic>_<animation>_spritesheet_NNNNN_.png

Run with (from the repo root):
    python drivers/run_w5_cosmetic_spritesheet.py
(or whatever ComfyUI-compatible Python interpreter you use; the script
only needs `urllib`, no extra dependencies).
"""
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8000"
INPUT_DIR = Path(r"F:/ComfyUI/input")
OUTPUT_DIR = Path(r"F:/ComfyUI/output")
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "w5-cosmetic-spritesheet" / "workflow_api.json"
COSMETICS_ROOT = OUTPUT_DIR / "character_pipeline" / "cosmetics"

# --- per-run knobs ---
COSMETIC_NAME = "green_eyes"
ANIMATION_NAME = "idle_breathing"
COSMETIC_DESCRIPTION = "eyes"  # SAM3 prompt — describe the body region, not the color
NUM_FRAMES = 33
CELL_SIZE = 128
GREYSCALE = True  # matches eye_style_2 precedent; iris layer is extracted separately from pass1 mp4
SAM_CONFIDENCE = 0.15
MASK_OFFSET = 1
MASK_SMOOTH = 1

# Explicit override — set to a full path to force-use a specific mp4 (e.g.
# a pass1 variant with a dilation tag). When None, the driver picks the
# latest pass2 mp4 under <cosmetic>/<animation>/, falling back to pass1.
# Example:
#   SOURCE_VIDEO = r"F:/ComfyUI/output/character_pipeline/cosmetics/yellow_hair/walking_right/pass1/yellow_hair_walking_right_d35_pass1_00001_.mp4"
SOURCE_VIDEO: str | None = None


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


def resolve_source_video() -> Path:
    if SOURCE_VIDEO:
        p = Path(SOURCE_VIDEO)
        if not p.exists():
            raise RuntimeError(f"SOURCE_VIDEO does not exist: {p}")
        return p
    anim_dir = COSMETICS_ROOT / COSMETIC_NAME / ANIMATION_NAME
    for sub in ("pass2", "pass1"):
        d = anim_dir / sub
        if not d.exists():
            continue
        candidates = sorted(
            [c for c in d.glob("*.mp4") if c.suffix.lower() == ".mp4"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise RuntimeError(
        f"no mp4 found under {anim_dir}/pass2 or {anim_dir}/pass1. "
        "Run W4 first or set SOURCE_VIDEO explicitly."
    )


def build_workflow(input_video_name: str) -> dict:
    """Load the base workflow and append the per-frame slice + stitch chain.

    The fixed JSON ends at node 25 (JoinImageWithAlpha -> RGBA batch). We
    append:
      100..100+N-1  ImageFromBatch per frame
      200..200+N-2  ImageStitch left-to-right
      300           SaveImage of the final strip

    Also:
      - Rewrite node 1's file to the staged video.
      - Rewrite node 3's length to NUM_FRAMES.
      - Rewrite the ImageScale width/height on both branches to CELL_SIZE.
      - Rewrite SAM3Segment + MaskEnhancer knobs.
      - Route node 25's image input to either node 28 (greyscale) or 20 (color).
    """
    wf = json.loads(WORKFLOW_PATH.read_text())
    wf["1"]["inputs"]["file"] = input_video_name
    wf["3"]["inputs"]["length"] = NUM_FRAMES

    wf["10"]["inputs"]["prompt"] = COSMETIC_DESCRIPTION
    wf["10"]["inputs"]["confidence_threshold"] = SAM_CONFIDENCE
    wf["11"]["inputs"]["mask_offset"] = MASK_OFFSET
    wf["11"]["inputs"]["smooth"] = MASK_SMOOTH

    for resize_nid in ("13", "20"):
        wf[resize_nid]["inputs"]["width"] = CELL_SIZE
        wf[resize_nid]["inputs"]["height"] = CELL_SIZE

    # Color vs greyscale: route JoinImageWithAlpha.image accordingly.
    # Alpha input stays on node 15 (InvertMask output) in both cases.
    wf["25"]["inputs"]["image"] = ["28", 0] if GREYSCALE else ["20", 0]
    wf["25"]["inputs"]["alpha"] = ["15", 0]

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
                f"character_pipeline/cosmetics/{COSMETIC_NAME}/{ANIMATION_NAME}/"
                f"{COSMETIC_NAME}_{ANIMATION_NAME}_spritesheet"
            ),
        },
        "_meta": {"title": "Save cosmetic sprite sheet"},
    }
    return wf


def main() -> None:
    src = resolve_source_video()
    staged = INPUT_DIR / f"cosmetic_{COSMETIC_NAME}_{ANIMATION_NAME}.mp4"
    shutil.copy2(src, staged)
    print(f"[W5] staged {src.name} -> {staged}")
    print(
        f"[W5] {COSMETIC_NAME}/{ANIMATION_NAME} | frames={NUM_FRAMES} | "
        f"cell={CELL_SIZE}px | greyscale={GREYSCALE} | prompt={COSMETIC_DESCRIPTION!r} | "
        f"conf={SAM_CONFIDENCE} | offset={MASK_OFFSET} | smooth={MASK_SMOOTH}"
    )
    wf = build_workflow(staged.name)
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
