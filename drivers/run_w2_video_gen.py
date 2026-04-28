"""Enqueue w2-video-gen/workflow_api.json for a single animation.

Takes the posed keyframe from Workflow 1 and generates a short mp4
using WAN 2.2 14B i2v (Q5_K_M GGUF) with two-stage MoE eviction.

Resolves ANIMATION_NAME automatically: finds the LATEST versioned dir
matching ANIMATION_NAME_BASE under F:/ComfyUI/output/character_pipeline/
base_animations/. If only <BASE> exists, uses that; if <BASE>_2 / <BASE>_3
exist, uses the highest numbered one. This pairs with W1's auto-versioning.

The keyframe is auto-discovered from that dir's poses/ folder (the latest
posed_*.png) and staged to F:/ComfyUI/input/ as
keyframe_<resolved_name>.png.

Output saved at:
    F:/ComfyUI/output/character_pipeline/base_animations/<resolved_name>/
        <resolved_name>_NNNNN_.mp4

Run with (from the repo root):
    python drivers/run_w2_video_gen.py
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
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "w2-video-gen" / "workflow_api.json"
ANIM_ROOT = OUTPUT_DIR / "character_pipeline" / "base_animations"

# --- per-run knobs ---
ANIMATION_NAME_BASE = "idle_breathing"
SEED = 92003
NUM_FRAMES = 33
FPS = 16

# WAN 2.2 requires length satisfies 4n+1 (9, 13, 17, 21, 25, 33, 49, 81, ...).
# Catch here instead of after ~2 min inside the KSampler.
assert (NUM_FRAMES - 1) % 4 == 0, (
    f"NUM_FRAMES={NUM_FRAMES} violates WAN's 4n+1 rule "
    "(valid: 9, 13, 17, 21, 25, 33, 49, 81)"
)

POS_PROMPT = (
    "A small chibi cartoon character stands perfectly still in place on a "
    "pure white background in a relaxed idle standing pose. The character "
    "is almost completely static — a near-frozen held pose with only the "
    "tiniest barely perceptible micro-motion, on the threshold of no "
    "motion at all. The arms hang relaxed at the sides and do not move, "
    "the head holds its position exactly, the torso stays still. The feet "
    "stay planted exactly in place and never move; the character does not "
    "bend, does not crouch, does not lean, does not turn, does not sway, "
    "does not walk, does not jump. Facing stays constant throughout — the "
    "character keeps the same side-profile direction in every frame. "
    "Extremely minimal motion, nearly static held pose, smooth and "
    "continuous, no jitter, no flicker, clean black line art outlines, "
    "flat colors, consistent lighting, pure white empty background."
)

NEG_PROMPT = (
    "heavy breathing, exaggerated breathing, deep breathing, panting, "
    "pronounced chest expansion, pronounced chest rise, large chest "
    "movement, inflating chest, deflating chest, heaving chest, "
    "swaying, bobbing, rocking, weight shifting, shifting weight, "
    "arm swing, arm swaying, head bobbing, head turning, head tilting, "
    "bending, crouching, leaning, squatting, kneeling, sitting, walking, "
    "jumping, turning, rotating, flipping, spinning, lunging, stretching, "
    "jerky motion, abrupt movement, sudden motion, twitching, shaking, "
    "flicker, jitter, warping, distorted, extra limbs, mangled anatomy, "
    "color shift, artifacts, blurry, face, eyes, mouth, nose, ears, blush, "
    "blush marks, pink cheeks, peach cheeks, cheek dots, freckles, facial "
    "features, hair, hat, clothing, clothes, shirt, pants, shoes, "
    "accessories, jewelry, shadow, drop shadow, cast shadow, ground "
    "shadow, floor shadow, shading below feet, dark shape under character, "
    "ellipse on ground, ground, floor, horizon, surface"
)


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
            "Run W1 first."
        )
    candidates.sort(key=lambda p: p[0])
    return candidates[-1][1]


def latest_posed_keyframe(animation_name: str) -> Path:
    poses_dir = ANIM_ROOT / animation_name / "poses"
    if not poses_dir.exists():
        raise RuntimeError(f"poses dir missing: {poses_dir}. Run W1 first.")
    candidates = sorted(
        poses_dir.glob(f"posed_{animation_name}*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"no posed_*.png in {poses_dir}. Run W1 first (or ANIMATION_NAME_BASE mismatch)."
        )
    return candidates[0]


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


def wait_for(prompt_id: str, timeout_s: int = 1800) -> dict:
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
        if time.time() - last_log > 20:
            print(f"  ...waiting for {prompt_id}")
            last_log = time.time()
        time.sleep(2)


def output_files(history_entry: dict) -> list[Path]:
    files: list[Path] = []
    for node_outputs in history_entry.get("outputs", {}).values():
        for key in ("videos", "images"):
            for spec in node_outputs.get(key, []) or []:
                sub = spec.get("subfolder", "")
                fn = spec.get("filename")
                if fn:
                    files.append(OUTPUT_DIR / sub / fn)
    return files


def build_workflow(animation_name: str, keyframe_image: str) -> dict:
    wf = json.loads(WORKFLOW_PATH.read_text())

    # Match by _meta.title rather than node id / prompt text — text content
    # can be edited by the user in a way that fools a content heuristic.
    # Titles are stable in the JSON we author.
    load_image = save_video = cond_node = pos_text = neg_text = None
    sampler_nodes_list: list[str] = []
    for nid, node in wf.items():
        ct = node.get("class_type")
        title = node.get("_meta", {}).get("title", "").lower()
        if ct == "LoadImage":
            load_image = nid
        elif ct == "CLIPTextEncode":
            if "positive" in title:
                pos_text = nid
            elif "negative" in title:
                neg_text = nid
        elif ct == "KSamplerAdvanced":
            sampler_nodes_list.append(nid)
        elif ct == "SaveVideo":
            save_video = nid
        elif ct == "WanImageToVideo":
            cond_node = nid

    missing = [
        n for n, v in [
            ("load_image", load_image), ("pos_text", pos_text),
            ("neg_text", neg_text), ("cond_node", cond_node),
            ("save_video", save_video),
        ] if v is None
    ]
    if missing or not sampler_nodes_list:
        raise RuntimeError(
            f"workflow JSON missing expected nodes: {missing or 'no KSamplerAdvanced'}. "
            "CLIPTextEncode _meta.title must contain 'positive' / 'negative'."
        )

    wf[load_image]["inputs"]["image"] = keyframe_image
    wf[pos_text]["inputs"]["text"] = POS_PROMPT
    wf[neg_text]["inputs"]["text"] = NEG_PROMPT
    for nid in sampler_nodes_list:
        wf[nid]["inputs"]["noise_seed"] = SEED
    wf[cond_node]["inputs"]["length"] = NUM_FRAMES
    wf[save_video]["inputs"]["filename_prefix"] = (
        f"character_pipeline/base_animations/{animation_name}/{animation_name}"
    )
    return wf


def main() -> int:
    animation_name = resolve_latest_animation_name(ANIMATION_NAME_BASE)
    src_keyframe = latest_posed_keyframe(animation_name)
    staged_name = f"keyframe_{animation_name}.png"
    staged_path = INPUT_DIR / staged_name
    shutil.copy2(src_keyframe, staged_path)

    print(
        f"[W2] {animation_name} | keyframe={src_keyframe.name} -> {staged_name} | "
        f"seed={SEED} | frames={NUM_FRAMES} | fps={FPS}"
    )
    wf = build_workflow(animation_name, staged_name)
    pid = post_prompt(wf)
    print(f"  enqueued: {pid}")
    history = wait_for(pid)
    saved = output_files(history)
    if not saved:
        raise RuntimeError("no output produced")
    for p in saved:
        print(f"  saved -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
