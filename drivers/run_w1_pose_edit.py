"""Enqueue w1-pose-edit/workflow_api.json for a single animation.

Drives Workflow 1 of the ComfyUI animation generation pipeline:
Qwen-Image-Edit 2509 (Q5_K_M GGUF) + InstantX Qwen-Image ControlNet-Union
on the OpenPose sub-type. Takes a character PNG plus a pre-rendered
OpenPose skeleton PNG and produces a single posed keyframe.

The VL text encoder (~7 GB) and the Qwen-Edit UNET (~15 GB) never sit
in VRAM at the same time: VRAMUnloadClip drops the encoder after text
encoding, VRAMUnloadModel drops the UNET after sampling. Re-enqueuing
with a fresh seed reuses the cached conditionings and only re-runs the
KSampler + UNET unload; a prompt change invalidates the loader chain
so the encoder reloads fresh on the next run.

Inputs expected at:
    F:/ComfyUI/input/<SOURCE_IMAGE>
    F:/ComfyUI/input/<POSE_SKELETON>

Output saved at:
    F:/ComfyUI/output/character_pipeline/base_animations/<resolved_name>/poses/
        posed_<resolved_name>_NNNNN_.png

Run with (from the repo root):
    python drivers/run_w1_pose_edit.py
(or whatever ComfyUI-compatible Python interpreter you use; the script
only needs `urllib`, no extra dependencies).

Usage rules:
- Character must face RIGHT (side profile, nose pointing to viewer's right).
- Keyframe must have NO ground shadow, NO floor, NO cast shadow.
- If POSE_SKELETON and source match those rules, seed the run. If not,
  re-roll the seed up to ~10 times. If ~10 seeds all fail the same row,
  fix the pose JSON or positive prompt, then seed-roll again.
- Each run creates a fresh versioned animation dir. If
  base_animations/<BASE>/ exists, W1 saves into base_animations/<BASE>_2/;
  then <BASE>_3, <BASE>_4, etc. W2 and W3 auto-pick up the LATEST
  versioned dir for this base name.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8000"
INPUT_DIR = Path(r"F:/ComfyUI/input")
OUTPUT_DIR = Path(r"F:/ComfyUI/output")
WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "w1-pose-edit" / "workflow_api.json"
ANIM_ROOT = OUTPUT_DIR / "character_pipeline" / "base_animations"

# --- per-run knobs (edit for each animation) ---
ANIMATION_NAME_BASE = "idle_breathing"
SOURCE_IMAGE = "source_idle_breathing.png"
POSE_SKELETON = "pose_idle_breathing.png"
SEED = 30123
CONTROLNET_STRENGTH = 1.6

POS_PROMPT = (
    "The same chibi cartoon character from the reference image in a "
    "relaxed idle standing pose, oriented toward the right (the "
    "character's body and nose point to the viewer's right) in a gentle "
    "3/4 view - the character is mostly pointed right but the face plane "
    "is rotated only about 70 degrees from the camera, NOT a full 90-"
    "degree side profile. Arms hang relaxed at the sides. Feet planted "
    "at the bottom of the frame. Keep the identical character identity, "
    "skin tone, clean black line art outlines, flat colors, pure white "
    "empty background. "
    "CRITICAL - the head interior must be COMPLETELY FEATURELESS: a "
    "perfectly smooth blank round bald scalp with absolutely NO face, "
    "NO eye, NO eyes, NO eye dots, NO pupils, NO iris, NO eyeballs, NO "
    "mouth, NO lip, NO lips, NO teeth, NO tongue, NO nostril, NO ear, "
    "NO ear curve, NO C-shaped line on the head, NO curves drawn inside "
    "the head outline, NO blush, NO cheek marks, NO pink patch, NO peach "
    "patch, NO freckles, NO eyebrows, NO lashes, NO facial feature of "
    "any kind whatsoever. The inside of the head is an unbroken "
    "smooth expanse of the same flat skin-tone color - only the outer "
    "head silhouette contour line is present. The only protrusion is a "
    "tiny unmarked nose bump on the right outline. "
    "Body is naked with no clothing, no hair, no accessories. The "
    "character floats freely against a pure white background with "
    "ABSOLUTELY NO shadow, NO ground shadow, NO drop shadow, NO cast "
    "shadow, NO dark oval or ellipse under the feet, NO ground, NO "
    "floor, NO horizon, NO surface - pure uniform white pixels continue "
    "beneath and around the feet."
)

# Non-empty negative prompt works in this stack (unlike ConditioningZeroOut,
# which collapses the model). Lists the facial features we keep seeing leak
# despite a strong positive.
NEG_PROMPT = (
    "face, eyes, eye, eyeball, eyeballs, pupil, pupils, iris, mouth, "
    "lip, lips, teeth, tongue, nose detail, nostril, ear, ears, ear curve, "
    "C-shape on head, curved line on head, eyebrow, eyebrows, eyelash, "
    "eyelashes, blush, blush mark, blush marks, pink cheek, pink cheeks, "
    "peach cheek, peach cheeks, cheek dot, cheek dots, freckle, freckles, "
    "facial features, makeup, expression, smile, frown, hair, hat, "
    "clothing, clothes, shirt, pants, shoes, accessories, jewelry, "
    "shadow, drop shadow, cast shadow, ground shadow, floor shadow, "
    "shading below feet, dark shape under character, ellipse on ground, "
    "ground, floor, horizon, surface"
)


def _dir_has_completed_video(d: Path) -> bool:
    """A dir 'completed' a pipeline run if it contains a *.mp4 — W2 has run."""
    return any(d.glob("*.mp4"))


def _version_index(name: str, base: str) -> int:
    """Return the numeric version of an animation dir name.

    <base>     -> 1
    <base>_2   -> 2
    <base>_17  -> 17
    anything else -> 0 (not a match)
    """
    if name == base:
        return 1
    m = re.match(rf"^{re.escape(base)}_(\d+)$", name)
    return int(m.group(1)) if m else 0


def resolve_next_animation_name(base: str) -> str:
    """Return the target versioned dir name for this W1 run.

    Rule:
      - If there is a dir matching <base> or <base>_N that exists but has
        no .mp4, stay on the highest-numbered such dir. (W1 seed re-rolls
        in an in-progress run — don't create a new version.)
      - Otherwise, create the next unused <base>_N slot (first N such that
        the dir does not exist).
    """
    in_progress: list[tuple[int, str]] = []
    max_existing = 0
    if ANIM_ROOT.exists():
        for child in ANIM_ROOT.iterdir():
            if not child.is_dir():
                continue
            idx = _version_index(child.name, base)
            if idx == 0:
                continue
            max_existing = max(max_existing, idx)
            if not _dir_has_completed_video(child):
                in_progress.append((idx, child.name))
    if in_progress:
        in_progress.sort(key=lambda p: p[0])
        return in_progress[-1][1]
    next_idx = max_existing + 1
    return base if next_idx == 1 else f"{base}_{next_idx}"


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


def wait_for(prompt_id: str, timeout_s: int = 900) -> dict:
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


def build_workflow(animation_name: str) -> dict:
    wf = json.loads(WORKFLOW_PATH.read_text())

    # Match nodes by filename prefix / title instead of node id order — the
    # JSON can be re-exported with arbitrary ids and we must not silently
    # swap source vs pose (would produce a posed skeleton, not a character).
    source_load = pose_load = pos_text = neg_text = sampler = save = cnet_apply = None
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct == "LoadImage":
            img = node["inputs"].get("image", "")
            if img.startswith("source_"):
                source_load = nid
            elif img.startswith("pose_"):
                pose_load = nid
        elif ct == "TextEncodeQwenImageEditPlus":
            title = node.get("_meta", {}).get("title", "").lower()
            if "positive" in title:
                pos_text = nid
            elif "negative" in title:
                neg_text = nid
        elif ct == "KSampler":
            sampler = nid
        elif ct == "SaveImage":
            save = nid
        elif ct == "ControlNetApplyAdvanced":
            cnet_apply = nid
    missing = [
        n for n, v in [
            ("source_load", source_load), ("pose_load", pose_load),
            ("pos_text", pos_text), ("neg_text", neg_text),
            ("sampler", sampler), ("save", save), ("cnet_apply", cnet_apply),
        ] if v is None
    ]
    if missing:
        raise RuntimeError(
            f"workflow JSON missing expected nodes: {missing}. "
            "LoadImage inputs must start with source_ / pose_; "
            "TextEncodeQwenImageEditPlus _meta.title must contain 'positive' / 'negative'."
        )

    wf[source_load]["inputs"]["image"] = SOURCE_IMAGE
    wf[pose_load]["inputs"]["image"] = POSE_SKELETON
    wf[pos_text]["inputs"]["prompt"] = POS_PROMPT
    wf[neg_text]["inputs"]["prompt"] = NEG_PROMPT
    wf[sampler]["inputs"]["seed"] = SEED
    wf[cnet_apply]["inputs"]["strength"] = CONTROLNET_STRENGTH
    wf[save]["inputs"]["filename_prefix"] = (
        f"character_pipeline/base_animations/{animation_name}/poses/"
        f"posed_{animation_name}"
    )
    return wf


def main() -> int:
    # CLI overrides for fast seed-sweeping without rewriting the file.
    #   python run_w1_pose_edit.py [SEED] [CNET_STRENGTH]
    global SEED, CONTROLNET_STRENGTH
    argv = sys.argv[1:]
    if argv:
        SEED = int(argv[0])
    if len(argv) >= 2:
        CONTROLNET_STRENGTH = float(argv[1])

    animation_name = resolve_next_animation_name(ANIMATION_NAME_BASE)
    (ANIM_ROOT / animation_name / "poses").mkdir(parents=True, exist_ok=True)
    print(
        f"[W1] {animation_name} | src={SOURCE_IMAGE} | pose={POSE_SKELETON} | "
        f"seed={SEED} | cnet={CONTROLNET_STRENGTH}"
    )
    wf = build_workflow(animation_name)
    pid = post_prompt(wf)
    print(f"  enqueued: {pid}")
    history = wait_for(pid)
    saved = output_files(history)
    if not saved:
        raise RuntimeError("no output produced")
    for p in saved:
        print(f"  saved -> {p}")
    print(f"  animation_dir -> {ANIM_ROOT / animation_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
