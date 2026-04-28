"""Enqueue the two-pass Workflow 4 pipeline for a single (cosmetic, animation) pair.

Two-pass bootstrap:
  Pass 1 — rough inpaint. SAM3 on the cosmetic reference image only produces a
    single static hair mask; VACE broadcasts it across all 33 frames and
    paints hair approximately everywhere it should be. Output is rough
    (blending artifacts in dilated halo zones) and feeds pass 2's SAM3.
  Pass 2 — clean inpaint. SAM3 per-frame on the pass 1 rough video
    produces precise per-frame hair masks that track character motion.
    VACE inpaints the ORIGINAL base video using these precise masks; the
    composite preserves base pixels bit-exact everywhere outside the hair.

Strict VRAM discipline (SAM3 and VACE DiT never coexist):
  Within each pass:
    - SAM3Segment runs with unload_model=true (evicts internally +
      torch.cuda.empty_cache before returning).
    - KSampler inputs are ordered {positive, negative, latent_image, model}
      — ComfyUI walks dict in insertion order, so the conditioning chain
      (which transitively triggers SAM3) executes BEFORE the model chain
      (UnetLoaderGGUF -> LoRA -> SD3Sampling).
    - VRAMUnloadClip evicts UMT5 after text encode.
    - VRAMUnloadModel evicts VACE DiT after KSampler.
  Between passes:
    - Driver POSTs /free with {unload_models:true, free_memory:true}
      to purge any residual models from pass 1 before pass 2's SAM3 loads.

Directory layout (per run):
  F:/ComfyUI/output/character_pipeline/cosmetics/{cosmetic}/{animation}/
    pass1/{cosmetic}_{animation}_pass1_NNNNN_.mp4   rough bootstrap
    pass2/{cosmetic}_{animation}_pass2_NNNNN_.mp4   final clean output
    masks/pass1_static_mask_NNNNN_.png              single static mask
    masks/pass2_precise_mask_NNNNN_.png             33 per-frame masks

Run with (from the repo root):
    python drivers/run_w4_cosmetic_inpaint.py
(or whatever ComfyUI-compatible Python interpreter you use; the script
only needs `urllib`, no extra dependencies).
"""
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8000"
INPUT_DIR = Path(r"F:/ComfyUI/input")
OUTPUT_DIR = Path(r"F:/ComfyUI/output")
PASS1_WORKFLOW = Path(__file__).parent.parent / "workflows" / "w4-cosmetic-inpaint" / "pass1_rough_api.json"
PASS2_WORKFLOW = Path(__file__).parent.parent / "workflows" / "w4-cosmetic-inpaint" / "pass2_clean_api.json"

# --- per-run knobs ---
COSMETIC_NAME = "green_eyes"
ANIMATION_NAME = "in-air"  # re-running with CG_PASS1_MASK_DILATE=70 to fix bottom-clip

# Optional variant tag. When non-empty, outputs + staged pass1 mp4 get a
# dedicated subdir/suffix so parallel experiments don't collide.
#   ""      → character_pipeline/cosmetics/{cos}/{anim}/{pass1,pass2,masks}/   (default)
#   "p1d25" → character_pipeline/cosmetics/{cos}/{anim}/p1d25/{pass1,pass2,masks}/
RUN_TAG = os.environ.get("CG_RUN_TAG", "")

# When set to an existing pass 1 mp4 path, skip pass 1 entirely and reuse
# that output as pass 2's SAM3 input. Useful for iterating on pass 2
# params (SAM_CONFIDENCE, PASS2_MASK_DILATE) without rerunning the
# expensive pass 1 inference. Set to None to run pass 1 normally.
PASS1_MP4_OVERRIDE = None

# Pass 2 is optional — it's a polish step that tightens the mask around the
# actual cosmetic pixels, eliminating any blending from the pass 1 dilation
# halo. Set False when pass 1 output already looks clean (no cosmetic-
# background merging, no halo glow). Set True when you see pass 1 bleeding
# cosmetic color into adjacent skin/body/background. See the workflow doc's
# "Chosen: two-pass bootstrap (pass 2 optional)" section.
RUN_PASS2 = False

SOURCE_VIDEO = Path(
    r"F:/ComfyUI/output/character_pipeline/base_animations/in-air/in-air_00001_.mp4"
)
COSMETIC_REF = Path(
    r"F:/ComfyUI/output/character_pipeline/cosmetics/green-eyes_512.png"
)

COSMETIC_DESCRIPTION = "eyes, eyebrows"
# Lower = more generous segmentation, catches thinner strands at the cost
# of possibly including body/background pixels. Range 0.05-0.95. Default
# 0.3 misses wispy outer strands (front-hair, ear-obscuring); 0.2 usually
# catches them.
SAM_CONFIDENCE = 0.2

# Dilation knobs — see workflow-4-cosmetic-inpaint.md "Mask dilation" section
# for tuning guidance. Pass 1 has wide tolerance (dilation doesn't appear
# in the final output); pass 2 is the sensitive one.
PASS1_MASK_DILATE = int(os.environ.get("CG_PASS1_MASK_DILATE", "25"))
PASS1_MASK_SMOOTH = 2
PASS2_MASK_DILATE = 10
PASS2_MASK_SMOOTH = 1

SEED = int(os.environ.get("CG_SEED", "12345"))
NUM_FRAMES = 9
FPS = 16
# Steps are split per pass. At LoRA 0.8, 4 steps regressed visibly from 6
# (measured 2026-04-20) — pass 1 stays at 6 minimum. Pass 2 stays at 4
# because it's structural replacement inside a precise mask, not new
# detail generation.
PASS1_STEPS = 10
PASS2_STEPS = 4
CFG = 1.0
SHIFT = 5
STRENGTH = 1.0
LORA_STRENGTH = 1.0

# Pass 1 dual-sampler knobs (CausVid V2 + high-CFG early steps, per Kijai
# discussion #27). Sampler 1 runs steps 0→PASS1_CFG_SPLIT_STEP at
# PASS1_CFG_HIGH for reference-fidelity pull; sampler 2 runs
# PASS1_CFG_SPLIT_STEP→PASS1_STEPS at CFG (= 1.0) for CausVid-distilled
# denoise. Pass 2 still uses the single KSampler in its workflow JSON.
PASS1_CFG_HIGH = 4.0
PASS1_CFG_SPLIT_STEP = 4

POS_PROMPT = (
    "cute chibi character with large anime eyes, bright green irises, round "
    "sclera, small white specular highlights, crisp clean lineart, flat "
    "cel-shaded coloring"
)

NEG_PROMPT = (
    "blurry, distorted, artifacts, warping, jitter, color shift, ghosting, "
    "deformed, flicker, closed eyes, squinting, different character"
)

assert (NUM_FRAMES - 1) % 4 == 0, (
    f"NUM_FRAMES={NUM_FRAMES} violates WAN's 4n+1 rule (valid: 9, 13, 17, 21, 25, 33, 49, 81)"
)


def post_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{COMFY_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read().decode() or "{}")


def post_prompt(workflow: dict) -> str:
    resp = post_json("/prompt", {"prompt": workflow})
    if "prompt_id" not in resp:
        raise RuntimeError(f"comfyui rejected workflow: {resp}")
    return resp["prompt_id"]


def free_memory(label: str) -> None:
    """Purge all resident models. Abort on failure — the SAM3/VACE
    non-coexistence invariant depends on this."""
    try:
        post_json("/free", {"unload_models": True, "free_memory": True})
        print(f"[W4] {label}: /free OK (models + memory cleared)")
    except Exception as e:
        raise RuntimeError(
            f"/free failed at '{label}': {e}. Aborting — running without "
            "/free risks SAM3 and VACE DiT coexisting in VRAM (OOM)."
        )


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
        time.sleep(2)  # always sleep, including on the HTTP-error path


def _tag_suffix() -> str:
    """File-name suffix that segregates variants (`_p1d25`, etc.); empty by default."""
    return f"_{RUN_TAG}" if RUN_TAG else ""


def _output_root() -> str:
    """ComfyUI SaveImage/SaveVideo filename_prefix root for this run."""
    root = f"character_pipeline/cosmetics/{COSMETIC_NAME}/{ANIMATION_NAME}"
    if RUN_TAG:
        root = f"{root}/{RUN_TAG}"
    return root


def _pass1_staged_name() -> str:
    return f"pass1_{ANIMATION_NAME}{_tag_suffix()}.mp4"


def clear_stale_staged_pass1() -> None:
    """Remove any leftover pass1 staged file from a previous run. Prevents
    silent reuse if a rerun aborts before staging a fresh pass1 output."""
    stale = INPUT_DIR / _pass1_staged_name()
    if stale.exists():
        stale.unlink()
        print(f"[W4] removed stale staged pass1: input/{stale.name}")


def output_files(history_entry: dict, node_id: str) -> list[Path]:
    """Return all files saved by a specific node in the history entry.

    Filtering by node id (instead of by position or extension) is
    important: SaveImage nodes for mask debug also appear in outputs, and
    depending on them being "last" is fragile.
    """
    files: list[Path] = []
    node_outputs = history_entry.get("outputs", {}).get(node_id, {})
    for key in ("videos", "images"):
        for spec in node_outputs.get(key, []) or []:
            sub = spec.get("subfolder", "")
            fn = spec.get("filename")
            if fn:
                files.append(OUTPUT_DIR / sub / fn)
    return files


# Node ids for the SaveVideo nodes in each pass workflow (see JSONs).
PASS1_SAVEVIDEO_NID = "28"
PASS2_SAVEVIDEO_NID = "28"


def stage_inputs() -> tuple[str, str]:
    """Copy base video + cosmetic ref into F:/ComfyUI/input/ with normalized names.
    Also removes any stale pass1 staged file so a partial rerun cannot reuse it."""
    if not SOURCE_VIDEO.exists():
        raise RuntimeError(f"SOURCE_VIDEO not found: {SOURCE_VIDEO}")
    if not COSMETIC_REF.exists():
        raise RuntimeError(f"COSMETIC_REF not found: {COSMETIC_REF}")

    clear_stale_staged_pass1()

    staged_video = INPUT_DIR / f"base_{ANIMATION_NAME}.mp4"
    staged_ref = INPUT_DIR / f"cosmetic_ref_{COSMETIC_NAME}{COSMETIC_REF.suffix.lower()}"

    shutil.copy2(SOURCE_VIDEO, staged_video)
    shutil.copy2(COSMETIC_REF, staged_ref)
    print(f"[W4] staged base: {SOURCE_VIDEO.name} -> input/{staged_video.name}")
    print(f"[W4] staged ref:  {COSMETIC_REF.name} -> input/{staged_ref.name}")
    return staged_video.name, staged_ref.name


def stage_pass1_for_pass2(pass1_mp4: Path) -> str:
    """Copy pass1 output mp4 into F:/ComfyUI/input/ so pass2's LoadVideo can read it."""
    staged = INPUT_DIR / _pass1_staged_name()
    shutil.copy2(pass1_mp4, staged)
    print(f"[W4] staged pass1 -> input/{staged.name}")
    return staged.name


def build_pass1(base_video: str, cosmetic_ref: str) -> dict:
    wf = json.loads(PASS1_WORKFLOW.read_text())
    wf["1"]["inputs"]["file"] = base_video
    wf["5"]["inputs"]["image"] = cosmetic_ref
    wf["7"]["inputs"]["prompt"] = COSMETIC_DESCRIPTION
    wf["7"]["inputs"]["confidence_threshold"] = SAM_CONFIDENCE
    wf["10"]["inputs"]["mask_offset"] = PASS1_MASK_DILATE
    wf["10"]["inputs"]["smooth"] = PASS1_MASK_SMOOTH
    wf["12"]["inputs"]["filename_prefix"] = f"{_output_root()}/masks/pass1_static_mask"
    wf["14"]["inputs"]["text"] = POS_PROMPT
    wf["15"]["inputs"]["text"] = NEG_PROMPT
    wf["18"]["inputs"]["length"] = NUM_FRAMES
    wf["18"]["inputs"]["strength"] = STRENGTH
    wf["20"]["inputs"]["strength_model"] = LORA_STRENGTH
    wf["21"]["inputs"]["shift"] = SHIFT
    # Sampler 1: high CFG, early steps (reference-fidelity pull).
    wf["22"]["inputs"]["noise_seed"] = SEED
    wf["22"]["inputs"]["steps"] = PASS1_STEPS
    wf["22"]["inputs"]["cfg"] = PASS1_CFG_HIGH
    wf["22"]["inputs"]["start_at_step"] = 0
    wf["22"]["inputs"]["end_at_step"] = PASS1_CFG_SPLIT_STEP
    # Sampler 2: low CFG (CausVid-distilled), remaining steps.
    wf["29"]["inputs"]["noise_seed"] = SEED
    wf["29"]["inputs"]["steps"] = PASS1_STEPS
    wf["29"]["inputs"]["cfg"] = CFG
    wf["29"]["inputs"]["start_at_step"] = PASS1_CFG_SPLIT_STEP
    wf["29"]["inputs"]["end_at_step"] = PASS1_STEPS
    wf["27"]["inputs"]["fps"] = FPS
    wf["28"]["inputs"]["filename_prefix"] = (
        f"{_output_root()}/pass1/{COSMETIC_NAME}_{ANIMATION_NAME}{_tag_suffix()}_pass1"
    )
    return wf


def build_pass2(base_video: str, pass1_video: str, cosmetic_ref: str) -> dict:
    wf = json.loads(PASS2_WORKFLOW.read_text())
    wf["1"]["inputs"]["file"] = base_video
    wf["4"]["inputs"]["file"] = pass1_video
    wf["7"]["inputs"]["image"] = cosmetic_ref
    wf["9"]["inputs"]["prompt"] = COSMETIC_DESCRIPTION
    wf["9"]["inputs"]["confidence_threshold"] = SAM_CONFIDENCE
    wf["10"]["inputs"]["mask_offset"] = PASS2_MASK_DILATE
    wf["10"]["inputs"]["smooth"] = PASS2_MASK_SMOOTH
    wf["12"]["inputs"]["filename_prefix"] = f"{_output_root()}/masks/pass2_precise_mask"
    wf["14"]["inputs"]["text"] = POS_PROMPT
    wf["15"]["inputs"]["text"] = NEG_PROMPT
    wf["18"]["inputs"]["length"] = NUM_FRAMES
    wf["18"]["inputs"]["strength"] = STRENGTH
    wf["20"]["inputs"]["strength_model"] = LORA_STRENGTH
    wf["21"]["inputs"]["shift"] = SHIFT
    wf["22"]["inputs"]["seed"] = SEED
    wf["22"]["inputs"]["steps"] = PASS2_STEPS
    wf["22"]["inputs"]["cfg"] = CFG
    wf["27"]["inputs"]["fps"] = FPS
    wf["28"]["inputs"]["filename_prefix"] = (
        f"{_output_root()}/pass2/{COSMETIC_NAME}_{ANIMATION_NAME}{_tag_suffix()}_pass2"
    )
    return wf


def main() -> int:
    base_video, cosmetic_ref = stage_inputs()

    print(
        f"[W4] {COSMETIC_NAME} x {ANIMATION_NAME} | concept='{COSMETIC_DESCRIPTION}' | "
        f"seed={SEED} | frames={NUM_FRAMES} | "
        f"dilate=pass1:{PASS1_MASK_DILATE}px / pass2:{PASS2_MASK_DILATE}px"
    )

    # --- Pass 1 (or skip, if override provided) ---
    if PASS1_MP4_OVERRIDE is not None:
        if not PASS1_MP4_OVERRIDE.exists():
            raise RuntimeError(f"PASS1_MP4_OVERRIDE not found: {PASS1_MP4_OVERRIDE}")
        pass1_mp4 = PASS1_MP4_OVERRIDE
        print(f"[W4] === pass 1 skipped; reusing {pass1_mp4.name} ===")
    else:
        free_memory("before pass1")
        print("[W4] === pass 1: rough inpaint for mask bootstrap ===")
        t1 = time.time()
        wf1 = build_pass1(base_video, cosmetic_ref)
        pid1 = post_prompt(wf1)
        print(f"  pass1 enqueued: {pid1}")
        hist1 = wait_for(pid1)
        pass1_videos = output_files(hist1, PASS1_SAVEVIDEO_NID)
        if not pass1_videos:
            raise RuntimeError(f"pass1 node {PASS1_SAVEVIDEO_NID} produced no video")
        pass1_mp4 = pass1_videos[0]
        print(f"  pass1 output: {pass1_mp4}  (wall: {time.time()-t1:.1f}s)")

    if not RUN_PASS2:
        print("[W4] === pass 2 skipped (RUN_PASS2=False); stopping after pass 1 ===")
        print(f"  pass1 output: {pass1_mp4}")
        return 0

    # --- Between-pass staging + /free ---
    pass1_staged_name = stage_pass1_for_pass2(pass1_mp4)

    # --- Pass 2 ---
    free_memory("between passes")
    print("[W4] === pass 2: clean inpaint with precise per-frame masks ===")
    t2 = time.time()
    wf2 = build_pass2(base_video, pass1_staged_name, cosmetic_ref)
    pid2 = post_prompt(wf2)
    print(f"  pass2 enqueued: {pid2}")
    hist2 = wait_for(pid2)
    pass2_videos = output_files(hist2, PASS2_SAVEVIDEO_NID)
    if not pass2_videos:
        raise RuntimeError(f"pass2 node {PASS2_SAVEVIDEO_NID} produced no video")
    pass2_mp4 = pass2_videos[0]
    print(f"  pass2 output: {pass2_mp4}  (wall: {time.time()-t2:.1f}s)")

    print("[W4] done")
    print(f"  final deliverable: {pass2_mp4}")
    print(f"  pass1 bootstrap:   {pass1_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
