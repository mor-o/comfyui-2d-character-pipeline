"""Extract the iris-only sprite sheet for a green-iris eye cosmetic.

**This is a post-processing step for eye cosmetics only.** It runs after
[Workflow 4] has produced the cosmetic pass1 mp4 and is independent of
[Workflow 5] — which still produces the full eye sprite sheet from the
same pass1 mp4 via its usual SAM3 segmentation.

Why the iris needs its own extraction:
  Eye cosmetics render in two paired layers — a full eye sheet (sclera,
  lashes, highlights, baked iris) and a separate near-white greyscale
  iris sheet on top. The iris layer is what the player's chosen eye
  colour tints at runtime (Sprite.tint multiplies a near-white source to
  a saturated target), so its pixels must be tight on the iris disc —
  no sclera, no lashes, no highlights. SAM3 with prompts like "iris" or
  "pupil" proved unreliable at isolating just that sub-region across
  frames.

How this script works:
  The cosmetic reference image is authored with a BRIGHT GREEN iris —
  a hue that's absent from sclera (white), lashes (black), and specular
  highlights (white). A global colour-key on the pass1 mp4 frames
  (`G > R + GREEN_MARGIN` AND `G > B + GREEN_MARGIN`) keeps the iris
  pixels and rejects everything else. No segmentation, no masks — the
  hue does the work. The kept pixels are flattened to a near-white
  greyscale (their luminance pulled from the green channel and lifted
  toward 1.0) so runtime tint produces bright iris colours.

Inputs (edit at top of file):
  COSMETIC_NAME   staging slug — e.g. "green_eyes"
  ANIMATION_NAME  W4 anim slug — "idle_breathing", "walking_right", "in-air"
  PASS1_MP4       (optional) explicit pass1 mp4 path, or None to auto-pick
                  the latest under character_pipeline/cosmetics/<cos>/<anim>/pass1/
  NUM_FRAMES      must match W4 (33 for idle/walking, 9 for in-air)
  CELL_SIZE       128
  GREEN_MARGIN    G - R and G - B must exceed this for a pixel to count
                  as iris
  IRIS_LIFT_MIN   min value in [0,1] for the near-white greyscale map

Output (RGBA, horizontal strip NUM_FRAMES × CELL_SIZE):
  character_pipeline/cosmetics/<cosmetic>/<animation>/<cosmetic>_<animation>_iris_spritesheet.png

Run with (from the repo root):
    python drivers/run_iris_greenkey.py
(or whatever ComfyUI-compatible Python interpreter you use; the script
only needs `urllib`, no extra dependencies).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(r"F:/ComfyUI/output")
COSMETICS_ROOT = OUTPUT_DIR / "character_pipeline" / "cosmetics"

# --- per-run knobs ---
COSMETIC_NAME = "green_eyes"
ANIMATION_NAME = "idle_breathing"
PASS1_MP4: str | None = None  # None = auto-pick latest under pass1/
NUM_FRAMES = 33
CELL_SIZE = 128

# Green-dominance margin. A pixel counts as iris if G > R + MARGIN AND
# G > B + MARGIN. 20 rejects warm sclera anti-aliasing while catching the
# darker iris edges.
GREEN_MARGIN = 20

# Iris greyscale lift. Map green-channel luminance [0, 1] → [LIFT_MIN, 1.0]
# so runtime Sprite.tint yields bright, saturated iris colours.
IRIS_LIFT_MIN = 0.55


def resolve_pass1() -> Path:
    if PASS1_MP4:
        p = Path(PASS1_MP4)
        if not p.exists():
            raise RuntimeError(f"PASS1_MP4 does not exist: {p}")
        return p
    pass1_dir = COSMETICS_ROOT / COSMETIC_NAME / ANIMATION_NAME / "pass1"
    mp4s = sorted(pass1_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError(
            f"no pass1 mp4 under {pass1_dir}. Run W4 first or set PASS1_MP4."
        )
    return mp4s[0]


def read_frames(mp4: Path, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != n:
        raise RuntimeError(f"{mp4.name}: expected {n} frames, got {len(frames)}")
    return np.stack(frames, 0)


def build_iris_sheet(frames: np.ndarray) -> np.ndarray:
    """Green-key at source res per frame, then downscale + stitch → RGBA strip.

    Keying at source resolution preserves thin iris edges that would
    otherwise blur into sclera white during a downscale. The kept
    luminance (from the green channel) is lifted toward white so the
    runtime Sprite.tint yields saturated iris colours.
    """
    r = frames[..., 0].astype(int)
    g = frames[..., 1].astype(int)
    b = frames[..., 2].astype(int)
    iris = (g > r + GREEN_MARGIN) & (g > b + GREEN_MARGIN) & (g > 40)

    v = np.clip(g.astype(float) / 255.0, 0.0, 1.0)
    v_lift = IRIS_LIFT_MIN + (1.0 - IRIS_LIFT_MIN) * v
    v_u8 = (v_lift * 255).astype(np.uint8)

    # Per-frame RGBA at source res: near-white greyscale inside iris, else 0.
    rgba_src = np.zeros(frames.shape[:-1] + (4,), dtype=np.uint8)
    rgba_src[..., 0] = v_u8
    rgba_src[..., 1] = v_u8
    rgba_src[..., 2] = v_u8
    rgba_src[..., 3] = np.where(iris, 255, 0)
    # Kill RGB on transparent pixels (keeps PNG cleaner).
    rgba_src[~iris] = 0

    # Per-frame downscale with INTER_AREA (clean alpha edges at chibi res).
    rgba_small = np.stack(
        [cv2.resize(f, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
         for f in rgba_src],
        0,
    )
    return np.concatenate(list(rgba_small), axis=1)


def main() -> int:
    pass1 = resolve_pass1()
    out_dir = COSMETICS_ROOT / COSMETIC_NAME / ANIMATION_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_iris = out_dir / f"{COSMETIC_NAME}_{ANIMATION_NAME}_iris_spritesheet.png"

    print(f"[iris] {COSMETIC_NAME}/{ANIMATION_NAME} | frames={NUM_FRAMES} | cell={CELL_SIZE}")
    print(f"  pass1: {pass1}")

    frames = read_frames(pass1, NUM_FRAMES)
    iris = build_iris_sheet(frames)
    Image.fromarray(iris, "RGBA").save(out_iris)
    a = iris[..., 3]
    print(f"  iris -> {out_iris}  ({int((a > 0).sum())} opaque / {a.size})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
