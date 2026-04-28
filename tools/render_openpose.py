"""Render OpenPose-standard skeleton PNGs from canonical pose JSON files.

Matches the output of `controlnet_aux.util.draw_bodypose()` byte-for-byte:
thin limbs (stickwidth=4), 60% opacity limb overlay, 4px joint circles,
COCO-18 keypoint order. This is the exact algorithm the ComfyUI OpenPose
preprocessor uses, so skeletons produced here look identical to what
ControlNet was trained on — no bespoke rendering, no hand-tuned line
widths. It also reads the same JSON shape as openposeai.com and other
OpenPose editors, so poses can be authored visually and dropped in.

Input JSON (OpenPose standard):
    {
      "canvas_width": 512,
      "canvas_height": 512,
      "people": [
        {"pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ..., x17,y17,c17]}
      ]
    }

Keypoints are COCO-18:
    0 nose        9  r_knee
    1 neck        10 r_ankle
    2 r_shoulder  11 l_hip
    3 r_elbow     12 l_knee
    4 r_wrist     13 l_ankle
    5 l_shoulder  14 r_eye
    6 l_elbow     15 l_eye
    7 l_wrist     16 r_ear
    8 r_hip       17 l_ear

Confidence <= 0 marks a missing keypoint (skipped in rendering).

Usage:
    python render_openpose.py <pose.json> [out.png]
    python render_openpose.py <dir/>            # render every *.json in dir
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LIMB_SEQ: list[tuple[int, int]] = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
LIMB_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
JOINT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
]

STICKWIDTH = 4
JOINT_RADIUS = 4
LIMB_OPACITY = 0.6
ELLIPSE_POLY_STEPS = 36


def _ellipse_polygon(
    cx: float, cy: float, length: float, stick_w: int, angle_deg: float
) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    pts: list[tuple[float, float]] = []
    for i in range(ELLIPSE_POLY_STEPS):
        t = 2 * math.pi * i / ELLIPSE_POLY_STEPS
        ex = length / 2 * math.cos(t)
        ey = stick_w * math.sin(t)
        pts.append((ex * cos_a - ey * sin_a + cx, ex * sin_a + ey * cos_a + cy))
    return pts


def render_pose(
    keypoints: list[tuple[float, float, float] | None],
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    for (a, b), color in zip(LIMB_SEQ, LIMB_COLORS):
        ka, kb = keypoints[a], keypoints[b]
        if ka is None or kb is None:
            continue
        xa, ya = ka[0], ka[1]
        xb, yb = kb[0], kb[1]
        length = math.hypot(xb - xa, yb - ya)
        if length < 1:
            continue
        angle = math.degrees(math.atan2(yb - ya, xb - xa))
        cx, cy = (xa + xb) / 2, (ya + yb) / 2
        poly = _ellipse_polygon(cx, cy, length, STICKWIDTH, angle)

        layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        ImageDraw.Draw(layer).polygon(poly, fill=color)
        layer_arr = np.array(layer, dtype=np.float32)
        mask = (layer_arr.sum(axis=2) > 0)[..., None]
        canvas = np.where(
            mask,
            canvas * (1 - LIMB_OPACITY) + layer_arr * LIMB_OPACITY,
            canvas,
        )

    img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for kp, color in zip(keypoints, JOINT_COLORS):
        if kp is None:
            continue
        x, y = kp[0], kp[1]
        r = JOINT_RADIUS
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


def load_pose_json(
    path: Path,
) -> tuple[list[tuple[float, float, float] | None], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    canvas_w = int(data.get("canvas_width", 512))
    canvas_h = int(data.get("canvas_height", 512))
    flat = data["people"][0]["pose_keypoints_2d"]
    if len(flat) != 18 * 3:
        raise ValueError(
            f"{path}: expected 54 values (18 keypoints × 3), got {len(flat)}"
        )
    kps: list[tuple[float, float, float] | None] = []
    for i in range(18):
        x, y, c = flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]
        kps.append(None if c <= 0 else (float(x), float(y), float(c)))
    return kps, canvas_w, canvas_h


def render_file(in_path: Path, out_path: Path) -> None:
    kps, w, h = load_pose_json(in_path)
    img = render_pose(kps, w, h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  {in_path.name} -> {out_path}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: render_openpose.py <pose.json|dir> [out.png]")
        return 1
    target = Path(argv[1])
    if target.is_dir():
        for json_path in sorted(target.glob("*.json")):
            render_file(json_path, json_path.with_suffix(".png"))
    else:
        out = Path(argv[2]) if len(argv) > 2 else target.with_suffix(".png")
        render_file(target, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
