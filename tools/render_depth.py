"""Render a sparse hand/foot depth companion map for an OpenPose skeleton JSON.

Matches the `depth_pose` output style openposeai.com emits alongside its
OpenPose skeleton: only the hands and feet are rendered as small filled
volumes with grayscale shading by depth, everything else stays pure
black. This is deliberate:

- The InstantX Qwen-Image ControlNet-Union's depth sub-type was trained
  on dense volumetric depth (MiDaS / ZoeDepth output), so a sparse map
  with only a few bright/dark regions is read as "the rest of the scene
  has no depth constraint" rather than as a thin-line skeleton depth.
- Hands and feet carry the bulk of the directional information — if we
  can communicate "near hand here, far hand here, near foot here, far
  foot here" the ControlNet can disambiguate which side of the body
  faces the camera.
- Sparse rendering avoids the noise introduced by a full depth-shaded
  skeleton (which caused seed 46003's start keyframe to flip left and
  leak a face on its first depth-stacked run).

Depth values come from the JSON's `pose_keypoints_z` field (parallel to
`pose_keypoints_2d`, one float per keypoint in [0, 1] where 0 = far and
1 = near). Only indices 4, 7 (wrists) and 10, 13 (ankles) actually
contribute to the output. Other keypoints' z-values are ignored by this
renderer but remain meaningful for any future dense-depth variant.

Usage:
    python render_depth.py <pose.json|dir> [out.png]
    python render_depth.py <dir/>            # render every *.json in dir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HAND_JOINT_IDS = (4, 7)
FOOT_JOINT_IDS = (10, 13)

HAND_RADIUS = 22
FOOT_RADIUS_X = 26
FOOT_RADIUS_Y = 14


def _gray(z: float) -> tuple[int, int, int]:
    v = max(0, min(255, int(round(z * 255))))
    return (v, v, v)


def render_depth(
    keypoints: list[tuple[float, float, float] | None],
    z_values: list[float],
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    img = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    extremities: list[tuple[int, float]] = []
    for i in (*HAND_JOINT_IDS, *FOOT_JOINT_IDS):
        if keypoints[i] is None:
            continue
        extremities.append((i, z_values[i]))
    extremities.sort(key=lambda t: t[1])

    for i, z in extremities:
        kp = keypoints[i]
        assert kp is not None
        x, y = kp[0], kp[1]
        color = _gray(z)
        if i in HAND_JOINT_IDS:
            r = HAND_RADIUS
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            draw.ellipse(
                [
                    x - FOOT_RADIUS_X, y - FOOT_RADIUS_Y,
                    x + FOOT_RADIUS_X, y + FOOT_RADIUS_Y,
                ],
                fill=color,
            )

    return img


def load_pose_json(
    path: Path,
) -> tuple[list[tuple[float, float, float] | None], list[float], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    canvas_w = int(data.get("canvas_width", 512))
    canvas_h = int(data.get("canvas_height", 512))
    person = data["people"][0]
    flat = person["pose_keypoints_2d"]
    if len(flat) != 18 * 3:
        raise ValueError(
            f"{path}: expected 54 values in pose_keypoints_2d (18 kp x 3), got {len(flat)}"
        )
    z = person.get("pose_keypoints_z")
    if z is None:
        z = [0.5] * 18
    elif len(z) != 18:
        raise ValueError(
            f"{path}: expected 18 values in pose_keypoints_z, got {len(z)}"
        )
    kps: list[tuple[float, float, float] | None] = []
    for i in range(18):
        x, y, c = flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]
        kps.append(None if c <= 0 else (float(x), float(y), float(c)))
    return kps, [float(v) for v in z], canvas_w, canvas_h


def render_file(in_path: Path, out_path: Path) -> None:
    kps, z, w, h = load_pose_json(in_path)
    img = render_depth(kps, z, w, h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  {in_path.name} -> {out_path}")


def _out_path_for(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + "_depth.png")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: render_depth.py <pose.json|dir> [out.png]")
        return 1
    target = Path(argv[1])
    if target.is_dir():
        for json_path in sorted(target.glob("*.json")):
            render_file(json_path, _out_path_for(json_path))
    else:
        out = Path(argv[2]) if len(argv) > 2 else _out_path_for(target)
        render_file(target, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
