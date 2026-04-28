"""Inspect a start/end keyframe pair against the base-animation-pipeline
acceptance checklist.

Reports pass/fail per row for the measurable checks (bbox envelope, feet
alignment, head anchor, canvas edge cropping) and flags visual-inspection
rows (face leaks, ground shadow, clothing leaks) that still need the human
eye. Thresholds match
[base-animation-pipeline.md § keyframe acceptance + seed re-roll].

Usage:
    python inspect_keyframes.py <start.png> <end.png>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

CHAR_PIXEL_THRESHOLD = 240
CANVAS_EDGE_MARGIN = 5
HEAD_REGION_FRACTION = 0.4

THRESHOLDS = {
    "bbox_height_drift_pct": 3.0,
    "head_top_drift_px": 25,
    "feet_y_drift_px": 10,
    "head_x_mid_offset_px": 10,
    "feet_y_canvas_margin": 5,
    "head_top_canvas_margin": 5,
}


def measure(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    canvas_w, canvas_h = img.size
    rgb = np.array(img)
    mask = rgb.min(axis=2) < CHAR_PIXEL_THRESHOLD
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise RuntimeError(f"{path}: no character pixels found")
    top, bot = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())

    head_cutoff = top + int(HEAD_REGION_FRACTION * (bot - top))
    head_xs = xs[ys < head_cutoff]
    head_x_mid = int(round(float(head_xs.mean()))) if len(head_xs) else (left + right) // 2

    below_feet = mask[bot + 1 : min(bot + 20, canvas_h)]
    shadow_pixel_count = int(below_feet.sum())

    return {
        "path": path,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "bbox": (left, top, right, bot),
        "bbox_w": right - left + 1,
        "bbox_h": bot - top + 1,
        "feet_y": bot,
        "head_top_y": top,
        "head_x_mid": head_x_mid,
        "shadow_pixel_count": shadow_pixel_count,
    }


def inspect_pair(start: dict, end: dict) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    canvas_h = start["canvas_h"]
    canvas_w = start["canvas_w"]

    h_drift = abs(start["bbox_h"] - end["bbox_h"])
    h_drift_pct = 100 * h_drift / max(start["bbox_h"], end["bbox_h"])
    ok = h_drift_pct <= THRESHOLDS["bbox_height_drift_pct"]
    results.append((
        "7a bbox_height drift",
        ok,
        f"{h_drift} px ({h_drift_pct:.1f}%) vs threshold {THRESHOLDS['bbox_height_drift_pct']}%",
    ))

    top_drift = abs(start["head_top_y"] - end["head_top_y"])
    ok = top_drift <= THRESHOLDS["head_top_drift_px"]
    results.append((
        "7b head_top_y drift",
        ok,
        f"{top_drift} px vs threshold {THRESHOLDS['head_top_drift_px']} px",
    ))

    feet_drift = abs(start["feet_y"] - end["feet_y"])
    ok = feet_drift <= THRESHOLDS["feet_y_drift_px"]
    results.append((
        "7c feet_y drift",
        ok,
        f"{feet_drift} px vs threshold {THRESHOLDS['feet_y_drift_px']} px",
    ))

    mid = canvas_w // 2
    for name, frame in (("start", start), ("end", end)):
        off = abs(frame["head_x_mid"] - mid)
        ok = off <= THRESHOLDS["head_x_mid_offset_px"]
        results.append((
            f"7d head_x_mid offset ({name})",
            ok,
            f"{off} px from canvas midline ({frame['head_x_mid']} vs {mid})",
        ))

    for name, frame in (("start", start), ("end", end)):
        room_below = canvas_h - 1 - frame["feet_y"]
        ok = room_below >= THRESHOLDS["feet_y_canvas_margin"]
        results.append((
            f"4a feet not cropped ({name})",
            ok,
            f"{room_below} px below feet (canvas {canvas_h})",
        ))
        ok_top = frame["head_top_y"] >= THRESHOLDS["head_top_canvas_margin"]
        results.append((
            f"4b head not cropped ({name})",
            ok_top,
            f"{frame['head_top_y']} px above head top",
        ))
        edge_gap_left = frame["bbox"][0]
        edge_gap_right = canvas_w - 1 - frame["bbox"][2]
        ok_sides = edge_gap_left >= 5 and edge_gap_right >= 5
        results.append((
            f"4c sides not cropped ({name})",
            ok_sides,
            f"left gap {edge_gap_left} px, right gap {edge_gap_right} px",
        ))

    for name, frame in (("start", start), ("end", end)):
        ok = frame["shadow_pixel_count"] == 0
        results.append((
            f"3 no shadow below feet ({name})",
            ok,
            f"{frame['shadow_pixel_count']} non-background pixels within 20 px below feet",
        ))

    return results


def format_frame(tag: str, frame: dict) -> str:
    l, t, r, b = frame["bbox"]
    return (
        f"  {tag:>5}: {Path(frame['path']).name}\n"
        f"         canvas={frame['canvas_w']}x{frame['canvas_h']}  bbox=({l},{t})-({r},{b})  "
        f"size={frame['bbox_w']}x{frame['bbox_h']}\n"
        f"         feet_y={frame['feet_y']}  head_top_y={frame['head_top_y']}  "
        f"head_x_mid={frame['head_x_mid']}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: inspect_keyframes.py <start.png> <end.png>")
        return 1
    start = measure(Path(argv[1]))
    end = measure(Path(argv[2]))

    print("frames:")
    print(format_frame("start", start))
    print(format_frame("end", end))

    print("\nautomated checks:")
    results = inspect_pair(start, end)
    failed = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}: {detail}")

    print("\nvisual-inspection rows (require human eye — open both PNGs and confirm):")
    print("  [1] Head is perfectly smooth and blank (no eyes, mouth, nose, ears, blush, eyebrows)")
    print("  [2] Body is naked (no shirt, shoes, hair, hat)")
    print("  [5] Body is upright and not rotated")
    print("  [6] Facing direction matches between start and end (check by FEET, not head)")

    if failed:
        print(f"\n>>> {failed} automated check(s) failed — re-roll or re-author before proceeding.")
        return 2
    print("\n>>> all automated checks pass. Run visual inspection before staging for W2.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
