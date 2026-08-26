#!/usr/bin/env python3
"""Select screen-corner control points without assuming machine-axis orientation.

The user only identifies the four visible parts by screen order (top to bottom)
and clicks each part's screen TL, TR, BR, BL outer corners. Machine part identity
and X/Y orientation are intentionally NOT assigned here; the calibration audit
compares those hypotheses later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile

SCREEN_PARTS = ("screen_part_A_topmost", "screen_part_B", "screen_part_C", "screen_part_D_bottommost")
SCREEN_CORNERS = ("screen_top_left", "screen_top_right", "screen_bottom_right", "screen_bottom_left")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record screen-corner controls for orientation-agnostic calibration")
    p.add_argument("--tiff", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--layer-z", type=int, default=125)
    p.add_argument("--led", type=int, default=3)
    p.add_argument("--display-max-px", type=int, default=1000)
    p.add_argument("--percentiles", nargs=2, type=float, default=[1.0, 99.5])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def frame_info(path: Path) -> tuple[str, tuple[int, ...]]:
    with tifffile.TiffFile(path) as tif:
        return str(tif.series[0].axes), tuple(int(x) for x in tif.series[0].shape)


def load_frame(path: Path, axes: str, shape: tuple[int, ...], z: int, led: int) -> np.ndarray:
    if not {"T", "Z", "Y", "X"}.issubset(axes):
        raise ValueError(f"Expected TZYX-compatible TIFF, got axes={axes!r}")
    if not 1 <= z <= shape[axes.index("Z")] or not 1 <= led <= shape[axes.index("T")]:
        raise ValueError("Requested layer or LED is outside the TIFF range")
    data = tifffile.memmap(path, series=0, mode="r")
    idx: list[Any] = []
    for axis in axes:
        if axis == "T": idx.append(led - 1)
        elif axis == "Z": idx.append(z - 1)
        elif axis == "C": idx.append(0)
        elif axis in {"Y", "X"}: idx.append(slice(None))
        else: raise ValueError(f"Unsupported TIFF axis {axis}")
    out = np.asarray(data[tuple(idx)])
    if out.ndim != 2: raise ValueError(f"Expected 2D frame, got {out.shape}")
    return out


def main() -> None:
    a = args()
    path, output = a.tiff.resolve(), a.output_json.resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    if output.exists() and not a.overwrite:
        raise FileExistsError(f"Output file already exists: {output}. Review it or use --overwrite deliberately.")
    if a.display_max_px <= 0 or not 0 <= a.percentiles[0] < a.percentiles[1] <= 100:
        raise ValueError("Invalid display size or percentile range")
    output.parent.mkdir(parents=True, exist_ok=True)
    axes, shape = frame_info(path)
    frame = load_frame(path, axes, shape, a.layer_z, a.led)
    h, w = frame.shape
    scale = min(1.0, a.display_max_px / max(h, w))
    dw, dh = w * scale, h * scale
    low, high = np.percentile(frame, a.percentiles)

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.imshow(frame, cmap="gray", vmin=low, vmax=high, origin="upper", extent=(0, dw, dh, 0), interpolation="nearest")
    ax.set_xlim(0, dw); ax.set_ylim(dh, 0); ax.set_aspect("equal")
    ax.set_xlabel(f"Screen display x (scale={scale:.4f} × raw pixel)")
    ax.set_ylabel(f"Screen display y (scale={scale:.4f} × raw pixel)")
    selected: list[dict[str, Any]] = []
    print("Click only visible screen geometry. Do NOT infer machine X/Y or part01..04 here.")
    for part_i, screen_part in enumerate(SCREEN_PARTS, start=1):
        for corner_i, screen_corner in enumerate(SCREEN_CORNERS, start=1):
            n = (part_i - 1) * 4 + corner_i
            ax.set_title(f"Click {n}/16: {screen_part} | {screen_corner}\nVisible part outer corner in SCREEN coordinates only")
            fig.canvas.draw_idle()
            print(f"[{n}/16] Click {screen_part}, {screen_corner}")
            click = plt.ginput(1, timeout=-1, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
            if len(click) != 1:
                plt.close(fig); raise RuntimeError("Selection cancelled; no JSON was written")
            sx, sy = map(float, click[0])
            rx, ry = min(max(sx / scale, 0.0), w - 1.0), min(max(sy / scale, 0.0), h - 1.0)
            selected.append({"screen_part": screen_part, "screen_corner": screen_corner, "raw_camera_x_px": rx, "raw_camera_y_px": ry})
            ax.plot(sx, sy, "cx", ms=8, mew=2); ax.text(sx + 4, sy + 4, str(n), color="cyan", fontsize=8)
    ax.set_title("All screen-corner controls selected. Close this window to save JSON.")
    fig.canvas.draw_idle(); plt.show(block=True); plt.close(fig)
    payload = {
        "schema": "screen_corners_v2_orientation_agnostic",
        "purpose": "Screen-corner controls only; machine part identity and machine-axis orientation are resolved by the calibration hypothesis audit.",
        "raw_input_policy": "TIFF read through tifffile.memmap(mode='r'); no raw data modified.",
        "reference_frame": {"tiff_path": str(path), "axes": axes, "shape": list(shape), "stage": "B", "layer_z": a.layer_z, "led": a.led, "display_percentiles": list(map(float, a.percentiles)), "raw_dimensions_px": [w, h], "display_dimensions_px": [dw, dh], "display_scale_to_raw_pixel": scale},
        "screen_part_order": list(SCREEN_PARTS), "screen_corner_order": list(SCREEN_CORNERS), "control_points": selected,
        "status": "screen_controls_ready_for_part_and_orientation_hypothesis_audit",
        "notes": ["screen_part_A is the topmost visible part and screen_part_D is the bottommost visible part.", "Raw pixels outside later projected sparse support remain unknown, not negative labels."],
    }
    with output.open("w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2); f.write("\n")
    print(f"Saved 16 screen-corner controls to: {output}")
    print("Next: run audit_machine_camera_calibration.py to compare machine-part/orientation hypotheses.")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise
