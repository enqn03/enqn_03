#!/usr/bin/env python3
"""Read-only AMMT A/B pair audit.

Purpose
-------
Verify that LayerCameraAfterSpreading (A) and LayerCameraBurned (B) can be
safely paired by the same (manufacturing layer z, LED direction t) before
building ROI, normalization, causal Dataset, or fusion localization code.

Safety guarantees
-----------------
* Input TIFF files are opened using tifffile.memmap(..., mode='r').
* The script does NOT modify, move, rename, or delete either input TIFF.
* It writes only three small results below --output-dir.
* It refuses to overwrite existing result files unless --overwrite is set.

Install
-------
python -m pip install numpy tifffile matplotlib

Example from ammt_project root
------------------------------
python audit_ab_pairs.py \
  --tiff-a raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --tiff-b raw_original/layer_camera/LayerCameraBurned.tif \
  --output-dir processed/audit_ab_pairs \
  --z-values 1 10 20 125 230 250
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile


@dataclass(frozen=True)
class HyperstackInfo:
    path: str
    axes: str
    shape: tuple[int, ...]
    dtype: str
    page_count: int
    height: int
    width: int
    layers_z: int
    leds_t: int
    size_bytes: int


def inspect_tiff(path: Path) -> HyperstackInfo:
    """Read metadata only; do not decode the full image stack."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(v) for v in series.shape)
        imagej = tif.imagej_metadata or {}
        return HyperstackInfo(
            path=str(path.resolve()),
            axes=axes,
            shape=shape,
            dtype=str(np.dtype(series.dtype)),
            page_count=len(tif.pages),
            height=int(shape[axes.index("Y")]),
            width=int(shape[axes.index("X")]),
            layers_z=int(imagej.get("slices", 1)),
            leds_t=int(imagej.get("frames", 1)),
            size_bytes=path.stat().st_size,
        )


def validate_compatible(a: HyperstackInfo, b: HyperstackInfo) -> None:
    """Fail early if A/B cannot be paired pixel-for-pixel."""
    compared = ("axes", "shape", "dtype", "height", "width", "layers_z", "leds_t")
    conflicts = [name for name in compared if getattr(a, name) != getattr(b, name)]
    if conflicts:
        details = "; ".join(f"{name}: A={getattr(a, name)!r}, B={getattr(b, name)!r}" for name in conflicts)
        raise ValueError(f"A/B hyperstack이 호환되지 않습니다. {details}")
    if not {"T", "Z", "Y", "X"}.issubset(set(a.axes)):
        raise ValueError(f"지원하지 않는 logical axes입니다: {a.axes}. T, Z, Y, X가 필요합니다.")


def read_frame(data: np.memmap, info: HyperstackInfo, z: int, led: int) -> np.ndarray:
    """Return one 2D uint16 frame. z and led are 1-based."""
    if not 1 <= z <= info.layers_z:
        raise IndexError(f"z는 1..{info.layers_z} 범위여야 합니다: {z}")
    if not 1 <= led <= info.leds_t:
        raise IndexError(f"led는 1..{info.leds_t} 범위여야 합니다: {led}")

    index: list[Any] = []
    for axis in info.axes:
        if axis == "T":
            index.append(led - 1)
        elif axis == "Z":
            index.append(z - 1)
        elif axis == "C":
            index.append(0)
        elif axis in {"Y", "X"}:
            index.append(slice(None))
        else:
            raise ValueError(f"지원하지 않는 axis: {info.axes}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"2D frame이 필요하지만 {frame.shape}를 받았습니다.")
    return frame


def percentile_display(frame: np.ndarray) -> np.ndarray:
    """Display-only scaling. It does not alter the raw image or training input."""
    low, high = np.percentile(frame, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    return np.clip((frame.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def frame_metrics(a: np.ndarray, b: np.ndarray, z: int, led: int) -> dict[str, Any]:
    """Metrics quantify A/B difference but do not label it as a defect."""
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    diff = b32 - a32
    abs_diff = np.abs(diff)
    # Pearson correlation uses a low-memory 8x8 spatial subsample for the audit.
    a_small, b_small = a[::8, ::8].astype(np.float64), b[::8, ::8].astype(np.float64)
    correlation = float(np.corrcoef(a_small.ravel(), b_small.ravel())[0, 1])
    return {
        "layer_z": z,
        "led_t": led,
        "a_mean": float(a32.mean()),
        "b_mean": float(b32.mean()),
        "a_std": float(a32.std()),
        "b_std": float(b32.std()),
        "a_full_scale_fraction": float(np.mean(a == np.iinfo(a.dtype).max)),
        "b_full_scale_fraction": float(np.mean(b == np.iinfo(b.dtype).max)),
        "mean_abs_difference": float(abs_diff.mean()),
        "rmse_difference": float(np.sqrt(np.mean(diff * diff))),
        "sampled_pixel_correlation": correlation,
    }


def ensure_new(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"결과 파일이 이미 존재합니다: {path}\n검토 후 덮어쓰려면 --overwrite를 명시하세요.")


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_contact_sheet(
    path: Path,
    data_a: np.memmap,
    data_b: np.memmap,
    info: HyperstackInfo,
    z_values: list[int],
    led: int,
    overwrite: bool,
) -> None:
    """Save A, B, and abs(B-A) panels for each sampled manufacturing layer."""
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(z_values), 3, figsize=(11, 3.5 * len(z_values)), squeeze=False)

    for row, z in enumerate(z_values):
        a = read_frame(data_a, info, z=z, led=led)
        b = read_frame(data_b, info, z=z, led=led)
        difference = np.abs(b.astype(np.float32) - a.astype(np.float32))
        difference_display = percentile_display(difference)

        for axis, image, title in (
            (axes[row, 0], percentile_display(a), f"A: AfterSpreading | z={z}, LED={led}"),
            (axes[row, 1], percentile_display(b), f"B: Burned | z={z}, LED={led}"),
            (axes[row, 2], difference_display, f"|B-A| candidate map | z={z}"),
        ):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(title, fontsize=9)
            axis.set_axis_off()

    fig.suptitle("AMMT A/B pair audit — display-only percentile scaling", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only A/B pair audit for AMMT LayerCamera TIFF files")
    parser.add_argument("--tiff-a", required=True, type=Path, help="AfterSpreading A TIFF 경로")
    parser.add_argument("--tiff-b", required=True, type=Path, help="Burned B TIFF 경로")
    parser.add_argument("--output-dir", required=True, type=Path, help="소형 audit 결과를 저장할 폴더")
    parser.add_argument("--z-values", nargs="+", type=int, default=[1, 10, 20, 125, 230, 250], help="초기·중기·후기 pair를 확인할 1-based z 번호")
    parser.add_argument("--contact-led", type=int, default=1, help="A/B contact sheet에 쓸 LED 번호")
    parser.add_argument("--overwrite", action="store_true", help="기존 audit 결과를 검토 후 덮어쓸 때만 사용")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_a = args.tiff_a.expanduser().resolve()
    path_b = args.tiff_b.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"A/B TIFF 경로를 찾지 못했습니다. A={path_a}, B={path_b}")

    print("[1/4] A/B TIFF header와 ImageJ metadata를 읽습니다.")
    info_a, info_b = inspect_tiff(path_a), inspect_tiff(path_b)
    validate_compatible(info_a, info_b)
    info = info_a
    print(f"A: axes={info_a.axes}, shape={info_a.shape}, dtype={info_a.dtype}")
    print(f"B: axes={info_b.axes}, shape={info_b.shape}, dtype={info_b.dtype}")

    z_values = sorted(set(args.z_values))
    for z in z_values:
        if not 1 <= z <= info.layers_z:
            raise ValueError(f"z-values는 1..{info.layers_z} 범위여야 합니다: {z}")
    if not 1 <= args.contact_led <= info.leds_t:
        raise ValueError(f"contact-led는 1..{info.leds_t} 범위여야 합니다.")

    output_csv = output_dir / "ab_pair_metrics.csv"
    output_json = output_dir / "ab_pair_audit_summary.json"
    output_png = output_dir / "ab_pair_contact_sheet.png"
    for output in (output_csv, output_json, output_png):
        ensure_new(output, args.overwrite)

    print("[2/4] read-only memmap으로 A/B hyperstack을 엽니다.")
    data_a = tifffile.memmap(path_a, series=0, mode="r")
    data_b = tifffile.memmap(path_b, series=0, mode="r")
    print(f"A memmap={tuple(data_a.shape)}, B memmap={tuple(data_b.shape)}")

    print(f"[3/4] z={z_values}, LED=1..{info.leds_t}의 A/B pair metrics를 계산합니다.")
    rows: list[dict[str, Any]] = []
    for z in z_values:
        for led in range(1, info.leds_t + 1):
            frame_a = read_frame(data_a, info, z=z, led=led)
            frame_b = read_frame(data_b, info, z=z, led=led)
            rows.append(frame_metrics(frame_a, frame_b, z=z, led=led))

    print(f"[4/4] LED={args.contact_led}의 A/B/|B-A| QC contact sheet를 생성합니다.")
    write_csv(output_csv, rows, args.overwrite)
    write_json(
        output_json,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Read-only audit of A/B frame pairing before ROI and model preprocessing",
            "a_hyperstack": asdict(info_a),
            "b_hyperstack": asdict(info_b),
            "z_values": z_values,
            "metric_leds": list(range(1, info.leds_t + 1)),
            "contact_led": args.contact_led,
            "guarantee": "Both TIFF files are opened by tifffile.memmap(mode='r') and are never modified.",
            "warning": "Absolute A/B difference is a process-change candidate map, not a defect label.",
        },
        args.overwrite,
    )
    make_contact_sheet(output_png, data_a, data_b, info, z_values, args.contact_led, args.overwrite)

    print("\n완료. 원본 A/B TIFF는 수정되지 않았습니다.")
    print("생성 파일:")
    for output in (output_csv, output_json, output_png):
        print(f"- {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"오류: {error}", file=sys.stderr)
        raise
