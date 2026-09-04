utf-8
#!/usr/bin/env python3
"""AMMT LayerCamera TIFF: read-only memmap audit.
This script is intentionally conservative:
  * It NEVER writes to, renames, moves, or deletes the input TIFF.
  * It opens the ImageJ hyperstack through tifffile.memmap(mode='r').
  * It reads only one 2D frame at a time.
  * It writes only small CSV/JSON/PNG audit results under --output-dir.
  * It refuses to overwrite an existing result file unless --overwrite is set.
Install:
    python -m pip install numpy tifffile matplotlib
Example from the ammt_project root:
    python src/audit_layer_camera.py \
      --tiff raw_original/layer_camera/LayerCameraAfterSpreading.tif \
      --output-dir processed/audit_after_spreading \
      --z-start 1 --z-end 20 --contact-led 1
"""
from __future__ import annotations
import argparse
import csv
import hashlib
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
class TiffInfo:
    """Logical ImageJ hyperstack description without decoding all pixel data."""
    path: str
    size_bytes: int
    sha256: str
    axes: str
    shape: tuple[int, ...]
    dtype: str
    page_count: int
    width: int
    height: int
    channels: int
    layers_z: int
    leds_t: int
def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Calculate SHA-256 in chunks without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
def read_tiff_info(tiff_path: Path, calculate_sha256: bool) -> TiffInfo:
    """Read header/metadata only. No 6 GB image array is decoded."""
    with tifffile.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        imagej = tif.imagej_metadata or {}
        axes = str(series.axes)
        shape = tuple(int(v) for v in series.shape)
        page0 = tif.pages[0]
        width = int(shape[axes.index("X")]) if "X" in axes else int(page0.imagewidth)
        height = int(shape[axes.index("Y")]) if "Y" in axes else int(page0.imagelength)
        channels = int(imagej.get("channels", 1))
        layers_z = int(imagej.get("slices", 1))
        leds_t = int(imagej.get("frames", 1))
        if layers_z < 1 or leds_t < 1:
            raise ValueError("ImageJ metadata에서 Z/T 축 정보를 찾지 못했습니다.")
        return TiffInfo(
            path=str(tiff_path.resolve()),
            size_bytes=tiff_path.stat().st_size,
            sha256=sha256_file(tiff_path) if calculate_sha256 else "SKIPPED",
            axes=axes,
            shape=shape,
            dtype=str(np.dtype(series.dtype)),
            page_count=len(tif.pages),
            width=width,
            height=height,
            channels=channels,
            layers_z=layers_z,
            leds_t=leds_t,
        )
def map_frame(data: np.memmap, info: TiffInfo, z: int, led: int) -> np.ndarray:
    """Return one uint16 [H, W] frame for 1-based manufacturing z and LED t.
    Handles both normal logical axes (for example TZYX) and flattened ImageJ
    stack axes (for example QYX) used by the AMMT TIFF implementation.
    """
    if not 1 <= z <= info.layers_z:
        raise IndexError(f"z는 1..{info.layers_z} 범위여야 합니다: {z}")
    if not 1 <= led <= info.leds_t:
        raise IndexError(f"LED는 1..{info.leds_t} 범위여야 합니다: {led}")
    if "Z" in info.axes and "T" in info.axes:
        index: list[Any] = []
        for axis in info.axes:
            if axis == "Z":
                index.append(z - 1)
            elif axis == "T":
                index.append(led - 1)
            elif axis == "C":
                index.append(0)
            elif axis in ("Y", "X"):
                index.append(slice(None))
            else:
                raise ValueError(f"지원하지 않는 TIFF axis: {info.axes}")
        frame = data[tuple(index)]
    elif data.ndim == 3 and data.shape[0] == info.layers_z * info.leds_t:
        flat_index = (led - 1) * info.layers_z + (z - 1)
        frame = data[flat_index]
    else:
        raise ValueError(
            "자동 frame mapping 실패: "
            f"axes={info.axes}, shape={tuple(data.shape)}, "
            f"metadata(Z={info.layers_z}, T={info.leds_t})"
        )
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"예상한 2D frame이 아닙니다: shape={frame.shape}")
    return frame
def ensure_new_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"이미 결과 파일이 있습니다: {path}\n"
            "원본·기존 결과를 보호하기 위해 중단합니다. 덮어쓰려면 --overwrite를 명시하세요."
        )
def robust_display(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Create display-only [0,1] image; raw uint16 frame remains unchanged."""
    p1, p998 = np.percentile(frame, [1.0, 99.8])
    if p998 <= p1:
        p998 = p1 + 1.0
    shown = np.clip((frame.astype(np.float32) - p1) / (p998 - p1), 0.0, 1.0)
    return shown, float(p1), float(p998)
def frame_statistics(frame: np.ndarray, z: int, led: int) -> dict[str, Any]:
    """Compute per-frame values without creating a full processed dataset."""
    p1, p50, p99, p998 = np.percentile(frame, [1.0, 50.0, 99.0, 99.8])
    dtype_max = float(np.iinfo(frame.dtype).max)
    return {
        "layer_z": z,
        "led_t": led,
        "dtype": str(frame.dtype),
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": float(frame.mean(dtype=np.float64)),
        "std": float(frame.std(dtype=np.float64)),
        "p01": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "p998": float(p998),
        "zero_fraction": float(np.mean(frame == 0)),
        "full_scale_fraction": float(np.mean(frame == dtype_max)),
    }
def write_inventory(path: Path, info: TiffInfo, overwrite: bool) -> None:
    ensure_new_file(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(info).keys()))
        writer.writeheader()
        row = asdict(info)
        row["shape"] = "x".join(map(str, info.shape))
        writer.writerow(row)
def write_statistics(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_new_file(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def write_summary(path: Path, info: TiffInfo, args: argparse.Namespace, overwrite: bool) -> None:
    ensure_new_file(path, overwrite)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only audit of AMMT LayerCamera TIFF through tifffile.memmap",
        "input_tiff": asdict(info),
        "audit_parameters": {
            "z_start": args.z_start,
            "z_end": args.z_end,
            "leds": args.leds,
            "contact_led": args.contact_led,
            "contact_count": args.contact_count,
            "sha256_calculated": not args.skip_sha256,
        },
        "guarantee": "Input TIFF is opened through memmap mode='r' and is never modified by this script.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def write_contact_sheet(
    path: Path,
    data: np.memmap,
    info: TiffInfo,
    z_values: list[int],
    led: int,
    overwrite: bool,
) -> None:
    ensure_new_file(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(5, len(z_values))
    rows = math.ceil(len(z_values) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    for axis, z in zip(axes.flat, z_values):
        frame = map_frame(data, info, z=z, led=led)
        shown, p1, p998 = robust_display(frame)
        axis.imshow(shown, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"z={z}, LED={led}\np1={p1:.0f}, p99.8={p998:.0f}", fontsize=8)
        axis.set_axis_off()
    for axis in axes.flat[len(z_values):]:
        axis.set_axis_off()
    fig.suptitle("AMMT LayerCamera QC contact sheet (display-only percentile scaling)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only memmap audit for AMMT LayerCamera TIFF")
    parser.add_argument("--tiff", required=True, type=Path, help="원본 LayerCamera TIFF 파일 경로")
    parser.add_argument("--output-dir", required=True, type=Path, help="작은 audit 결과를 저장할 새/기존 폴더")
    parser.add_argument("--z-start", type=int, default=1, help="감사 시작 제조 layer (1-based)")
    parser.add_argument("--z-end", type=int, default=20, help="감사 종료 제조 layer (inclusive)")
    parser.add_argument("--leds", nargs="+", type=int, default=[1, 2, 3], help="통계를 낼 LED 번호")
    parser.add_argument("--contact-led", type=int, default=1, help="contact sheet에 표시할 LED 번호")
    parser.add_argument("--contact-count", type=int, default=20, help="contact sheet의 frame 수")
    parser.add_argument("--skip-sha256", action="store_true", help="처음 빠른 구조 확인 시에만 SHA-256 계산을 생략")
    parser.add_argument("--overwrite", action="store_true", help="동일 이름의 audit 결과만 덮어씀; 원본 TIFF에는 적용되지 않음")
    return parser.parse_args()
def main() -> None:
    args = parse_arguments()
    tiff_path = args.tiff.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not tiff_path.is_file():
        raise FileNotFoundError(f"원본 TIFF를 찾을 수 없습니다: {tiff_path}")
    if args.z_start < 1 or args.z_end < args.z_start:
        raise ValueError("z 범위는 1 이상이고 z_end >= z_start여야 합니다.")
    if args.contact_count < 1:
        raise ValueError("contact-count는 1 이상이어야 합니다.")
    info = read_tiff_info(tiff_path, calculate_sha256=not args.skip_sha256)
    if args.z_end > info.layers_z:
        raise ValueError(f"z-end는 {info.layers_z} 이하이어야 합니다.")
    if not 1 <= args.contact_led <= info.leds_t:
        raise ValueError(f"contact-led는 1..{info.leds_t} 범위여야 합니다.")
    for led in args.leds:
        if not 1 <= led <= info.leds_t:
            raise ValueError(f"LED는 1..{info.leds_t} 범위여야 합니다: {led}")
    inventory_path = output_dir / "raw_inventory.csv"
    stats_path = output_dir / "frame_stats.csv"
    summary_path = output_dir / "audit_summary.json"
    contact_path = output_dir / "qc_contact_sheet.png"
    for path in (inventory_path, stats_path, summary_path, contact_path):
        ensure_new_file(path, args.overwrite)
    print("[1/4] 원본 TIFF header와 ImageJ metadata 확인")
    print(info)
    print("[2/4] tifffile.memmap(mode='r') 생성: 전체 TIFF를 RAM에 올리지 않습니다.")
    data = tifffile.memmap(tiff_path, series=0, mode="r")
    print(f"memmap shape={tuple(data.shape)}, dtype={data.dtype}")
    rows: list[dict[str, Any]] = []
    z_values = list(range(args.z_start, args.z_end + 1))
    print(f"[3/4] frame statistics: z={args.z_start}..{args.z_end}, LEDs={args.leds}")
    for z in z_values:
        for led in args.leds:
            frame = map_frame(data, info, z=z, led=led)
            rows.append(frame_statistics(frame, z=z, led=led))
    sample_count = min(args.contact_count, len(z_values))
    contact_z = sorted(set(int(v) for v in np.linspace(args.z_start, args.z_end, sample_count)))
    print(f"[4/4] QC contact sheet: z={contact_z}, LED={args.contact_led}")
    write_inventory(inventory_path, info, args.overwrite)
    write_statistics(stats_path, rows, args.overwrite)
    write_summary(summary_path, info, args, args.overwrite)
    write_contact_sheet(contact_path, data, info, contact_z, args.contact_led, args.overwrite)
    print("\n완료. 원본 TIFF는 읽기 전용 memmap으로만 접근했으며 수정되지 않았습니다.")
    print("생성된 audit 파일:")
    for path in (inventory_path, stats_path, summary_path, contact_path):
        print(f"- {path}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"오류: {error}", file=sys.stderr)
        raise
