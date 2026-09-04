utf-8
#!/usr/bin/env python3
"""Compare several AMMT layer-camera ROI candidates without modifying raw data.
The script opens the A (AfterSpreading) and B (Burned) ImageJ hyperstacks via
``tifffile.memmap(..., mode="r")``.  It measures sampled-frame brightness and
full-scale saturation in several candidate rectangles, then creates only small
CSV, JSON and PNG audit products under ``--output-dir``.
This is a screening audit, not a final ROI-selection or normalization step.
Its results must be reviewed before any config file or training dataset is made.
Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/audit_roi_candidates.py \
  --tiff-a raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --tiff-b raw_original/layer_camera/LayerCameraBurned.tif \
  --output-dir processed/roi_candidates
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import tifffile
FULL_SCALE = np.iinfo(np.uint16).max
DEFAULT_Z_VALUES = (1, 10, 20, 125, 230, 250)
DEFAULT_CANDIDATES: dict[str, tuple[int, int, int, int]] = {
    "wide_250_250_1750_1750": (250, 250, 1750, 1750),
    "inner_350_350_1650_1650": (350, 350, 1650, 1650),
    "inner_450_450_1550_1550": (450, 450, 1550, 1550),
    "upper_350_250_1650_1550": (350, 250, 1650, 1550),
    "lower_350_450_1650_1750": (350, 450, 1650, 1750),
}
@dataclass(frozen=True)
class StackInfo:
    """Minimal structure required to index the ImageJ hyperstack safely."""
    axes: str
    shape: tuple[int, ...]
    dtype: str
    width: int
    height: int
    layers: int
    leds: int
@dataclass(frozen=True)
class Candidate:
    """A named raw-camera rectangle."""
    candidate_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    @property
    def area_pixels(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)
def inspect_stack(path: Path) -> StackInfo:
    """Read metadata only; do not decode or rewrite any TIFF pixels."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        imagej = tif.imagej_metadata or {}
    required_axes = {"T", "Z", "Y", "X"}
    if not required_axes.issubset(set(axes)):
        raise ValueError(f"Expected a TZYX-compatible ImageJ stack, got axes={axes!r}")
    if dtype != np.dtype(np.uint16):
        raise ValueError(f"This audit expects uint16 data, got {dtype}")
    return StackInfo(
        axes=axes,
        shape=shape,
        dtype=str(dtype),
        width=shape[axes.index("X")],
        height=shape[axes.index("Y")],
        layers=int(imagej.get("slices", 1)),
        leds=int(imagej.get("frames", 1)),
    )
def validate_pair(a: StackInfo, b: StackInfo) -> None:
    """Reject the audit if A and B cannot be indexed as corresponding frames."""
    for field in ("axes", "shape", "dtype", "width", "height", "layers", "leds"):
        if getattr(a, field) != getattr(b, field):
            raise ValueError(f"A/B stack mismatch at {field}: {getattr(a, field)!r} != {getattr(b, field)!r}")
def read_frame(data: np.memmap, info: StackInfo, z: int, led: int) -> np.ndarray:
    """Return one raw uint16 frame using 1-based layer and LED indices."""
    if not 1 <= z <= info.layers:
        raise ValueError(f"layer z must be 1..{info.layers}, got {z}")
    if not 1 <= led <= info.leds:
        raise ValueError(f"LED must be 1..{info.leds}, got {led}")
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
            raise ValueError(f"Unsupported stack axis {axis!r} in {info.axes!r}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape={frame.shape}")
    return frame
def validate_candidates(candidates: list[Candidate], info: StackInfo) -> None:
    if not candidates:
        raise ValueError("At least one ROI candidate is required")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ROI candidate identifiers must be unique")
    for candidate in candidates:
        if not (0 <= candidate.x0 < candidate.x1 <= info.width):
            raise ValueError(f"Invalid x range for {candidate.candidate_id}: {candidate.roi}")
        if not (0 <= candidate.y0 < candidate.y1 <= info.height):
            raise ValueError(f"Invalid y range for {candidate.candidate_id}: {candidate.roi}")
def ensure_new(path: Path, overwrite: bool) -> None:
    """Avoid silently replacing an earlier audit result."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Review it or use --overwrite.")
def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    ensure_new(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
def display_scale(frame: np.ndarray) -> np.ndarray:
    """Scale only the QC view; never use this for model input normalization."""
    low, high = np.percentile(frame, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return np.clip((frame.astype(np.float32) - low) / (high - low), 0.0, 1.0)
def draw_candidates(axis: plt.Axes, candidates: list[Candidate]) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(candidates)))
    for color, candidate in zip(colors, candidates, strict=True):
        rectangle = plt.Rectangle(
            (candidate.x0, candidate.y0),
            candidate.x1 - candidate.x0,
            candidate.y1 - candidate.y0,
            linewidth=1.7,
            edgecolor=color,
            facecolor="none",
            label=candidate.candidate_id,
        )
        axis.add_patch(rectangle)
def aggregate_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize sampled layers per candidate, stage and LED."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in frame_rows:
        key = (str(row["candidate_id"]), str(row["stage"]), int(row["led_t"]))
        groups.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (candidate_id, stage, led), rows in sorted(groups.items()):
        saturation = np.asarray([float(row["roi_full_scale_fraction"]) for row in rows])
        valid = np.asarray([float(row["roi_valid_fraction"]) for row in rows])
        means = np.asarray([float(row["roi_mean"]) for row in rows])
        pair_mad = np.asarray([float(row["pair_mean_abs_difference"]) for row in rows])
        x0, y0, x1, y1 = (int(rows[0][field]) for field in ("roi_x0", "roi_y0", "roi_x1", "roi_y1"))
        aggregate.append(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "led_t": led,
                "sampled_layer_count": len(rows),
                "roi_x0": x0,
                "roi_y0": y0,
                "roi_x1": x1,
                "roi_y1": y1,
                "roi_area_pixels": int(rows[0]["roi_area_pixels"]),
                "mean_full_scale_fraction": float(saturation.mean()),
                "max_full_scale_fraction": float(saturation.max()),
                "mean_valid_fraction": float(valid.mean()),
                "min_valid_fraction": float(valid.min()),
                "mean_intensity": float(means.mean()),
                "mean_pair_abs_difference": float(pair_mad.mean()),
            }
        )
    return aggregate
def candidate_screening_summary(aggregate: list[dict[str, Any]], candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Provide transparent descriptive rankings; this function never selects a final ROI."""
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate.candidate_id: [] for candidate in candidates}
    for row in aggregate:
        by_candidate[str(row["candidate_id"])].append(row)
    summary: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = by_candidate[candidate.candidate_id]
        mean_saturation = np.asarray([float(row["mean_full_scale_fraction"]) for row in rows])
        worst_saturation = np.asarray([float(row["max_full_scale_fraction"]) for row in rows])
        mean_valid = np.asarray([float(row["mean_valid_fraction"]) for row in rows])
        summary.append(
            {
                "candidate_id": candidate.candidate_id,
                "roi": {"x0": candidate.x0, "y0": candidate.y0, "x1": candidate.x1, "y1": candidate.y1},
                "area_pixels": candidate.area_pixels,
                "mean_full_scale_fraction_across_stage_led": float(mean_saturation.mean()),
                "worst_sample_full_scale_fraction": float(worst_saturation.max()),
                "mean_valid_fraction_across_stage_led": float(mean_valid.mean()),
            }
        )
    ranked = sorted(
        summary,
        key=lambda row: (row["mean_full_scale_fraction_across_stage_led"], row["worst_sample_full_scale_fraction"]),
    )
    rank_by_id = {str(row["candidate_id"]): index + 1 for index, row in enumerate(ranked)}
    for row in summary:
        row["saturation_screen_rank_low_to_high"] = rank_by_id[str(row["candidate_id"])]
    return summary
def save_qc(
    path: Path,
    a_data: np.memmap,
    b_data: np.memmap,
    info: StackInfo,
    candidates: list[Candidate],
    sat_count_a_led1: np.ndarray,
    sat_count_b_led1: np.ndarray,
    z_values: tuple[int, ...],
    aggregate: list[dict[str, Any]],
    reference_z: int,
    overwrite: bool,
) -> None:
    """Create a deterministic data-driven QC plot, without generating imagery."""
    ensure_new(path, overwrite)
    frame_a = read_frame(a_data, info, reference_z, led=1)
    frame_b = read_frame(b_data, info, reference_z, led=1)
    sampled_count = len(z_values)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    image_panels = (
        (axes[0, 0], display_scale(frame_a), f"A reference frame | z={reference_z}, LED=1", "gray"),
        (axes[0, 1], display_scale(frame_b), f"B reference frame | z={reference_z}, LED=1", "gray"),
        (axes[0, 2], sat_count_a_led1 / sampled_count, "A LED=1 saturation frequency", "magma"),
        (axes[1, 0], sat_count_b_led1 / sampled_count, "B LED=1 saturation frequency", "magma"),
    )
    for axis, image, title, cmap in image_panels:
        im = axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        draw_candidates(axis, candidates)
        axis.set_title(title)
        axis.set_axis_off()
        if "frequency" in title:
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="fraction at 65535")
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        fontsize=8,
        frameon=True,
        title="ROI candidates (raw pixel rectangles)",
        title_fontsize=8,
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    positions = np.arange(len(candidate_ids))
    colors = {"A": "#2a6fbb", "B": "#d55e00"}
    for led in range(1, info.leds + 1):
        for stage in ("A", "B"):
            values = []
            for candidate_id in candidate_ids:
                row = next(
                    item
                    for item in aggregate
                    if item["candidate_id"] == candidate_id and item["stage"] == stage and item["led_t"] == led
                )
                values.append(100.0 * float(row["mean_full_scale_fraction"]))
            offset = ((led - 2) * 0.26) + (-0.09 if stage == "A" else 0.09)
            axes[1, 1].bar(
                positions + offset,
                values,
                width=0.16,
                color=colors[stage],
                alpha=0.42 + 0.18 * led,
                label=f"{stage} LED {led}",
            )
    axes[1, 1].set_title("Mean sampled full-scale saturation by ROI")
    axes[1, 1].set_ylabel("percent of ROI pixels at 65535")
    axes[1, 1].set_xticks(positions, [str(index + 1) for index in positions])
    axes[1, 1].set_xlabel("candidate index (legend above)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=7, ncol=2)
    worst_values = []
    mean_values = []
    for candidate_id in candidate_ids:
        rows = [item for item in aggregate if item["candidate_id"] == candidate_id]
        worst_values.append(100.0 * max(float(item["max_full_scale_fraction"]) for item in rows))
        mean_values.append(100.0 * np.mean([float(item["mean_full_scale_fraction"]) for item in rows]))
    axes[1, 2].bar(positions - 0.18, mean_values, width=0.34, label="mean over stage/LED", color="#009e73")
    axes[1, 2].bar(positions + 0.18, worst_values, width=0.34, label="worst sampled frame", color="#cc79a7")
    axes[1, 2].set_title("Saturation screening summary")
    axes[1, 2].set_ylabel("full-scale saturation (%)")
    axes[1, 2].set_xticks(positions, [str(index + 1) for index in positions])
    axes[1, 2].set_xlabel("candidate index (legend above)")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].legend(fontsize=8)
    fig.suptitle(
        "AMMT ROI candidate screening — descriptive audit only; no final ROI is selected",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AMMT ROI candidate screening audit")
    parser.add_argument("--tiff-a", required=True, type=Path, help="AfterSpreading TIFF (A)")
    parser.add_argument("--tiff-b", required=True, type=Path, help="Burned TIFF (B)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--z-values", nargs="+", type=int, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--reference-z", type=int, default=125, help="Layer used only for the QC reference frame")
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=None,
        help="Optional JSON object: {candidate_id: [x0, y0, x1, y1]}. Defaults to five built-in screening ROIs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs after review")
    return parser.parse_args()
def load_candidates(path: Path | None) -> list[Candidate]:
    source: dict[str, Any]
    if path is None:
        source = DEFAULT_CANDIDATES
    else:
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if not isinstance(source, dict):
            raise ValueError("--candidates-json must contain a JSON object")
    candidates: list[Candidate] = []
    for identifier, coordinates in source.items():
        if not isinstance(identifier, str) or not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            raise ValueError("Each candidate must map a string ID to [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (int(value) for value in coordinates)
        candidates.append(Candidate(identifier, x0, y0, x1, y1))
    return candidates
def main() -> None:
    args = parse_args()
    path_a = args.tiff_a.resolve()
    path_b = args.tiff_b.resolve()
    output_dir = args.output_dir.resolve()
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"Missing TIFF input. A={path_a}, B={path_b}")
    info_a = inspect_stack(path_a)
    info_b = inspect_stack(path_b)
    validate_pair(info_a, info_b)
    info = info_a
    candidates = load_candidates(args.candidates_json)
    validate_candidates(candidates, info)
    z_values = tuple(sorted(set(int(value) for value in args.z_values)))
    if not z_values or any(z < 1 or z > info.layers for z in z_values):
        raise ValueError(f"--z-values must be within 1..{info.layers}")
    if not 1 <= args.reference_z <= info.layers:
        raise ValueError(f"--reference-z must be within 1..{info.layers}")
    frame_csv = output_dir / "roi_candidate_comparison.csv"
    aggregate_csv = output_dir / "roi_candidate_aggregate.csv"
    summary_json = output_dir / "roi_candidate_summary.json"
    qc_png = output_dir / "roi_candidate_qc.png"
    for output in (frame_csv, aggregate_csv, summary_json, qc_png):
        ensure_new(output, args.overwrite)
    print("[1/3] Opening A/B TIFF files through read-only memmap.")
    a_data = tifffile.memmap(path_a, series=0, mode="r")
    b_data = tifffile.memmap(path_b, series=0, mode="r")
    print(f"A/B shape={a_data.shape}; sampled layers={list(z_values)}; candidates={len(candidates)}")
    sat_count_a_led1 = np.zeros((info.height, info.width), dtype=np.uint16)
    sat_count_b_led1 = np.zeros((info.height, info.width), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    print("[2/3] Measuring sampled frames. Raw TIFF files remain unchanged.")
    for z in z_values:
        for led in range(1, info.leds + 1):
            frame_a = read_frame(a_data, info, z, led)
            frame_b = read_frame(b_data, info, z, led)
            if led == 1:
                sat_count_a_led1 += frame_a == FULL_SCALE
                sat_count_b_led1 += frame_b == FULL_SCALE
            for candidate in candidates:
                roi_slice = np.s_[candidate.y0:candidate.y1, candidate.x0:candidate.x1]
                a_roi = frame_a[roi_slice]
                b_roi = frame_b[roi_slice]
                pair_mad = float(np.abs(b_roi.astype(np.float32) - a_roi.astype(np.float32)).mean())
                for stage, roi in (("A", a_roi), ("B", b_roi)):
                    saturation = float(np.mean(roi == FULL_SCALE))
                    rows.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "stage": stage,
                            "layer_z": z,
                            "led_t": led,
                            "roi_x0": candidate.x0,
                            "roi_y0": candidate.y0,
                            "roi_x1": candidate.x1,
                            "roi_y1": candidate.y1,
                            "roi_area_pixels": candidate.area_pixels,
                            "roi_mean": float(roi.mean()),
                            "roi_std": float(roi.std()),
                            "roi_p01": float(np.percentile(roi, 1.0)),
                            "roi_p50": float(np.percentile(roi, 50.0)),
                            "roi_p99": float(np.percentile(roi, 99.0)),
                            "roi_full_scale_fraction": saturation,
                            "roi_valid_fraction": float(1.0 - saturation),
                            "pair_mean_abs_difference": pair_mad,
                        }
                    )
    aggregate = aggregate_rows(rows)
    screening = candidate_screening_summary(aggregate, candidates)
    summary: dict[str, Any] = {
        "audit_type": "ROI candidate comparison; descriptive only, not final ROI selection",
        "raw_inputs": {"after_spreading": str(path_a), "burned": str(path_b)},
        "raw_input_policy": "Opened with tifffile.memmap(mode='r'); raw TIFF bytes are never modified.",
        "stack": asdict(info),
        "sampled_layers_z": list(z_values),
        "sampled_leds_t": list(range(1, info.leds + 1)),
        "sampled_frame_pairs": len(z_values) * info.leds,
        "reference_qc_frame": {"layer_z": args.reference_z, "led_t": 1},
        "candidate_rectangles_raw_pixels": {candidate.candidate_id: list(candidate.roi) for candidate in candidates},
        "screening_metrics": screening,
        "decision_boundary": (
            "Do not select a final ROI from saturation alone. Review build coverage, persistent saturation location, "
            "and later XCT/annotation coverage before creating configs/roi_v1.yaml."
        ),
        "next_required_review": "Define saturation validity-mask and training-only stage/LED normalization policy after review.",
    }
    print("[3/3] Writing only small CSV, JSON and deterministic PNG QC outputs.")
    write_csv(frame_csv, rows, args.overwrite)
    write_csv(aggregate_csv, aggregate, args.overwrite)
    write_json(summary_json, summary, args.overwrite)
    save_qc(
        qc_png,
        a_data,
        b_data,
        info,
        candidates,
        sat_count_a_led1,
        sat_count_b_led1,
        z_values,
        aggregate,
        args.reference_z,
        args.overwrite,
    )
    print("Done. Raw TIFF files were opened read-only and were not modified.")
    for output in (frame_csv, aggregate_csv, summary_json, qc_png):
        print(f"- {output}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
