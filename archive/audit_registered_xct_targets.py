#!/usr/bin/env python3
"""Read-only audit of registered XCT sparse weak-target support for AMMT X4.
The 2025 NIST registered X4 data contains one headerless 40-column CSV per
part and manufacturing layer. Each row is a sparse XYPT/melt-pool measurement
location, not a dense layer-camera pixel. Columns 38–40 carry original,
3x3x3-filtered and 5x5x5-filtered XCT voxel values.
This audit answers the questions that must be settled before generating any
camera heatmap target:
1. Are all part/layer CSV files present and structurally valid?
2. Where is sparse machine-coordinate supervision supported?
3. Which layers have finite XCT values, including the train-only history?
4. What are the train-only distributions of the three XCT voxel responses?
It does NOT create a camera-space heatmap, a binary defect label, or a model
training tensor. Locations not represented by registered CSV rows remain
unknown rather than negative supervision.
Example
-------
cd ~/ammt_project
/usr/local/bin/python3 src/audit_registered_xct_targets.py \
  --registered-root raw_original/registered_xct \
  --manifest manifests/causal_sequence_manifest.csv \
  --output-dir processed/xct_target_audit
Raw registered CSVs are opened only for reading. Outputs are small summary
artifacts under processed/ and can be regenerated.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import matplotlib.pyplot as plt
import numpy as np
EXPECTED_COLUMN_COUNT = 40
COLUMN_PART_NUMBER = 1
COLUMN_BUILD_TIME_US = 2
COLUMN_COMMAND_X_MM = 3
COLUMN_COMMAND_Y_MM = 4
COLUMN_REAL_X_MM = 7
COLUMN_REAL_Y_MM = 8
COLUMN_XCT_ORIGINAL = 38
COLUMN_XCT_3X3X3 = 39
COLUMN_XCT_5X5X5 = 40
TARGET_COLUMNS = {
    "xct_original": COLUMN_XCT_ORIGINAL,
    "xct_3x3x3": COLUMN_XCT_3X3X3,
    "xct_5x5x5": COLUMN_XCT_5X5X5,
}
COLUMN_DICTIONARY = {
    "part_number": COLUMN_PART_NUMBER,
    "build_time_us": COLUMN_BUILD_TIME_US,
    "command_x_mm": COLUMN_COMMAND_X_MM,
    "command_y_mm": COLUMN_COMMAND_Y_MM,
    "real_x_mm": COLUMN_REAL_X_MM,
    "real_y_mm": COLUMN_REAL_Y_MM,
    **TARGET_COLUMNS,
}
@dataclass
class LayerInventory:
    part: str
    layer_z: int
    csv_path: str
    rows_total: int
    rows_schema_40: int
    rows_bad_schema: int
    command_x_min_mm: float | None
    command_x_max_mm: float | None
    command_y_min_mm: float | None
    command_y_max_mm: float | None
    xct_original_finite_count: int
    xct_3x3x3_finite_count: int
    xct_5x5x5_finite_count: int
    xct_original_finite_fraction: float
    xct_3x3x3_finite_fraction: float
    xct_5x5x5_finite_fraction: float
    xct_5x5x5_mean: float | None
    xct_5x5x5_min: float | None
    xct_5x5x5_max: float | None
@dataclass
class DistributionSummary:
    scope: str
    part: str
    target_column: str
    source_column_1_based: int
    finite_count: int
    p01: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit registered XCT sparse-target support without modifying source CSV files"
    )
    parser.add_argument("--registered-root", required=True, type=Path, help="Directory containing part01..part04")
    parser.add_argument("--manifest", required=True, type=Path, help="Causal sequence manifest CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="New processed output directory")
    parser.add_argument(
        "--qc-layer",
        type=int,
        default=100,
        help="Manufacturing layer used for sparse machine-XY support QC (default: 100)",
    )
    parser.add_argument(
        "--max-qc-points-per-part",
        type=int,
        default=10000,
        help="Maximum spatial points plotted for each part at --qc-layer (default: 10000)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()
def parse_finite_float(value: str) -> float | None:
    """Convert numeric CSV text to finite float; NaN/empty/nonfinite becomes None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
def nan_or_float(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)
def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite only after review.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
def read_train_history_layers(manifest_path: Path) -> tuple[list[int], int]:
    """Return the unique train-only causal history layers and train-row count."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    history_layers: set[int] = set()
    train_rows = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"split", "history_layer_z"}
        missing = expected.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("split") != "train":
                continue
            train_rows += 1
            try:
                history_layers.update(int(token) for token in str(row["history_layer_z"]).split(";"))
            except ValueError as error:
                raise ValueError(f"Invalid train history in manifest row: {row}") from error
    if not history_layers:
        raise ValueError("Manifest contains no train history layers")
    return sorted(history_layers), train_rows
def find_part_directories(registered_root: Path) -> list[Path]:
    if not registered_root.is_dir():
        raise FileNotFoundError(f"Missing registered XCT root: {registered_root}")
    parts = sorted(path for path in registered_root.iterdir() if path.is_dir() and re.fullmatch(r"part\d+", path.name))
    if not parts:
        raise ValueError(f"No partNN directories found under: {registered_root}")
    return parts
def layer_number_from_path(path: Path) -> int:
    matched = re.fullmatch(r"L(\d{4})\.csv", path.name)
    if matched is None:
        raise ValueError(f"Expected LNNNN.csv filename, got {path.name}")
    return int(matched.group(1))
def read_csv_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.reader(handle)
def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p01", "p05", "p50", "p95", "p99", "mean", "std", "minimum", "maximum")}
    data = np.asarray(values, dtype=np.float64)
    return {
        "p01": float(np.percentile(data, 1)),
        "p05": float(np.percentile(data, 5)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=0)),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
    }
def audit_layer_csv(
    part: str,
    layer_z: int,
    csv_path: Path,
    collect_train_values: bool,
    collect_qc_points: bool,
    max_qc_points: int,
) -> tuple[LayerInventory, dict[str, list[float]], np.ndarray]:
    """Audit one headerless 40-column file and optionally collect small in-memory samples."""
    total_rows = 0
    schema_rows = 0
    bad_rows = 0
    xs: list[float] = []
    ys: list[float] = []
    target_values: dict[str, list[float]] = {name: [] for name in TARGET_COLUMNS}
    finite_counts: dict[str, int] = {name: 0 for name in TARGET_COLUMNS}
    xct5_values: list[float] = []
    qc_rows: list[tuple[float, float, float]] = []
    for row in read_csv_rows(csv_path):
        total_rows += 1
        if len(row) != EXPECTED_COLUMN_COUNT:
            bad_rows += 1
            continue
        schema_rows += 1
        x = parse_finite_float(row[COLUMN_COMMAND_X_MM - 1])
        y = parse_finite_float(row[COLUMN_COMMAND_Y_MM - 1])
        if x is not None:
            xs.append(x)
        if y is not None:
            ys.append(y)
        finite_current: dict[str, float | None] = {}
        for name, column_index in TARGET_COLUMNS.items():
            value = parse_finite_float(row[column_index - 1])
            finite_current[name] = value
            if value is not None:
                finite_counts[name] += 1
                if collect_train_values:
                    target_values[name].append(value)
        if finite_current["xct_5x5x5"] is not None:
            xct5_values.append(float(finite_current["xct_5x5x5"]))
            if collect_qc_points and x is not None and y is not None:
                qc_rows.append((x, y, float(finite_current["xct_5x5x5"])))
    if collect_qc_points and len(qc_rows) > max_qc_points:
        indices = np.linspace(0, len(qc_rows) - 1, num=max_qc_points, dtype=np.int64)
        qc_rows = [qc_rows[index] for index in indices]
    denominator = schema_rows if schema_rows else 1
    xct5_summary = _quantiles(xct5_values)
    inventory = LayerInventory(
        part=part,
        layer_z=layer_z,
        csv_path=str(csv_path),
        rows_total=total_rows,
        rows_schema_40=schema_rows,
        rows_bad_schema=bad_rows,
        command_x_min_mm=nan_or_float(min(xs) if xs else None),
        command_x_max_mm=nan_or_float(max(xs) if xs else None),
        command_y_min_mm=nan_or_float(min(ys) if ys else None),
        command_y_max_mm=nan_or_float(max(ys) if ys else None),
        xct_original_finite_count=finite_counts["xct_original"],
        xct_3x3x3_finite_count=finite_counts["xct_3x3x3"],
        xct_5x5x5_finite_count=finite_counts["xct_5x5x5"],
        xct_original_finite_fraction=finite_counts["xct_original"] / denominator,
        xct_3x3x3_finite_fraction=finite_counts["xct_3x3x3"] / denominator,
        xct_5x5x5_finite_fraction=finite_counts["xct_5x5x5"] / denominator,
        xct_5x5x5_mean=xct5_summary["mean"],
        xct_5x5x5_min=xct5_summary["minimum"],
        xct_5x5x5_max=xct5_summary["maximum"],
    )
    qc_array = np.asarray(qc_rows, dtype=np.float32) if qc_rows else np.empty((0, 3), dtype=np.float32)
    return inventory, target_values, qc_array
def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def build_distribution_rows(train_values_by_part: dict[str, dict[str, list[float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_values: dict[str, list[float]] = {name: [] for name in TARGET_COLUMNS}
    for part, values_by_target in sorted(train_values_by_part.items()):
        for target_name, values in values_by_target.items():
            all_values[target_name].extend(values)
            summary = _quantiles(values)
            rows.append(
                asdict(
                    DistributionSummary(
                        scope="train_history_layers",
                        part=part,
                        target_column=target_name,
                        source_column_1_based=TARGET_COLUMNS[target_name],
                        finite_count=len(values),
                        **summary,
                    )
                )
            )
    for target_name, values in all_values.items():
        summary = _quantiles(values)
        rows.append(
            asdict(
                DistributionSummary(
                    scope="train_history_layers",
                    part="all_parts",
                    target_column=target_name,
                    source_column_1_based=TARGET_COLUMNS[target_name],
                    finite_count=len(values),
                    **summary,
                )
            )
        )
    return rows
def make_qc_figure(
    output_path: Path,
    inventories: list[LayerInventory],
    train_values_by_part: dict[str, dict[str, list[float]]],
    qc_points_by_part: dict[str, np.ndarray],
    qc_layer: int,
) -> None:
    """Render support coverage and response distributions; no heatmap is implied."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    part_names = sorted(qc_points_by_part)
    spatial_axis = axes[0, 0]
    scatter_handle = None
    for part in part_names:
        points = qc_points_by_part[part]
        if points.size == 0:
            continue
        scatter_handle = spatial_axis.scatter(
            points[:, 0], points[:, 1], c=points[:, 2], s=4, alpha=0.6, cmap="viridis", label=part
        )
    spatial_axis.set_title(f"Sparse registered support: layer {qc_layer:04d} (XCT 5x5x5 value)")
    spatial_axis.set_xlabel("Command X [mm]")
    spatial_axis.set_ylabel("Command Y [mm]")
    spatial_axis.set_aspect("equal", adjustable="box")
    spatial_axis.legend(loc="best", title="Part", markerscale=3)
    if scatter_handle is not None:
        fig.colorbar(scatter_handle, ax=spatial_axis, label="XCT voxel value")
    else:
        spatial_axis.text(0.5, 0.5, "No finite XCT 5x5x5 values at requested QC layer", ha="center", va="center")
    coverage_axis = axes[0, 1]
    for part in sorted({item.part for item in inventories}):
        subset = [item for item in inventories if item.part == part]
        coverage_axis.plot(
            [item.layer_z for item in subset],
            [item.xct_5x5x5_finite_fraction for item in subset],
            linewidth=1.2,
            label=part,
        )
    coverage_axis.set_title("Finite XCT 5x5x5 support by manufacturing layer")
    coverage_axis.set_xlabel("Layer z")
    coverage_axis.set_ylabel("Finite-row fraction")
    coverage_axis.set_ylim(-0.02, 1.02)
    coverage_axis.legend(loc="best", title="Part")
    histogram_axis = axes[1, 0]
    all_values: dict[str, list[float]] = {name: [] for name in TARGET_COLUMNS}
    for values_by_target in train_values_by_part.values():
        for target_name, values in values_by_target.items():
            all_values[target_name].extend(values)
    for target_name, values in all_values.items():
        if values:
            histogram_axis.hist(values, bins=80, density=True, histtype="step", linewidth=1.5, label=target_name)
    histogram_axis.set_title("Train-only XCT response distribution across all parts")
    histogram_axis.set_xlabel("XCT voxel value")
    histogram_axis.set_ylabel("Density")
    histogram_axis.legend(loc="best")
    count_axis = axes[1, 1]
    for part in sorted({item.part for item in inventories}):
        subset = [item for item in inventories if item.part == part]
        count_axis.plot(
            [item.layer_z for item in subset],
            [item.xct_5x5x5_finite_count for item in subset],
            linewidth=1.2,
            label=part,
        )
    count_axis.set_title("Finite sparse target points per layer")
    count_axis.set_xlabel("Layer z")
    count_axis.set_ylabel("Finite XCT 5x5x5 rows")
    count_axis.legend(loc="best", title="Part")
    fig.suptitle(
        "Registered XCT sparse-target audit: machine-coordinate support, not a camera-pixel heatmap",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
def main() -> None:
    args = parse_args()
    if args.qc_layer < 1:
        raise ValueError("--qc-layer must be >= 1")
    if args.max_qc_points_per_part < 1:
        raise ValueError("--max-qc-points-per-part must be >= 1")
    registered_root = args.registered_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    train_layers, train_row_count = read_train_history_layers(manifest_path)
    train_layer_set = set(train_layers)
    part_directories = find_part_directories(registered_root)
    prepare_output_directory(output_dir, overwrite=args.overwrite)
    inventories: list[LayerInventory] = []
    train_values_by_part: dict[str, dict[str, list[float]]] = {}
    qc_points_by_part: dict[str, np.ndarray] = {}
    missing_expected_train_files: list[str] = []
    for part_dir in part_directories:
        part = part_dir.name
        csv_paths = sorted(part_dir.glob("L*.csv"), key=layer_number_from_path)
        if not csv_paths:
            raise ValueError(f"No LNNNN.csv files found in {part_dir}")
        found_layers = {layer_number_from_path(path) for path in csv_paths}
        for z in train_layers:
            if z not in found_layers:
                missing_expected_train_files.append(str(part_dir / f"L{z:04d}.csv"))
        train_values_by_part[part] = {name: [] for name in TARGET_COLUMNS}
        qc_points_by_part[part] = np.empty((0, 3), dtype=np.float32)
        for csv_path in csv_paths:
            layer_z = layer_number_from_path(csv_path)
            inventory, values, qc_points = audit_layer_csv(
                part=part,
                layer_z=layer_z,
                csv_path=csv_path,
                collect_train_values=layer_z in train_layer_set,
                collect_qc_points=layer_z == args.qc_layer,
                max_qc_points=args.max_qc_points_per_part,
            )
            inventories.append(inventory)
            if layer_z in train_layer_set:
                for target_name, target_values in values.items():
                    train_values_by_part[part][target_name].extend(target_values)
            if layer_z == args.qc_layer:
                qc_points_by_part[part] = qc_points
    inventory_rows = [asdict(item) for item in inventories]
    distribution_rows = build_distribution_rows(train_values_by_part)
    write_csv(output_dir / "xct_target_inventory.csv", inventory_rows)
    write_csv(output_dir / "xct_target_train_statistics.csv", distribution_rows)
    make_qc_figure(
        output_path=output_dir / "xct_target_qc.png",
        inventories=inventories,
        train_values_by_part=train_values_by_part,
        qc_points_by_part=qc_points_by_part,
        qc_layer=args.qc_layer,
    )
    all_train_counts = {
        target_name: int(sum(len(values_by_target[target_name]) for values_by_target in train_values_by_part.values()))
        for target_name in TARGET_COLUMNS
    }
    all_layer_count = len(inventories)
    all_schema_valid = sum(item.rows_schema_40 for item in inventories)
    all_schema_invalid = sum(item.rows_bad_schema for item in inventories)
    summary = {
        "audit_type": "registered XCT sparse weak-target support audit; not a dense heatmap or binary defect label",
        "raw_input_policy": "Registered XCT CSV files are read only; no raw CSV, TIFF or metadata file is modified.",
        "registered_root": str(registered_root),
        "manifest": str(manifest_path),
        "registered_csv_schema": {
            "expected_column_count": EXPECTED_COLUMN_COUNT,
            "column_dictionary_1_based": COLUMN_DICTIONARY,
            "xct_target_interpretation": {
                "xct_original": "Column 38: original registered XCT voxel value",
                "xct_3x3x3": "Column 39: 3x3x3 mean-filtered registered XCT voxel value",
                "xct_5x5x5": "Column 40: 5x5x5 mean-filtered registered XCT voxel value",
            },
        },
        "sparse_supervision_policy": {
            "reference_coordinate_system": "machine/XYPT command coordinates in millimetres",
            "supported_locations": "Only CSV rows with finite target values",
            "unsupported_camera_pixels": "Unknown; never interpret as a negative defect label",
            "candidate_continuous_response": "xct_5x5x5",
            "binary_defect_threshold": "Not set by this audit",
            "camera_heatmap": "Not generated; requires separate machine-XY to raw-camera-pixel calibration validation",
        },
        "train_only_scope": {
            "train_manifest_row_count": train_row_count,
            "unique_train_history_layers": train_layers,
            "unique_train_history_layer_count": len(train_layers),
            "train_finite_target_counts": all_train_counts,
            "validation_and_test_target_values_used": False,
        },
        "inventory": {
            "part_directories": [path.name for path in part_directories],
            "audited_part_layer_csv_count": all_layer_count,
            "schema_valid_row_count": all_schema_valid,
            "schema_invalid_row_count": all_schema_invalid,
            "missing_expected_train_layer_files": missing_expected_train_files,
            "qc_layer": args.qc_layer,
            "max_qc_points_per_part": args.max_qc_points_per_part,
        },
        "outputs": {
            "inventory_csv": "xct_target_inventory.csv",
            "train_statistics_csv": "xct_target_train_statistics.csv",
            "qc_png": "xct_target_qc.png",
        },
        "next_dependency": "Validate machine-XY to raw-camera-pixel calibration before generating any camera-space weak heatmap.",
    }
    with (output_dir / "xct_target_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("Registered XCT sparse-target audit completed. No raw CSV, TIFF, camera heatmap or label file was modified/created.")
    print(f"- audited part/layer CSV files: {all_layer_count}")
    print(f"- train history layers used for distribution statistics: {len(train_layers)} ({train_layers[0]}..{train_layers[-1]})")
    print(f"- train finite target counts: {all_train_counts}")
    print(f"- output directory: {output_dir}")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
