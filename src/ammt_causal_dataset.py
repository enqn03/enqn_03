utf-8
#!/usr/bin/env python3
"""Causal AMMT layer-camera PyTorch Dataset with saturation validity masks.
The Dataset consumes a stage-specific TIFF (A or B), the causal manifest, and
``configs/normalization_v1.yaml``. It returns an image sequence made only from
manifest history layers, so no future manufacturing layer is accessed.
Returned tensors
----------------
* ``intensity_history``: [K, 3, H, W], float32 normalized LED intensities.
* ``validity_mask_history``: [K, 3, H, W], float32 mask where 1 means the raw
  pixel is not full-scale saturated and 0 means raw pixel == 65535.
* ``model_input_history``: [K, 6, H, W], concatenated intensity and mask.
The existing 3-channel architecture skeleton can use ``intensity_history``
while applying ``validity_mask_history`` in the loss. A later mask-aware
backbone can consume ``model_input_history`` directly.
No labels are generated here. A/B difference, XCT projection and manually
reviewed masks remain separate future target-design work.
Example: inspect one causal A-stage training sample
----------------------------------------------------
cd ~/ammt_project
/usr/local/bin/python3 src/ammt_causal_dataset.py \
  --stage A \
  --tiff raw_original/layer_camera/LayerCameraAfterSpreading.tif \
  --manifest manifests/causal_sequence_manifest.csv \
  --normalization-config configs/normalization_v1.yaml \
  --split train \
  --index 0 \
  --resize 256 256
The script never writes an image, tensor, crop or mask to disk. It only prints
an inspection summary for the requested Dataset item.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
try:
    import yaml
except ImportError as error:                                                   
    raise RuntimeError(
        "PyYAML is required to load configs/normalization_v1.yaml. "
        "Install it with: /usr/local/bin/python3 -m pip install PyYAML"
    ) from error
@dataclass(frozen=True)
class StackInfo:
    """Metadata required to index one ImageJ logical hyperstack."""
    axes: str
    shape: tuple[int, ...]
    dtype: str
    width: int
    height: int
    layers: int
    leds: int
@dataclass(frozen=True)
class Roi:
    """Raw-camera pixel rectangle with x1/y1 as exclusive bounds."""
    x0: int
    y0: int
    x1: int
    y1: int
    @property
    def width(self) -> int:
        return self.x1 - self.x0
    @property
    def height(self) -> int:
        return self.y1 - self.y0
    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
@dataclass(frozen=True)
class ManifestRow:
    """One causal endpoint plus its checked history-layer indices."""
    sample_id: str
    split: str
    endpoint_layer_z: int
    history_layer_z: tuple[int, ...]
    sequence_length_k: int
def inspect_stack(path: Path) -> StackInfo:
    """Read TIFF/ImageJ metadata without decoding the full stack."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        imagej = tif.imagej_metadata or {}
    required_axes = {"T", "Z", "Y", "X"}
    if not required_axes.issubset(set(axes)):
        raise ValueError(f"Expected TZYX-compatible stack, got axes={axes!r}")
    if dtype != np.dtype(np.uint16):
        raise ValueError(f"Expected uint16 TIFF data, got {dtype}")
    return StackInfo(
        axes=axes,
        shape=shape,
        dtype=str(dtype),
        width=shape[axes.index("X")],
        height=shape[axes.index("Y")],
        layers=int(imagej.get("slices", 1)),
        leds=int(imagej.get("frames", 1)),
    )
def read_frame(data: np.memmap, info: StackInfo, z: int, led: int) -> np.ndarray:
    """Read exactly one uint16 frame using 1-based manufacturing layer and LED."""
    if not 1 <= z <= info.layers:
        raise ValueError(f"z must be 1..{info.layers}, got {z}")
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
            raise ValueError(f"Unsupported axis {axis!r} in {info.axes!r}")
    frame = np.asarray(data[tuple(index)])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape={frame.shape}")
    return frame
def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and provide a clear error for malformed config files."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Normalization config must be a YAML mapping: {path}")
    return loaded
def parse_manifest(path: Path, split: str) -> list[ManifestRow]:
    """Read and validate only rows belonging to the requested split."""
    required = {"sample_id", "split", "endpoint_layer_z", "history_layer_z", "sequence_length_k"}
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for raw in reader:
            if raw.get("split") != split:
                continue
            try:
                history = tuple(int(value) for value in str(raw["history_layer_z"]).split(";"))
                row = ManifestRow(
                    sample_id=str(raw["sample_id"]),
                    split=str(raw["split"]),
                    endpoint_layer_z=int(raw["endpoint_layer_z"]),
                    history_layer_z=history,
                    sequence_length_k=int(raw["sequence_length_k"]),
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid manifest row: {raw}") from error
            validate_manifest_row(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest has no rows for split={split!r}")
    return rows
def validate_manifest_row(row: ManifestRow) -> None:
    """Enforce causal ordering in every sample before TIFF access."""
    if row.sequence_length_k < 1:
        raise ValueError(f"{row.sample_id}: K must be >= 1")
    if len(row.history_layer_z) != row.sequence_length_k:
        raise ValueError(f"{row.sample_id}: history length does not match K")
    if tuple(sorted(row.history_layer_z)) != row.history_layer_z:
        raise ValueError(f"{row.sample_id}: history must be ordered ascending")
    if any(current - previous != 1 for previous, current in zip(row.history_layer_z, row.history_layer_z[1:])):
        raise ValueError(f"{row.sample_id}: history layers must be contiguous")
    if row.history_layer_z[-1] != row.endpoint_layer_z:
        raise ValueError(f"{row.sample_id}: history must end at endpoint layer")
    if any(z > row.endpoint_layer_z for z in row.history_layer_z):
        raise ValueError(f"{row.sample_id}: future layer found in history")
def mapping_value(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Normalization config missing key: {key}")
    return mapping[key]
def roi_from_config(config: dict[str, Any]) -> Roi:
    working_roi = mapping_value(config, "working_roi")
    if not isinstance(working_roi, dict):
        raise ValueError("working_roi must be a mapping")
    coordinates = mapping_value(working_roi, "coordinates_raw_camera_pixels")
    if not isinstance(coordinates, list) or len(coordinates) != 4:
        raise ValueError("working_roi.coordinates_raw_camera_pixels must be [x0,y0,x1,y1]")
    return Roi(*(int(value) for value in coordinates))
class AMMTCausalStageDataset(Dataset[dict[str, Any]]):
    """Stage-specific causal Dataset with normalized intensity and validity mask.
    Parameters
    ----------
    stage:
        ``"A"`` for AfterSpreading or ``"B"`` for Burned. A/B are intentionally
        loaded independently; this Dataset does not construct B-A targets.
    tiff_path:
        TIFF for the selected stage only.
    manifest_path:
        Causal sample index. Only the selected ``split`` is exposed.
    normalization_config_path:
        ``configs/normalization_v1.yaml`` with train-only p01/p99 values.
    split:
        One of ``train``, ``validation`` or ``test``.
    resize_hw:
        Output ``(height, width)`` after crop. Set to ``None`` only when full
        working-ROI resolution is required and sufficient memory is available.
    """
    def __init__(
        self,
        stage: str,
        tiff_path: str | Path,
        manifest_path: str | Path,
        normalization_config_path: str | Path,
        split: str,
        resize_hw: tuple[int, int] | None = (256, 256),
    ) -> None:
        super().__init__()
        if stage not in {"A", "B"}:
            raise ValueError("stage must be 'A' or 'B'")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation' or 'test'")
        if resize_hw is not None and (len(resize_hw) != 2 or any(int(value) < 1 for value in resize_hw)):
            raise ValueError("resize_hw must be (height, width) with positive values, or None")
        self.stage = stage
        self.stage_id = 0 if stage == "A" else 1
        self.split = split
        self.tiff_path = Path(tiff_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.normalization_config_path = Path(normalization_config_path).resolve()
        if not self.tiff_path.is_file():
            raise FileNotFoundError(f"Missing TIFF: {self.tiff_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.normalization_config_path.is_file():
            raise FileNotFoundError(f"Missing normalization config: {self.normalization_config_path}")
        self.info = inspect_stack(self.tiff_path)
        self.config = load_yaml_mapping(self.normalization_config_path)
        self.roi = roi_from_config(self.config)
        self._validate_roi()
        self.full_scale_value = int(mapping_value(mapping_value(self.config, "raw_input"), "full_scale_value"))
        self.rows = parse_manifest(self.manifest_path, self.split)
        self.sequence_length_k = self.rows[0].sequence_length_k
        if any(row.sequence_length_k != self.sequence_length_k for row in self.rows):
            raise ValueError("Manifest rows in one split have inconsistent sequence_length_k")
        if any(z < 1 or z > self.info.layers for row in self.rows for z in row.history_layer_z):
            raise ValueError("Manifest contains a history layer outside TIFF bounds")
        self.resize_hw = None if resize_hw is None else (int(resize_hw[0]), int(resize_hw[1]))
        self._normalization = self._load_stage_normalization()
        self._data: np.memmap | None = None
    def _validate_roi(self) -> None:
        if not (0 <= self.roi.x0 < self.roi.x1 <= self.info.width):
            raise ValueError(f"ROI x range does not fit TIFF width={self.info.width}: {self.roi}")
        if not (0 <= self.roi.y0 < self.roi.y1 <= self.info.height):
            raise ValueError(f"ROI y range does not fit TIFF height={self.info.height}: {self.roi}")
    def _load_stage_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        normalization = mapping_value(self.config, "normalization")
        parameters = mapping_value(normalization, "parameters")
        stage_parameters = mapping_value(parameters, self.stage)
        p01: list[float] = []
        scale: list[float] = []
        for led in range(1, self.info.leds + 1):
            led_parameters = mapping_value(stage_parameters, f"led_{led}")
            low = float(mapping_value(led_parameters, "p01"))
            denominator = float(mapping_value(led_parameters, "scale_p99_minus_p01"))
            if denominator <= 0:
                raise ValueError(f"Non-positive normalization scale for {self.stage} LED {led}")
            p01.append(low)
            scale.append(denominator)
        return np.asarray(p01, dtype=np.float32), np.asarray(scale, dtype=np.float32)
    def _ensure_open(self) -> np.memmap:
        if self._data is None:
            self._data = tifffile.memmap(self.tiff_path, series=0, mode="r")
        return self._data
    def __getstate__(self) -> dict[str, Any]:
        """Avoid pickling an active memmap when DataLoader starts worker processes."""
        state = dict(self.__dict__)
        state["_data"] = None
        return state
    def __len__(self) -> int:
        return len(self.rows)
    def _raw_led_stack(self, z: int) -> np.ndarray:
        """Read and crop three LED frames at one causal layer endpoint/history step."""
        data = self._ensure_open()
        y_slice = slice(self.roi.y0, self.roi.y1)
        x_slice = slice(self.roi.x0, self.roi.x1)
        channels = [read_frame(data, self.info, z=z, led=led)[y_slice, x_slice] for led in range(1, self.info.leds + 1)]
        return np.stack(channels, axis=0)
    def _normalize_and_mask(self, raw_led_stack: np.ndarray) -> tuple[Tensor, Tensor]:
        """Return [3,H,W] normalized intensity and aligned validity mask tensors."""
        if raw_led_stack.shape[0] != self.info.leds:
            raise ValueError(f"Expected {self.info.leds} LED channels, got {raw_led_stack.shape}")
        valid_mask = raw_led_stack < self.full_scale_value
        raw_float = raw_led_stack.astype(np.float32, copy=False)
        intensity = (raw_float - self._normalization[0][:, None, None]) / self._normalization[1][:, None, None]
        intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32, copy=False)
        mask = valid_mask.astype(np.float32, copy=False)
        return torch.from_numpy(intensity), torch.from_numpy(mask)
    def _resize_pair(self, intensity: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.resize_hw is None:
            return intensity, mask
        if tuple(intensity.shape[-2:]) == self.resize_hw:
            return intensity, mask
        intensity_out = F.interpolate(
            intensity.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask_out = F.interpolate(mask.unsqueeze(0), size=self.resize_hw, mode="nearest").squeeze(0)
        return intensity_out, mask_out
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        intensity_steps: list[Tensor] = []
        mask_steps: list[Tensor] = []
        for z in row.history_layer_z:
            raw = self._raw_led_stack(z)
            intensity, mask = self._normalize_and_mask(raw)
            intensity, mask = self._resize_pair(intensity, mask)
            intensity_steps.append(intensity)
            mask_steps.append(mask)
        intensity_history = torch.stack(intensity_steps, dim=0).contiguous()
        validity_mask_history = torch.stack(mask_steps, dim=0).contiguous()
        model_input_history = torch.cat([intensity_history, validity_mask_history], dim=1).contiguous()
        output_height, output_width = (intensity_history.shape[-2], intensity_history.shape[-1])
        metadata = {
            "sample_id": row.sample_id,
            "stage": self.stage,
            "split": row.split,
            "endpoint_layer_z": row.endpoint_layer_z,
            "history_layer_z": list(row.history_layer_z),
            "working_roi_raw_camera_pixels": self.roi.as_dict(),
            "output_height": output_height,
            "output_width": output_width,
            "raw_pixels_per_output_pixel_y": self.roi.height / output_height,
            "raw_pixels_per_output_pixel_x": self.roi.width / output_width,
        }
        return {
            "intensity_history": intensity_history,
            "validity_mask_history": validity_mask_history,
            "model_input_history": model_input_history,
            "stage_id": torch.tensor(self.stage_id, dtype=torch.long),
            "endpoint_layer_z": torch.tensor(row.endpoint_layer_z, dtype=torch.long),
            "history_layer_z": torch.tensor(row.history_layer_z, dtype=torch.long),
            "metadata": metadata,
        }
def _json_safe_sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    """Produce a small terminal summary without writing a derived artifact."""
    intensity = sample["intensity_history"]
    mask = sample["validity_mask_history"]
    model_input = sample["model_input_history"]
    return {
        "intensity_history_shape": list(intensity.shape),
        "intensity_dtype": str(intensity.dtype),
        "intensity_min": float(intensity.min()),
        "intensity_max": float(intensity.max()),
        "validity_mask_history_shape": list(mask.shape),
        "validity_mask_dtype": str(mask.dtype),
        "valid_fraction": float(mask.mean()),
        "model_input_history_shape": list(model_input.shape),
        "stage_id": int(sample["stage_id"]),
        "endpoint_layer_z": int(sample["endpoint_layer_z"]),
        "history_layer_z": [int(value) for value in sample["history_layer_z"].tolist()],
        "metadata": sample["metadata"],
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one AMMT causal Dataset item without writing files")
    parser.add_argument("--stage", required=True, choices=["A", "B"])
    parser.add_argument("--tiff", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization-config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=0, help="Index inside the selected split")
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--full-resolution", action="store_true", help="Do not resize the working ROI; requires much more memory")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    resize_hw = None if args.full_resolution else (int(args.resize[0]), int(args.resize[1]))
    dataset = AMMTCausalStageDataset(
        stage=args.stage,
        tiff_path=args.tiff,
        manifest_path=args.manifest,
        normalization_config_path=args.normalization_config,
        split=args.split,
        resize_hw=resize_hw,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"--index must be 0..{len(dataset) - 1} for split={args.split}")
    sample = dataset[args.index]
    print(json.dumps(_json_safe_sample_summary(sample), ensure_ascii=False, indent=2))
    print("Dataset inspection complete. Raw TIFF files were opened read-only and no output file was written.")
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
