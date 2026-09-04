utf-8
#!/usr/bin/env python3
"""NIST AMMT LayerCamera ImageJ TIFF hyperstack -> PyTorch Dataset.
This version is designed for the AMMT TIFF implementation in which tifffile
reports page_count=1 even though ImageJ metadata describes 250 layers x 3 LED
views. The 6 GB pixel block is memory-mapped; no 6 GB ndarray is loaded into
RAM. Only a requested [z, LED] 2D frame is materialized.
Install:
    python -m pip install tifffile numpy matplotlib torch
Run:
    python ammt_tiff_pytorch_dataset_memmap.py \
      --tiff "LayerCameraAfterSpreading.tif" \
      --z 125 --led 1 --sequence-len 8 --resize 256 256
Dataset item : [T_sequence, C_LED, H, W]
DataLoader batch: [B, T_sequence, C_LED, H, W]
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
@dataclass(frozen=True)
class HyperstackInfo:
    """TIFF series description without reading pixel data."""
    axes: str
    shape: tuple[int, ...]
    height: int
    width: int
    n_channels: int
    n_z: int
    n_t: int
    page_count: int
    dtype: np.dtype
class AMMTTiffMemmapReader:
    """Read one AMMT ImageJ hyperstack frame at a time with numpy.memmap."""
    def __init__(self, tiff_path: str | Path) -> None:
        self.path = Path(tiff_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"TIFF 파일을 찾을 수 없습니다: {self.path}")
        self.info = self._read_header()
        self._data: Optional[np.memmap] = None
    def _read_header(self) -> HyperstackInfo:
        with tifffile.TiffFile(self.path) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = tuple(int(v) for v in series.shape)
            imagej = tif.imagej_metadata or {}
            n_channels = int(imagej.get("channels", 1))
            n_z = int(imagej.get("slices", 1))
            n_t = int(imagej.get("frames", 1))
            page0 = tif.pages[0]
            height = shape[axes.index("Y")] if "Y" in axes else int(page0.imagelength)
            width = shape[axes.index("X")] if "X" in axes else int(page0.imagewidth)
            if n_z < 1 or n_t < 1:
                raise ValueError("ImageJ metadata에서 Z/T 축 정보를 읽지 못했습니다.")
            return HyperstackInfo(
                axes=axes,
                shape=shape,
                height=height,
                width=width,
                n_channels=n_channels,
                n_z=n_z,
                n_t=n_t,
                page_count=len(tif.pages),
                dtype=np.dtype(series.dtype),
            )
    def _ensure_memmap(self) -> np.memmap:
        if self._data is None:
            try:
                self._data = tifffile.memmap(self.path, series=0, mode="r")
            except ValueError as exc:
                raise RuntimeError(
                    "이 TIFF는 메모리 매핑을 지원하지 않는 형태입니다. "
                    "원본 파일을 수정하지 말고 TIFF가 완전히 내려받아졌는지 확인하세요."
                ) from exc
        return self._data
    def close(self) -> None:
        self._data = None
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state
    def _validate_indices(self, z: int, led: int) -> None:
        if not 1 <= z <= self.info.n_z:
            raise IndexError(f"z(layer)는 1..{self.info.n_z} 범위여야 합니다: {z}")
        if not 1 <= led <= self.info.n_t:
            raise IndexError(f"LED(t)는 1..{self.info.n_t} 범위여야 합니다: {led}")
    def read_frame(self, z: int, led: int, channel: int = 1) -> np.ndarray:
        """Return exactly one 2D uint16 frame for 1-based (z, LED).
        Primary path supports tifffile logical axes such as TZYX. Fallback
        supports a generic I/Q leading stack axis with 750 frames.
        """
        self._validate_indices(z, led)
        if channel != 1:
            raise ValueError("AMMT LayerCamera TIFF은 실질적으로 channel=1만 사용합니다.")
        data = self._ensure_memmap()
        axes = self.info.axes
        if "Z" in axes and "T" in axes:
            index = []
            for axis in axes:
                if axis == "Z":
                    index.append(z - 1)
                elif axis == "T":
                    index.append(led - 1)
                elif axis == "C":
                    index.append(0)
                elif axis in ("Y", "X"):
                    index.append(slice(None))
                else:
                    raise ValueError(f"지원하지 않는 TIFF 축입니다: {axes}")
            frame = data[tuple(index)]
        elif data.ndim == 3 and data.shape[0] == self.info.n_z * self.info.n_t:
            flat_index = (led - 1) * self.info.n_z + (z - 1)
            frame = data[flat_index, :, :]
        else:
            raise ValueError(
                "자동 축 매핑에 실패했습니다. "
                f"axes={axes}, shape={tuple(data.shape)}, metadata "
                f"(Z={self.info.n_z}, T={self.info.n_t})"
            )
        frame = np.asarray(frame)
        if frame.ndim != 2:
            raise ValueError(f"2D 이미지가 아닌 결과가 나왔습니다: shape={frame.shape}")
        return frame
def show_frame(
    reader: AMMTTiffMemmapReader,
    z: int,
    led: int,
    percentile_range: Tuple[float, float] = (1.0, 99.8),
) -> np.ndarray:
    """Extract and visualize one frame with robust contrast for uint16 data."""
    frame = reader.read_frame(z=z, led=led)
    vmin, vmax = np.percentile(frame, percentile_range)
    if vmax <= vmin:
        vmax = vmin + 1
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"LayerCamera | z(layer)={z}, t(LED)={led}")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.8, label="16-bit grayscale value")
    fig.tight_layout()
    plt.show()
    return frame
class AMMTLayerSequenceDataset(Dataset):
    """Create PyTorch 4D layer-time tensors from lazy TIFF reads.
    A sample is [T_sequence, C_LED, H, W]. The `z` axis is the process-time
    axis. LED direction is a channel axis, not a time axis.
    """
    def __init__(
        self,
        tiff_path: str | Path,
        sequence_length: int = 8,
        leds: Sequence[int] = (1, 2, 3),
        stride: int = 1,
        crop_yx: Optional[Tuple[int, int, int, int]] = None,
        resize_hw: Optional[Tuple[int, int]] = (256, 256),
    ) -> None:
        self.reader = AMMTTiffMemmapReader(tiff_path)
        self.sequence_length = int(sequence_length)
        self.leds = tuple(int(v) for v in leds)
        self.stride = int(stride)
        self.crop_yx = crop_yx
        self.resize_hw = resize_hw
        if not 1 <= self.sequence_length <= self.reader.info.n_z:
            raise ValueError(f"sequence_length는 1..{self.reader.info.n_z} 범위여야 합니다.")
        if self.stride < 1:
            raise ValueError("stride는 1 이상이어야 합니다.")
        if not self.leds or len(set(self.leds)) != len(self.leds):
            raise ValueError("중복 없이 최소 하나의 LED 번호를 지정하세요.")
        for led in self.leds:
            self.reader._validate_indices(1, led)
        if crop_yx is not None:
            y0, y1, x0, x1 = crop_yx
            if not (0 <= y0 < y1 <= self.reader.info.height and
                    0 <= x0 < x1 <= self.reader.info.width):
                raise ValueError("crop_yx=(y0, y1, x0, x1)가 원본 범위를 벗어났습니다.")
        self.starts = list(range(1, self.reader.info.n_z - self.sequence_length + 2, self.stride))
    def __len__(self) -> int:
        return len(self.starts)
    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        if self.crop_yx is not None:
            y0, y1, x0, x1 = self.crop_yx
            frame = frame[y0:y1, x0:x1]
        max_value = float(np.iinfo(frame.dtype).max)
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(torch.float32) / max_value
        tensor = tensor[None, None, :, :]                
        if self.resize_hw is not None:
            tensor = F.interpolate(
                tensor,
                size=self.resize_hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor[0, 0]          
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        z_start = self.starts[index]
        timesteps = []
        for z in range(z_start, z_start + self.sequence_length):
            led_channels = [
                self._prepare_frame(self.reader.read_frame(z=z, led=led))
                for led in self.leds
            ]
            timesteps.append(torch.stack(led_channels, dim=0))             
        return {
            "x": torch.stack(timesteps, dim=0),                
            "z_start": z_start,
            "z_end": z_start + self.sequence_length - 1,
        }
def main() -> None:
    parser = argparse.ArgumentParser(description="AMMT TIFF memmap reader and PyTorch Dataset demo")
    parser.add_argument("--tiff", required=True, help="LayerCamera*.tif 경로")
    parser.add_argument("--z", type=int, default=1, help="시각화할 제조 layer: 1-based")
    parser.add_argument("--led", type=int, default=1, help="시각화할 LED: 1-based")
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--leds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"), default=[256, 256])
    parser.add_argument("--workers", type=int, default=0, help="먼저 0으로 테스트 후 2로 증가")
    args = parser.parse_args()
    reader = AMMTTiffMemmapReader(args.tiff)
    print("\n[TIFF hyperstack 정보]")
    print(reader.info)
    print("메모리 매핑은 프레임을 요청할 때만 2D uint16 배열로 읽습니다.")
    frame = show_frame(reader, z=args.z, led=args.led)
    print(f"선택 프레임: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
    dataset = AMMTLayerSequenceDataset(
        tiff_path=args.tiff,
        sequence_length=args.sequence_len,
        leds=args.leds,
        resize_hw=tuple(args.resize),
    )
    sample = dataset[0]
    print("\n[Dataset 샘플]")
    print(f"샘플 수: {len(dataset)}")
    print(f"x.shape: {tuple(sample['x'].shape)}  # [T, C, H, W]")
    print(f"layer 구간: z={sample['z_start']}..{sample['z_end']}")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    batch = next(iter(loader))
    print(f"batch['x'].shape: {tuple(batch['x'].shape)}  # [B, T, C, H, W]")
if __name__ == "__main__":
    main()
