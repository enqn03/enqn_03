"""Shared backbone + A/B heads skeleton for AMMT LayerCamera streams.
This is an architecture skeleton, not a finished training script.
Input convention
----------------
history: [batch, K, 3, height, width]
    K is a causal window: K-1 previous manufacturing layers plus the current
    layer. The 3 channels are the three LED observations of ONE process stage.
Operational rules
-----------------
* forward_a() runs at the A stage (after spreading, before laser exposure).
* forward_b() runs at the B stage (after laser scan).
* forward_fusion() is allowed only after the B image exists.
* A and B are NEVER used as each other's denoising targets.
* The same denoiser weights are reused for A and B; the stage embedding lets
  the network model their illumination/domain shift.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
class ConvNormAct(nn.Module):
    """3x3 convolution, GroupNorm, and SiLU activation."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
class ConvLSTMCell(nn.Module):
    """One causal ConvLSTM cell operating on encoded image features."""
    def __init__(self, input_ch: int, hidden_ch: int) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(input_ch + hidden_ch, 4 * hidden_ch, kernel_size=3, padding=1)
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor] | None) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = torch.zeros(
                x.shape[0], self.hidden_ch, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c
class StageFiLM(nn.Module):
    """Applies a learned A/B stage condition to the shared bottleneck.
    stage_id: 0 for A (after spreading), 1 for B (burned).
    """
    def __init__(self, channels: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, embedding_dim)
        self.to_scale_shift = nn.Linear(embedding_dim, 2 * channels)
    def forward(self, x: Tensor, stage_id: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(self.embedding(stage_id)).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + scale) + shift
class FrameEncoder(nn.Module):
    """U-Net encoder that processes one 3-LED frame at a time."""
    def __init__(self, input_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.level1 = nn.Sequential(ConvNormAct(input_channels, base_channels), ConvNormAct(base_channels, base_channels))
        self.level2 = nn.Sequential(ConvNormAct(base_channels, 2 * base_channels, stride=2), ConvNormAct(2 * base_channels, 2 * base_channels))
        self.level3 = nn.Sequential(ConvNormAct(2 * base_channels, 4 * base_channels, stride=2), ConvNormAct(4 * base_channels, 4 * base_channels))
    def forward(self, frame: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        skip1 = self.level1(frame)                   
        skip2 = self.level2(skip1)                         
        encoded = self.level3(skip2)                         
        return skip1, skip2, encoded
class UNetDecoder(nn.Module):
    """Decoder uses the current frame's spatial skip connections."""
    def __init__(self, base_channels: int, output_channels: int = 3) -> None:
        super().__init__()
        self.up2 = nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2)
        self.refine2 = nn.Sequential(ConvNormAct(4 * base_channels, 2 * base_channels), ConvNormAct(2 * base_channels, 2 * base_channels))
        self.up1 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=2, stride=2)
        self.refine1 = nn.Sequential(ConvNormAct(2 * base_channels, base_channels), ConvNormAct(base_channels, base_channels))
        self.residual_out = nn.Conv2d(base_channels, output_channels, kernel_size=1)
    def forward(self, temporal_feature: Tensor, skip1: Tensor, skip2: Tensor) -> Tensor:
        x = self.up2(temporal_feature)
        x = self.refine2(torch.cat([x, skip2], dim=1))
        x = self.up1(x)
        x = self.refine1(torch.cat([x, skip1], dim=1))
        return self.residual_out(x)
class SharedCausalDenoiser(nn.Module):
    """Shared A/B denoiser with causal temporal aggregation.
    The model receives only current and earlier layers. It outputs a denoised
    version of the current frame and a compact latent vector for a task head.
    """
    def __init__(self, input_channels: int = 3, base_channels: int = 32, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = FrameEncoder(input_channels=input_channels, base_channels=base_channels)
        bottleneck_channels = 4 * base_channels
        self.stage_film = StageFiLM(bottleneck_channels)
        self.temporal_cell = ConvLSTMCell(bottleneck_channels, bottleneck_channels)
        self.decoder = UNetDecoder(base_channels=base_channels, output_channels=input_channels)
        self.latent_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, latent_dim),
            nn.SiLU(inplace=True),
        )
    def forward(self, history: Tensor, stage_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (denoised_current_frame, latent).
        history has shape [B, K, C=3, H, W].
        stage_id has shape [B] and contains 0 (A) or 1 (B).
        """
        if history.ndim != 5:
            raise ValueError(f"history must be [B,K,C,H,W], got {tuple(history.shape)}")
        batch, steps, _, _, _ = history.shape
        if stage_id.shape != (batch,):
            raise ValueError(f"stage_id must be [B], got {tuple(stage_id.shape)}")
        state = None
        final_skip1 = final_skip2 = None
        for t in range(steps):
            skip1, skip2, encoded = self.encoder(history[:, t])
            encoded = self.stage_film(encoded, stage_id)
            state = self.temporal_cell(encoded, state)
            if t == steps - 1:
                final_skip1, final_skip2 = skip1, skip2
        temporal_feature, _ = state
        assert final_skip1 is not None and final_skip2 is not None
        residual = self.decoder(temporal_feature, final_skip1, final_skip2)
        denoised = history[:, -1] + residual                                  
        latent = self.latent_projection(temporal_feature)
        return denoised, latent
class AnomalyHead(nn.Module):
    """A or B task head: pixel anomaly logits plus one layer-level risk logit."""
    def __init__(self, image_channels: int = 3, latent_dim: int = 128, hidden_channels: int = 32) -> None:
        super().__init__()
        self.image_path = nn.Sequential(
            ConvNormAct(image_channels, hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
        )
        self.latent_to_map = nn.Linear(latent_dim, hidden_channels)
        self.pixel_classifier = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.risk_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels + latent_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
    def forward(self, denoised: Tensor, latent: Tensor) -> Dict[str, Tensor]:
        feature = self.image_path(denoised)
        latent_map = self.latent_to_map(latent)[:, :, None, None]
        fused = feature + latent_map
        anomaly_map_logits = self.pixel_classifier(fused)
        risk_logit = self.risk_classifier(torch.cat([F.adaptive_avg_pool2d(fused, 1).flatten(1), latent], dim=1))
        return {"anomaly_map_logits": anomaly_map_logits, "risk_logit": risk_logit}
class FusionHead(nn.Module):
    """Post-B A/B fusion head for root-cause and XCT quality-risk prediction."""
    def __init__(self, image_channels: int = 3, latent_dim: int = 128, sensor_dim: int = 10, root_classes: int = 3) -> None:
        super().__init__()
        self.image_path = nn.Sequential(
            ConvNormAct(image_channels, 32),
            ConvNormAct(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        combined_dim = 64 + 2 * latent_dim + sensor_dim
        self.mlp = nn.Sequential(nn.Linear(combined_dim, 128), nn.SiLU(inplace=True), nn.Dropout(0.1))
        self.root_cause_logits = nn.Linear(128, root_classes)                           
        self.quality_risk_logit = nn.Linear(128, 1)
    def forward(self, denoised_a: Tensor, latent_a: Tensor, denoised_b: Tensor, latent_b: Tensor, sensor: Tensor) -> Dict[str, Tensor]:
        difference = torch.abs(denoised_b - denoised_a)
        image_feature = self.image_path(difference)
        feature = self.mlp(torch.cat([image_feature, latent_a, latent_b, sensor], dim=1))
        return {
            "difference_image": difference,
            "root_cause_logits": self.root_cause_logits(feature),
            "quality_risk_logit": self.quality_risk_logit(feature),
        }
@dataclass
class StageOutput:
    denoised: Tensor
    latent: Tensor
    anomaly_map_logits: Tensor
    risk_logit: Tensor
class AMMTSharedABSystem(nn.Module):
    """Complete deployment skeleton.
    The denoiser is deliberately ONE shared module. A/B-specific logic exists
    only in their separate task heads and their stage IDs.
    """
    def __init__(self, sensor_dim: int = 10, base_channels: int = 32, latent_dim: int = 128) -> None:
        super().__init__()
        self.shared_denoiser = SharedCausalDenoiser(base_channels=base_channels, latent_dim=latent_dim)
        self.a_head = AnomalyHead(latent_dim=latent_dim)
        self.b_head = AnomalyHead(latent_dim=latent_dim)
        self.fusion_head = FusionHead(latent_dim=latent_dim, sensor_dim=sensor_dim)
    def _stage_forward(self, history: Tensor, stage: str) -> StageOutput:
        if stage not in {"A", "B"}:
            raise ValueError("stage must be 'A' or 'B'")
        stage_value = 0 if stage == "A" else 1
        stage_id = torch.full((history.shape[0],), stage_value, dtype=torch.long, device=history.device)
        denoised, latent = self.shared_denoiser(history, stage_id)
        head = self.a_head if stage == "A" else self.b_head
        prediction = head(denoised, latent)
        return StageOutput(denoised=denoised, latent=latent, **prediction)
    def forward_a(self, history_a: Tensor) -> StageOutput:
        """Call at A time. This path is available before laser exposure."""
        return self._stage_forward(history_a, stage="A")
    def forward_b(self, history_b: Tensor) -> StageOutput:
        """Call only after B is captured, i.e. after the laser scan."""
        return self._stage_forward(history_b, stage="B")
    def forward_fusion(self, history_a: Tensor, history_b: Tensor, sensor: Tensor) -> Dict[str, Tensor]:
        """Post-B analysis. Do not call this for A-time early warning."""
        a = self.forward_a(history_a)
        b = self.forward_b(history_b)
        fusion = self.fusion_head(a.denoised, a.latent, b.denoised, b.latent, sensor)
        return {"a": a, "b": b, "fusion": fusion}
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AMMTSharedABSystem(sensor_dim=10, base_channels=16, latent_dim=64).to(device)
    batch, steps, height, width = 2, 4, 256, 256
    history_a = torch.randn(batch, steps, 3, height, width, device=device)
    history_b = torch.randn(batch, steps, 3, height, width, device=device)
    daq_xypt = torch.randn(batch, 10, device=device)
    a = model.forward_a(history_a)
    b = model.forward_b(history_b)
    result = model.forward_fusion(history_a, history_b, daq_xypt)
    print("A denoised:", tuple(a.denoised.shape))
    print("A anomaly map:", tuple(a.anomaly_map_logits.shape))
    print("B denoised:", tuple(b.denoised.shape))
    print("B anomaly map:", tuple(b.anomaly_map_logits.shape))
    print("A/B difference:", tuple(result["fusion"]["difference_image"].shape))
    print("root-cause logits:", tuple(result["fusion"]["root_cause_logits"].shape))
    print("quality-risk logit:", tuple(result["fusion"]["quality_risk_logit"].shape))
