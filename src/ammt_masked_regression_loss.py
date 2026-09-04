utf-8
#!/usr/bin/env python3
"""Support-mask weighted continuous regression loss for AMMT weak targets.
The target is a direction-unresolved, XCT-derived continuous quality response.
Only pixels with weak_support_mask == 1 contribute to the loss. Therefore an
endpoint with no sparse XCT support has exactly zero target loss and cannot be
silently converted into a normal/negative example.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
@dataclass(frozen=True)
class MaskedRegressionLossResult:
    """Loss scalar and detached accounting values for one batch."""
    loss: Tensor
    supervised_pixel_count: int
    supervised_sample_count: int
def _validate_shapes(prediction: Tensor, target: Tensor, support_mask: Tensor) -> None:
    if prediction.shape != target.shape or prediction.shape != support_mask.shape:
        raise ValueError(
            "prediction, target, and support_mask must have identical shapes; "
            f"got {tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(support_mask.shape)}."
        )
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            "expected [batch, 1, height, width] tensors for the AMMT candidate map; "
            f"got {tuple(prediction.shape)}."
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors.")
class SupportMaskedSmoothL1Loss(nn.Module):
    """Mean Smooth L1 only over sparse XCT-supported pixels.
    The reduction is a global supported-pixel mean across the batch:
        sum(mask * smooth_l1(prediction, target)) / sum(mask)
    Samples with zero support contribute neither numerator nor denominator. If
    the whole batch has zero support, the returned scalar is differentiable but
    exactly zero, with zero gradient for prediction. This makes an early XCT
    unavailable endpoint safe to pass through a normal training loop.
    """
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}.")
        self.beta = float(beta)
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        support_mask: Tensor,
    ) -> MaskedRegressionLossResult:
        _validate_shapes(prediction, target, support_mask)
        mask = (support_mask > 0).to(dtype=prediction.dtype)
        per_pixel = F.smooth_l1_loss(prediction, target.to(dtype=prediction.dtype), beta=self.beta, reduction="none")
        supervised_pixel_count_tensor = mask.sum()
        supervised_sample_count_tensor = (mask.flatten(start_dim=1).sum(dim=1) > 0).sum()
        if int(supervised_pixel_count_tensor.detach().item()) == 0:
            loss = prediction.sum() * 0.0
        else:
            loss = (per_pixel * mask).sum() / supervised_pixel_count_tensor
        return MaskedRegressionLossResult(
            loss=loss,
            supervised_pixel_count=int(supervised_pixel_count_tensor.detach().item()),
            supervised_sample_count=int(supervised_sample_count_tensor.detach().item()),
        )
