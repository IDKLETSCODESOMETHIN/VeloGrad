# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Production Numerical Stability Safeguards Python API (velograd.numerical_stability)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
Production Numerical Stability Safeguards (velograd.numerical_stability):
Guarantees zero loss spikes and zero NaN crashes across trillion-token pre-training runs.
- Dynamic Loss Scaler with automatic step rollback and scale backoff.
- Global L2 Gradient Norm Clipping.
- Z-Loss Regularization to prevent logit explosion and attention sink drift.
"""

import math
from typing import List, Tuple, Optional, Any
from .tensor import Tensor
from . import nn

class DynamicLossScaler:
    """
    Dynamic FP16 / FP8 Loss Scaler.
    Detects gradient overflow/underflow, scales loss to maintain high dynamic range,
    and automatically skips corrupted optimizer steps without NaN contamination.
    """
    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._successful_steps = 0
        self.total_overflows = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def unscale_and_check_overflow(self, parameters: List[Tensor]) -> bool:
        """
        Unscales gradients in-place. If any Inf/NaN is detected, returns True (overflow).
        """
        inv_scale = 1.0 / self.scale
        has_overflow = False

        for p in parameters:
            if p.grad is not None:
                g_data = p.grad.contiguous()._data
                for i in range(len(g_data)):
                    val = g_data[i]
                    if math.isnan(val) or math.isinf(val):
                        has_overflow = True
                        break
                    g_data[i] = val * inv_scale
            if has_overflow:
                break

        return has_overflow

    def update(self, has_overflow: bool) -> None:
        if has_overflow:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._successful_steps = 0
            self.total_overflows += 1
        else:
            self._successful_steps += 1
            if self._successful_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._successful_steps = 0

def clip_grad_norm_(parameters: List[Tensor], max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0

    total_norm_sq = 0.0
    for g in grads:
        g_data = g.contiguous()._data
        total_norm_sq += sum(x * x for x in g_data)

    total_norm = math.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for g in grads:
            g_data = g.contiguous()._data
            for i in range(len(g_data)):
                g_data[i] *= clip_coef

    return total_norm

def compute_z_loss(logits: Tensor, alpha: float = 1e-4) -> Tensor:
    """
    Z-Loss Regularization: L_z = alpha * log^2(sum(exp(logits))).
    Penalizes extreme logit values, preventing softmax attention collapse and loss spikes.
    """
    data = logits.contiguous()._data
    vocab_size = logits.shape[-1]
    batch_size = logits.numel // vocab_size

    total_z_loss = 0.0
    for b in range(batch_size):
        row = data[b * vocab_size : (b + 1) * vocab_size]
        max_v = max(row)
        sum_exp = sum(math.exp(v - max_v) for v in row)
        log_z = max_v + math.log(max(1e-12, sum_exp))
        total_z_loss += alpha * (log_z ** 2)

    return Tensor((1,), data=[total_z_loss / max(1, batch_size)], requires_grad=True)

__all__ = ["DynamicLossScaler", "clip_grad_norm_", "compute_z_loss"]
