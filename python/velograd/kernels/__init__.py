# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Fused Computational SIMD & CUDA Kernels Engine (velograd.kernels)
# COPYRIGHT (C) 2026 Ayushman Singh Rathore
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

from typing import Tuple, Optional, Any
import numpy as np
from .fused import (
    fused_rmsnorm_residual_forward,
    fused_rmsnorm_residual_backward,
    fused_rope_forward,
    fused_streaming_cross_entropy,
    fused_topk_moe_routing,
    fused_adamw_step as _fused_adamw_step_raw
)

def fused_rmsnorm_residual(x: Any, residual: Optional[Any] = None, weight: Optional[Any] = None, eps: float = 1e-6) -> Any:
    from ..tensor import Tensor
    x_arr = np.asarray(x.contiguous()._data, dtype=np.float32).reshape(x.shape) if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    res_arr = None
    if residual is not None:
        res_arr = np.asarray(residual.contiguous()._data, dtype=np.float32).reshape(residual.shape) if isinstance(residual, Tensor) else np.asarray(residual, dtype=np.float32)
    w_arr = None
    if weight is not None:
        w_arr = np.asarray(weight.contiguous()._data, dtype=np.float32).reshape(weight.shape) if isinstance(weight, Tensor) else np.asarray(weight, dtype=np.float32)

    normed, _, _ = fused_rmsnorm_residual_forward(x_arr, res_arr, w_arr, eps)
    if isinstance(x, Tensor):
        return Tensor(normed.shape, data=normed.flatten().tolist(), requires_grad=x.requires_grad)
    return normed

def fused_rope(x: Any, k: Optional[Any] = None, head_dim: Optional[int] = None, base: float = 10000.0) -> Any:
    from ..tensor import Tensor
    x_arr = np.asarray(x.contiguous()._data, dtype=np.float32).reshape(x.shape) if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
    k_arr = np.asarray(k.contiguous()._data, dtype=np.float32).reshape(k.shape) if isinstance(k, Tensor) else (np.asarray(k, dtype=np.float32) if k is not None else None)

    q_rot, k_rot = fused_rope_forward(x_arr, k_arr, head_dim=head_dim, theta=base)
    if isinstance(x, Tensor):
        out_q = Tensor(q_rot.shape, data=q_rot.flatten().tolist(), requires_grad=x.requires_grad)
        if k is not None and isinstance(k, Tensor):
            out_k = Tensor(k_rot.shape, data=k_rot.flatten().tolist(), requires_grad=k.requires_grad)
            return out_q, out_k
        return out_q
    if k is not None:
        return q_rot, k_rot
    return q_rot

def fused_cross_entropy_streaming(logits: Any, targets: Any) -> Tuple[float, Any]:
    from ..tensor import Tensor
    l_arr = np.asarray(logits.contiguous()._data, dtype=np.float32).reshape(logits.shape) if isinstance(logits, Tensor) else np.asarray(logits, dtype=np.float32)
    if isinstance(targets, Tensor):
        t_arr = np.asarray(targets.contiguous()._data, dtype=np.int64)
    elif isinstance(targets, list):
        t_arr = np.asarray(targets, dtype=np.int64)
    else:
        t_arr = np.asarray(targets, dtype=np.int64)

    loss, grad = fused_streaming_cross_entropy(l_arr, t_arr)
    if isinstance(logits, Tensor):
        return float(loss), Tensor(grad.shape, data=grad.flatten().tolist())
    return float(loss), grad

def fused_moe_routing(router_logits: Any, top_k: int = 2) -> Tuple[Any, Any]:
    from ..tensor import Tensor
    r_arr = np.asarray(router_logits.contiguous()._data, dtype=np.float32).reshape(router_logits.shape) if isinstance(router_logits, Tensor) else np.asarray(router_logits, dtype=np.float32)
    indices, weights, _ = fused_topk_moe_routing(r_arr, top_k)
    if isinstance(router_logits, Tensor):
        return Tensor(indices.shape, data=indices.flatten().tolist()), Tensor(weights.shape, data=weights.flatten().tolist())
    return indices, weights

def fused_adamw_step(param: Any, grad: Any, exp_avg: Any, exp_avg_sq: Any, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01, step: int = 1) -> None:
    from ..tensor import Tensor
    p_arr = np.asarray(param.contiguous()._data if isinstance(param, Tensor) else param, dtype=np.float32)
    g_arr = np.asarray(grad.contiguous()._data if isinstance(grad, Tensor) else grad, dtype=np.float32)
    m_arr = np.asarray(exp_avg.contiguous()._data if isinstance(exp_avg, Tensor) else exp_avg, dtype=np.float32)
    v_arr = np.asarray(exp_avg_sq.contiguous()._data if isinstance(exp_avg_sq, Tensor) else exp_avg_sq, dtype=np.float32)

    _fused_adamw_step_raw([p_arr], [g_arr], [m_arr], [v_arr], step, lr, beta1, beta2, eps, weight_decay)

    if isinstance(param, Tensor):
        param._data = p_arr.flatten().tolist()
    if isinstance(exp_avg, Tensor):
        exp_avg._data = m_arr.flatten().tolist()
    if isinstance(exp_avg_sq, Tensor):
        exp_avg_sq._data = v_arr.flatten().tolist()

fused_adamw = fused_adamw_step

__all__ = [
    "fused_rmsnorm_residual",
    "fused_rmsnorm_residual_forward",
    "fused_rmsnorm_residual_backward",
    "fused_rope",
    "fused_rope_forward",
    "fused_cross_entropy_streaming",
    "fused_streaming_cross_entropy",
    "fused_moe_routing",
    "fused_topk_moe_routing",
    "fused_adamw",
    "fused_adamw_step"
]
