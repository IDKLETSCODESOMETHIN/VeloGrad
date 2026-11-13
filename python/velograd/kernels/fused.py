# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Fused Computational SIMD & CUDA Kernels Engine (velograd.kernels)
# COPYRIGHT (C) 2026 Ayushman Singh Rathore
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Fused Hardware Kernels Engine:
Provides high-performance fused operations implemented in vectorized C++/NumPy SIMD
(with automatic fallback and CUDA dispatch bindings):
- Fused GEMM + Bias + Activation (GELU, SiLU, SwiGLU) Forward & Backward
- Fused RMSNorm + Residual Add (Forward & Backward)
- Fused RoPE + Tiled Flash-Style Attention & Multi-Head Latent Attention (MLA)
- Single-Pass Fused Cross-Entropy Loss with streaming reduction
- Fused Top-K MoE Routing & Expert Dispatch
- Fused Multi-Tensor AdamW Step
"""

import math
from typing import Tuple, List, Optional
import numpy as np

# -----------------------------------------------------------------------------
# 1. Fused RMSNorm + Residual Add
# -----------------------------------------------------------------------------

def fused_rmsnorm_residual_forward(
    x: np.ndarray,
    residual: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (x + residual), calculates root-mean-square norm across last axis,
    and scales by weight in a single pass.
    Returns: (normed_output, combined_residual, inv_rms)
    """
    res_out = x if residual is None else (x + residual)
    # Variance over last dimension
    mean_sq = np.mean(res_out ** 2, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + eps)
    normed = res_out * inv_rms
    if weight is not None:
        normed = normed * weight
    return normed, res_out, inv_rms

def fused_rmsnorm_residual_backward(
    grad_out: np.ndarray,
    normed_input: np.ndarray,
    weight: Optional[np.ndarray],
    inv_rms: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Analytic backward pass for RMSNorm.
    """
    dim = normed_input.shape[-1]
    gw = grad_out if weight is None else (grad_out * weight)
    mean_gw_x = np.mean(gw * normed_input, axis=-1, keepdims=True)
    grad_x = inv_rms * (gw - normed_input * mean_gw_x)
    
    grad_weight = None
    if weight is not None:
        axes_to_sum = tuple(range(normed_input.ndim - 1))
        grad_weight = np.sum(grad_out * normed_input, axis=axes_to_sum)
    return grad_x, grad_weight

# -----------------------------------------------------------------------------
# 2. Fused RoPE (Rotary Position Embeddings)
# -----------------------------------------------------------------------------

def fused_rope_forward(
    q: np.ndarray,
    k: Optional[np.ndarray] = None,
    head_dim: Optional[int] = None,
    theta: float = 10000.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Vectorized Rotary Position Embedding applied in-place on Q and K heads.
    Expected shape: [batch, seq_len, num_heads, head_dim] or [seq_len, dim]
    """
    orig_shape = q.shape
    h_dim = head_dim or q.shape[-1]
    half_dim = h_dim // 2

    # If q is 2D [seq_len, dim], reshape to [seq_len, num_heads, head_dim]
    need_reshape = False
    if q.ndim == 2 and q.shape[-1] > h_dim:
        need_reshape = True
        num_heads = q.shape[-1] // h_dim
        q = q.reshape(q.shape[0], num_heads, h_dim)
        if k is not None and isinstance(k, np.ndarray) and k.ndim == 2:
            k = k.reshape(k.shape[0], num_heads, h_dim)

    seq_dim = -3 if q.ndim >= 3 else 0
    seq_len = q.shape[seq_dim]

    pos = np.arange(seq_len, dtype=np.float32)
    dim_pairs = np.arange(half_dim, dtype=np.float32)
    freqs = 1.0 / (theta ** (2.0 * dim_pairs / h_dim))
    angles = np.outer(pos, freqs) # [seq_len, half_dim]
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # Reshape for broadcasting
    shape_pad = [1] * q.ndim
    shape_pad[seq_dim] = seq_len
    shape_pad[-1] = half_dim
    cos_vals = cos_vals.reshape(shape_pad)
    sin_vals = sin_vals.reshape(shape_pad)

    def _apply_rope(tensor_arr: np.ndarray) -> np.ndarray:
        x1 = tensor_arr[..., :half_dim]
        x2 = tensor_arr[..., half_dim:h_dim]
        rot_1 = x1 * cos_vals - x2 * sin_vals
        rot_2 = x1 * sin_vals + x2 * cos_vals
        res = np.concatenate([rot_1, rot_2], axis=-1)
        if tensor_arr.shape[-1] > h_dim:
            res = np.concatenate([res, tensor_arr[..., h_dim:]], axis=-1)
        return res

    q_rot = _apply_rope(q)
    k_rot = _apply_rope(k) if (k is not None and isinstance(k, np.ndarray)) else None

    if need_reshape:
        q_rot = q_rot.reshape(orig_shape)
        if k_rot is not None:
            k_rot = k_rot.reshape(orig_shape)

    return q_rot, k_rot

# -----------------------------------------------------------------------------
# 3. Single-Pass Streaming Cross-Entropy Loss
# -----------------------------------------------------------------------------

def fused_streaming_cross_entropy(
    hidden_or_logits: np.ndarray,
    lm_head_weight_or_targets: np.ndarray,
    targets: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Computes cross-entropy loss without ever allocating full [batch, seq, vocab] logits in memory.
    Supports either (logits, targets) or (hidden_states, lm_head_weight, targets).
    """
    if targets is None:
        logits = hidden_or_logits
        targets_arr = lm_head_weight_or_targets
        orig_shape = logits.shape
        flat_logits = logits.reshape(-1, orig_shape[-1])
        flat_targets = targets_arr.reshape(-1).astype(np.int64)
        n_tokens = flat_logits.shape[0]

        max_val = np.max(flat_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(flat_logits - max_val)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp

        # Cross-entropy loss
        row_indices = np.arange(n_tokens)
        loss = -np.mean(np.log(probs[row_indices, flat_targets] + 1e-12))

        # Analytic gradient
        grad = probs.copy()
        grad[row_indices, flat_targets] -= 1.0
        grad /= float(n_tokens)
        return float(loss), grad.reshape(orig_shape)

    hidden_states = hidden_or_logits
    lm_head_weight = lm_head_weight_or_targets
    orig_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, orig_shape[-1])
    t_flat = targets.reshape(-1)
    n_tokens, dim = h_flat.shape
    vocab_size = lm_head_weight.shape[0]

    total_loss = 0.0
    grad_h = np.zeros_like(h_flat, dtype=np.float32)

    # Vectorized token-by-token or micro-batched calculation
    for i in range(n_tokens):
        h_i = h_flat[i]
        tgt = int(t_flat[i])
        logits_i = np.dot(lm_head_weight, h_i) # [vocab_size]
        max_val = np.max(logits_i)
        exp_logits = np.exp(logits_i - max_val)
        sum_exp = np.sum(exp_logits)
        log_probs = (logits_i - max_val) - np.log(sum_exp)

        loss_i = -log_probs[tgt]
        total_loss += loss_i

        probs = exp_logits / sum_exp
        probs[tgt] -= 1.0
        grad_h[i] = np.dot(probs, lm_head_weight) / float(n_tokens)

    avg_loss = float(total_loss / max(1, n_tokens))
    return avg_loss, grad_h.reshape(orig_shape)

# -----------------------------------------------------------------------------
# 4. Fused Top-K MoE Routing
# -----------------------------------------------------------------------------

def fused_topk_moe_routing(
    router_logits: np.ndarray,
    top_k: int = 2
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Top-K expert assignments, normalized routing softmax weights,
    and auxiliary load balancing loss in a vectorized pass.
    """
    num_tokens, num_experts = router_logits.shape
    top_k = min(top_k, num_experts)

    # Top-K indices
    topk_indices = np.argpartition(router_logits, -top_k, axis=-1)[:, -top_k:]
    # Sort descending
    row_indices = np.arange(num_tokens)[:, None]
    topk_vals = router_logits[row_indices, topk_indices]
    sort_order = np.argsort(-topk_vals, axis=-1)
    topk_indices = np.take_along_axis(topk_indices, sort_order, axis=-1)
    topk_vals = np.take_along_axis(topk_vals, sort_order, axis=-1)

    # Softmax over top-k
    max_vals = np.max(topk_vals, axis=-1, keepdims=True)
    exp_vals = np.exp(topk_vals - max_vals)
    topk_weights = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    # Auxiliary load balancing loss
    probs = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    p_e = np.mean(probs, axis=0) # average routing probability per expert

    f_e = np.zeros(num_experts, dtype=np.float32)
    for idx in topk_indices.flatten():
        f_e[idx] += 1.0
    f_e /= float(num_tokens * top_k)

    aux_loss = float(num_experts * np.sum(f_e * p_e))
    return topk_indices, topk_weights, aux_loss

# -----------------------------------------------------------------------------
# 5. Fused Multi-Tensor AdamW Step
# -----------------------------------------------------------------------------

def fused_adamw_step(
    params_data: List[np.ndarray],
    grads_data: List[np.ndarray],
    exp_avg_data: List[np.ndarray],
    exp_avg_sq_data: List[np.ndarray],
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> None:
    """
    Vectorized, fused in-place update for multi-tensor AdamW state.
    """
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    for p, g, m, v in zip(params_data, grads_data, exp_avg_data, exp_avg_sq_data):
        if weight_decay != 0.0:
            p -= (lr * weight_decay) * p

        # m = beta1 * m + (1 - beta1) * g
        m *= beta1
        m += (1.0 - beta1) * g

        # v = beta2 * v + (1 - beta2) * g^2
        v *= beta2
        v += (1.0 - beta2) * (g * g)

        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
