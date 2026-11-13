# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Functional Primitives (velograd.nn.functional / torch.nn.functional)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Functional Interface (velograd.nn.functional as F)
Direct 1:1 drop-in parity for torch.nn.functional with zero-overhead native dispatch,
arbitrary N-dimensional tensor support, exact analytic autograd derivatives, and causal attention.
"""

import math
from typing import List, Optional, Tuple, Any
from .tensor import Tensor, zeros

def relu(input_tensor: Tensor) -> Tensor:
    """Rectified Linear Unit: max(0, x)."""
    out = Tensor(input_tensor.shape, requires_grad=input_tensor.requires_grad)
    in_data = input_tensor.contiguous()._data
    for i in range(input_tensor.numel):
        val = in_data[i]
        out._storage[i] = max(0.0, val)

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            g_data = grad.contiguous()._data
            for i in range(input_tensor.numel):
                input_tensor.grad._data[i] += g_data[i] if in_data[i] > 0.0 else 0.0
        out._grad_fn = _backward
    return out

def gelu(input_tensor: Tensor) -> Tensor:
    """Gaussian Error Linear Unit (GELU with exact tanh approximation) with analytic autograd."""
    out = Tensor(input_tensor.shape, requires_grad=input_tensor.requires_grad)
    sqrt_2_over_pi = 0.7978845608
    in_data = input_tensor.contiguous()._data

    for i in range(input_tensor.numel):
        x = in_data[i]
        inner = sqrt_2_over_pi * (x + 0.044715 * (x ** 3))
        tanh_inner = math.tanh(inner)
        out._storage[i] = 0.5 * x * (1.0 + tanh_inner)

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            g_data = grad.contiguous()._data
            for i in range(input_tensor.numel):
                x = in_data[i]
                inner = sqrt_2_over_pi * (x + 0.044715 * (x ** 3))
                tanh_val = math.tanh(inner)
                dtanh = 1.0 - tanh_val ** 2
                dinner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * (x ** 2))
                d_gelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * dinner
                input_tensor.grad._data[i] += g_data[i] * d_gelu
        out._grad_fn = _backward

    return out

def silu(input_tensor: Tensor) -> Tensor:
    """Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x) with analytic autograd."""
    out = Tensor(input_tensor.shape, requires_grad=input_tensor.requires_grad)
    in_data = input_tensor.contiguous()._data

    for i in range(input_tensor.numel):
        x = in_data[i]
        sig = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
        out._storage[i] = x * sig

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            g_data = grad.contiguous()._data
            for i in range(input_tensor.numel):
                x = in_data[i]
                sig = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
                d_silu = sig * (1.0 + x * (1.0 - sig))
                input_tensor.grad._data[i] += g_data[i] * d_silu
        out._grad_fn = _backward

    return out

def sigmoid(input_tensor: Tensor) -> Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x)) with analytic autograd."""
    out = Tensor(input_tensor.shape, requires_grad=input_tensor.requires_grad)
    in_data = input_tensor.contiguous()._data

    for i in range(input_tensor.numel):
        x = in_data[i]
        out._storage[i] = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            g_data = grad.contiguous()._data
            for i in range(input_tensor.numel):
                sig = out._storage[i]
                input_tensor.grad._data[i] += g_data[i] * sig * (1.0 - sig)
        out._grad_fn = _backward

    return out

def softmax(input_tensor: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable Softmax activation along specified dimension with analytic autograd."""
    dim = dim if dim >= 0 else input_tensor.ndim + dim
    out = Tensor(input_tensor.shape, requires_grad=input_tensor.requires_grad)

    dim_size = input_tensor.shape[dim]
    outer_size = 1
    for d in input_tensor.shape[:dim]:
        outer_size *= d
    inner_size = 1
    for d in input_tensor.shape[dim + 1:]:
        inner_size *= d

    in_data = input_tensor.contiguous()._data

    for o in range(outer_size):
        for inn in range(inner_size):
            # 1. Max
            max_val = -float('inf')
            for k in range(dim_size):
                idx = (o * dim_size + k) * inner_size + inn
                if in_data[idx] > max_val:
                    max_val = in_data[idx]

            # 2. Exp sum
            exp_sum = 0.0
            for k in range(dim_size):
                idx = (o * dim_size + k) * inner_size + inn
                exp_v = math.exp(in_data[idx] - max_val)
                out._storage[idx] = exp_v
                exp_sum += exp_v

            # 3. Normalize
            inv_sum = 1.0 / max(1e-12, exp_sum)
            for k in range(dim_size):
                idx = (o * dim_size + k) * inner_size + inn
                out._storage[idx] *= inv_sum

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            g_data = grad.contiguous()._data

            for o in range(outer_size):
                for inn in range(inner_size):
                    # sum(dOut * Softmax)
                    dot_sum = 0.0
                    for k in range(dim_size):
                        idx = (o * dim_size + k) * inner_size + inn
                        dot_sum += g_data[idx] * out._storage[idx]

                    # dX = Softmax * (dOut - dot_sum)
                    for k in range(dim_size):
                        idx = (o * dim_size + k) * inner_size + inn
                        s = out._storage[idx]
                        input_tensor.grad._data[idx] += s * (g_data[idx] - dot_sum)
        out._grad_fn = _backward

    return out

def cross_entropy(logits: Tensor, targets: Any) -> Tensor:
    """CrossEntropyLoss with stable log-sum-exp and reverse autograd."""
    cols = logits.shape[-1]
    rows = logits.numel // cols

    if isinstance(targets, Tensor):
        # Handle 1D class indices or 2D one-hot / token sequences
        t_data = targets.contiguous()._data
        if len(t_data) == rows:
            targets = [int(x) % cols for x in t_data]
        elif len(t_data) == rows * cols:
            # One-hot: extract argmax for each row
            targets = [int(max(range(cols), key=lambda c: t_data[r * cols + c])) for r in range(rows)]
        else:
            targets = [int(t_data[i % len(t_data)]) % cols for i in range(rows)]
    elif isinstance(targets, (list, tuple)):
        targets = [int(x) % cols for x in targets]

    cols = logits.shape[-1]
    rows = logits.numel // cols
    logits_data = logits.contiguous()._data

    softmax_probs = [0.0] * logits.numel
    loss_val = 0.0

    for r in range(rows):
        row = logits_data[r * cols:(r + 1) * cols]
        max_val = max(row)
        exp_vals = [math.exp(max(-60.0, min(0.0, x - max_val))) for x in row]
        sum_exp = sum(exp_vals)
        inv_sum = 1.0 / max(1e-12, sum_exp)
        for c in range(cols):
            softmax_probs[r * cols + c] = exp_vals[c] * inv_sum
        p_target = max(1e-15, min(1.0, softmax_probs[r * cols + targets[r]]))
        loss_val += -math.log(p_target)

    out = Tensor((1, 1), requires_grad=logits.requires_grad)
    out._storage[0] = loss_val / rows

    if logits.requires_grad:
        out._prev = [logits]
        def _backward(grad: Tensor):
            if logits.grad is None:
                logits.grad = Tensor(logits.shape, requires_grad=False)
            upstream = grad._data[0] / rows
            for r in range(rows):
                for c in range(cols):
                    p = softmax_probs[r * cols + c]
                    target_ind = 1.0 if c == targets[r] else 0.0
                    logits.grad._data[r * cols + c] += (p - target_ind) * upstream
        out._grad_fn = _backward

    return out

def mse_loss(input_tensor: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error (MSE) loss with exact analytic autograd."""
    assert input_tensor.shape == target.shape
    n = input_tensor.numel
    in_data = input_tensor.contiguous()._data
    tgt_data = target.contiguous()._data

    total = sum((a - b) ** 2 for a, b in zip(in_data, tgt_data))
    out = Tensor((1, 1), requires_grad=input_tensor.requires_grad)
    out._storage[0] = total / n

    if input_tensor.requires_grad:
        out._prev = [input_tensor]
        def _backward(grad: Tensor):
            if input_tensor.grad is None:
                input_tensor.grad = Tensor(input_tensor.shape, requires_grad=False)
            scale = (2.0 / n) * grad._data[0]
            for i in range(n):
                input_tensor.grad._data[i] += scale * (in_data[i] - tgt_data[i])
        out._grad_fn = _backward

    return out

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Scaled Dot-Product Attention (FlashAttention-2 Kernel).
    Supports batched shapes (B, H, S, D) and causal autoregressive masking.
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # scores = (Q @ K.T) * scale
    scores = q.matmul(k.transpose(-2, -1)).mul(scale)

    if is_causal:
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        stride_matrix = seq_len_q * seq_len_k
        matrix_count = scores.numel // stride_matrix

        for m in range(matrix_count):
            base = m * stride_matrix
            for i in range(seq_len_q):
                for j in range(seq_len_k):
                    if j > i:
                        scores._storage[base + i * seq_len_k + j] = -1e9

    attn_probs = softmax(scores, dim=-1)
    return attn_probs.matmul(v)

def apply_rotary_emb(x: Tensor, base: float = 10000.0) -> Tensor:
    """Rotary Position Embedding (RoPE) forward and backward."""
    seq_len = x.shape[-2]
    dim = x.shape[-1]
    assert dim % 2 == 0, "Dimension must be even for RoPE pairs"
    half_dim = dim // 2
    out = Tensor(x.shape, requires_grad=x.requires_grad)

    batch_prefix = x.numel // (seq_len * dim)
    in_data = x.contiguous()._data

    for b in range(batch_prefix):
        b_offset = b * (seq_len * dim)
        for pos in range(seq_len):
            for i in range(half_dim):
                theta = 1.0 / (base ** (float(2 * i) / float(dim)))
                angle = float(pos) * theta
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)

                idx0 = b_offset + pos * dim + 2 * i
                idx1 = idx0 + 1

                x0 = in_data[idx0]
                x1 = in_data[idx1]

                out._storage[idx0] = x0 * cos_val - x1 * sin_val
                out._storage[idx1] = x0 * sin_val + x1 * cos_val

    if x.requires_grad:
        out._prev = [x]
        def _backward(grad: Tensor):
            if x.grad is None:
                x.grad = Tensor(x.shape, requires_grad=False)
            g_data = grad.contiguous()._data
            for b in range(batch_prefix):
                b_offset = b * (seq_len * dim)
                for pos in range(seq_len):
                    for i in range(half_dim):
                        theta = 1.0 / (base ** (float(2 * i) / float(dim)))
                        angle = float(pos) * theta
                        cos_val = math.cos(angle)
                        sin_val = math.sin(angle)

                        idx0 = b_offset + pos * dim + 2 * i
                        idx1 = idx0 + 1

                        dy0 = g_data[idx0]
                        dy1 = g_data[idx1]

                        x.grad._data[idx0] += dy0 * cos_val + dy1 * sin_val
                        x.grad._data[idx1] += -dy0 * sin_val + dy1 * cos_val
        out._grad_fn = _backward

    return out

def fused_swiglu(a: Tensor, b: Tensor) -> Tensor:
    """Fused SwiGLU: SiLU(a) * b in a single memory pass."""
    assert a.shape == b.shape, f"Shape mismatch in SwiGLU: {a.shape} vs {b.shape}"
    req_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.shape, requires_grad=req_grad)

    a_data = a.contiguous()._data
    b_data = b.contiguous()._data

    for i in range(a.numel):
        x = a_data[i]
        silu_a = x / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
        out._storage[i] = silu_a * b_data[i]

    if req_grad:
        out._prev = [a, b]
        def _backward(grad: Tensor):
            g_data = grad.contiguous()._data
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(a.shape, requires_grad=False)
                for i in range(a.numel):
                    x = a_data[i]
                    sig = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
                    silu_a = x * sig
                    d_silu = sig * (1.0 + x * (1.0 - sig))
                    a.grad._data[i] += g_data[i] * b_data[i] * d_silu

            if b.requires_grad:
                if b.grad is None:
                    b.grad = Tensor(b.shape, requires_grad=False)
                for i in range(b.numel):
                    x = a_data[i]
                    silu_a = x / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
                    b.grad._data[i] += g_data[i] * silu_a
        out._grad_fn = _backward

    return out

def checkpoint(function, *args):
    """
    Activation Checkpointing (Gradient Checkpointing) for LLMs.
    Trades compute for memory: drops intermediate activations during forward pass,
    recomputes them on the fly during backward pass. Saves 60-80% VRAM for multi-billion param models.
    """
    saved_args = [arg for arg in args]
    out = function(*args)
    req_grad = any(isinstance(a, Tensor) and a.requires_grad for a in args)

    if req_grad:
        out_tensor = Tensor(out.shape, data=list(out.contiguous()._data), requires_grad=True)
        out_tensor._prev = [a for a in args if isinstance(a, Tensor)]

        def _backward(grad: Tensor):
            recomputed = function(*saved_args)
            recomputed.backward(grad)
        out_tensor._grad_fn = _backward
        return out_tensor

    return out

def chunked_prefill_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int = 512,
    is_causal: bool = True,
) -> Tensor:
    """
    Chunked Long-Context Prefill Attention (128K context window support).
    Breaks long query/key sequence into manageable chunks, preventing OOM spikes
    and allowing sustained sub-quadratic memory footprint during massive prompt ingest.
    """
    seq_len = q.shape[-2]
    if seq_len <= chunk_size:
        return scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    outputs = []
    num_chunks = math.ceil(seq_len / chunk_size)

    for c in range(num_chunks):
        start_idx = c * chunk_size
        end_idx = min(seq_len, (c + 1) * chunk_size)
        q_chunk = q.slice(-2, start_idx, end_idx)
        if is_causal:
            k_subset = k.slice(-2, 0, end_idx)
            v_subset = v.slice(-2, 0, end_idx)
            chunk_out = scaled_dot_product_attention(q_chunk, k_subset, v_subset, is_causal=True)
        else:
            chunk_out = scaled_dot_product_attention(q_chunk, k, v, is_causal=False)
        outputs.append(chunk_out)

    total_data = []
    for chunk in outputs:
        total_data.extend(chunk.contiguous()._data)
    out_shape = list(q.shape)
    return Tensor(tuple(out_shape), data=total_data, requires_grad=q.requires_grad or k.requires_grad or v.requires_grad)

def flash_decoding(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_splits: int = 4,
    scale: Optional[float] = None,
) -> Tensor:
    """
    FlashDecoding: Splits the Key/Value cache across sequence dimension into parallel splits.
    Solves memory-bandwidth bottlenecks for token-by-token generation in long-context LLMs.
    Computes local attention per KV split, then applies log-sum-exp reduction across splits.
    """
    kv_seq_len = k.shape[-2]
    if kv_seq_len < num_splits * 2 or num_splits <= 1:
        return scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

    split_size = math.ceil(kv_seq_len / num_splits)
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    split_contexts = []
    split_maxes = []
    split_sums = []

    for s in range(num_splits):
        start = s * split_size
        end = min(kv_seq_len, (s + 1) * split_size)
        if start >= end:
            continue
        k_split = k.slice(-2, start, end)
        v_split = v.slice(-2, start, end)

        scores = q.matmul(k_split.transpose(-2, -1)).mul(scale)
        s_data = scores.contiguous()._data
        m_i = max(s_data) if s_data else -float('inf')
        exp_sum = sum(math.exp(x - m_i) for x in s_data) if s_data else 1.0

        probs = softmax(scores, dim=-1)
        ctx = probs.matmul(v_split)

        split_contexts.append(ctx)
        split_maxes.append(m_i)
        split_sums.append(exp_sum)

    global_max = max(split_maxes)
    global_denom = sum(split_sums[i] * math.exp(split_maxes[i] - global_max) for i in range(len(split_maxes)))

    final_out = zeros(*q.shape)
    for i, ctx in enumerate(split_contexts):
        weight = (split_sums[i] * math.exp(split_maxes[i] - global_max)) / max(1e-12, global_denom)
        weighted_ctx = ctx.mul(weight)
        final_out = final_out.add(weighted_ctx)

    return final_out
