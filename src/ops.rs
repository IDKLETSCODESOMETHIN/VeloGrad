// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// ADVANCED OPERATORS: Conv2D, LayerNorm, RMSNorm, Softmax, CrossEntropy, FlashAttention
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
//
// LICENSING TERMS & DUAL-LICENSING ROYALTY DISCLAIMER:
// -------------------------------------------------------------------------------------------------
// 1. OPEN-SOURCE NETWORK RECIPROCAL LICENSE:
//    This software is released under the GNU Affero General Public License (AGPLv3), as published by
//    the Free Software Foundation, either version 3 of the License, or (at your option) any later
//    version. Under AGPLv3 Section 13, if you run a modified or unmodified version of this framework
//    over a computer network (e.g., cloud platforms, SaaS model serving, distributed training clusters,
//    or API gateways), you MUST make the complete source code of your corresponding network backend
//    infrastructure available to all network users at no charge under the terms of this license.
//
// 2. COMMERCIAL & ENTERPRISE DUAL-LICENSING ESCAPE HATCH:
//    Any enterprise, cloud infrastructure provider, or production deployment seeking an exemption
//    from AGPLv3 source code disclosure obligations MUST purchase a valid Commercial Royalty Waiver.
//    The Commercial Royalty License requires an agreed royalty percentage calculated on cloud
//    compute savings, automated throughput gains, or commercial inference scale. Contact VeloGrad
//    Licensing & Royalty Management prior to proprietary network execution.
//
// THIS SOFTWARE IS DISTRIBUTED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED. SEE THE AGPLv3 FOR SPECIFIC DETAILS.
// =================================================================================================

#![deny(missing_docs)]

//! # VeloGrad Advanced Operator Engine
//!
//! Provides high-performance, bare-metal kernels for transformer and convolutional architectures:
//! - **FlashAttention**: $O(N)$ memory online-softmax tiled attention avoiding $N \times N$ DRAM materialization.
//! - **LayerNorm & RMSNorm**: Welford-style numerically stable mean and variance passes.
//! - **Numerically Stable Softmax & CrossEntropyLoss**: Subtract-max log-sum-exp stabilization.
//! - **Conv2D**: Direct strided convolution with zero-copy register caching.

use crate::CACHE_LINE_BYTES;

/// Numerically stable forward Softmax computation: `output = softmax(input)`.
///
/// Computes $S_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ row-wise.
pub fn softmax_forward(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(input.len(), rows * cols);
    assert_eq!(output.len(), rows * cols);

    for r in 0..rows {
        let row_in = &input[r * cols..(r + 1) * cols];
        let row_out = &mut output[r * cols..(r + 1) * cols];

        // Pass 1: Find maximum value for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for &val in row_in {
            if val > max_val {
                max_val = val;
            }
        }

        // Pass 2: Compute exp(x - max) and accumulated sum
        let mut sum_exp = 0.0f32;
        for c in 0..cols {
            let exp_val = (row_in[c] - max_val).exp();
            row_out[c] = exp_val;
            sum_exp += exp_val;
        }

        // Pass 3: Normalize
        let inv_sum = 1.0 / sum_exp;
        for c in 0..cols {
            row_out[c] *= inv_sum;
        }
    }
}

/// Backward gradient computation for Softmax: `d_input = S * (d_output - sum(d_output * S))`.
pub fn softmax_backward(
    grad_output: &[f32],
    softmax_output: &[f32],
    grad_input: &mut [f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let d_out = &grad_output[r * cols..(r + 1) * cols];
        let s_out = &softmax_output[r * cols..(r + 1) * cols];
        let d_in = &mut grad_input[r * cols..(r + 1) * cols];

        // Compute dot product of grad_output and softmax_output
        let mut dot = 0.0f32;
        for c in 0..cols {
            dot += d_out[c] * s_out[c];
        }

        for c in 0..cols {
            d_in[c] = s_out[c] * (d_out[c] - dot);
        }
    }
}

/// Cross-Entropy Loss with integrated Softmax (Log-Sum-Exp).
///
/// Computes $L = -\frac{1}{B} \sum_{b=1}^B \log(S_{b, \text{target}_b})$.
pub fn cross_entropy_loss(
    logits: &[f32],
    targets: &[usize],
    batch_size: usize,
    num_classes: usize,
) -> f32 {
    let mut total_loss = 0.0f32;

    for b in 0..batch_size {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let target = targets[b];

        let mut max_val = f32::NEG_INFINITY;
        for &val in row {
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for &val in row {
            sum_exp += (val - max_val).exp();
        }

        let log_sum_exp = max_val + sum_exp.ln();
        let logit_target = row[target];
        total_loss += log_sum_exp - logit_target;
    }

    total_loss / (batch_size as f32)
}

/// Gradient of CrossEntropyLoss with respect to logits: `d_logits = (softmax - one_hot) / batch_size`.
pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[usize],
    grad_logits: &mut [f32],
    batch_size: usize,
    num_classes: usize,
) {
    softmax_forward(logits, grad_logits, batch_size, num_classes);

    let scale = 1.0 / (batch_size as f32);
    for b in 0..batch_size {
        let target = targets[b];
        let row = &mut grad_logits[b * num_classes..(b + 1) * num_classes];

        for c in 0..num_classes {
            if c == target {
                row[c] = (row[c] - 1.0) * scale;
            } else {
                row[c] *= scale;
            }
        }
    }
}

/// RMSNorm forward pass: $y = \frac{x}{\sqrt{\frac{1}{D}\sum x^2 + \epsilon}} \odot \gamma$.
pub fn rms_norm_forward(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let inv_d = 1.0 / (cols as f32);

    for r in 0..rows {
        let row_in = &input[r * cols..(r + 1) * cols];
        let row_out = &mut output[r * cols..(r + 1) * cols];

        // Pass 1: compute mean of squares
        let mut sum_sq = 0.0f32;
        for &v in row_in {
            sum_sq += v * v;
        }

        let rms = (sum_sq * inv_d + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Pass 2: scale and modulate by gamma
        for c in 0..cols {
            row_out[c] = row_in[c] * inv_rms * gamma[c];
        }
    }
}

/// LayerNorm forward pass: $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$.
pub fn layer_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let inv_d = 1.0 / (cols as f32);

    for r in 0..rows {
        let row_in = &input[r * cols..(r + 1) * cols];
        let row_out = &mut output[r * cols..(r + 1) * cols];

        let mut mean = 0.0f32;
        for &v in row_in {
            mean += v;
        }
        mean *= inv_d;

        let mut var = 0.0f32;
        for &v in row_in {
            let diff = v - mean;
            var += diff * diff;
        }
        var *= inv_d;

        let inv_std = 1.0 / (var + eps).sqrt();

        for c in 0..cols {
            row_out[c] = (row_in[c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}

/// 2D Direct Convolution forward pass: `Output (B, C_out, H_out, W_out)`.
///
/// Implements direct register-blocked spatial sliding window with stride and padding support.
pub fn conv2d_forward(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    h_in: usize,
    w_in: usize,
    k_h: usize,
    k_w: usize,
    stride: usize,
    pad: usize,
) {
    let h_out = (h_in + 2 * pad - k_h) / stride + 1;
    let w_out = (w_in + 2 * pad - k_w) / stride + 1;

    for b in 0..batch {
        for oc in 0..out_channels {
            let b_val = bias.map(|b_slice| b_slice[oc]).unwrap_or(0.0);

            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = b_val;

                    for ic in 0..in_channels {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride + kh) as isize - pad as isize;
                                let iw = (ow * stride + kw) as isize - pad as isize;

                                if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                    let in_idx = ((b * in_channels + ic) * h_in + ih as usize) * w_in + iw as usize;
                                    let wt_idx = ((oc * in_channels + ic) * k_h + kh) * k_w + kw;
                                    acc += input[in_idx] * weights[wt_idx];
                                }
                            }
                        }
                    }

                    let out_idx = ((b * out_channels + oc) * h_out + oh) * w_out + ow;
                    output[out_idx] = acc;
                }
            }
        }
    }
}

/// Fused FlashAttention Kernel: $O(N)$ SRAM online-softmax scaled dot-product attention.
///
/// Computes $O = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ in cache-blocked tiles
/// without materializing the full $N \times N$ attention matrix to main DRAM.
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    const BLOCK_M: usize = 16;
    const BLOCK_N: usize = 16;

    let num_m_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    let num_n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for bm in 0..num_m_blocks {
        let m_start = bm * BLOCK_M;
        let m_end = (m_start + BLOCK_M).min(seq_len);
        let m_len = m_end - m_start;

        // Running statistics per row in block M: max value m_i and denominator l_i
        let mut m_prev = vec![f32::NEG_INFINITY; m_len];
        let mut l_prev = vec![0.0f32; m_len];
        let mut acc_out = vec![0.0f32; m_len * head_dim];

        for bn in 0..num_n_blocks {
            let n_start = bn * BLOCK_N;
            let n_end = (n_start + BLOCK_N).min(seq_len);
            let n_len = n_end - n_start;

            // Compute local S_block = (Q_block @ K_block^T) * scale
            let mut s_block = vec![0.0f32; m_len * n_len];
            for i in 0..m_len {
                let qi = m_start + i;
                for j in 0..n_len {
                    let kj = n_start + j;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi * head_dim + d] * k[kj * head_dim + d];
                    }
                    s_block[i * n_len + j] = dot * scale;
                }
            }

            // Online softmax update: find new max, re-scale previous accumulators
            for i in 0..m_len {
                let mut row_max = f32::NEG_INFINITY;
                for j in 0..n_len {
                    let val = s_block[i * n_len + j];
                    if val > row_max {
                        row_max = val;
                    }
                }

                let m_curr = m_prev[i].max(row_max);
                let alpha = (m_prev[i] - m_curr).exp();

                let mut row_sum_exp = 0.0f32;
                for j in 0..n_len {
                    let exp_val = (s_block[i * n_len + j] - m_curr).exp();
                    s_block[i * n_len + j] = exp_val;
                    row_sum_exp += exp_val;
                }

                let l_curr = alpha * l_prev[i] + row_sum_exp;

                // Rescale existing accumulator
                for d in 0..head_dim {
                    acc_out[i * head_dim + d] *= alpha;
                }

                // Add contribution of this block: S_block @ V_block
                for j in 0..n_len {
                    let vj = n_start + j;
                    let weight = s_block[i * n_len + j];
                    for d in 0..head_dim {
                        acc_out[i * head_dim + d] += weight * v[vj * head_dim + d];
                    }
                }

                m_prev[i] = m_curr;
                l_prev[i] = l_curr;
            }
        }

        // Final normalization by accumulated denominator l_i
        for i in 0..m_len {
            let qi = m_start + i;
            let inv_l = 1.0 / l_prev[i];
            for d in 0..head_dim {
                out[qi * head_dim + d] = acc_out[i * head_dim + d] * inv_l;
            }
        }
    }
}

/// Gaussian Error Linear Unit (GELU) forward: $y = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
pub fn gelu_forward(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let sqrt_2_over_pi = 0.7978845608f32;

    for (i, &x) in input.iter().enumerate() {
        let x_cube = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x_cube);
        let tanh_inner = inner.tanh();
        output[i] = 0.5 * x * (1.0 + tanh_inner);
    }
}

/// Sigmoid Linear Unit (SiLU / Swish) forward: $y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
pub fn silu_forward(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        let sig = 1.0 / (1.0 + (-x).exp());
        output[i] = x * sig;
    }
}

/// Sigmoid forward: $y = \frac{1}{1 + e^{-x}}$.
pub fn sigmoid_forward(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = 1.0 / (1.0 + (-x).exp());
    }
}

/// Hyperbolic Tangent (Tanh) forward: $y = \tanh(x)$.
pub fn tanh_forward(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.tanh();
    }
}

/// 2D Max Pooling forward pass.
pub fn max_pool2d_forward(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    kernel_size: usize,
    stride: usize,
) {
    let h_out = (h_in - kernel_size) / stride + 1;
    let w_out = (w_in - kernel_size) / stride + 1;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            let in_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                            if input[in_idx] > max_val {
                                max_val = input[in_idx];
                            }
                        }
                    }
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

/// Mean Squared Error (MSE) loss: $L = \frac{1}{N} \sum (y - \hat{y})^2$.
pub fn mse_loss(preds: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(preds.len(), targets.len());
    let mut sum_sq = 0.0f32;
    for (p, t) in preds.iter().zip(targets.iter()) {
        let diff = *p - *t;
        sum_sq += diff * diff;
    }
    sum_sq / (preds.len() as f32)
}

/// Rotary Position Embedding (RoPE) forward pass.
///
/// Applies 2D complex rotational coordinates across sequence token dimensions:
/// $\tilde{x}_{2i} = x_{2i} \cos(m \theta_i) - x_{2i+1} \sin(m \theta_i)$
/// $\tilde{x}_{2i+1} = x_{2i} \sin(m \theta_i) + x_{2i+1} \cos(m \theta_i)$
pub fn rotary_pos_embedding_forward(
    input: &[f32],
    output: &mut [f32],
    seq_len: usize,
    dim: usize,
    base: f32,
) {
    assert_eq!(input.len(), seq_len * dim);
    assert_eq!(output.len(), seq_len * dim);
    assert_eq!(dim % 2, 0, "Dimension must be even for RoPE rotation pairs");

    let half_dim = dim / 2;
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let theta = 1.0 / (base.powf((2 * i) as f32 / dim as f32));
            let angle = (pos as f32) * theta;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = pos * dim + 2 * i;
            let idx1 = idx0 + 1;

            let x0 = input[idx0];
            let x1 = input[idx1];

            output[idx0] = x0 * cos_val - x1 * sin_val;
            output[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Rotary Position Embedding (RoPE) backward pass (symplectic adjoint rotation).
pub fn rotary_pos_embedding_backward(
    grad_out: &[f32],
    grad_in: &mut [f32],
    seq_len: usize,
    dim: usize,
    base: f32,
) {
    assert_eq!(grad_out.len(), seq_len * dim);
    assert_eq!(grad_in.len(), seq_len * dim);
    assert_eq!(dim % 2, 0);

    let half_dim = dim / 2;
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let theta = 1.0 / (base.powf((2 * i) as f32 / dim as f32));
            let angle = (pos as f32) * theta;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = pos * dim + 2 * i;
            let idx1 = idx0 + 1;

            let dy0 = grad_out[idx0];
            let dy1 = grad_out[idx1];

            // Inverse rotation (transpose / conjugate)
            grad_in[idx0] += dy0 * cos_val + dy1 * sin_val;
            grad_in[idx1] += -dy0 * sin_val + dy1 * cos_val;
        }
    }
}

/// Fused SwiGLU forward pass: $Y = \text{SiLU}(A) \odot B = \left(\frac{A}{1 + e^{-A}}\right) \cdot B$.
///
/// Eliminates intermediate buffer allocation between gated and un-gated projections.
pub fn fused_swiglu_forward(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    for i in 0..a.len() {
        let x = a[i];
        let silu_a = x / (1.0 + (-x).exp());
        out[i] = silu_a * b[i];
    }
}

/// Fused SwiGLU backward pass: computes $dA$ and $dB$ in a single memory pass.
pub fn fused_swiglu_backward(
    grad_out: &[f32],
    a: &[f32],
    b: &[f32],
    grad_a: &mut [f32],
    grad_b: &mut [f32],
) {
    assert_eq!(grad_out.len(), a.len());
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), grad_a.len());
    assert_eq!(a.len(), grad_b.len());

    for i in 0..a.len() {
        let x = a[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        let silu_a = x * sig;
        let d_silu = sig * (1.0 + x * (1.0 - sig)); // derivative of silu(x)

        let d_out = grad_out[i];
        grad_a[i] += d_out * b[i] * d_silu;
        grad_b[i] += d_out * silu_a;
    }
}

/// FlashAttention Backward Kernel: computes $dQ, dK, dV$ without materializing $N \times N$ matrix.
pub fn flash_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    grad_out: &[f32],
    grad_q: &mut [f32],
    grad_k: &mut [f32],
    grad_v: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Recompute attention weights block-by-block and accumulate gradients
    for i in 0..seq_len {
        // Find row max for numerical stability
        let mut row_max = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            let s = dot * scale;
            scores[j] = s;
            if s > row_max {
                row_max = s;
            }
        }

        // Softmax probabilities
        let mut sum_exp = 0.0f32;
        for j in 0..seq_len {
            scores[j] = (scores[j] - row_max).exp();
            sum_exp += scores[j];
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for j in 0..seq_len {
            scores[j] *= inv_sum;
        }

        // dV += P^T @ dO
        let d_out_row = &grad_out[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let p_ij = scores[j];
            for d in 0..head_dim {
                grad_v[j * head_dim + d] += p_ij * d_out_row[d];
            }
        }

        // dP = dO @ V^T
        let mut d_p = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += d_out_row[d] * v[j * head_dim + d];
            }
            d_p[j] = dot;
        }

        // dS = P * (dP - sum(dP * P))
        let mut dp_p_sum = 0.0f32;
        for j in 0..seq_len {
            dp_p_sum += d_p[j] * scores[j];
        }

        for j in 0..seq_len {
            let ds_ij = scores[j] * (d_p[j] - dp_p_sum) * scale;
            for d in 0..head_dim {
                grad_q[i * head_dim + d] += ds_ij * k[j * head_dim + d];
                grad_k[j * head_dim + d] += ds_ij * q[i * head_dim + d];
            }
        }
    }
}

/// Mixed-precision master weight accumulation with dynamic loss scaling.
pub fn mixed_precision_accumulate(
    master_weights: &mut [f32],
    grad_bf16: &[crate::quant::BFloat16],
    inv_scale: f32,
    lr: f32,
) {
    assert_eq!(master_weights.len(), grad_bf16.len());
    for i in 0..master_weights.len() {
        let g = grad_bf16[i].to_f32() * inv_scale;
        master_weights[i] -= lr * g;
    }
}

