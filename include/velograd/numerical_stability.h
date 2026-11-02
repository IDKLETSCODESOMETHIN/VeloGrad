// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Production Numerical Stability & Loss Spike Safeguards (VeloGrad v5.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace velograd {
namespace numerical {

/**
 * Computes global L2 gradient norm across all model parameter gradients.
 */
inline float compute_global_gradient_norm(const std::vector<const float*>& grad_ptrs, const std::vector<size_t>& sizes) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < grad_ptrs.size(); ++i) {
        const float* g = grad_ptrs[i];
        size_t n = sizes[i];
        for (size_t j = 0; j < n; ++j) {
            sum_sq += static_cast<double>(g[j]) * static_cast<double>(g[j]);
        }
    }
    return static_cast<float>(std::sqrt(sum_sq));
}

/**
 * Clips gradients in-place such that their global norm does not exceed max_norm.
 * Prevents gradient explosion and catastrophic parameter corruptions.
 */
inline float clip_gradient_norm_inplace(
    std::vector<float*>& grad_ptrs,
    const std::vector<size_t>& sizes,
    float max_norm
) {
    std::vector<const float*> const_ptrs(grad_ptrs.begin(), grad_ptrs.end());
    float total_norm = compute_global_gradient_norm(const_ptrs, sizes);

    if (total_norm > max_norm && total_norm > 1e-6f) {
        float clip_coef = max_norm / total_norm;
        for (size_t i = 0; i < grad_ptrs.size(); ++i) {
            float* g = grad_ptrs[i];
            size_t n = sizes[i];
            for (size_t j = 0; j < n; ++j) {
                g[j] *= clip_coef;
            }
        }
    }
    return total_norm;
}

/**
 * Z-Loss Regularization: L_z = alpha * log^2(sum(exp(logits)))
 * Prevents logit drift towards extreme magnitudes, eliminating attention sinks and loss spikes.
 */
inline float compute_z_loss_and_gradients(
    const float* logits,
    float* grad_logits,
    size_t batch_size,
    size_t vocab_size,
    float alpha = 1e-4f
) {
    float total_z_loss = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = logits + (b * vocab_size);
        float* g_row = grad_logits ? (grad_logits + (b * vocab_size)) : nullptr;

        float max_val = -1e9f;
        for (size_t v = 0; v < vocab_size; ++v) {
            if (row[v] > max_val) max_val = row[v];
        }

        double sum_exp = 0.0;
        for (size_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(static_cast<double>(row[v] - max_val));
        }

        double log_z = max_val + std::log(std::max(1e-12, sum_exp));
        float row_z_loss = alpha * static_cast<float>(log_z * log_z);
        total_z_loss += row_z_loss;

        // Gradient of z-loss w.r.t logits: 2 * alpha * log_z * softmax(logits)
        if (g_row) {
            float coef = 2.0f * alpha * static_cast<float>(log_z);
            double inv_sum = 1.0 / std::max(1e-12, sum_exp);
            for (size_t v = 0; v < vocab_size; ++v) {
                float sm = static_cast<float>(std::exp(static_cast<double>(row[v] - max_val)) * inv_sum);
                g_row[v] += coef * sm;
            }
        }
    }

    return total_z_loss / static_cast<float>(batch_size);
}

} // namespace numerical
} // namespace velograd
