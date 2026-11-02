// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Unpadded Ragged Attention & Dynamic Token Packing (VeloGrad v4.2)
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
namespace ragged {

/// Descriptor for unpadded continuous ragged batch
struct RaggedBatchDescriptor {
    int32_t total_tokens{0};
    int32_t num_sequences{0};
    std::vector<int32_t> sequence_lengths;
    std::vector<int32_t> cumulative_offsets; // Size: num_sequences + 1
};

/**
 * Packs multiple variable-length sequences into a single continuous 1D token stream.
 * Eliminates all zero/pad tokens and prevents Tensor Core waste.
 */
inline RaggedBatchDescriptor pack_ragged_sequences(
    const std::vector<std::vector<int32_t>>& sequences,
    std::vector<int32_t>& out_packed_tokens
) {
    RaggedBatchDescriptor desc;
    desc.num_sequences = static_cast<int32_t>(sequences.size());
    desc.sequence_lengths.resize(desc.num_sequences);
    desc.cumulative_offsets.resize(desc.num_sequences + 1, 0);

    size_t total = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        desc.sequence_lengths[i] = static_cast<int32_t>(sequences[i].size());
        total += sequences[i].size();
    }
    desc.total_tokens = static_cast<int32_t>(total);
    out_packed_tokens.resize(total);

    int32_t cursor = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        desc.cumulative_offsets[i] = cursor;
        std::copy(sequences[i].begin(), sequences[i].end(), out_packed_tokens.begin() + cursor);
        cursor += static_cast<int32_t>(sequences[i].size());
    }
    desc.cumulative_offsets[desc.num_sequences] = cursor;

    return desc;
}

/**
 * Variable-Length Unpadded Attention Kernel.
 * Computes self-attention across packed continuous token streams using offset lookups.
 * Tensor Cores compute exactly the valid causal tokens per sequence with zero padding overhead.
 */
inline void unpadded_ragged_attention_forward(
    const float* __restrict__ q_packed,        // [total_tokens, num_heads * head_dim]
    const float* __restrict__ k_packed,        // [total_tokens, num_heads * head_dim]
    const float* __restrict__ v_packed,        // [total_tokens, num_heads * head_dim]
    const int32_t* __restrict__ cu_offsets,    // [num_seqs + 1]
    int32_t num_sequences,
    int32_t num_heads,
    int32_t head_dim,
    float scale,
    float* __restrict__ out_packed             // [total_tokens, num_heads * head_dim]
) noexcept {
    int32_t hidden_dim = num_heads * head_dim;

    for (int32_t s = 0; s < num_sequences; ++s) {
        int32_t seq_start = cu_offsets[s];
        int32_t seq_end = cu_offsets[s + 1];
        int32_t seq_len = seq_end - seq_start;

        if (seq_len <= 0) continue;

        // Process attention for this unpadded sequence
        for (int32_t h = 0; h < num_heads; ++h) {
            int32_t h_offset = h * head_dim;

            // Q @ K^T causal attention with online softmax
            for (int32_t i = 0; i < seq_len; ++i) {
                const float* q_tok = q_packed + (seq_start + i) * hidden_dim + h_offset;
                float* out_tok = out_packed + (seq_start + i) * hidden_dim + h_offset;

                std::vector<float> scores(i + 1);
                float max_score = -1e9f;

                for (int32_t j = 0; j <= i; ++j) {
                    const float* k_tok = k_packed + (seq_start + j) * hidden_dim + h_offset;
                    float dot = 0.0f;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        dot += q_tok[d] * k_tok[d];
                    }
                    float s_val = dot * scale;
                    scores[j] = s_val;
                    if (s_val > max_score) max_score = s_val;
                }

                float sum_exp = 0.0f;
                for (int32_t j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / std::max(1e-12f, sum_exp);

                // Compute output = Softmax @ V
                for (int32_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int32_t j = 0; j <= i; ++j) {
                        const float* v_tok = v_packed + (seq_start + j) * hidden_dim + h_offset;
                        acc += scores[j] * inv_sum * v_tok[d];
                    }
                    out_tok[d] = acc;
                }
            }
        }
    }
}

} // namespace ragged
} // namespace velograd
