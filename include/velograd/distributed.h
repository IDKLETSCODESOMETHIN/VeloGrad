// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Distributed Multi-GPU Parallelism & Custom Ring-AllReduce Engine (VeloGrad v4.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <future>

namespace velograd {
namespace distributed {

enum class ParallelismType : uint8_t {
    COLUMN_PARALLEL = 0,
    ROW_PARALLEL = 1,
    EXPERT_PARALLEL = 2
};

/// High-Speed Direct NVLink / Inter-Process Shared Memory Ring Buffer
struct alignas(64) IPCMemoryRing {
    int32_t rank;
    int32_t world_size;
    float* peer_ptrs[8]; // Direct P2P pointers across up to 8 GPUs / local ranks
    std::atomic<uint32_t> sync_flags[8];
};

/// Distributed Communicator & Peer-to-Peer Engine
class DistributedRuntime {
public:
    static DistributedRuntime& get_instance();

    void initialize(int32_t rank, int32_t world_size);
    void finalize();

    int32_t get_rank() const noexcept { return rank_; }
    int32_t get_world_size() const noexcept { return world_size_; }

    /// Bare-metal Ring-AllReduce: executes scatter-reduce + allgather without NCCL dependency
    void allreduce_sum(float* data, size_t count);

    /// Non-blocking Reduce-Scatter: reduces sum across world and leaves rank i with chunk i
    void reduce_scatter_sum(const float* in_data, float* out_scattered_chunk, size_t total_count);

    /// All-Gather: collects disjoint chunks from all ranks into a contiguous global buffer
    void allgather(const float* local_chunk, float* global_data, size_t chunk_count);

    /// MoE P2P Dispatch: routes tokens to target expert GPUs over direct NVLink memory rings
    void moe_dispatch_p2p(
        const float* tokens,
        const int32_t* expert_assignments,
        size_t num_tokens,
        size_t token_dim,
        float* routed_tokens,
        size_t expert_capacity
    );

    /// Broadcast from root rank
    void broadcast(float* data, size_t count, int32_t root_rank = 0);

    /// Peer-to-peer direct memory copy over NVLink / Shared Memory
    void p2p_send(const float* src, size_t count, int32_t dst_rank);
    void p2p_recv(float* dst, size_t count, int32_t src_rank);

    /// Barrier synchronization
    void barrier();

    /// Warp-Specialized 100% Comm-Compute Overlap:
    /// Concurrently executes GEMM computation while Producer threads stream NVLink P2P rings.
    void execute_overlapped_gemm_allreduce(
        const float* local_input,
        const float* weight,
        float* output,
        size_t M,
        size_t K,
        size_t N
    );

private:
    DistributedRuntime();
    ~DistributedRuntime();

    int32_t rank_{0};
    int32_t world_size_{1};
    bool initialized_{false};
    std::mutex comm_mutex_;
    std::vector<std::vector<float>> simulated_ipc_rings_;
};

/// Tensor-Parallel Linear Layer (Column-Parallel or Row-Parallel)
class TensorParallelLinear {
public:
    TensorParallelLinear(
        size_t in_features,
        size_t out_features,
        ParallelismType type,
        int32_t rank,
        int32_t world_size
    );

    /// Forward pass executing local slice GEMM + fused Ring-AllReduce if Row-Parallel
    void forward(const float* x, float* out, size_t batch_size);

    ParallelismType get_type() const noexcept { return type_; }
    size_t get_local_in_features() const noexcept { return local_in_features_; }
    size_t get_local_out_features() const noexcept { return local_out_features_; }

    std::vector<float>& get_weights() { return local_weights_; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t local_in_features_;
    size_t local_out_features_;
    ParallelismType type_;
    int32_t rank_;
    int32_t world_size_;
    std::vector<float> local_weights_;
};

/// Distributed Expert Parallelism (EP) Dispatcher & Gather Router
class DistributedMoERouter {
public:
    DistributedMoERouter(size_t num_experts, size_t dim, int32_t rank, int32_t world_size, float capacity_factor = 1.25f);

    /// Computes token assignments, groups tokens by target GPU rank, and exchanges via All-to-All
    void dispatch_and_route(
        const float* hidden_states,     // [num_tokens, dim]
        const int32_t* expert_indices,  // [num_tokens, top_k]
        const float* expert_weights,    // [num_tokens, top_k]
        size_t num_tokens,
        size_t top_k,
        std::vector<float>& out_dispatched_tokens,
        std::vector<int32_t>& out_token_indices_map
    );

private:
    size_t num_experts_;
    size_t dim_;
    int32_t rank_;
    int32_t world_size_;
    float capacity_factor_;
    size_t experts_per_rank_;
};

} // namespace distributed
} // namespace velograd
