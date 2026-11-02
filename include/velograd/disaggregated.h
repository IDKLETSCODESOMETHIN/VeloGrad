// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Prefill-Decode Disaggregation & Dynamic Workload Scheduler (VeloGrad v4.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <future>
#include <chrono>

namespace velograd {
namespace scheduler {

struct InferenceRequest {
    std::string request_id;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    size_t max_new_tokens{16};
    size_t tokens_generated{0};
    bool is_prefill_completed{false};
    bool is_finished{false};
    uint64_t arrival_time_us{0};
    uint64_t ttft_us{0}; // Time to first token
    std::vector<int32_t> allocated_physical_blocks;
};

/// High-Speed Prefill-to-Decode RDMA / IPC Transmission Message
struct KVBlockDescriptor {
    std::string request_id;
    int32_t physical_block_id;
    int32_t block_size;
    int32_t num_valid_tokens;
    uint64_t rdma_remote_addr{0};
};

/// Chunked Prefill Worker Node
class ChunkedPrefillWorker {
public:
    ChunkedPrefillWorker(int32_t node_id, size_t chunk_size = 512);

    /// Processes prompt context in chunked slices, streaming KV blocks to decode worker
    bool process_prefill_chunk(InferenceRequest& request, std::vector<KVBlockDescriptor>& out_streamed_blocks);

private:
    int32_t node_id_;
    size_t chunk_size_;
};

/// Low-Latency Decode Worker Node with Speculative Tree Verification
class DecodeWorker {
public:
    DecodeWorker(int32_t node_id);

    /// Ingests streamed KV blocks from prefill worker via RDMA/IPC
    void ingest_kv_blocks(const std::vector<KVBlockDescriptor>& blocks);

    /// Executes one continuous decoding step with speculative candidate verification
    int32_t step_decode(InferenceRequest& request);

private:
    int32_t node_id_;
    std::vector<KVBlockDescriptor> resident_blocks_;
};

/// Dynamic Workload Scheduler: Disaggregates prefill and decode tasks across nodes
class DisaggregatedScheduler {
public:
    DisaggregatedScheduler(size_t num_prefill_workers = 2, size_t num_decode_workers = 4);

    void submit_request(const std::string& request_id, const std::vector<int32_t>& prompt, size_t max_tokens);

    /// Main scheduler iteration loop executing prefill chunking and decode iterations without head-of-line blocking
    size_t schedule_iteration();

    size_t get_active_prefill_count() const noexcept;
    size_t get_active_decode_count() const noexcept;
    size_t get_completed_requests_count() const noexcept;

    std::vector<InferenceRequest> drain_completed_requests();

private:
    uint64_t current_micros() const noexcept;

    mutable std::mutex queue_mutex_;
    std::queue<InferenceRequest> prefill_queue_;
    std::vector<InferenceRequest> active_decode_pool_;
    std::vector<InferenceRequest> completed_requests_;

    std::vector<std::unique_ptr<ChunkedPrefillWorker>> prefill_workers_;
    std::vector<std::unique_ptr<DecodeWorker>> decode_workers_;
};

} // namespace scheduler
} // namespace velograd
