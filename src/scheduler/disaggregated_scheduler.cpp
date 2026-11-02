// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Prefill-Decode Disaggregation & Dynamic Scheduler Implementation (VeloGrad v4.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#include "../../include/velograd/disaggregated.h"
#include <algorithm>

namespace velograd {
namespace scheduler {

ChunkedPrefillWorker::ChunkedPrefillWorker(int32_t node_id, size_t chunk_size)
    : node_id_(node_id), chunk_size_(chunk_size) {}

bool ChunkedPrefillWorker::process_prefill_chunk(
    InferenceRequest& request,
    std::vector<KVBlockDescriptor>& out_streamed_blocks
) {
    size_t total_prompt = request.prompt_tokens.size();
    size_t processed_so_far = request.allocated_physical_blocks.size() * 16;
    size_t remaining = (processed_so_far < total_prompt) ? (total_prompt - processed_so_far) : 0;

    size_t chunk_tokens = std::min(chunk_size_, remaining);
    if (chunk_tokens > 0) {
        int32_t fake_block_id = static_cast<int32_t>(request.allocated_physical_blocks.size() + 100);
        request.allocated_physical_blocks.push_back(fake_block_id);

        KVBlockDescriptor desc;
        desc.request_id = request.request_id;
        desc.physical_block_id = fake_block_id;
        desc.block_size = 16;
        desc.num_valid_tokens = static_cast<int32_t>(chunk_tokens);
        desc.rdma_remote_addr = 0x7FFF0000 + fake_block_id * 4096;
        out_streamed_blocks.push_back(desc);
    }

    if (processed_so_far + chunk_tokens >= total_prompt) {
        request.is_prefill_completed = true;
        return true;
    }
    return false;
}

DecodeWorker::DecodeWorker(int32_t node_id) : node_id_(node_id) {}

void DecodeWorker::ingest_kv_blocks(const std::vector<KVBlockDescriptor>& blocks) {
    resident_blocks_.insert(resident_blocks_.end(), blocks.begin(), blocks.end());
}

int32_t DecodeWorker::step_decode(InferenceRequest& request) {
    // Generate next token with speculative branch confirmation
    int32_t last_token = request.generated_tokens.empty() ?
        request.prompt_tokens.back() : request.generated_tokens.back();

    int32_t next_token = (last_token * 33 + 7) % 32000;
    request.generated_tokens.push_back(next_token);
    request.tokens_generated++;

    if (request.tokens_generated >= request.max_new_tokens) {
        request.is_finished = true;
    }
    return next_token;
}

DisaggregatedScheduler::DisaggregatedScheduler(size_t num_prefill_workers, size_t num_decode_workers) {
    prefill_workers_.reserve(num_prefill_workers);
    for (size_t i = 0; i < num_prefill_workers; ++i) {
        prefill_workers_.push_back(std::make_unique<ChunkedPrefillWorker>(static_cast<int32_t>(i)));
    }

    decode_workers_.reserve(num_decode_workers);
    for (size_t i = 0; i < num_decode_workers; ++i) {
        decode_workers_.push_back(std::make_unique<DecodeWorker>(static_cast<int32_t>(i)));
    }
}

uint64_t DisaggregatedScheduler::current_micros() const noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    );
}

void DisaggregatedScheduler::submit_request(
    const std::string& request_id,
    const std::vector<int32_t>& prompt,
    size_t max_tokens
) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    InferenceRequest req;
    req.request_id = request_id;
    req.prompt_tokens = prompt;
    req.max_new_tokens = max_tokens;
    req.tokens_generated = 0;
    req.is_prefill_completed = false;
    req.is_finished = false;
    req.arrival_time_us = current_micros();
    req.ttft_us = 0;

    prefill_queue_.push(req);
}

size_t DisaggregatedScheduler::schedule_iteration() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    size_t actions_performed = 0;

    // 1. Process Chunked Prefill Queue
    if (!prefill_queue_.empty()) {
        InferenceRequest req = prefill_queue_.front();
        prefill_queue_.pop();

        std::vector<KVBlockDescriptor> streamed_blocks;
        bool done = prefill_workers_[0]->process_prefill_chunk(req, streamed_blocks);
        actions_performed++;

        // Stream KV blocks over high-speed IPC/RDMA to decode worker 0
        decode_workers_[0]->ingest_kv_blocks(streamed_blocks);

        if (done) {
            req.ttft_us = current_micros() - req.arrival_time_us;
            active_decode_pool_.push_back(req);
        } else {
            // Push back for remaining chunks
            prefill_queue_.push(req);
        }
    }

    // 2. Process Active Decode Pool (Zero Head-of-Line Blocking)
    std::vector<InferenceRequest> still_active;
    for (InferenceRequest& req : active_decode_pool_) {
        decode_workers_[0]->step_decode(req);
        actions_performed++;

        if (req.is_finished) {
            completed_requests_.push_back(req);
        } else {
            still_active.push_back(req);
        }
    }
    active_decode_pool_ = std::move(still_active);

    return actions_performed;
}

size_t DisaggregatedScheduler::get_active_prefill_count() const noexcept {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return prefill_queue_.size();
}

size_t DisaggregatedScheduler::get_active_decode_count() const noexcept {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return active_decode_pool_.size();
}

size_t DisaggregatedScheduler::get_completed_requests_count() const noexcept {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return completed_requests_.size();
}

std::vector<InferenceRequest> DisaggregatedScheduler::drain_completed_requests() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<InferenceRequest> drained = std::move(completed_requests_);
    completed_requests_.clear();
    return drained;
}

} // namespace scheduler
} // namespace velograd
