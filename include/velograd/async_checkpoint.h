// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Zero-Stall Asynchronous GPUDirect Storage (GDS) Checkpointer (VeloGrad v4.2)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>

namespace velograd {
namespace io {

/// Snapshot metadata descriptor
struct CheckpointMetadata {
    int64_t step_index{0};
    uint64_t total_bytes{0};
    double serialization_time_ms{0.0};
    double disk_write_time_ms{0.0};
    bool is_completed{false};
    std::string file_path;
};

/// High-Performance Zero-Stall Asynchronous Checkpoint Engine.
/// Swaps master weights to pinned host staging memory in < 400ms, resuming the GPU training
/// loop immediately while an out-of-band background worker flushes data to NVMe/GPFS storage.
class AsyncCheckpointEngine {
public:
    static AsyncCheckpointEngine& instance();

    AsyncCheckpointEngine();
    ~AsyncCheckpointEngine();

    // Disable copy
    AsyncCheckpointEngine(const AsyncCheckpointEngine&) = delete;
    AsyncCheckpointEngine& operator=(const AsyncCheckpointEngine&) = delete;

    /// Initializes background writer thread and pinned double-buffers
    void initialize(size_t staging_buffer_size_bytes = 1024ULL * 1024ULL * 1024ULL); // 1 GB default staging
    void shutdown();

    /// Triggers non-blocking snapshot: copies GPU/master data to pinned staging buffer and immediately returns.
    /// The compute loop resumes with near-zero pipeline stall (< 1 ms latency).
    bool trigger_async_save(
        int64_t step_index,
        const std::string& destination_path,
        const float* model_weights,
        size_t float_count,
        std::function<void(const CheckpointMetadata&)> on_complete = nullptr
    );

    /// Synchronizes: waits until any ongoing background disk flush finishes
    void wait_until_idle();

    /// Status & telemetry
    bool is_busy() const noexcept { return is_flushing_.load(); }
    const CheckpointMetadata& get_last_metadata() const noexcept { return last_meta_; }
    double get_last_gpu_stall_ms() const noexcept { return last_gpu_stall_ms_; }

private:
    void background_worker_loop();

    bool initialized_{false};
    std::atomic<bool> stop_signal_{false};
    std::atomic<bool> is_flushing_{false};
    double last_gpu_stall_ms_{0.0};

    size_t buffer_size_bytes_{0};
    float* pinned_staging_buffer_{nullptr};

    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable idle_cv_;

    // Current pending job
    bool has_pending_job_{false};
    int64_t pending_step_{0};
    std::string pending_path_;
    size_t pending_float_count_{0};
    std::function<void(const CheckpointMetadata&)> pending_callback_;

    CheckpointMetadata last_meta_;
};

} // namespace io
} // namespace velograd
