// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Spatial Multitasking & Silicon Co-Execution Runtime (VeloGrad v4.3)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <functional>

namespace velograd {
namespace coexecution {

/// Stream Priority Channels for Spatial Multitasking
enum class StreamPriority : uint8_t {
    HIGH_TENSOR_CORE = 0,    // High Priority: Foreground GEMM Matrix Math
    LOW_STANDARD_ALU = 1     // Low Priority: Background Tokenization, Embedding, Speculative Draft
};

/// Hardware Silicon Co-Execution Manager
/// Exploits the idle 70% of the GPU die (FP32/INT32 ALUs, Copy Engines, SFUs)
/// concurrently while Tensor Cores execute heavy GEMM matrix multiplication.
class SpatialMultitasker {
public:
    static SpatialMultitasker& instance();

    SpatialMultitasker();
    ~SpatialMultitasker();

    // Disable copy
    SpatialMultitasker(const SpatialMultitasker&) = delete;
    SpatialMultitasker& operator=(const SpatialMultitasker&) = delete;

    /// Dispatches a high-priority Tensor Core task (GEMM forward pass)
    template <typename Func, typename... Args>
    auto launch_tensor_core_task(Func&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<Func, Args...>::type> 
    {
        using return_type = typename std::invoke_result<Func, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tensor_tasks_count_++;
        }
        std::thread([task]() { (*task)(); }).detach();
        return res;
    }

    /// Dispatches a background ALU task (tokenization, draft model scoring, embedding prep)
    template <typename Func, typename... Args>
    auto launch_alu_background_task(Func&& f, Args&&... args)
        -> std::future<typename std::invoke_result<Func, Args...>::type>
    {
        using return_type = typename std::invoke_result<Func, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            alu_tasks_count_++;
        }
        std::thread([task]() { (*task)(); }).detach();
        return res;
    }

    /// Telemetry
    uint64_t get_tensor_tasks_count() const noexcept { return tensor_tasks_count_.load(); }
    uint64_t get_alu_tasks_count() const noexcept { return alu_tasks_count_.load(); }

private:
    std::mutex queue_mutex_;
    std::atomic<uint64_t> tensor_tasks_count_{0};
    std::atomic<uint64_t> alu_tasks_count_{0};
};

} // namespace coexecution
} // namespace velograd
