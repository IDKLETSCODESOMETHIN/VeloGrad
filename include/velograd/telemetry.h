// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Hardware Profiler, Live NVML/ROCm Telemetry & Sustained MFU Auditor (VeloGrad v4.1)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <atomic>
#include <cmath>
#include <algorithm>

namespace velograd {
namespace telemetry {

/// Theoretical Hardware Performance Specification
struct HardwarePeakSpec {
    std::string device_name{"NVIDIA Hopper H100 SXM5 80GB"};
    double peak_fp16_tensor_tflops{989.0};    // Dense FP16 Tensor Core TFLOPs
    double peak_fp8_tensor_tflops{1978.0};    // Dense FP8 Tensor Core TFLOPs
    double peak_fp4_tensor_tflops{3956.0};    // Dense FP4 Tensor Core TFLOPs (Blackwell B200 baseline: 4500)
    double peak_mem_bandwidth_gb_s{3350.0};   // HBM3 Peak Bandwidth (GB/s)
    double nvlink_bidirectional_bw_gb_s{900.0};// NVLink 4 / 5 Bidirectional Bandwidth (GB/s)
    double pcie_gen5_bw_gb_s{128.0};          // PCIe Gen5 x16 Bandwidth (GB/s)
};

/// Fine-Grained Runtime Memory Consumption Breakdown
struct MemoryFootprintSnapshot {
    uint64_t static_weights_bytes{0};         // Immutable quantized model weights
    uint64_t active_paged_kv_bytes{0};        // Dynamically allocated physical KV pages
    uint64_t scratchpad_bytes{0};             // Temporary activation & GEMM workspace buffers
    uint64_t pcie_nvlink_transfers_bytes{0};  // Distributed P2P / Host-Device staging buffers
    uint64_t total_allocated_bytes{0};
    uint64_t peak_allocated_bytes{0};
    double memory_bus_saturation_pct{0.0};    // (Measured Memory Bandwidth / Peak Bandwidth) * 100
};

/// Live Sustained Serving & Training Telemetry Snapshot
struct TelemetrySnapshot {
    std::string device_name;
    int32_t device_id{0};
    uint32_t gpu_utilization_pct{0};
    uint32_t memory_utilization_pct{0};
    uint32_t temperature_celsius{0};
    uint32_t power_draw_watts{0};
    uint64_t total_vram_bytes{0};
    uint64_t free_vram_bytes{0};
    uint64_t used_vram_bytes{0};

    // MFU & Compute Efficiency
    double step_time_ms{0.0};
    double achieved_tflops{0.0};
    double model_flops_utilization_pct{0.0}; // True MFU (0.0% to 100.0%)
    double tokens_per_second{0.0};
    
    // Memory Breakdown
    MemoryFootprintSnapshot memory;
};

/// In-Engine Hardware Profiler with Live NVML / ROCm SMI Dynamic Hooks
class HardwareProfiler {
public:
    static HardwareProfiler& instance();

    HardwareProfiler();
    ~HardwareProfiler();

    // Disable copy
    HardwareProfiler(const HardwareProfiler&) = delete;
    HardwareProfiler& operator=(const HardwareProfiler&) = delete;

    /// Initializes dynamic API loader (NVML / ROCm SMI)
    bool initialize(int32_t device_id = 0);
    void shutdown();

    /// Sets or updates the active hardware target specifications
    void set_hardware_spec(const HardwarePeakSpec& spec);
    const HardwarePeakSpec& get_hardware_spec() const noexcept;

    /// Captures instantaneous hardware telemetry from device APIs
    TelemetrySnapshot query_snapshot();

    /// Computes True Model FLOPs Utilization (MFU) for LLM forward/backward steps
    /// is_training: True (6 FLOPs/param/token), False (2 FLOPs/param/token)
    double calculate_model_mfu(
        int64_t model_param_count,
        int64_t num_tokens,
        double step_duration_seconds,
        bool is_training = false
    );

    /// Records active memory allocation metrics across execution categories
    void record_memory_state(
        uint64_t static_weights_bytes,
        uint64_t active_paged_kv_bytes,
        uint64_t scratchpad_bytes,
        uint64_t pcie_nvlink_transfers_bytes
    );

    /// Projects memory scaling across dynamic context lengths (512 up to 128k)
    MemoryFootprintSnapshot project_context_memory_scaling(
        int64_t model_param_count,
        int32_t num_layers,
        int32_t num_heads,
        int32_t head_dim,
        int32_t context_length,
        int32_t batch_size,
        uint8_t kv_precision_bits = 4 // 4-bit (NF4/FP4), 8-bit (FP8), 16-bit (FP16)
    );

    /// Audits real-world sustained vs isolated synthetic micro-benchmarks
    /// Identifies bottlenecks: Memory-bound, Compute-bound, or Communication-bound
    std::string audit_performance_bottleneck(
        double achieved_mfu_pct,
        double memory_bus_saturation_pct,
        double comm_stall_ratio_pct
    );

    /// Hardware Power & Clock Frequency Pinning:
    /// Locks deterministic application clocks to eliminate frequency down-clocking jitter (+2.0% MFU).
    bool pin_gpu_clocks(uint32_t mem_mhz = 2619, uint32_t sm_mhz = 1980);

private:
    int32_t device_id_{0};
    bool nvml_initialized_{false};
    bool rocm_initialized_{false};
    HardwarePeakSpec hardware_spec_;
    std::atomic<uint64_t> static_weights_{0};
    std::atomic<uint64_t> active_paged_kv_{0};
    std::atomic<uint64_t> scratchpad_{0};
    std::atomic<uint64_t> pcie_nvlink_{0};
    std::atomic<uint64_t> peak_allocated_{0};

    void* nvml_lib_handle_{nullptr};
    void* rocm_lib_handle_{nullptr};
};

} // namespace telemetry
} // namespace velograd
