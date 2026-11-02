// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Enterprise Cluster Fault Tolerance & Elastic Node Recovery (VeloGrad v5.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace velograd {
namespace fault_tolerance {

enum class NodeHealthStatus : uint8_t {
    HEALTHY = 0,
    DEGRADED = 1,
    DEAD = 2,
    ECC_UNRECOVERABLE = 3
};

struct NodeDescriptor {
    int32_t rank{-1};
    std::string ip_address;
    uint64_t last_heartbeat_timestamp_ms{0};
    NodeHealthStatus status{NodeHealthStatus::HEALTHY};
    uint64_t total_steps_completed{0};
    uint32_t ecc_single_bit_errors{0};
    uint32_t ecc_double_bit_errors{0};
};

/// Enterprise Cluster Watchdog & Elastic Node Recovery Engine.
/// Detects physical hardware bitflips, dropped InfiniBand links, and dead ranks.
/// Automatically executes safe checkpoint rollbacks and dynamic cluster re-sharding.
class ElasticClusterOrchestrator {
public:
    static ElasticClusterOrchestrator& instance();

    ElasticClusterOrchestrator();
    ~ElasticClusterOrchestrator();

    // Disable copy
    ElasticClusterOrchestrator(const ElasticClusterOrchestrator&) = delete;
    ElasticClusterOrchestrator& operator=(const ElasticClusterOrchestrator&) = delete;

    /// Registers a node with heartbeat monitoring
    void register_node(int32_t rank, const std::string& ip_address);

    /// Records periodic keep-alive heartbeat from a rank
    void record_heartbeat(int32_t rank, uint64_t step_index);

    /// Reports hardware ECC or PCIe bus error
    void report_hardware_fault(int32_t rank, bool is_double_bit_unrecoverable);

    /// Scans all ranks for dropped nodes (timeout > heartbeat_timeout_ms)
    std::vector<int32_t> scan_cluster_health(uint64_t heartbeat_timeout_ms = 5000);

    /// Executes emergency cluster re-sharding upon node failure:
    /// Evicts dead rank and recalculates partition boundaries for surviving ranks.
    bool trigger_emergency_reshard(
        const std::vector<int32_t>& failed_ranks,
        std::function<void(int32_t new_world_size)> on_reshard_complete = nullptr
    );

    /// Query cluster state
    int32_t get_active_world_size() const noexcept;
    bool is_cluster_healthy() const noexcept;
    NodeHealthStatus get_node_status(int32_t rank) const;

private:
    uint64_t current_time_ms() const noexcept;

    mutable std::mutex cluster_mutex_;
    std::unordered_map<int32_t, NodeDescriptor> nodes_;
    std::atomic<bool> cluster_healthy_{true};
    std::atomic<int32_t> active_world_size_{0};
};

} // namespace fault_tolerance
} // namespace velograd
