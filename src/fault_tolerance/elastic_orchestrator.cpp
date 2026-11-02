// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Enterprise Elastic Orchestrator Implementation (VeloGrad v5.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#include "velograd/fault_tolerance.h"
#include <iostream>
#include <algorithm>

namespace velograd {
namespace fault_tolerance {

ElasticClusterOrchestrator& ElasticClusterOrchestrator::instance() {
    static ElasticClusterOrchestrator orchestrator;
    return orchestrator;
}

ElasticClusterOrchestrator::ElasticClusterOrchestrator() = default;

ElasticClusterOrchestrator::~ElasticClusterOrchestrator() = default;

uint64_t ElasticClusterOrchestrator::current_time_ms() const noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    );
}

void ElasticClusterOrchestrator::register_node(int32_t rank, const std::string& ip_address) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    NodeDescriptor desc;
    desc.rank = rank;
    desc.ip_address = ip_address;
    desc.last_heartbeat_timestamp_ms = current_time_ms();
    desc.status = NodeHealthStatus::HEALTHY;
    desc.total_steps_completed = 0;
    desc.ecc_single_bit_errors = 0;
    desc.ecc_double_bit_errors = 0;

    nodes_[rank] = desc;
    active_world_size_ = static_cast<int32_t>(nodes_.size());
}

void ElasticClusterOrchestrator::record_heartbeat(int32_t rank, uint64_t step_index) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    auto it = nodes_.find(rank);
    if (it != nodes_.end()) {
        it->second.last_heartbeat_timestamp_ms = current_time_ms();
        it->second.total_steps_completed = step_index;
        if (it->second.status == NodeHealthStatus::DEGRADED) {
            it->second.status = NodeHealthStatus::HEALTHY;
        }
    }
}

void ElasticClusterOrchestrator::report_hardware_fault(int32_t rank, bool is_double_bit_unrecoverable) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    auto it = nodes_.find(rank);
    if (it != nodes_.end()) {
        if (is_double_bit_unrecoverable) {
            it->second.ecc_double_bit_errors++;
            it->second.status = NodeHealthStatus::ECC_UNRECOVERABLE;
            cluster_healthy_ = false;
        } else {
            it->second.ecc_single_bit_errors++;
            it->second.status = NodeHealthStatus::DEGRADED;
        }
    }
}

std::vector<int32_t> ElasticClusterOrchestrator::scan_cluster_health(uint64_t heartbeat_timeout_ms) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    std::vector<int32_t> failed_ranks;
    uint64_t now = current_time_ms();

    for (auto& pair : nodes_) {
        NodeDescriptor& node = pair.second;
        if (node.status == NodeHealthStatus::ECC_UNRECOVERABLE) {
            failed_ranks.push_back(node.rank);
            continue;
        }

        uint64_t elapsed = now - node.last_heartbeat_timestamp_ms;
        if (elapsed > heartbeat_timeout_ms) {
            node.status = NodeHealthStatus::DEAD;
            failed_ranks.push_back(node.rank);
        }
    }

    if (!failed_ranks.empty()) {
        cluster_healthy_ = false;
    }

    return failed_ranks;
}

bool ElasticClusterOrchestrator::trigger_emergency_reshard(
    const std::vector<int32_t>& failed_ranks,
    std::function<void(int32_t new_world_size)> on_reshard_complete
) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    if (failed_ranks.empty()) return true;

    // Evict failed ranks from active topology
    for (int32_t rank : failed_ranks) {
        nodes_.erase(rank);
    }

    active_world_size_ = static_cast<int32_t>(nodes_.size());
    cluster_healthy_ = (active_world_size_ > 0);

    if (on_reshard_complete && active_world_size_ > 0) {
        on_reshard_complete(active_world_size_);
    }

    return (active_world_size_ > 0);
}

int32_t ElasticClusterOrchestrator::get_active_world_size() const noexcept {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    return active_world_size_.load();
}

bool ElasticClusterOrchestrator::is_cluster_healthy() const noexcept {
    return cluster_healthy_.load();
}

NodeHealthStatus ElasticClusterOrchestrator::get_node_status(int32_t rank) const {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    auto it = nodes_.find(rank);
    if (it != nodes_.end()) {
        return it->second.status;
    }
    return NodeHealthStatus::DEAD;
}

} // namespace fault_tolerance
} // namespace velograd
