// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Bare-Metal InfiniBand RDMA Verbs C-API Driver (VeloGrad v5.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

namespace velograd {
namespace rdma {

enum class RDMAOpType : uint8_t {
    READ = 0,
    WRITE = 1,
    SEND = 2,
    RECV = 3
};

struct RDMAMemoryRegion {
    void* addr{nullptr};
    size_t length{0};
    uint32_t lkey{0};
    uint32_t rkey{0};
};

struct RDMAQueuePairConfig {
    int32_t local_rank{0};
    int32_t remote_rank{0};
    uint32_t qp_num{0};
    uint16_t lid{0};
    uint8_t gid[16]{0};
};

/// High-Speed Bare-Metal InfiniBand & RoCEv2 Verbs Interconnect.
/// Dynamically hooks libibverbs / librdmacm for direct hardware memory registration
/// and zero-copy InfiniBand Queue Pair (QP) DMA transfers.
class InfiniBandRDMAEngine {
public:
    static InfiniBandRDMAEngine& instance();

    InfiniBandRDMAEngine();
    ~InfiniBandRDMAEngine();

    // Disable copy
    InfiniBandRDMAEngine(const InfiniBandRDMAEngine&) = delete;
    InfiniBandRDMAEngine& operator=(const InfiniBandRDMAEngine&) = delete;

    /// Attempts dynamic initialization of InfiniBand device context (e.g. mlx5_0)
    bool initialize(const std::string& dev_name = "mlx5_0", uint8_t ib_port = 1);
    void shutdown();

    /// Registers a contiguous host/GPU memory buffer for direct RDMA transfers
    RDMAMemoryRegion register_memory(void* buffer, size_t length);

    /// Deregisters an RDMA memory region
    void deregister_memory(const RDMAMemoryRegion& region);

    /// Establishes an RDMA Queue Pair connection with a remote peer
    bool connect_queue_pair(const RDMAQueuePairConfig& local_cfg, const RDMAQueuePairConfig& remote_cfg);

    /// Issues asynchronous zero-copy RDMA Write / Send across InfiniBand
    bool post_rdma_transfer(
        int32_t remote_rank,
        RDMAOpType op,
        const void* local_buf,
        size_t bytes,
        uint64_t remote_addr = 0,
        uint32_t rkey = 0
    );

    /// Polls completion queue for finished transfers
    int32_t poll_completions(int32_t max_completions = 16);

    bool is_available() const noexcept { return is_initialized_; }
    const std::string& get_device_name() const noexcept { return active_device_; }

private:
    bool is_initialized_{false};
    std::string active_device_;
    uint8_t port_{1};
    void* ib_ctx_{nullptr};
    void* pd_{nullptr};
    std::mutex rdma_mutex_;
    std::vector<RDMAQueuePairConfig> peers_;
};

} // namespace rdma
} // namespace velograd
