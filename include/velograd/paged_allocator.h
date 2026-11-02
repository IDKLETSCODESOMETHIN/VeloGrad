// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Hardware-Level Zero-Fragmentation Paged Memory & Radix Prefix Cache (VeloGrad v4.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <chrono>

namespace velograd {
namespace memory {

constexpr size_t PAGE_ALIGNMENT = 64;

/// Sub-block allocation granularity
enum class BlockGranularity : uint8_t {
    BLOCK_4_TOKENS = 4,
    BLOCK_8_TOKENS = 8,
    BLOCK_16_TOKENS = 16
};

/// Physical Memory Block Descriptor
struct alignas(PAGE_ALIGNMENT) PhysicalPage {
    int32_t physical_block_id{-1};
    int32_t block_size{16};          // 4, 8, or 16
    int32_t head_dim{64};
    int32_t num_tokens{0};
    std::atomic<uint32_t> ref_count{0}; // Atomic reference count for prefix sharing
    bool is_evicted_to_host{false};
    uint64_t last_accessed_timestamp{0};

    // Device / GPU memory pointers
    float* dev_k_ptr{nullptr};
    float* dev_v_ptr{nullptr};

    // Host pinned memory pointers for eviction
    float* host_pinned_k_ptr{nullptr};
    float* host_pinned_v_ptr{nullptr};
};

/// Radix Tree Node for Sub-Microsecond Prefix KV Reuse
struct RadixNode {
    std::vector<int32_t> prefix_tokens;
    std::vector<int32_t> physical_block_ids;
    std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;
    std::weak_ptr<RadixNode> parent;
    bool is_locked{false};
};

/// High-Performance Hardware-Level Paged Table Allocator & Radix Prefix Cache
class PagedAllocator {
public:
    explicit PagedAllocator(
        size_t total_gpu_blocks,
        size_t total_host_blocks,
        BlockGranularity granularity = BlockGranularity::BLOCK_16_TOKENS,
        int32_t head_dim = 64
    );

    ~PagedAllocator();

    // Disable copy
    PagedAllocator(const PagedAllocator&) = delete;
    PagedAllocator& operator=(const PagedAllocator&) = delete;

    /// Allocates physical blocks for a given virtual request ID
    std::vector<int32_t> allocate_blocks(const std::string& request_id, size_t num_tokens);

    /// Appends a token vector to the physical block assigned to request_id
    bool append_token(const std::string& request_id, const float* k_vec, const float* v_vec);

    /// Radix Tree Prefix Lookup: Finds existing physical blocks matching prefix tokens
    std::vector<int32_t> match_prefix(const std::vector<int32_t>& prompt_tokens, size_t& out_matched_token_count);

    /// Registers a completed sequence prefix into the immutable Radix Tree with ref counting
    void insert_prefix(const std::vector<int32_t>& prompt_tokens, const std::vector<int32_t>& block_ids);

    /// Copy-On-Write (CoW) Branching: creates a lightweight branch of source_request_id with zero duplicate KV compute
    bool branch_request_cow(const std::string& source_request_id, const std::string& new_request_id);

    /// Frees blocks assigned to request_id, updating reference counts
    void free_request(const std::string& request_id);

    /// Asynchronous host-GPU paging eviction: Swaps cold physical pages to host pinned memory
    size_t evict_cold_pages_to_host(size_t max_blocks_to_evict);

    /// Transparently restores evicted page back to GPU memory
    bool restore_page_to_gpu(int32_t physical_block_id);

    /// Introspection
    size_t get_free_gpu_blocks() const noexcept;
    size_t get_total_gpu_blocks() const noexcept;
    size_t get_evicted_blocks_count() const noexcept;
    float get_fragmentation_ratio() const noexcept;

private:
    uint64_t current_time_micros() const noexcept;

    size_t total_gpu_blocks_;
    size_t total_host_blocks_;
    int32_t block_size_;
    int32_t head_dim_;

    mutable std::mutex mutex_;
    std::vector<PhysicalPage> gpu_pages_;
    std::vector<int32_t> free_gpu_block_ids_;
    std::unordered_map<std::string, std::vector<int32_t>> virtual_to_physical_map_;

    // Radix Tree Prefix Cache Root
    std::shared_ptr<RadixNode> radix_root_;
};

} // namespace memory
} // namespace velograd
