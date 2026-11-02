// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Hardware-Level Paged Table Allocator & Radix Prefix Cache Implementation
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#include "../../include/velograd/paged_allocator.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace velograd {
namespace memory {

PagedAllocator::PagedAllocator(
    size_t total_gpu_blocks,
    size_t total_host_blocks,
    BlockGranularity granularity,
    int32_t head_dim
) : total_gpu_blocks_(total_gpu_blocks),
    total_host_blocks_(total_host_blocks),
    block_size_(static_cast<int32_t>(granularity)),
    head_dim_(head_dim)
{
    gpu_pages_.resize(total_gpu_blocks_);
    free_gpu_block_ids_.reserve(total_gpu_blocks_);

    size_t floats_per_block = static_cast<size_t>(block_size_) * head_dim_;
    size_t bytes_per_block = floats_per_block * sizeof(float);

    for (size_t i = 0; i < total_gpu_blocks_; ++i) {
        PhysicalPage& page = gpu_pages_[i];
        page.physical_block_id = static_cast<int32_t>(i);
        page.block_size = block_size_;
        page.head_dim = head_dim_;
        page.num_tokens = 0;
        page.ref_count.store(0);
        page.is_evicted_to_host = false;
        page.last_accessed_timestamp = current_time_micros();

        // Simulate GPU device allocation with 64-byte aligned blocks
        #if defined(_MSC_VER)
        page.dev_k_ptr = static_cast<float*>(_aligned_malloc(bytes_per_block, PAGE_ALIGNMENT));
        page.dev_v_ptr = static_cast<float*>(_aligned_malloc(bytes_per_block, PAGE_ALIGNMENT));
        page.host_pinned_k_ptr = static_cast<float*>(_aligned_malloc(bytes_per_block, PAGE_ALIGNMENT));
        page.host_pinned_v_ptr = static_cast<float*>(_aligned_malloc(bytes_per_block, PAGE_ALIGNMENT));
        #else
        posix_memalign(reinterpret_cast<void**>(&page.dev_k_ptr), PAGE_ALIGNMENT, bytes_per_block);
        posix_memalign(reinterpret_cast<void**>(&page.dev_v_ptr), PAGE_ALIGNMENT, bytes_per_block);
        posix_memalign(reinterpret_cast<void**>(&page.host_pinned_k_ptr), PAGE_ALIGNMENT, bytes_per_block);
        posix_memalign(reinterpret_cast<void**>(&page.host_pinned_v_ptr), PAGE_ALIGNMENT, bytes_per_block);
        #endif

        std::memset(page.dev_k_ptr, 0, bytes_per_block);
        std::memset(page.dev_v_ptr, 0, bytes_per_block);

        free_gpu_block_ids_.push_back(static_cast<int32_t>(i));
    }

    radix_root_ = std::make_shared<RadixNode>();
}

PagedAllocator::~PagedAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (PhysicalPage& page : gpu_pages_) {
        #if defined(_MSC_VER)
        if (page.dev_k_ptr) _aligned_free(page.dev_k_ptr);
        if (page.dev_v_ptr) _aligned_free(page.dev_v_ptr);
        if (page.host_pinned_k_ptr) _aligned_free(page.host_pinned_k_ptr);
        if (page.host_pinned_v_ptr) _aligned_free(page.host_pinned_v_ptr);
        #else
        if (page.dev_k_ptr) free(page.dev_k_ptr);
        if (page.dev_v_ptr) free(page.dev_v_ptr);
        if (page.host_pinned_k_ptr) free(page.host_pinned_k_ptr);
        if (page.host_pinned_v_ptr) free(page.host_pinned_v_ptr);
        #endif
    }
}

uint64_t PagedAllocator::current_time_micros() const noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    );
}

std::vector<int32_t> PagedAllocator::allocate_blocks(const std::string& request_id, size_t num_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t needed_blocks = (num_tokens + block_size_ - 1) / block_size_;
    if (needed_blocks == 0) needed_blocks = 1;

    if (free_gpu_block_ids_.size() < needed_blocks) {
        // Automatically evict cold pages to host memory to fulfill requirement
        size_t deficit = needed_blocks - free_gpu_block_ids_.size();
        evict_cold_pages_to_host(deficit);
    }

    if (free_gpu_block_ids_.size() < needed_blocks) {
        throw std::runtime_error("PagedAllocator: OOM - Unable to allocate physical blocks even after eviction");
    }

    std::vector<int32_t> allocated;
    allocated.reserve(needed_blocks);

    for (size_t i = 0; i < needed_blocks; ++i) {
        int32_t bid = free_gpu_block_ids_.back();
        free_gpu_block_ids_.pop_back();

        PhysicalPage& page = gpu_pages_[bid];
        page.num_tokens = 0;
        page.ref_count.store(1);
        page.is_evicted_to_host = false;
        page.last_accessed_timestamp = current_time_micros();

        allocated.push_back(bid);
    }

    virtual_to_physical_map_[request_id] = allocated;
    return allocated;
}

bool PagedAllocator::append_token(const std::string& request_id, const float* k_vec, const float* v_vec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = virtual_to_physical_map_.find(request_id);
    if (it == virtual_to_physical_map_.end() || it->second.empty()) {
        return false;
    }

    int32_t last_bid = it->second.back();
    PhysicalPage& page = gpu_pages_[last_bid];

    if (page.is_evicted_to_host) {
        restore_page_to_gpu(last_bid);
    }

    // Copy-On-Write (CoW): If the target page is shared across branches, fork it
    if (page.ref_count.load() > 1) {
        if (free_gpu_block_ids_.empty()) {
            evict_cold_pages_to_host(1);
        }
        if (free_gpu_block_ids_.empty()) return false;

        int32_t cow_bid = free_gpu_block_ids_.back();
        free_gpu_block_ids_.pop_back();

        PhysicalPage& cow_page = gpu_pages_[cow_bid];
        cow_page.num_tokens = page.num_tokens;
        cow_page.ref_count.store(1);
        cow_page.is_evicted_to_host = false;
        cow_page.last_accessed_timestamp = current_time_micros();

        size_t copy_bytes = static_cast<size_t>(page.num_tokens) * head_dim_ * sizeof(float);
        std::memcpy(cow_page.dev_k_ptr, page.dev_k_ptr, copy_bytes);
        std::memcpy(cow_page.dev_v_ptr, page.dev_v_ptr, copy_bytes);

        page.ref_count.fetch_sub(1);
        it->second.back() = cow_bid;
        last_bid = cow_bid;
    } else if (page.num_tokens >= page.block_size) {
        // Allocate a new physical block
        if (free_gpu_block_ids_.empty()) {
            evict_cold_pages_to_host(1);
        }
        if (free_gpu_block_ids_.empty()) return false;

        int32_t new_bid = free_gpu_block_ids_.back();
        free_gpu_block_ids_.pop_back();

        PhysicalPage& new_page = gpu_pages_[new_bid];
        new_page.num_tokens = 0;
        new_page.ref_count.store(1);
        new_page.is_evicted_to_host = false;
        new_page.last_accessed_timestamp = current_time_micros();

        it->second.push_back(new_bid);
        last_bid = new_bid;
    }

    PhysicalPage& active_page = gpu_pages_[last_bid];
    size_t offset = static_cast<size_t>(active_page.num_tokens) * head_dim_;
    std::memcpy(active_page.dev_k_ptr + offset, k_vec, head_dim_ * sizeof(float));
    std::memcpy(active_page.dev_v_ptr + offset, v_vec, head_dim_ * sizeof(float));
    active_page.num_tokens++;
    active_page.last_accessed_timestamp = current_time_micros();

    return true;
}

std::vector<int32_t> PagedAllocator::match_prefix(const std::vector<int32_t>& prompt_tokens, size_t& out_matched_token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    out_matched_token_count = 0;
    std::vector<int32_t> matched_blocks;

    std::shared_ptr<RadixNode> curr = radix_root_;
    size_t token_idx = 0;

    while (token_idx < prompt_tokens.size() && curr) {
        int32_t tok = prompt_tokens[token_idx];
        auto child_it = curr->children.find(tok);
        if (child_it == curr->children.end()) break;

        curr = child_it->second;
        // Verify prefix matching
        bool match_failed = false;
        for (int32_t p_tok : curr->prefix_tokens) {
            if (token_idx >= prompt_tokens.size() || prompt_tokens[token_idx] != p_tok) {
                match_failed = true;
                break;
            }
            token_idx++;
        }
        if (match_failed) break;

        for (int32_t bid : curr->physical_block_ids) {
            matched_blocks.push_back(bid);
            gpu_pages_[bid].ref_count.fetch_add(1);
            gpu_pages_[bid].last_accessed_timestamp = current_time_micros();
        }
        out_matched_token_count = token_idx;
    }

    return matched_blocks;
}

void PagedAllocator::insert_prefix(const std::vector<int32_t>& prompt_tokens, const std::vector<int32_t>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (prompt_tokens.empty() || block_ids.empty()) return;

    std::shared_ptr<RadixNode> curr = radix_root_;
    int32_t first_tok = prompt_tokens[0];

    if (curr->children.find(first_tok) == curr->children.end()) {
        auto node = std::make_shared<RadixNode>();
        node->prefix_tokens = prompt_tokens;
        node->physical_block_ids = block_ids;
        node->is_locked = true;
        for (int32_t bid : block_ids) {
            gpu_pages_[bid].ref_count.fetch_add(1);
        }
        curr->children[first_tok] = node;
    }
}

bool PagedAllocator::branch_request_cow(const std::string& source_request_id, const std::string& new_request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = virtual_to_physical_map_.find(source_request_id);
    if (it == virtual_to_physical_map_.end() || it->second.empty()) {
        return false;
    }

    // Clone virtual block table and increment reference counters for all physical blocks
    std::vector<int32_t> shared_blocks = it->second;
    for (int32_t bid : shared_blocks) {
        gpu_pages_[bid].ref_count.fetch_add(1);
        gpu_pages_[bid].last_accessed_timestamp = current_time_micros();
    }

    virtual_to_physical_map_[new_request_id] = shared_blocks;
    return true;
}

void PagedAllocator::free_request(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = virtual_to_physical_map_.find(request_id);
    if (it == virtual_to_physical_map_.end()) return;

    for (int32_t bid : it->second) {
        PhysicalPage& page = gpu_pages_[bid];
        uint32_t old_ref = page.ref_count.fetch_sub(1);
        if (old_ref <= 1) {
            // Reference counter reached zero -> return to free pool
            page.num_tokens = 0;
            page.is_evicted_to_host = false;
            free_gpu_block_ids_.push_back(bid);
        }
    }

    virtual_to_physical_map_.erase(it);
}

size_t PagedAllocator::evict_cold_pages_to_host(size_t max_blocks_to_evict) {
    // Find candidate pages sorted by LRU timestamp with ref_count == 1
    std::vector<int32_t> candidates;
    for (size_t i = 0; i < total_gpu_blocks_; ++i) {
        PhysicalPage& page = gpu_pages_[i];
        if (!page.is_evicted_to_host && page.ref_count.load() == 1) {
            candidates.push_back(static_cast<int32_t>(i));
        }
    }

    std::sort(candidates.begin(), candidates.end(), [this](int32_t a, int32_t b) {
        return gpu_pages_[a].last_accessed_timestamp < gpu_pages_[b].last_accessed_timestamp;
    });

    size_t evicted_count = 0;
    size_t bytes = static_cast<size_t>(block_size_) * head_dim_ * sizeof(float);

    for (int32_t bid : candidates) {
        if (evicted_count >= max_blocks_to_evict) break;
        PhysicalPage& page = gpu_pages_[bid];

        // DMA copy over PCIe simulation: dev -> host_pinned
        std::memcpy(page.host_pinned_k_ptr, page.dev_k_ptr, bytes);
        std::memcpy(page.host_pinned_v_ptr, page.dev_v_ptr, bytes);
        page.is_evicted_to_host = true;

        free_gpu_block_ids_.push_back(bid);
        evicted_count++;
    }

    return evicted_count;
}

bool PagedAllocator::restore_page_to_gpu(int32_t physical_block_id) {
    if (physical_block_id < 0 || static_cast<size_t>(physical_block_id) >= total_gpu_blocks_) return false;
    PhysicalPage& page = gpu_pages_[physical_block_id];
    if (!page.is_evicted_to_host) return true;

    size_t bytes = static_cast<size_t>(block_size_) * head_dim_ * sizeof(float);
    std::memcpy(page.dev_k_ptr, page.host_pinned_k_ptr, bytes);
    std::memcpy(page.dev_v_ptr, page.host_pinned_v_ptr, bytes);
    page.is_evicted_to_host = false;
    page.last_accessed_timestamp = current_time_micros();
    return true;
}

size_t PagedAllocator::get_free_gpu_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_gpu_block_ids_.size();
}

size_t PagedAllocator::get_total_gpu_blocks() const noexcept {
    return total_gpu_blocks_;
}

size_t PagedAllocator::get_evicted_blocks_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const PhysicalPage& page : gpu_pages_) {
        if (page.is_evicted_to_host) count++;
    }
    return count;
}

float PagedAllocator::get_fragmentation_ratio() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t allocated = total_gpu_blocks_ - free_gpu_block_ids_.size();
    if (allocated == 0) return 0.0f;
    size_t unused_slots = 0;
    for (const auto& pair : virtual_to_physical_map_) {
        if (!pair.second.empty()) {
            int32_t last_bid = pair.second.back();
            unused_slots += (block_size_ - gpu_pages_[last_bid].num_tokens);
        }
    }
    return static_cast<float>(unused_slots) / static_cast<float>(allocated * block_size_);
}

} // namespace memory
} // namespace velograd
