// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE: Ahead-of-Time (AOT) Graph Engine & Standalone Binary Compiler (VeloGrad v4.0)
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// AGPLv3 + Commercial Royalty Waiver Dual License
// =================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace velograd {
namespace compiler {

enum class OpType : uint8_t {
    LINEAR = 0,
    RMSNORM = 1,
    ROPE = 2,
    FLASH_ATTENTION = 3,
    SWIGLU = 4,
    FUSED_TRANSFORMER_BLOCK = 5
};

struct GraphNode {
    std::string node_name;
    OpType op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<size_t> input_shapes;
    std::vector<size_t> output_shapes;
    size_t memory_offset{0};
    size_t memory_bytes{0};
};

struct MemoryPlan {
    size_t total_arena_bytes{0};
    size_t peak_working_memory{0};
    std::unordered_map<std::string, size_t> tensor_offsets;
};

/// Ahead-of-Time (AOT) Graph Compiler
class AOTGraphCompiler {
public:
    AOTGraphCompiler();

    /// Registers an operator node into the execution graph
    void add_node(const GraphNode& node);

    /// Optimization Pass 1: Operator Fusion (merges RMSNorm + RoPE + FlashAttn + SwiGLU)
    size_t run_fusion_passes();

    /// Optimization Pass 2: Static Memory Address Planning (liveness analysis)
    MemoryPlan plan_memory_arena();

    /// Compiles and serializes the complete model graph into a standalone C-ABI shared library
    bool export_standalone_c_abi(const std::string& output_lib_path, const MemoryPlan& plan);

    size_t get_node_count() const noexcept { return nodes_.size(); }
    const std::vector<GraphNode>& get_nodes() const noexcept { return nodes_; }

private:
    std::vector<GraphNode> nodes_;
};

} // namespace compiler
} // namespace velograd

// =================================================================================================
// C-ABI EXPORTS FOR STANDALONE RUNTIME (libvelograd.so / velograd.dll)
// =================================================================================================

extern "C" {
    #if defined(_WIN32)
    #define VELOGRAD_API __declspec(dllexport)
    #else
    #define VELOGRAD_API __attribute__((visibility("default")))
    #endif

    typedef void* velograd_engine_handle_t;

    VELOGRAD_API velograd_engine_handle_t velograd_create_engine(size_t total_memory_bytes);
    VELOGRAD_API int32_t velograd_forward_step(velograd_engine_handle_t handle, const float* in, float* out, size_t count);
    VELOGRAD_API void velograd_destroy_engine(velograd_engine_handle_t handle);
    VELOGRAD_API const char* velograd_get_version();
}
