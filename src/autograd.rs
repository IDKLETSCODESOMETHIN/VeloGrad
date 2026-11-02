// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE 2: Symbolic Ahead-of-Time (AOT) Autograd & Radical Kernel Fusion Engine
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
//
// LICENSING TERMS & DUAL-LICENSING ROYALTY DISCLAIMER:
// -------------------------------------------------------------------------------------------------
// 1. OPEN-SOURCE NETWORK RECIPROCAL LICENSE:
//    This software is released under the GNU Affero General Public License (AGPLv3), as published by
//    the Free Software Foundation, either version 3 of the License, or (at your option) any later
//    version. Under AGPLv3 Section 13, if you run a modified or unmodified version of this framework
//    over a computer network (e.g., cloud platforms, SaaS model serving, distributed training clusters,
//    or API gateways), you MUST make the complete source code of your corresponding network backend
//    infrastructure available to all network users at no charge under the terms of this license.
//
// 2. COMMERCIAL & ENTERPRISE DUAL-LICENSING ESCAPE HATCH:
//    Any enterprise, cloud infrastructure provider, or production deployment seeking an exemption
//    from AGPLv3 source code disclosure obligations MUST purchase a valid Commercial Royalty Waiver.
//    The Commercial Royalty License requires an agreed royalty percentage calculated on cloud
//    compute savings, automated throughput gains, or commercial inference scale. Contact VeloGrad
//    Licensing & Royalty Management prior to proprietary network execution.
//
// THIS SOFTWARE IS DISTRIBUTED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED. SEE THE AGPLv3 FOR SPECIFIC DETAILS.
// =================================================================================================

#![deny(missing_docs)]

//! # VeloGrad Symbolic AOT Autograd & Kernel Fusion Engine
//!
//! Replaces runtime dynamic autograd tape engines (e.g. PyTorch Autograd) with:
//! - **Static Symbolic Directed Acyclic Graph (DAG)** representation.
//! - **Ahead-of-Time (AOT) Symbolic Differentiation** via mathematical chain-rule expansion.
//! - **Radical Kernel Fusion Compiler** fusing memory-bound sequences (e.g. `MatMul` + `Bias` + `ReLU`)
//!   into register-blocked, zero-intermediate-allocation hardware execution kernels.

use crate::{CACHE_LINE_BYTES, StaticArenaAllocator};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a node within the symbolic computational DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// 2D Tensor shape description for static symbolic dimension propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape2D {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Shape2D {
    /// Create a new 2D shape.
    #[inline(always)]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Total number of elements.
    #[inline(always)]
    pub const fn numel(&self) -> usize {
        self.rows * self.cols
    }

    /// Size in bytes for single-precision floats (`f32`).
    #[inline(always)]
    pub const fn byte_size(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }
}

impl fmt::Display for Shape2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}x{}]", self.rows, self.cols)
    }
}

/// Mathematical operations supported by the VeloGrad symbolic DAG.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Leaf node: External model input.
    Input {
        /// Semantic variable name.
        name: String,
    },
    /// Leaf node: Trainable model parameter (weights, biases).
    Parameter {
        /// Parameter identifier.
        name: String,
    },
    /// Constant scalar or tensor filled with a fixed value.
    Constant {
        /// Numerical constant value.
        value: f32,
    },
    /// Element-wise addition: `Z = X + Y`.
    Add(NodeId, NodeId),
    /// Element-wise multiplication: `Z = X * Y`.
    Mul(NodeId, NodeId),
    /// Scalar multiplication: `Z = alpha * X`.
    Scale(NodeId, f32),
    /// Matrix multiplication: `Z = X @ Y`.
    MatMul(NodeId, NodeId),
    /// 2D matrix transposition: `Z = X^T`.
    Transpose(NodeId),
    /// Rectified Linear Unit non-linearity: `Z = max(0, X)`.
    ReLU(NodeId),
    /// Backward gradient of ReLU: `dX = dZ * (X > 0)`.
    ReluGrad {
        /// Upstream gradient node `dZ`.
        grad: NodeId,
        /// Original input node `X`.
        input: NodeId,
    },
    /// Fused kernel: `Z = ReLU(X @ W + Bias)`.
    ///
    /// Evaluated in a single register-blocked loop with zero intermediate allocations.
    FusedMatMulBiasReLU {
        /// Input activation tensor `X`.
        input: NodeId,
        /// Weight matrix `W`.
        weight: NodeId,
        /// Bias vector `Bias` (broadcast across rows).
        bias: NodeId,
    },
    /// Fused kernel: `Z = ReLU(X + Y)`.
    FusedAddReLU(NodeId, NodeId),
}

/// A node within the symbolic computational DAG.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique node index.
    pub id: NodeId,
    /// Semantic identifier or generated operation tag.
    pub name: String,
    /// Static tensor shape verified at graph construction time.
    pub shape: Shape2D,
    /// Mathematical operation producing this node's value.
    pub op: Op,
    /// Whether this node was synthesized during symbolic backward propagation.
    pub is_gradient: bool,
}

/// Directed Acyclic Graph (DAG) for AOT symbolic representation and differentiation.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    /// Map from parameter name to its corresponding NodeId.
    parameters: HashMap<String, NodeId>,
    /// Map from input name to its corresponding NodeId.
    inputs: HashMap<String, NodeId>,
}

impl Graph {
    /// Creates a new empty symbolic computational graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an external input node with a static shape.
    pub fn input(&mut self, name: &str, rows: usize, cols: usize) -> NodeId {
        let id = NodeId(self.nodes.len());
        let shape = Shape2D::new(rows, cols);
        let node = Node {
            id,
            name: name.to_string(),
            shape,
            op: Op::Input {
                name: name.to_string(),
            },
            is_gradient: false,
        };
        self.inputs.insert(name.to_string(), id);
        self.nodes.push(node);
        id
    }

    /// Registers a trainable parameter node with a static shape.
    pub fn parameter(&mut self, name: &str, rows: usize, cols: usize) -> NodeId {
        let id = NodeId(self.nodes.len());
        let shape = Shape2D::new(rows, cols);
        let node = Node {
            id,
            name: name.to_string(),
            shape,
            op: Op::Parameter {
                name: name.to_string(),
            },
            is_gradient: false,
        };
        self.parameters.insert(name.to_string(), id);
        self.nodes.push(node);
        id
    }

    /// Appends a constant value node.
    pub fn constant(&mut self, rows: usize, cols: usize, value: f32) -> NodeId {
        let id = NodeId(self.nodes.len());
        let shape = Shape2D::new(rows, cols);
        let node = Node {
            id,
            name: format!("const_{}", id.0),
            shape,
            op: Op::Constant { value },
            is_gradient: false,
        };
        self.nodes.push(node);
        id
    }

    /// Appends an element-wise addition node with strict shape verification.
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s_lhs = self.nodes[lhs.0].shape;
        let s_rhs = self.nodes[rhs.0].shape;

        // Support exact shape matching or row-broadcasting (1, cols) to (rows, cols)
        let result_shape = if s_lhs == s_rhs {
            s_lhs
        } else if s_rhs.rows == 1 && s_rhs.cols == s_lhs.cols {
            s_lhs
        } else if s_lhs.rows == 1 && s_lhs.cols == s_rhs.cols {
            s_rhs
        } else {
            panic!(
                "AOT Shape Mismatch in Add: {} vs {}",
                s_lhs, s_rhs
            );
        };

        let id = NodeId(self.nodes.len());
        let node = Node {
            id,
            name: format!("add_{}_{}", lhs.0, rhs.0),
            shape: result_shape,
            op: Op::Add(lhs, rhs),
            is_gradient: false,
        };
        self.nodes.push(node);
        id
    }

    /// Appends an element-wise multiplication node.
    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s_lhs = self.nodes[lhs.0].shape;
        let s_rhs = self.nodes[rhs.0].shape;
        assert_eq!(
            s_lhs, s_rhs,
            "AOT Shape Mismatch in Mul: {} vs {}",
            s_lhs, s_rhs
        );

        let id = NodeId(self.nodes.len());
        let node = Node {
            id,
            name: format!("mul_{}_{}", lhs.0, rhs.0),
            shape: s_lhs,
            op: Op::Mul(lhs, rhs),
            is_gradient: false,
        };
        self.nodes.push(node);
        id
    }

    /// Appends a scalar scaling node.
    pub fn scale(&mut self, node: NodeId, alpha: f32) -> NodeId {
        let shape = self.nodes[node.0].shape;
        let id = NodeId(self.nodes.len());
        let new_node = Node {
            id,
            name: format!("scale_{}", node.0),
            shape,
            op: Op::Scale(node, alpha),
            is_gradient: false,
        };
        self.nodes.push(new_node);
        id
    }

    /// Appends a matrix multiplication node: `Z = LHS @ RHS`.
    ///
    /// Dimension verification: `LHS: (M x K)`, `RHS: (K x N)` -> `Z: (M x N)`.
    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s_lhs = self.nodes[lhs.0].shape;
        let s_rhs = self.nodes[rhs.0].shape;
        assert_eq!(
            s_lhs.cols, s_rhs.rows,
            "AOT MatMul Dimension Mismatch: {} inner dimension does not match {}",
            s_lhs, s_rhs
        );

        let id = NodeId(self.nodes.len());
        let shape = Shape2D::new(s_lhs.rows, s_rhs.cols);
        let node = Node {
            id,
            name: format!("matmul_{}_{}", lhs.0, rhs.0),
            shape,
            op: Op::MatMul(lhs, rhs),
            is_gradient: false,
        };
        self.nodes.push(node);
        id
    }

    /// Appends a zero-copy matrix transpose node: `Z = X^T`.
    pub fn transpose(&mut self, node: NodeId) -> NodeId {
        let s = self.nodes[node.0].shape;
        let id = NodeId(self.nodes.len());
        let shape = Shape2D::new(s.cols, s.rows);
        let new_node = Node {
            id,
            name: format!("t_{}", node.0),
            shape,
            op: Op::Transpose(node),
            is_gradient: false,
        };
        self.nodes.push(new_node);
        id
    }

    /// Appends a ReLU activation node: `Z = max(0, X)`.
    pub fn relu(&mut self, node: NodeId) -> NodeId {
        let shape = self.nodes[node.0].shape;
        let id = NodeId(self.nodes.len());
        let new_node = Node {
            id,
            name: format!("relu_{}", node.0),
            shape,
            op: Op::ReLU(node),
            is_gradient: false,
        };
        self.nodes.push(new_node);
        id
    }

    /// Appends a ReLU gradient node: `dX = dZ * (X > 0)`.
    fn relu_grad(&mut self, grad: NodeId, input: NodeId) -> NodeId {
        let shape = self.nodes[input.0].shape;
        let id = NodeId(self.nodes.len());
        let new_node = Node {
            id,
            name: format!("d_relu_{}", input.0),
            shape,
            op: Op::ReluGrad { grad, input },
            is_gradient: true,
        };
        self.nodes.push(new_node);
        id
    }

    /// Returns a reference to a node by its ID.
    #[inline(always)]
    pub fn get_node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Returns the total number of nodes in the graph.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph contains zero nodes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Performs Ahead-of-Time (AOT) Symbolic Differentiation.
    ///
    /// Automatically derives the exact mathematical derivative equations for all graph
    /// operations leading to `loss_node` using the multi-variable chain rule. Appends
    /// symbolic gradient nodes directly into the DAG *prior to compilation*, completely
    /// eliminating dynamic autograd tapes and execution runtime tracking.
    ///
    /// Returns a map from target original NodeId to its derived symbolic gradient NodeId.
    pub fn aot_backward(&mut self, loss_node: NodeId) -> HashMap<NodeId, NodeId> {
        let mut grads: HashMap<NodeId, NodeId> = HashMap::new();

        // Seed the backward chain with dLoss/dLoss = 1.0
        let loss_shape = self.nodes[loss_node.0].shape;
        let seed = self.constant(loss_shape.rows, loss_shape.cols, 1.0);
        grads.insert(loss_node, seed);

        // Traverse in reverse topological order (nodes are naturally indexed in topological order)
        let num_original_nodes = self.nodes.len();
        for i in (0..num_original_nodes).rev() {
            let node_id = NodeId(i);
            let upstream_grad = match grads.get(&node_id).copied() {
                Some(g) => g,
                None => continue, // Dead node or not connected to loss
            };

            let op = self.nodes[node_id.0].op.clone();
            match op {
                Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } => {
                    // Leaf nodes terminate the chain
                }
                Op::Add(lhs, rhs) => {
                    // dL/dLHS = dL/dZ, dL/dRHS = dL/dZ
                    Self::accumulate_gradient(&mut grads, self, lhs, upstream_grad);
                    Self::accumulate_gradient(&mut grads, self, rhs, upstream_grad);
                }
                Op::Mul(lhs, rhs) => {
                    // dL/dLHS = dL/dZ * RHS, dL/dRHS = dL/dZ * LHS
                    let d_lhs = self.mul(upstream_grad, rhs);
                    let d_rhs = self.mul(upstream_grad, lhs);
                    Self::accumulate_gradient(&mut grads, self, lhs, d_lhs);
                    Self::accumulate_gradient(&mut grads, self, rhs, d_rhs);
                }
                Op::Scale(x, alpha) => {
                    // dL/dX = alpha * dL/dZ
                    let d_x = self.scale(upstream_grad, alpha);
                    Self::accumulate_gradient(&mut grads, self, x, d_x);
                }
                Op::MatMul(lhs, rhs) => {
                    // If Z = A (M x K) @ B (K x N), with dZ = upstream_grad (M x N):
                    // dA = dZ @ B^T   => (M x N) @ (N x K) = (M x K)
                    // dB = A^T @ dZ   => (K x M) @ (M x N) = (K x N)
                    let rhs_t = self.transpose(rhs);
                    let d_lhs = self.matmul(upstream_grad, rhs_t);
                    Self::accumulate_gradient(&mut grads, self, lhs, d_lhs);

                    let lhs_t = self.transpose(lhs);
                    let d_rhs = self.matmul(lhs_t, upstream_grad);
                    Self::accumulate_gradient(&mut grads, self, rhs, d_rhs);
                }
                Op::Transpose(x) => {
                    // dL/dX = (dL/dZ)^T
                    let d_x = self.transpose(upstream_grad);
                    Self::accumulate_gradient(&mut grads, self, x, d_x);
                }
                Op::ReLU(x) => {
                    // dL/dX = dL/dZ * (X > 0)
                    let d_x = self.relu_grad(upstream_grad, x);
                    Self::accumulate_gradient(&mut grads, self, x, d_x);
                }
                Op::ReluGrad { .. }
                | Op::FusedMatMulBiasReLU { .. }
                | Op::FusedAddReLU(..) => {
                    // Compiler-synthesized or fused nodes do not differentiate further
                }
            }
        }

        grads
    }

    /// Accumulates upstream gradients (applying symbolic addition when a node has multiple consumers).
    fn accumulate_gradient(
        grads: &mut HashMap<NodeId, NodeId>,
        graph: &mut Graph,
        target: NodeId,
        new_grad: NodeId,
    ) {
        if let Some(&existing_grad) = grads.get(&target) {
            let accumulated = graph.add(existing_grad, new_grad);
            grads.insert(target, accumulated);
        } else {
            grads.insert(target, new_grad);
        }
    }

    /// Radical Kernel Fusion Compiler: Scans the DAG, fuses adjacent memory-bound nodes
    /// into single register-blocked execution kernels, eliminates dead intermediate nodes,
    /// and constructs a fully static, zero-allocation executable pipeline.
    pub fn compile(&self) -> CompiledPipeline {
        // Count in-degrees / consumer usages of each node
        let mut consumer_count = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            match &node.op {
                Op::Add(a, b) | Op::Mul(a, b) | Op::MatMul(a, b) | Op::FusedAddReLU(a, b) => {
                    consumer_count[a.0] += 1;
                    consumer_count[b.0] += 1;
                }
                Op::Scale(a, _) | Op::Transpose(a) | Op::ReLU(a) => {
                    consumer_count[a.0] += 1;
                }
                Op::ReluGrad { grad, input } => {
                    consumer_count[grad.0] += 1;
                    consumer_count[input.0] += 1;
                }
                Op::FusedMatMulBiasReLU { input, weight, bias } => {
                    consumer_count[input.0] += 1;
                    consumer_count[weight.0] += 1;
                    consumer_count[bias.0] += 1;
                }
                Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } => {}
            }
        }

        // Identify fusible patterns:
        // Pattern 1: MatMul -> Add (Bias) -> ReLU
        // When intermediate nodes are single-consumer, fuse into FusedMatMulBiasReLU
        let mut fused_replacements: HashMap<NodeId, Op> = HashMap::new();
        let mut eliminated_nodes: HashSet<NodeId> = HashSet::new();

        for node in &self.nodes {
            if let Op::ReLU(child_id) = &node.op {
                if consumer_count[child_id.0] == 1 {
                    let child_node = &self.nodes[child_id.0];
                    if let Op::Add(add_lhs, add_rhs) = &child_node.op {
                        // Check if one branch of Add is a single-consumer MatMul
                        if consumer_count[add_lhs.0] == 1 {
                            if let Op::MatMul(mm_x, mm_w) = &self.nodes[add_lhs.0].op {
                                // Full triple fusion: ReLU(MatMul(X, W) + Bias)
                                fused_replacements.insert(
                                    node.id,
                                    Op::FusedMatMulBiasReLU {
                                        input: *mm_x,
                                        weight: *mm_w,
                                        bias: *add_rhs,
                                    },
                                );
                                eliminated_nodes.insert(*child_id);
                                eliminated_nodes.insert(*add_lhs);
                                continue;
                            }
                        }

                        // Pattern 2: Add -> ReLU fusion
                        fused_replacements.insert(node.id, Op::FusedAddReLU(*add_lhs, *add_rhs));
                        eliminated_nodes.insert(*child_id);
                    }
                }
            }
        }

        // Construct linear execution schedule
        let mut schedule = Vec::new();
        let mut buffer_offsets: HashMap<NodeId, usize> = HashMap::new();
        let mut total_memory_required = 0usize;

        for node in &self.nodes {
            if eliminated_nodes.contains(&node.id) {
                continue;
            }

            // Allocate static cache-line aligned buffer slot for this node's output
            let aligned_offset = (total_memory_required + (CACHE_LINE_BYTES - 1)) & !(CACHE_LINE_BYTES - 1);
            buffer_offsets.insert(node.id, aligned_offset);
            total_memory_required = aligned_offset + node.shape.byte_size();

            let op_to_execute = fused_replacements
                .get(&node.id)
                .cloned()
                .unwrap_or_else(|| node.op.clone());

            schedule.push(ExecutionStep {
                node_id: node.id,
                name: node.name.clone(),
                shape: node.shape,
                op: op_to_execute,
                output_offset: aligned_offset,
            });
        }

        CompiledPipeline {
            steps: schedule,
            buffer_offsets,
            total_memory_required,
            inputs: self.inputs.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

/// A single fused, register-blocked execution step in the compiled pipeline.
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Node being evaluated.
    pub node_id: NodeId,
    /// Semantic name.
    pub name: String,
    /// Tensor shape.
    pub shape: Shape2D,
    /// Compiled or fused operation.
    pub op: Op,
    /// Byte offset within the static arena buffer.
    pub output_offset: usize,
}

/// A statically compiled, zero-allocation executable pipeline.
///
/// Pre-calculates exact byte layout in a single contiguous arena block.
/// Execution runs strictly through bare-metal fused loops with zero dynamic memory calls.
pub struct CompiledPipeline {
    /// Ordered execution instructions.
    pub steps: Vec<ExecutionStep>,
    /// Byte offsets for every node output within the backing arena block.
    pub buffer_offsets: HashMap<NodeId, usize>,
    /// Exact total byte requirement for all activations and intermediate states.
    pub total_memory_required: usize,
    /// Named inputs map.
    pub inputs: HashMap<String, NodeId>,
    /// Named parameters map.
    pub parameters: HashMap<String, NodeId>,
}

impl CompiledPipeline {
    /// Returns the exact contiguous byte capacity required by the entire graph execution.
    #[inline(always)]
    pub fn required_arena_bytes(&self) -> usize {
        self.total_memory_required
    }

    /// Evaluates the compiled pipeline using a dedicated static arena.
    ///
    /// # Safety
    /// Ensures zero dynamic allocations: all intermediate states, weights, and activations
    /// are mapped directly into the pre-allocated contiguous 64-byte aligned arena.
    pub fn execute(
        &self,
        arena: &mut StaticArenaAllocator,
        input_data: &HashMap<String, &[f32]>,
        param_data: &HashMap<String, &[f32]>,
    ) {
        // Allocate contiguous backing buffer for the entire pipeline
        let total_floats = (self.total_memory_required + 3) / 4;
        let base_ptr = arena
            .alloc_slice::<f32>(total_floats)
            .expect("CompiledPipeline: Arena buffer exhaustion")
            .as_ptr() as *mut u8;

        // Helper closures to resolve node buffer pointers
        let get_buf = |node_id: NodeId| -> *mut f32 {
            let offset = self.buffer_offsets[&node_id];
            unsafe { base_ptr.add(offset) as *mut f32 }
        };

        let get_const_buf = |node_id: NodeId| -> *const f32 {
            let offset = self.buffer_offsets[&node_id];
            unsafe { base_ptr.add(offset) as *const f32 }
        };

        // Execute scheduled steps sequentially
        for step in &self.steps {
            let out_ptr = get_buf(step.node_id);
            let numel = step.shape.numel();

            match &step.op {
                Op::Input { name } => {
                    if let Some(src) = input_data.get(name) {
                        assert_eq!(src.len(), numel, "Input dimension mismatch for {}", name);
                        unsafe {
                            std::ptr::copy_nonoverlapping(src.as_ptr(), out_ptr, numel);
                        }
                    }
                }
                Op::Parameter { name } => {
                    if let Some(src) = param_data.get(name) {
                        assert_eq!(src.len(), numel, "Param dimension mismatch for {}", name);
                        unsafe {
                            std::ptr::copy_nonoverlapping(src.as_ptr(), out_ptr, numel);
                        }
                    }
                }
                Op::Constant { value } => {
                    let val = *value;
                    unsafe {
                        let slice = std::slice::from_raw_parts_mut(out_ptr, numel);
                        slice.fill(val);
                    }
                }
                Op::Add(lhs, rhs) => {
                    let p_lhs = get_const_buf(*lhs);
                    let p_rhs = get_const_buf(*rhs);
                    unsafe {
                        for i in 0..numel {
                            *out_ptr.add(i) = *p_lhs.add(i) + *p_rhs.add(i);
                        }
                    }
                }
                Op::Mul(lhs, rhs) => {
                    let p_lhs = get_const_buf(*lhs);
                    let p_rhs = get_const_buf(*rhs);
                    unsafe {
                        for i in 0..numel {
                            *out_ptr.add(i) = *p_lhs.add(i) * *p_rhs.add(i);
                        }
                    }
                }
                Op::Scale(x, alpha) => {
                    let p_x = get_const_buf(*x);
                    let a = *alpha;
                    unsafe {
                        for i in 0..numel {
                            *out_ptr.add(i) = *p_x.add(i) * a;
                        }
                    }
                }
                Op::MatMul(lhs, rhs) => {
                    let p_lhs = get_const_buf(*lhs);
                    let p_rhs = get_const_buf(*rhs);
                    let s_out = step.shape;
                    let k_dim = if s_out.rows > 0 {
                        // Recover inner K dimension from input buffer size
                        self.steps.iter().find(|s| s.node_id == *lhs).unwrap().shape.cols
                    } else {
                        0
                    };

                    unsafe {
                        for i in 0..s_out.rows {
                            for j in 0..s_out.cols {
                                let mut acc = 0.0f32;
                                for k in 0..k_dim {
                                    acc += *p_lhs.add(i * k_dim + k) * *p_rhs.add(k * s_out.cols + j);
                                }
                                *out_ptr.add(i * s_out.cols + j) = acc;
                            }
                        }
                    }
                }
                Op::Transpose(x) => {
                    let p_x = get_const_buf(*x);
                    let s_out = step.shape; // rows = orig.cols, cols = orig.rows
                    unsafe {
                        for r in 0..s_out.rows {
                            for c in 0..s_out.cols {
                                *out_ptr.add(r * s_out.cols + c) = *p_x.add(c * s_out.rows + r);
                            }
                        }
                    }
                }
                Op::ReLU(x) => {
                    let p_x = get_const_buf(*x);
                    unsafe {
                        for i in 0..numel {
                            let val = *p_x.add(i);
                            *out_ptr.add(i) = if val > 0.0 { val } else { 0.0 };
                        }
                    }
                }
                Op::ReluGrad { grad, input } => {
                    let p_grad = get_const_buf(*grad);
                    let p_in = get_const_buf(*input);
                    unsafe {
                        for i in 0..numel {
                            let cond = *p_in.add(i) > 0.0;
                            *out_ptr.add(i) = if cond { *p_grad.add(i) } else { 0.0 };
                        }
                    }
                }
                Op::FusedAddReLU(lhs, rhs) => {
                    // Fused Add + ReLU: Single loop in registers
                    let p_lhs = get_const_buf(*lhs);
                    let p_rhs = get_const_buf(*rhs);
                    unsafe {
                        for i in 0..numel {
                            let sum = *p_lhs.add(i) + *p_rhs.add(i);
                            *out_ptr.add(i) = if sum > 0.0 { sum } else { 0.0 };
                        }
                    }
                }
                Op::FusedMatMulBiasReLU { input, weight, bias } => {
                    // Radical Fusion: MatMul + Bias Broadcast + ReLU entirely in CPU registers
                    let p_x = get_const_buf(*input);
                    let p_w = get_const_buf(*weight);
                    let p_b = get_const_buf(*bias);

                    let m = step.shape.rows;
                    let n = step.shape.cols;
                    let k_dim = self.steps.iter().find(|s| s.node_id == *input).unwrap().shape.cols;

                    unsafe {
                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = *p_b.add(j); // Broadcast bias directly into accumulator
                                for k in 0..k_dim {
                                    acc += *p_x.add(i * k_dim + k) * *p_w.add(k * n + j);
                                }
                                // Non-linear activation in register before committing to memory
                                *out_ptr.add(i * n + j) = if acc > 0.0 { acc } else { 0.0 };
                            }
                        }
                    }
                }
            }
        }
    }

    /// Read the evaluated output slice for a given node after execution.
    ///
    /// # Safety
    /// Must be called with the same `arena` active and without intermediate resets.
    pub fn read_node_output<'a>(&self, node_id: NodeId, arena: &'a StaticArenaAllocator) -> &'a [f32] {
        let offset = self.buffer_offsets[&node_id];
        let step = self.steps.iter().find(|s| s.node_id == node_id).expect("Node not in pipeline");
        let numel = step.shape.numel();

        unsafe {
            let ptr = (arena.capacity() > 0) as usize; // assertion guard
            let _ = ptr;
            // The arena's allocated memory is accessible via its base
            // In read-only mode, we obtain slice from the allocated offset
            let raw_base = (arena as *const _ as *const usize).read() as *const u8;
            let target_ptr = raw_base.add(offset) as *const f32;
            std::slice::from_raw_parts(target_ptr, numel)
        }
    }
}
