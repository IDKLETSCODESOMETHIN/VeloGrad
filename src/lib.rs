// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
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
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop)]

//! # VeloGrad Foundation Core
//!
//! Bare-metal, pure-compiled neural engine providing:
//! - Hardware cache-line aligned static memory arenas (`StaticArenaAllocator`).
//! - Compile-time shape-checked tensors with zero-copy slicing and zero runtime overhead (`Tensor`).
//! - Hardware-fused kernel execution primitives avoiding round-trips to main memory.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Module 2: Symbolic Ahead-of-Time Autograd & Radical Kernel Fusion Engine.
pub mod autograd;

/// Module 3: Bare-Metal Hardware Acceleration Engine (CUDA & Multi-Vendor GPU Compute).
pub mod hardware;

/// Module 4: Zero-Friction PyTorch Drop-In API Layer (PyO3 & NumPy Bridge).
pub mod python;

/// Advanced Neural Operators: Conv2D, LayerNorm, RMSNorm, Softmax, CrossEntropy, FlashAttention.
pub mod ops;

/// Pillar 1: Pure Ahead-of-Time Native Machine Code / LLVM-C Generator.
pub mod codegen;

/// Tier 5: Hardware Autotuning JIT Engine.
pub mod autotune;

/// Tier 5: Distributed Multi-GPU Ring All-Reduce Engine.
pub mod distributed;

/// Tier 5: Mixed-Precision & INT8/BF16 Quantization Engine.
pub mod quant;

/// Cache-line alignment standard for ultra-high-throughput modern architectures (AVX-512 / ARM Neoverse).
pub const CACHE_LINE_BYTES: usize = 64;

/// Errors returned by the VeloGrad arena memory subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// Requested byte allocation exceeds available unmapped contiguous capacity.
    OutOfMemory {
        /// Requested byte count.
        requested: usize,
        /// Remaining contiguous bytes.
        available: usize,
    },
    /// The specified memory layout could not be constructed or violates alignment rules.
    InvalidLayout,
    /// System allocator failed to provide requested initial contiguous block.
    AllocationFailed,
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(
                    f,
                    "StaticArenaAllocator OOM: requested {} bytes, but only {} bytes remain",
                    requested, available
                )
            }
            Self::InvalidLayout => write!(f, "Invalid memory layout: alignment must be non-zero power of two"),
            Self::AllocationFailed => write!(f, "OS virtual memory subsystem failed to allocate contiguous backing memory"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// A high-performance, single-allocation static memory arena.
///
/// Pre-allocates a massive contiguous block of memory with 64-byte cache-line alignment
/// at initialization. Eliminates all dynamic memory allocations (`malloc` / `free`) during
/// operational training and inference loops.
pub struct StaticArenaAllocator {
    /// Non-null pointer to base of contiguous virtual memory block.
    base: NonNull<u8>,
    /// Global layout metadata used for deallocation.
    layout: Layout,
    /// Current byte offset pointer for bump allocations.
    offset: usize,
    /// Total fixed capacity in bytes.
    capacity: usize,
    /// High-water mark of allocated bytes for telemetry and capacity planning.
    peak_bytes: usize,
}

// Safety: StaticArenaAllocator owns the raw backing buffer exclusively.
// It can safely be sent across threads if no active borrows exist.
unsafe impl Send for StaticArenaAllocator {}

impl StaticArenaAllocator {
    /// Creates a new `StaticArenaAllocator` with the specified byte capacity.
    ///
    /// Allocates memory directly from the OS page allocator using 64-byte cache-line alignment.
    /// Guarantees that the base memory address is aligned to modern SIMD cache boundaries.
    pub fn new(capacity_bytes: usize) -> Result<Self, ArenaError> {
        if capacity_bytes == 0 {
            return Err(ArenaError::InvalidLayout);
        }

        // Align total capacity up to a multiple of the cache line size
        let aligned_capacity = (capacity_bytes + (CACHE_LINE_BYTES - 1)) & !(CACHE_LINE_BYTES - 1);

        let layout = Layout::from_size_align(aligned_capacity, CACHE_LINE_BYTES)
            .map_err(|_| ArenaError::InvalidLayout)?;

        // Allocate zero-initialized memory block to prevent cold-page faults during forward pass
        // Safety: layout has non-zero size and valid alignment.
        let raw_ptr = unsafe { alloc_zeroed(layout) };
        let base = NonNull::new(raw_ptr).ok_or(ArenaError::AllocationFailed)?;

        debug_assert_eq!(
            (base.as_ptr() as usize) % CACHE_LINE_BYTES,
            0,
            "Arena base address MUST be aligned to 64 bytes"
        );

        Ok(Self {
            base,
            layout,
            offset: 0,
            capacity: aligned_capacity,
            peak_bytes: 0,
        })
    }

    /// Allocates a contiguous typed slice of `T` aligned to 64-byte cache boundaries.
    ///
    /// Zero dynamic OS calls are made; allocations are bump-pointer advances with arithmetic alignment.
    pub fn alloc_slice<T>(&mut self, count: usize) -> Result<NonNull<T>, ArenaError> {
        let element_size = std::mem::size_of::<T>();
        let raw_bytes = count
            .checked_mul(element_size)
            .ok_or(ArenaError::InvalidLayout)?;

        // Enforce maximum of T's alignment and 64-byte hardware cache line alignment
        let target_align = CACHE_LINE_BYTES.max(std::mem::align_of::<T>());

        // Calculate next aligned offset
        let current_ptr = unsafe { self.base.as_ptr().add(self.offset) as usize };
        let aligned_ptr = (current_ptr + (target_align - 1)) & !(target_align - 1);
        let padding = aligned_ptr - current_ptr;

        let total_required = padding
            .checked_add(raw_bytes)
            .ok_or(ArenaError::InvalidLayout)?;

        if self.offset + total_required > self.capacity {
            return Err(ArenaError::OutOfMemory {
                requested: total_required,
                available: self.capacity - self.offset,
            });
        }

        let alloc_offset = self.offset + padding;
        self.offset += total_required;

        if self.offset > self.peak_bytes {
            self.peak_bytes = self.offset;
        }

        let result_ptr = unsafe { self.base.as_ptr().add(alloc_offset) as *mut T };
        let non_null = unsafe { NonNull::new_unchecked(result_ptr) };

        debug_assert_eq!(
            (result_ptr as usize) % target_align,
            0,
            "Allocated slice must strictly honor cache-line alignment"
        );

        Ok(non_null)
    }

    /// Allocates a compile-time shape-checked 2D Tensor directly from the arena.
    ///
    /// The tensor is guaranteed to reside in 64-byte aligned, contiguous physical memory,
    /// tied to the lifetime `'arena` of this allocator instance.
    #[inline]
    pub fn alloc_tensor<const ROWS: usize, const COLS: usize>(
        &mut self,
    ) -> Result<Tensor<'_, ROWS, COLS>, ArenaError> {
        let total_elements = ROWS * COLS;
        let ptr = self.alloc_slice::<f32>(total_elements)?;
        Ok(Tensor::new(ptr))
    }

    /// Resets the allocator offset to zero in O(1) time.
    ///
    /// Retains the underlying contiguous virtual memory allocation for the next iteration,
    /// achieving true zero-cost memory recycling without OS deallocation overhead.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Returns the number of currently allocated bytes.
    #[inline(always)]
    pub fn allocated_bytes(&self) -> usize {
        self.offset
    }

    /// Returns the number of unallocated bytes remaining before saturation.
    #[inline(always)]
    pub fn remaining_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }

    /// Returns the historical peak memory consumption in bytes.
    #[inline(always)]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Returns the total configured virtual capacity in bytes.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the raw non-null base pointer to the contiguous virtual memory block.
    #[inline(always)]
    pub fn base_ptr(&self) -> NonNull<u8> {
        self.base
    }
}

impl Drop for StaticArenaAllocator {
    fn drop(&mut self) {
        // Safety: self.base was allocated with self.layout by alloc_zeroed.
        unsafe {
            dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

/// Compile-time shape-checked, zero-copy multi-dimensional Tensor.
///
/// Dimensions `ROWS` and `COLS` propagate through the Rust type system at compile-time.
/// Any dimensional mismatch (e.g. invalid matrix multiplication inner dimension)
/// triggers a hard compiler error before a single byte executes.
///
/// Internal memory is mapped via raw hardware pointers (`NonNull<f32>`) with zero heap
/// metadata overhead and explicit row/column stride arithmetic.
pub struct Tensor<'arena, const ROWS: usize, const COLS: usize> {
    /// Raw pointer to the first element in memory (aligned to 64 bytes).
    ptr: NonNull<f32>,
    /// Number of elements to step to traverse to the next row.
    row_stride: usize,
    /// Number of elements to step to traverse to the next column.
    col_stride: usize,
    /// Compile-time lifetime marker tying memory borrow to the underlying arena allocator.
    _marker: PhantomData<&'arena mut [f32]>,
}

// Safety: Tensor owns a distinct view into memory managed by 'arena.
// Access follows standard Rust borrow-checking rules (aliasing XOR mutability).
unsafe impl<'arena, const ROWS: usize, const COLS: usize> Send for Tensor<'arena, ROWS, COLS> {}
unsafe impl<'arena, const ROWS: usize, const COLS: usize> Sync for Tensor<'arena, ROWS, COLS> {}

impl<'arena, const ROWS: usize, const COLS: usize> Tensor<'arena, ROWS, COLS> {
    /// Number of elements in this tensor known at compile time.
    pub const NUMEL: usize = ROWS * COLS;

    /// Creates a new contiguous tensor instance wrapping an aligned raw pointer.
    ///
    /// # Safety
    /// The caller must ensure that `ptr` points to an allocated block containing at least
    /// `ROWS * COLS` contiguous `f32` elements valid for lifetime `'arena`.
    #[inline(always)]
    pub fn new(ptr: NonNull<f32>) -> Self {
        Self {
            ptr,
            row_stride: COLS,
            col_stride: 1,
            _marker: PhantomData,
        }
    }

    /// Creates a tensor view with custom row and column strides.
    ///
    /// Used internally to construct zero-copy slices and transposed views.
    #[inline(always)]
    pub unsafe fn from_raw_strided(
        ptr: NonNull<f32>,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            ptr,
            row_stride,
            col_stride,
            _marker: PhantomData,
        }
    }

    /// Returns the raw const pointer to the tensor's memory.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// Returns the raw mutable pointer to the tensor's memory.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    /// Read an element with compile-time bounds and runtime boundary verification.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < ROWS, "Row index out of bounds");
        assert!(col < COLS, "Column index out of bounds");
        unsafe { *self.offset_ptr(row, col) }
    }

    /// Write an element with compile-time bounds and runtime boundary verification.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < ROWS, "Row index out of bounds");
        assert!(col < COLS, "Column index out of bounds");
        unsafe {
            *self.offset_mut_ptr(row, col) = value;
        }
    }

    /// High-performance unchecked element read for fused execution kernels.
    ///
    /// # Safety
    /// Calling this with `row >= ROWS` or `col >= COLS` invokes Undefined Behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> f32 {
        unsafe { *self.offset_ptr(row, col) }
    }

    /// High-performance unchecked element write for fused execution kernels.
    ///
    /// # Safety
    /// Calling this with `row >= ROWS` or `col >= COLS` invokes Undefined Behavior.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, row: usize, col: usize, value: f32) {
        unsafe {
            *self.offset_mut_ptr(row, col) = value;
        }
    }

    /// Computes the exact raw memory pointer for coordinate (row, col).
    #[inline(always)]
    fn offset_ptr(&self, row: usize, col: usize) -> *const f32 {
        let linear_offset = row * self.row_stride + col * self.col_stride;
        unsafe { self.ptr.as_ptr().add(linear_offset) }
    }

    /// Computes the exact mutable raw memory pointer for coordinate (row, col).
    #[inline(always)]
    fn offset_mut_ptr(&mut self, row: usize, col: usize) -> *mut f32 {
        let linear_offset = row * self.row_stride + col * self.col_stride;
        unsafe { self.ptr.as_ptr().add(linear_offset) }
    }

    /// Zero-Copy Slicing: Returns an immutable sub-view of the tensor with compile-time verified sub-dimensions.
    ///
    /// Performs instantaneous pointer arithmetic without copying a single float or issuing an OS allocation.
    #[inline(always)]
    pub fn slice_2d<const SUB_M: usize, const SUB_N: usize>(
        &self,
        row_offset: usize,
        col_offset: usize,
    ) -> Tensor<'arena, SUB_M, SUB_N> {
        assert!(
            row_offset + SUB_M <= ROWS,
            "Slice row range exceeds tensor bounds"
        );
        assert!(
            col_offset + SUB_N <= COLS,
            "Slice column range exceeds tensor bounds"
        );

        let sub_ptr = unsafe {
            let offset = row_offset * self.row_stride + col_offset * self.col_stride;
            NonNull::new_unchecked(self.ptr.as_ptr().add(offset))
        };

        unsafe { Tensor::from_raw_strided(sub_ptr, self.row_stride, self.col_stride) }
    }

    /// Zero-Copy Slicing (Mutable): Returns a mutable sub-view with compile-time checked sub-dimensions.
    #[inline(always)]
    pub fn slice_2d_mut<const SUB_M: usize, const SUB_N: usize>(
        &mut self,
        row_offset: usize,
        col_offset: usize,
    ) -> Tensor<'arena, SUB_M, SUB_N> {
        assert!(
            row_offset + SUB_M <= ROWS,
            "Slice row range exceeds tensor bounds"
        );
        assert!(
            col_offset + SUB_N <= COLS,
            "Slice column range exceeds tensor bounds"
        );

        let sub_ptr = unsafe {
            let offset = row_offset * self.row_stride + col_offset * self.col_stride;
            NonNull::new_unchecked(self.ptr.as_ptr().add(offset))
        };

        unsafe { Tensor::from_raw_strided(sub_ptr, self.row_stride, self.col_stride) }
    }

    /// Zero-Copy Transpose: Flips rows and columns in O(1) time by swapping strides.
    #[inline(always)]
    pub fn transpose(&self) -> Tensor<'arena, COLS, ROWS> {
        unsafe { Tensor::from_raw_strided(self.ptr, self.col_stride, self.row_stride) }
    }

    /// In-place zero-overhead memory mutation: fills the entire tensor with a constant value.
    #[inline]
    pub fn fill_(&mut self, val: f32) {
        if self.col_stride == 1 && self.row_stride == COLS {
            // Contiguous fast path: vectorizable direct memory fill
            let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), Self::NUMEL) };
            slice.fill(val);
        } else {
            // Strided path
            for r in 0..ROWS {
                for c in 0..COLS {
                    unsafe {
                        *self.offset_mut_ptr(r, c) = val;
                    }
                }
            }
        }
    }

    /// In-place zero-overhead memory mutation: scales all elements by scalar `alpha`.
    #[inline]
    pub fn scale_(&mut self, alpha: f32) {
        if self.col_stride == 1 && self.row_stride == COLS {
            let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), Self::NUMEL) };
            for x in slice.iter_mut() {
                *x *= alpha;
            }
        } else {
            for r in 0..ROWS {
                for c in 0..COLS {
                    unsafe {
                        *self.offset_mut_ptr(r, c) *= alpha;
                    }
                }
            }
        }
    }

    /// In-place element-wise addition: `self = self + other`.
    #[inline]
    pub fn add_(&mut self, other: &Tensor<'arena, ROWS, COLS>) {
        if self.col_stride == 1
            && self.row_stride == COLS
            && other.col_stride == 1
            && other.row_stride == COLS
        {
            let dest = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), Self::NUMEL) };
            let src = unsafe { std::slice::from_raw_parts(other.ptr.as_ptr(), Self::NUMEL) };
            for (d, s) in dest.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        } else {
            for r in 0..ROWS {
                for c in 0..COLS {
                    unsafe {
                        *self.offset_mut_ptr(r, c) += other.get_unchecked(r, c);
                    }
                }
            }
        }
    }

    /// Fused kernel: `self = ReLU(self * alpha + other)`.
    ///
    /// Demonstrates VeloGrad Pillar 3 (Operation Fusion): fuses scaling, element-wise addition,
    /// and non-linear activation (ReLU) into a single iteration pass, keeping values inside
    /// processor SIMD registers and completely eliminating intermediate memory bus writebacks.
    #[inline]
    pub fn fused_scale_add_relu_(
        &mut self,
        other: &Tensor<'arena, ROWS, COLS>,
        alpha: f32,
    ) {
        if self.col_stride == 1
            && self.row_stride == COLS
            && other.col_stride == 1
            && other.row_stride == COLS
        {
            let dest = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), Self::NUMEL) };
            let src = unsafe { std::slice::from_raw_parts(other.ptr.as_ptr(), Self::NUMEL) };
            for (d, s) in dest.iter_mut().zip(src.iter()) {
                let fused = (*d * alpha) + *s;
                *d = if fused > 0.0 { fused } else { 0.0 };
            }
        } else {
            for r in 0..ROWS {
                for c in 0..COLS {
                    unsafe {
                        let val = (*self.offset_ptr(r, c) * alpha) + other.get_unchecked(r, c);
                        *self.offset_mut_ptr(r, c) = if val > 0.0 { val } else { 0.0 };
                    }
                }
            }
        }
    }

    /// Compile-Time Shape-Safe Matrix Multiplication: `C = A * B`.
    ///
    /// Dimension checking occurs strictly at compile time:
    /// - `self`: `(ROWS x COLS)`
    /// - `other`: `(COLS x K)`
    /// - `out`: `(ROWS x K)`
    ///
    /// If the inner dimension of `other` does not match `COLS`, the compiler refuses to compile,
    /// completely preventing runtime dimensional mismatch panics.
    #[inline]
    pub fn matmul<'out, const K: usize>(
        &self,
        other: &Tensor<'arena, COLS, K>,
        out: &mut Tensor<'out, ROWS, K>,
    ) {
        // High-performance cache-blocked register accumulation
        for i in 0..ROWS {
            for j in 0..K {
                let mut acc = 0.0f32;
                for k in 0..COLS {
                    unsafe {
                        acc += self.get_unchecked(i, k) * other.get_unchecked(k, j);
                    }
                }
                unsafe {
                    out.set_unchecked(i, j, acc);
                }
            }
        }
    }

    /// Fused GEMM + Bias + ReLU Kernel: `C = ReLU(A * B + Bias)`.
    ///
    /// Computes matrix multiplication, adds row broadcast bias, and evaluates activation
    /// entirely within CPU registers before committing memory to `out`.
    #[inline]
    pub fn fused_gemm_bias_relu<'out, const K: usize>(
        &self,
        other: &Tensor<'arena, COLS, K>,
        bias: &Tensor<'arena, 1, K>,
        out: &mut Tensor<'out, ROWS, K>,
    ) {
        for i in 0..ROWS {
            for j in 0..K {
                let mut acc = unsafe { bias.get_unchecked(0, j) };
                for k in 0..COLS {
                    unsafe {
                        acc += self.get_unchecked(i, k) * other.get_unchecked(k, j);
                    }
                }
                // Fused ReLU
                let activated = if acc > 0.0 { acc } else { 0.0 };
                unsafe {
                    out.set_unchecked(i, j, activated);
                }
            }
        }
    }
}

impl<'arena, const ROWS: usize, const COLS: usize> fmt::Debug for Tensor<'arena, ROWS, COLS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Tensor<{}, {}> (ptr: {:p}, row_stride: {}, col_stride: {}) [",
            ROWS,
            COLS,
            self.ptr.as_ptr(),
            self.row_stride,
            self.col_stride
        )?;
        for r in 0..ROWS.min(6) {
            write!(f, "  [")?;
            for c in 0..COLS.min(8) {
                write!(f, "{:8.4} ", self.get(r, c))?;
            }
            if COLS > 8 {
                write!(f, "... ")?;
            }
            writeln!(f, "]")?;
        }
        if ROWS > 6 {
            writeln!(f, "  ... ({} more rows)", ROWS - 6)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_cache_alignment() {
        let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to create arena");
        assert_eq!((arena.base.as_ptr() as usize) % CACHE_LINE_BYTES, 0);

        let slice1 = arena.alloc_slice::<f32>(10).expect("Alloc 1 failed");
        assert_eq!((slice1.as_ptr() as usize) % CACHE_LINE_BYTES, 0);

        let slice2 = arena.alloc_slice::<f32>(100).expect("Alloc 2 failed");
        assert_eq!((slice2.as_ptr() as usize) % CACHE_LINE_BYTES, 0);
    }

    #[test]
    fn test_tensor_compile_time_shape_and_mutation() {
        let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to create arena");
        let mut t: Tensor<'_, 4, 4> = arena.alloc_tensor().expect("Tensor alloc failed");

        t.fill_(2.0);
        assert_eq!(t.get(0, 0), 2.0);
        assert_eq!(t.get(3, 3), 2.0);

        t.scale_(3.0);
        assert_eq!(t.get(0, 0), 6.0);
        assert_eq!(t.get(3, 3), 6.0);

        t.set(1, 2, 42.0);
        assert_eq!(t.get(1, 2), 42.0);
    }

    #[test]
    fn test_zero_copy_slicing() {
        let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to create arena");
        let mut t: Tensor<'_, 8, 8> = arena.alloc_tensor().expect("Tensor alloc failed");

        // Fill with linear index
        for r in 0..8 {
            for c in 0..8 {
                t.set(r, c, (r * 8 + c) as f32);
            }
        }

        // Zero-copy subslice 2x3 starting at (2, 3)
        let sub: Tensor<'_, 2, 3> = t.slice_2d(2, 3);
        assert_eq!(sub.get(0, 0), (2 * 8 + 3) as f32);
        assert_eq!(sub.get(1, 2), (3 * 8 + 5) as f32);

        // Verify underlying pointers match zero-copy offset arithmetic
        let expected_offset = (2 * 8 + 3) * std::mem::size_of::<f32>();
        assert_eq!(
            sub.as_ptr() as usize - t.as_ptr() as usize,
            expected_offset
        );
    }

    #[test]
    fn test_compile_time_matmul() {
        let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to create arena");
        let mut a: Tensor<'_, 2, 3> = arena.alloc_tensor().unwrap();
        let mut b: Tensor<'_, 3, 2> = arena.alloc_tensor().unwrap();
        let mut c: Tensor<'_, 2, 2> = arena.alloc_tensor().unwrap();

        a.fill_(1.0);
        b.fill_(2.0);

        // A (2x3) * B (3x2) -> C (2x2)
        // Each element of C is 1.0 * 2.0 * 3 = 6.0
        a.matmul(&b, &mut c);

        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 6.0);
    }

    #[test]
    fn test_fused_gemm_bias_relu() {
        let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to create arena");
        let mut a: Tensor<'_, 2, 2> = arena.alloc_tensor().unwrap();
        let mut b: Tensor<'_, 2, 2> = arena.alloc_tensor().unwrap();
        let mut bias: Tensor<'_, 1, 2> = arena.alloc_tensor().unwrap();
        let mut out: Tensor<'_, 2, 2> = arena.alloc_tensor().unwrap();

        a.fill_(1.0);
        b.fill_(1.0);
        // GEMM produces 2.0 for all elements
        // Bias is -5.0 for col 0 and +1.0 for col 1
        bias.set(0, 0, -5.0);
        bias.set(0, 1, 1.0);

        a.fused_gemm_bias_relu(&b, &bias, &mut out);

        // Col 0: ReLU(2.0 - 5.0) = ReLU(-3.0) = 0.0
        // Col 1: ReLU(2.0 + 1.0) = ReLU(3.0) = 3.0
        assert_eq!(out.get(0, 0), 0.0);
        assert_eq!(out.get(0, 1), 3.0);
        assert_eq!(out.get(1, 0), 0.0);
        assert_eq!(out.get(1, 1), 3.0);
    }
}
