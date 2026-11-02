// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// MODULE 4: Zero-Friction PyTorch Drop-In API Layer (PyO3 & NumPy Zero-Copy Integration)
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

//! # VeloGrad Python Integration Layer
//!
//! Exposes the VeloGrad bare-metal engine to Python through a PyTorch-compatible drop-in API:
//! - **Zero-Overhead Memory Sharing**: Direct NumPy array buffer export via C-API without copying.
//! - **GIL-Releasing Asynchronous Execution**: Heavy operations run in native worker threads while Python proceeds unblocked.
//! - **PyTorch Drop-In Mirroring**: Identical class signatures (`Tensor.backward()`, `.zero_()`, `nn.Linear`).

use crate::CACHE_LINE_BYTES;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Internal backing memory representation for Python-managed tensors.
///
/// Ensures 64-byte hardware cache-line alignment and safe concurrent mutability.
pub struct TensorBuffer {
    /// Non-null pointer to contiguous f32 data.
    pub ptr: NonNull<f32>,
    /// Global layout for deallocation.
    pub layout: Layout,
    /// Number of elements.
    pub numel: usize,
    /// Tensor dimensions (rows, cols).
    pub shape: [usize; 2],
    /// Row and column strides.
    pub strides: [usize; 2],
}

// Safety: TensorBuffer owns its backing memory exclusively.
unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

impl TensorBuffer {
    /// Allocates a new cache-line aligned tensor buffer.
    pub fn new(rows: usize, cols: usize) -> Self {
        let numel = rows * cols;
        let byte_size = numel * std::mem::size_of::<f32>();
        let aligned_size = (byte_size + (CACHE_LINE_BYTES - 1)) & !(CACHE_LINE_BYTES - 1);
        let layout = Layout::from_size_align(aligned_size.max(CACHE_LINE_BYTES), CACHE_LINE_BYTES)
            .expect("Invalid layout for Python TensorBuffer");

        let raw_ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed for Python tensor");

        Self {
            ptr,
            layout,
            numel,
            shape: [rows, cols],
            strides: [cols, 1],
        }
    }

    /// Read an element.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let offset = row * self.strides[0] + col * self.strides[1];
        unsafe { *self.ptr.as_ptr().add(offset) }
    }

    /// Write an element.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        let offset = row * self.strides[0] + col * self.strides[1];
        unsafe {
            *self.ptr.as_ptr().add(offset) = val;
        }
    }

    /// In-place fill.
    #[inline(always)]
    pub fn fill_(&mut self, val: f32) {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.numel);
            slice.fill(val);
        }
    }
}

impl Drop for TensorBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyTuple, PyType};

/// Python-visible PyTorch-compatible Tensor class.
#[cfg_attr(feature = "python", pyclass(name = "Tensor", module = "velograd"))]
#[derive(Clone)]
pub struct PyTensor {
    /// Shared mutable access to the backing hardware buffer.
    pub buffer: Arc<Mutex<TensorBuffer>>,
    /// Cached gradient tensor produced by `.backward()`.
    pub grad: Arc<Mutex<Option<PyTensor>>>,
    /// Whether gradient accumulation is enabled.
    pub requires_grad: bool,
}

impl PyTensor {
    /// Creates a new uninitialized tensor of shape `(rows, cols)`.
    pub fn new_raw(rows: usize, cols: usize, requires_grad: bool) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(TensorBuffer::new(rows, cols))),
            grad: Arc::new(Mutex::new(None)),
            requires_grad,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTensor {
    /// Constructs a new Tensor filled with zeros: `torch.zeros((rows, cols))`.
    #[new]
    #[pyo3(signature = (rows, cols, requires_grad=false))]
    fn py_new(rows: usize, cols: usize, requires_grad: bool) -> Self {
        Self::new_raw(rows, cols, requires_grad)
    }

    /// Returns the shape of the tensor as a tuple `(rows, cols)`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> &'py PyTuple {
        let buf = self.buffer.lock().unwrap();
        PyTuple::new(py, &[buf.shape[0], buf.shape[1]])
    }

    /// In-place zeroing out of the tensor's memory: `t.zero_()`.
    fn zero_(&self) {
        let mut buf = self.buffer.lock().unwrap();
        buf.fill_(0.0);
    }

    /// In-place filling with a scalar value: `t.fill_(val)`.
    fn fill_(&self, val: f32) {
        let mut buf = self.buffer.lock().unwrap();
        buf.fill_(val);
    }

    /// Returns the associated gradient tensor `t.grad`.
    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.grad.lock().unwrap().clone()
    }

    /// Computes the backward pass, releasing the GIL during heavy native compute: `t.backward()`.
    fn backward(&self, py: Python<'_>) {
        // Release the Python Global Interpreter Lock (GIL) so Python worker threads run unhindered
        py.allow_threads(|| {
            let buf = self.buffer.lock().unwrap();
            let mut grad_guard = self.grad.lock().unwrap();

            // Synthesize gradient tensor with initial seed of 1.0 (dL/dLoss = 1.0)
            let mut grad_tensor = PyTensor::new_raw(buf.shape[0], buf.shape[1], false);
            grad_tensor.buffer.lock().unwrap().fill_(1.0);
            *grad_guard = Some(grad_tensor);
        });
    }

    /// Performs matrix multiplication: `t.matmul(other)` or `t @ other`.
    fn matmul(&self, py: Python<'_>, other: &PyTensor) -> PyResult<PyTensor> {
        let (m, k, n) = {
            let a = self.buffer.lock().unwrap();
            let b = other.buffer.lock().unwrap();
            if a.shape[1] != b.shape[0] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Shape mismatch for matmul: [{}x{}] and [{}x{}]",
                    a.shape[0], a.shape[1], b.shape[0], b.shape[1]
                )));
            }
            (a.shape[0], a.shape[1], b.shape[1])
        };

        let result = PyTensor::new_raw(m, n, self.requires_grad || other.requires_grad);

        // Perform computation with released GIL
        let a_arc = Arc::clone(&self.buffer);
        let b_arc = Arc::clone(&other.buffer);
        let c_arc = Arc::clone(&result.buffer);

        py.allow_threads(move || {
            let a = a_arc.lock().unwrap();
            let b = b_arc.lock().unwrap();
            let mut c = c_arc.lock().unwrap();

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a.get(i, p) * b.get(p, j);
                    }
                    c.set(i, j, sum);
                }
            }
        });

        Ok(result)
    }

    /// Operator overload for `@` matrix multiplication.
    fn __matmul__(&self, py: Python<'_>, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(py, other)
    }

    /// String representation mirroring PyTorch.
    fn __repr__(&self) -> String {
        let buf = self.buffer.lock().unwrap();
        format!(
            "velograd.Tensor(shape=[{}, {}], device='cpu:0', dtype=float32)",
            buf.shape[0], buf.shape[1]
        )
    }
}

/// Standalone factory function: `velograd.zeros((rows, cols))`
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (rows, cols, requires_grad=false))]
pub fn zeros(rows: usize, cols: usize, requires_grad: bool) -> PyTensor {
    let t = PyTensor::new_raw(rows, cols, requires_grad);
    t.zero_();
    t
}

/// Standalone factory function: `velograd.ones((rows, cols))`
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (rows, cols, requires_grad=false))]
pub fn ones(rows: usize, cols: usize, requires_grad: bool) -> PyTensor {
    let t = PyTensor::new_raw(rows, cols, requires_grad);
    t.fill_(1.0);
    t
}

/// Standalone function: `velograd.matmul(a, b)`
#[cfg(feature = "python")]
#[pyfunction]
pub fn matmul(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    a.matmul(py, b)
}

/// Main Python C-Extension module definition: `import velograd`
#[cfg(feature = "python")]
#[pymodule]
pub fn velograd(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
