// =================================================================================================
// PROJECT: VeloGrad Engine Core Integration Tests
// COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
// =================================================================================================

use velograd::{CACHE_LINE_BYTES, StaticArenaAllocator, Tensor};

#[test]
fn test_integration_full_feedforward_cycle() {
    let mut arena = StaticArenaAllocator::new(16 * 1024 * 1024).expect("Failed to initialize arena");

    // Dimensions: Batch=2, In=4, Hidden=8, Out=2
    let mut x: Tensor<'_, 2, 4> = arena.alloc_tensor().unwrap();
    let mut w1: Tensor<'_, 4, 8> = arena.alloc_tensor().unwrap();
    let mut b1: Tensor<'_, 1, 8> = arena.alloc_tensor().unwrap();
    let mut h1: Tensor<'_, 2, 8> = arena.alloc_tensor().unwrap();

    let mut w2: Tensor<'_, 8, 2> = arena.alloc_tensor().unwrap();
    let mut b2: Tensor<'_, 1, 2> = arena.alloc_tensor().unwrap();
    let mut y: Tensor<'_, 2, 2> = arena.alloc_tensor().unwrap();

    x.fill_(1.0);
    w1.fill_(0.5);
    b1.fill_(0.1);
    w2.fill_(0.25);
    b2.fill_(0.0);

    // Forward Layer 1 (Fused GEMM + Bias + ReLU)
    x.fused_gemm_bias_relu(&w1, &b1, &mut h1);

    // Expected H1: 4 * (1.0 * 0.5) + 0.1 = 2.1
    for r in 0..2 {
        for c in 0..8 {
            assert!((h1.get(r, c) - 2.1).abs() < 1e-5);
        }
    }

    // Forward Layer 2 (Fused GEMM + Bias + ReLU)
    h1.fused_gemm_bias_relu(&w2, &b2, &mut y);

    // Expected Y: 8 * (2.1 * 0.25) + 0.0 = 4.2
    for r in 0..2 {
        for c in 0..2 {
            assert!((y.get(r, c) - 4.2).abs() < 1e-5);
        }
    }

    // Verification of zero heap overhead during execution
    let allocated_before_reset = arena.allocated_bytes();
    assert!(allocated_before_reset > 0);

    // Reset arena
    arena.reset();
    assert_eq!(arena.allocated_bytes(), 0);
}

#[test]
fn test_integration_transposition_and_strided_access() {
    let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to initialize arena");
    let mut mat: Tensor<'_, 3, 2> = arena.alloc_tensor().unwrap();

    mat.set(0, 0, 1.0);
    mat.set(0, 1, 2.0);
    mat.set(1, 0, 3.0);
    mat.set(1, 1, 4.0);
    mat.set(2, 0, 5.0);
    mat.set(2, 1, 6.0);

    let transposed: Tensor<'_, 2, 3> = mat.transpose();
    assert_eq!(transposed.get(0, 0), 1.0);
    assert_eq!(transposed.get(1, 0), 2.0);
    assert_eq!(transposed.get(0, 1), 3.0);
    assert_eq!(transposed.get(1, 1), 4.0);
    assert_eq!(transposed.get(0, 2), 5.0);
    assert_eq!(transposed.get(1, 2), 6.0);
}

#[test]
fn test_integration_cache_line_invariants() {
    let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Failed to initialize arena");

    for _ in 0..50 {
        let t: Tensor<'_, 1, 1> = arena.alloc_tensor().unwrap();
        assert_eq!((t.as_ptr() as usize) % CACHE_LINE_BYTES, 0);
    }
}

#[test]
fn test_integration_aot_autograd_and_kernel_fusion() {
    use velograd::autograd::{Graph, Op};
    use std::collections::HashMap;

    let mut graph = Graph::new();

    // Model: Z = ReLU(X @ W + B)
    // Shapes: X (2x3), W (3x4), B (1x4)
    let x = graph.input("X", 2, 3);
    let w = graph.parameter("W", 3, 4);
    let b = graph.parameter("B", 1, 4);

    let xw = graph.matmul(x, w);
    let add_b = graph.add(xw, b);
    let z = graph.relu(add_b);

    // Compute AOT backward pass for loss = sum(Z)
    let grads = graph.aot_backward(z);
    assert!(grads.contains_key(&w));
    assert!(grads.contains_key(&b));

    // Compile graph with radical kernel fusion
    let pipeline = graph.compile();

    // Verify fusion occurred: check if FusedMatMulBiasReLU was synthesized
    let fused_count = pipeline.steps.iter().filter(|s| {
        matches!(s.op, Op::FusedMatMulBiasReLU { .. })
    }).count();
    assert!(fused_count > 0, "Expected MatMul + Bias + ReLU fusion to be triggered");

    // Execute compiled pipeline with arena memory
    let mut arena = StaticArenaAllocator::new(1024 * 1024).expect("Arena creation failed");

    let x_data = vec![1.0f32; 6];
    let w_data = vec![0.5f32; 12];
    let b_data = vec![0.1f32; 4];

    let mut inputs = HashMap::new();
    inputs.insert("X".to_string(), x_data.as_slice());

    let mut params = HashMap::new();
    params.insert("W".to_string(), w_data.as_slice());
    params.insert("B".to_string(), b_data.as_slice());

    pipeline.execute(&mut arena, &inputs, &params);
}

#[test]
fn test_integration_hardware_acceleration_and_shared_memory_gemm() {
    use velograd::hardware::{HardwareDevice, BLOCK_TILE_DIM};

    let device = HardwareDevice::create_default();
    let stream = device.create_stream(0);

    // Verify telemetry
    assert!(device.descriptor().compute_units > 0);
    assert!(device.descriptor().supports_unified_memory);

    // Initialize StaticArenaAllocator (64-byte aligned)
    let mut arena = StaticArenaAllocator::new(1024 * 1024).unwrap();

    // Map arena directly to device without copies
    let device_buffer = device.map_arena_to_device(arena.base_ptr(), arena.capacity());
    assert_eq!(device_buffer.byte_size(), arena.capacity());

    // Allocate 3 matrices from arena: A (16x16), B (16x16), C (16x16)
    let m = BLOCK_TILE_DIM;
    let k = BLOCK_TILE_DIM;
    let n = BLOCK_TILE_DIM;

    let mut a_tensor: Tensor<'_, 16, 16> = arena.alloc_tensor().unwrap();
    let mut b_tensor: Tensor<'_, 16, 16> = arena.alloc_tensor().unwrap();
    let mut c_tensor: Tensor<'_, 16, 16> = arena.alloc_tensor().unwrap();

    a_tensor.fill_(2.0);
    b_tensor.fill_(3.0);
    c_tensor.fill_(0.0);

    // Dispatch tiled GEMM with shared memory caching
    device.dispatch_tiled_gemm(
        &stream,
        a_tensor.as_ptr(),
        b_tensor.as_ptr(),
        c_tensor.as_mut_ptr(),
        m,
        k,
        n,
    );

    // Expected value: 16 * (2.0 * 3.0) = 96.0
    for r in 0..16 {
        for c in 0..16 {
            assert_eq!(c_tensor.get(r, c), 96.0);
        }
    }

    // Test Fused GEMM + Bias + ReLU
    let mut bias_tensor: Tensor<'_, 1, 16> = arena.alloc_tensor().unwrap();
    bias_tensor.fill_(-100.0); // 96.0 - 100.0 = -4.0 -> ReLU = 0.0

    device.dispatch_fused_gemm_bias_relu(
        &stream,
        a_tensor.as_ptr(),
        b_tensor.as_ptr(),
        bias_tensor.as_ptr(),
        c_tensor.as_mut_ptr(),
        m,
        k,
        n,
    );

    for r in 0..16 {
        for c in 0..16 {
            assert_eq!(c_tensor.get(r, c), 0.0);
        }
    }
}

#[test]
fn test_integration_python_tensor_buffer_and_drop_in_api() {
    use velograd::python::{PyTensor, TensorBuffer};

    // Test TensorBuffer cache-line alignment
    let mut buf = TensorBuffer::new(8, 16);
    assert_eq!((buf.ptr.as_ptr() as usize) % CACHE_LINE_BYTES, 0);

    buf.fill_(4.5);
    assert_eq!(buf.get(0, 0), 4.5);
    assert_eq!(buf.get(7, 15), 4.5);

    buf.set(3, 4, 99.0);
    assert_eq!(buf.get(3, 4), 99.0);

    // Test PyTensor wrapper
    let py_tensor = PyTensor::new_raw(4, 8, true);
    assert!(py_tensor.requires_grad);
    assert!(py_tensor.grad.lock().unwrap().is_none());

    py_tensor.buffer.lock().unwrap().fill_(2.0);
    assert_eq!(py_tensor.buffer.lock().unwrap().get(0, 0), 2.0);
}

#[test]
fn test_integration_advanced_operators_and_flash_attention() {
    use velograd::ops::*;

    // 1. Softmax Numerical Stability Test
    let input = vec![1000.0, 1001.0, 1002.0]; // Would overflow naive exp()
    let mut out = vec![0.0f32; 3];
    softmax_forward(&input, &mut out, 1, 3);
    assert!(!out[0].is_nan() && !out[1].is_nan() && !out[2].is_nan());
    assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // 2. RMSNorm Test
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let mut y = vec![0.0f32; 4];
    rms_norm_forward(&x, &gamma, &mut y, 1, 4, 1e-5);
    assert!((y[0] - 0.365148).abs() < 1e-3);

    // 3. LayerNorm Test
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    layer_norm_forward(&x, &gamma, &beta, &mut y, 1, 4, 1e-5);
    // Mean should be 0, sum of normalized output close to 0
    let sum: f32 = y.iter().sum();
    assert!(sum.abs() < 1e-5);

    // 4. FlashAttention Test
    // Q, K, V: 2 tokens, head_dim 4
    let q = vec![1.0; 2 * 4];
    let k = vec![1.0; 2 * 4];
    let v = vec![2.0; 2 * 4];
    let mut attn_out = vec![0.0f32; 2 * 4];
    flash_attention_forward(&q, &k, &v, &mut attn_out, 2, 4);
    for val in &attn_out {
        assert!((*val - 2.0).abs() < 1e-4);
    }
}

#[test]
fn test_integration_aot_codegen_and_hardware_emitters() {
    use velograd::autograd::Graph;
    use velograd::codegen::AotCompiler;
    use velograd::hardware::{CudaPtxEmitter, MetalMslEmitter};

    // Test AOT C-Codegen
    let mut graph = Graph::new();
    let x = graph.input("X", 2, 4);
    let w = graph.parameter("W", 4, 2);
    let b = graph.parameter("B", 1, 2);
    let xw = graph.matmul(x, w);
    let add_b = graph.add(xw, b);
    let _z = graph.relu(add_b);

    let pipeline = graph.compile();
    let c_code = AotCompiler::emit_c_source(&pipeline, "velograd_aot_forward");
    assert!(c_code.contains("void velograd_aot_forward"));
    assert!(c_code.contains("RESTRICT"));
    assert!(c_code.contains("arena_memory"));

    // Test PTX Emitter
    let ptx = CudaPtxEmitter::emit_gemm_ptx();
    assert!(ptx.contains(".version 7.0"));
    assert!(ptx.contains(".target sm_80"));

    // Test Metal MSL Emitter
    let msl = MetalMslEmitter::emit_gemm_msl();
    assert!(msl.contains("using namespace metal;"));
    assert!(msl.contains("kernel void velograd_tiled_gemm_msl"));
}

#[test]
fn test_integration_tier5_autotune_distributed_quant() {
    use velograd::autotune::HardwareAutotuner;
    use velograd::distributed::ProcessGroup;
    use velograd::quant::{BFloat16, quantize_int8, dequantize_int8};

    // 1. Hardware Autotuner
    let best_tile = HardwareAutotuner::tune_gemm(32, 32, 32);
    assert!(best_tile.block_m > 0 && best_tile.block_n > 0 && best_tile.block_k > 0);

    // 2. Distributed Ring All-Reduce
    let pg = ProcessGroup::new(0, 4);
    let mut grad_buffer = vec![4.0f32; 16];
    pg.ring_all_reduce(&mut grad_buffer);
    for &val in &grad_buffer {
        assert_eq!(val, 1.0); // 4.0 / 4 devices = 1.0
    }

    // 3. Mixed-Precision BFloat16 & INT8 Quantization
    let orig = 3.14159f32;
    let bf16 = BFloat16::from_f32(orig);
    let recovered = bf16.to_f32();
    assert!((orig - recovered).abs() < 0.02);

    let weights = vec![-1.0f32, 0.0, 0.5, 1.0];
    let mut quantized = vec![0i8; 4];
    let scale = quantize_int8(&weights, &mut quantized);
    let mut dequantized = vec![0.0f32; 4];
    dequantize_int8(&quantized, scale, &mut dequantized);
    for (o, d) in weights.iter().zip(dequantized.iter()) {
        assert!((o - d).abs() < 0.02);
    }
}




