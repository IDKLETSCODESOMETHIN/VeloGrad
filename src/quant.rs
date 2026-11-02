// =================================================================================================
// PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
// TIER 5 MIXED-PRECISION & QUANTIZATION ENGINE (BF16, FP16, FP8)
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

//! # VeloGrad Mixed-Precision & Quantization Engine
//!
//! Provides ultra-fast lower-precision arithmetic:
//! - **BF16 (Bfloat16)**: 8-bit exponent, 7-bit mantissa (same dynamic range as FP32).
//! - **FP16 (Half-Precision)**: 5-bit exponent, 10-bit mantissa.
//! - **INT8 Dynamic Quantization**: Symmetric absolute max quantization for 4x memory savings.

/// Bfloat16 representation (16-bit brain floating point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct BFloat16(pub u16);

impl BFloat16 {
    /// Converts a 32-bit float to Bfloat16 with round-to-nearest-even.
    #[inline(always)]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        // Truncate bottom 16 bits with rounding
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        let rounded = bits.wrapping_add(rounding_bias);
        Self((rounded >> 16) as u16)
    }

    /// Converts a Bfloat16 back to standard 32-bit float.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// Symmetrically quantizes an f32 buffer into INT8 with an absolute-max scale factor.
pub fn quantize_int8(input: &[f32], output: &mut [i8]) -> f32 {
    assert_eq!(input.len(), output.len());

    let mut max_abs = 0.0f32;
    for &x in input {
        let abs = x.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0.0 {
        output.fill(0);
        return 1.0;
    }

    let scale = 127.0 / max_abs;
    for (i, &x) in input.iter().enumerate() {
        let q = (x * scale).round().clamp(-128.0, 127.0);
        output[i] = q as i8;
    }

    1.0 / scale
}

/// Dequantizes an INT8 buffer back into an f32 buffer using the inverse scale factor.
pub fn dequantize_int8(input: &[i8], scale: f32, output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &q) in input.iter().enumerate() {
        output[i] = (q as f32) * scale;
    }
}
