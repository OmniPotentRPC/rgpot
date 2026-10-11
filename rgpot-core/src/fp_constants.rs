// MIT License
// Copyright 2023--present rgpot developers

//! Binary64 bits of the fused exp constants in `CppCore/rgpot/simd/exp.hpp`.
//! `scripts/export_constants.py --check` fails if the header and these bits disagree.

/// 2^27 + 1, the Veltkamp split used by Dekker's product.
pub const DEKKER_SPLIT_BITS: u64 = 0x41a0000002000000;

pub const LN2_HI_BITS: u64 = 0x3fe62e42fee00000;
pub const LN2_LO_BITS: u64 = 0x3dea39ef35793c76;
pub const INV_LN2_BITS: u64 = 0x3ff71547652b82fe;
pub const P1_BITS: u64 = 0x3fc555555555553e;
pub const P2_BITS: u64 = 0xbf66c16c16bebd93;
pub const P3_BITS: u64 = 0x3f11566aaf25de2c;
pub const P4_BITS: u64 = 0xbebbbd41c5d26bf1;
pub const P5_BITS: u64 = 0x3e66376972bea4d0;
pub const OVERFLOW_BITS: u64 = 0x40862e42fefa39ef;
pub const UNDERFLOW_BITS: u64 = 0xc0874910d52d3051;
