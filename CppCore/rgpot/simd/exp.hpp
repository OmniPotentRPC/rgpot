#pragma once
// MIT License
// Copyright 2023--present rgpot developers

namespace rgpot::simd {

/// One exp for every path. The body lives in exp.cc (not inlined) so the
/// scalar kernel and every Highway target call the same instructions.
///
/// Cody-Waite reduction by a split ln(2), then the fdlibm degree-5
/// polynomial in separate multiplies and adds. No fused multiply-add.
/// On the fdlibm domain the error is at most 1 ulp. Overflow returns
/// +inf, underflow returns 0, a NaN returns a NaN, and exp(-0) is +1.
double exp(double x);

} // namespace rgpot::simd
