#pragma once
// MIT License
// Copyright 2023--present rgpot developers

namespace rgpot::simd {

// fdlibm e_exp constants. The scalar body and every vector lane use these
// values in the same order.
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;
inline constexpr double kInvLn2 = 1.44269504088896338700e+00;
inline constexpr double kP1 = 1.66666666666666019037e-01;
inline constexpr double kP2 = -2.77777777770155933842e-03;
inline constexpr double kP3 = 6.61375632143793436117e-05;
inline constexpr double kP4 = -1.65339022054652515390e-06;
inline constexpr double kP5 = 4.13813679705723846039e-08;
inline constexpr double kOverflow = 7.09782712893383973096e+02;
inline constexpr double kUnderflow = -7.45133219101941108420e+02;

/// Scalar exp. Cody-Waite reduction by a split ln(2), then the fdlibm
/// degree-5 polynomial in separate multiplies and adds. No fused
/// multiply-add. On the fdlibm domain the error is at most 1 ulp.
/// Overflow returns +inf, underflow returns 0, a NaN returns a NaN, and
/// exp(-0) is +1. Vector targets replay this function per lane.
double exp(double x);

} // namespace rgpot::simd
