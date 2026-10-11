#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/pods.hpp"

#include <cstddef>

namespace rgpot::simd {

/// Name of the target selected for this process ("scalar", "sse2", "sse4",
/// "avx2", "avx3", "avx3_zen4", "neon", "sve", "sve2", "sve_256", "rvv",
/// "emu128"). RGPOT_SIMD is read once, inside the call_once that caches
/// the choice.
const char *chosen_target_name();

/// Test hook. "scalar" forces the portable kernel. A known target the CPU
/// cannot run falls back to scalar. An unknown name is ignored.
/// Not for production paths: Highway's chosen target is process-wide.
void set_target_for_test(const char *name);

/// Targets this CPU can execute, scalar first. A name that is absent was
/// not exercised.
int runnable_target_count();
const char *runnable_target_name(int index);

double lj_force(const PairView &view, const LjParams &params, double *F,
                double strain[6]);
double morse_force(const PairView &view, const MorseParams &params, double *F,
                   double strain[6]);
void eam_force(long n, const double *R, const EamAtom *atoms, const double *box,
               double *F, double *U);

/// In-place orthorhombic minimum image, one component array per axis.
/// inv == 0 leaves that axis unfolded.
void fold_orthorhombic(double *x, double *y, double *z, std::size_t n,
                       double w0, double w1, double w2, double i0, double i1,
                       double i2);

} // namespace rgpot::simd
