// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ext.cpp
 * @author rgpot Developers
 * @date 2026-10-11
 * @brief Nanobind frontend for the rust-core-demo staticlib.
 *
 * F1 FREE_THREADED, F4 release the GIL around the native dot, F6
 * float64 C-contiguous inputs with noconvert, F9 the link line exports
 * only PyInit__core. Shape checks and the exception translators come
 * from the vendored rgbind header.
 */

#include <cstddef>

#include "rgbind/rgbind.hpp"

extern "C" {
double rust_core_demo_dot(const double *a, const double *b, std::size_t n);
}

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "rust-core-demo: a fixed-order dot product behind a C ABI.";
  rgbind::install_policy(m);
  m.def(
      "dot",
      [](rgbind::F64Array a, rgbind::F64Array b) {
        rgbind::check_f64_rank(a, 1, "a");
        rgbind::check_f64_rank(b, 1, "b");
        const auto n = static_cast<std::size_t>(b.shape(0));
        rgbind::check_f64_1d(b, n, "b");
        rgbind::check_f64_1d(a, n, "a");
        const double *ap = a.data();
        const double *bp = b.data();
        nb::gil_scoped_release rel;
        return rust_core_demo_dot(ap, bp, n);
      },
      nb::arg("a").noconvert(), nb::arg("b").noconvert(),
      "Dot product of two float64 vectors. The sum is index order.");
}
