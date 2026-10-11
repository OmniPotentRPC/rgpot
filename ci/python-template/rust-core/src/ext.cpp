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
 * only PyInit__core.
 */

#include <cstddef>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
double rust_core_demo_dot(const double *a, const double *b, std::size_t n);
}

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "rust-core-demo: a fixed-order dot product behind a C ABI.";
  m.def(
      "dot",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) {
        if (a.shape(0) != b.shape(0)) {
          const std::string msg = "a: expected shape (" + std::to_string(b.shape(0)) +
                                  ",), got (" + std::to_string(a.shape(0)) + ",)";
          throw nb::value_error(msg.c_str());
        }
        const double *ap = a.data();
        const double *bp = b.data();
        const std::size_t n = static_cast<std::size_t>(a.shape(0));
        nb::gil_scoped_release rel;
        return rust_core_demo_dot(ap, bp, n);
      },
      nb::arg("a").noconvert(), nb::arg("b").noconvert(),
      "Dot product of two float64 vectors. The sum is index order.");
}
