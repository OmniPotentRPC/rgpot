#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * In-process XC kernel contractions from libxckernel (arXiv:2608.26440).
 *
 * This is not a Potential. Inputs are grid collocation (chi, dchi, optional
 * lapl_chi / hess_chi), named Libxc derivative arrays, and (perturbed)
 * fields. Outputs are AO Fock-like matrices. Wrapping the contractions as
 * Potential<XcPot> would invent energy / forces from kernel matrices.
 *
 * There is no PotentialConfig.xckernel arm (rgpot-qf6b). The operands are
 * not ForceInput, the results are not PotentialResult, and no DFT host
 * yet configures this over potserv. Keep the API in-process.
 *
 * Term ownership is XC-only: Coulomb, HF, and range-separated exchange
 * stay host-owned. libxckernel never evaluates functionals; the host
 * passes coefficient-mixed derivative arrays.
 *
 * First slice: families lda, gga, mgga_tau; max_order 2 (Fock o1 + fxc o2).
 * Build with meson -Dwith_xckernel=true (default false).
 *
 * Reentrancy: one XcKernel instance may be used from one thread at a time.
 * Do not share one `out` / scratch buffer across threads; construct one
 * instance per thread if callers evaluate concurrently.
 */

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rgpot {

class XcKernelError : public std::runtime_error {
public:
  explicit XcKernelError(const std::string &what) : std::runtime_error(what) {}
};

class XcKernel {
public:
  using KernelFn = int (*)(std::int64_t npts, std::int64_t nbf,
                           const double *chi, const double *dchi,
                           const double *lapl_chi, const double *hess_chi,
                           const double *const *scal, double *out);

  /// Dispatch one catalog kernel by name (e.g. "xck_lda_r_o1").
  /// Resolves the ISO C ABI symbol and its self-describing scal_names table.
  explicit XcKernel(std::string name);

  [[nodiscard]] const std::string &name() const noexcept { return name_; }
  [[nodiscard]] const std::vector<std::string> &scal_names() const noexcept {
    return scal_names_;
  }

  /// Accumulate the (nbf, nbf) matrix into `out` (+=), matching the C ABI.
  /// chi is nbf*npts row-major; dchi is 3*nbf*npts.
  /// `scal` must supply every name in scal_names(); missing names throw
  /// before the C call.
  void accumulate(
      std::int64_t npts, std::int64_t nbf, const double *chi,
      const double *dchi, const double *lapl_chi, const double *hess_chi,
      const std::unordered_map<std::string, const double *> &scal,
      double *out) const;

  /// Fresh (nbf*nbf) row-major matrix from accumulate on a zero buffer.
  [[nodiscard]] std::vector<double> contract(
      std::int64_t npts, std::int64_t nbf, const double *chi,
      const double *dchi, const double *lapl_chi, const double *hess_chi,
      const std::unordered_map<std::string, const double *> &scal) const;

  /// Compiled first-slice o1/o2 C kernel names (GIAO / o3 / o4 excluded).
  [[nodiscard]] static std::vector<std::string> first_slice_names();

  /// True when -Dwith_xckernel=true compiled the first-slice C ABI in.
  [[nodiscard]] static bool available() noexcept;

private:
  std::string name_;
  std::vector<std::string> scal_names_;
  KernelFn fn_ = nullptr;
};

} // namespace rgpot
