#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace rgpot {

/**
 * In-process XC kernel contraction (libxckernel C ABI).
 *
 * This is not a geometry PES. Do not subclass Potential. Do not add a
 * PotentialConfig.xckernel arm (rgpot-qf6b, rgpot-nrve): operands are
 * collocation + named Libxc derivative arrays + (perturbed) fields, not
 * ForceInput; results are AO matrices, not PotentialResult energy/forces.
 * No DFT host sends chi / dchi / weights over potserv in this slice.
 *
 * Term ownership is XC-only (rgpot-nkyx). Coulomb, Hartree-Fock exact
 * exchange, and range-separated exchange stay host-owned. libxckernel never
 * evaluates functionals: the host mixes Libxc arrays and passes them in.
 *
 * First slice (rgpot-chjn): families lda, gga, mgga_tau; max_order 2
 * (Fock o1 + fxc o2). Dispatch is by kernel name; scalar operand order is
 * read from <name>_scal_names / <name>_n_scal, not hard-coded.
 *
 * Each XcKernel instance is a name + resolved ABI pointers. The C kernels
 * are reentrant on distinct out/scal buffers. Do not share one `out`
 * pointer across threads. Distinct instances may run concurrently.
 */
struct XcGrid {
  std::int64_t npts = 0;
  std::int64_t nbf = 0;
  const double *chi = nullptr;      //!< nbf * npts, row-major
  const double *dchi = nullptr;     //!< 3 * nbf * npts
  const double *lapl_chi = nullptr; //!< nbf * npts, or nullptr
  const double *hess_chi = nullptr; //!< 6 * nbf * npts, or nullptr
};

struct XcFields {
  std::vector<double> rho;      //!< (npts,)
  std::vector<double> sigma;    //!< (npts,)
  std::vector<double> tau;      //!< (npts,)
  std::vector<double> lapl;     //!< (npts,)
  std::vector<double> grad_rho; //!< 3 * npts, xyz packed
};

class XcKernel {
public:
  using KernelFn = int (*)(std::int64_t npts, std::int64_t nbf,
                           const double *chi, const double *dchi,
                           const double *lapl_chi, const double *hess_chi,
                           const double *const *scal, double *out);

  /// Resolve a first-slice catalog name, e.g. "xck_gga_r_o2".
  explicit XcKernel(std::string name);

  [[nodiscard]] const std::string &name() const noexcept { return m_name; }
  [[nodiscard]] std::vector<std::string> scalNames() const;
  [[nodiscard]] int nScal() const noexcept { return m_n_scal; }
  [[nodiscard]] int nFields() const noexcept { return m_n_fields; }

  /// Known first-slice contraction names (o1 Fock + o2 fxc). Empty if
  /// the translation unit was compiled without -Dwith_xckernel.
  [[nodiscard]] static std::vector<std::string> catalog();

  /// Assemble `scal` in <name>_scal_names order and accumulate into out
  /// (nbf * nbf, +=). Returns the C ABI rc (0 on success).
  int contract(const XcGrid &grid,
               const std::map<std::string, const double *> &scal,
               double *out) const;

  /// Convenience: build rho / sigma / tau / lapl / grad_rho from a
  /// symmetric AO density. Host still evaluates Libxc on these fields.
  [[nodiscard]] static XcFields fieldsFromDensity(const XcGrid &grid,
                                                  const double *P);

private:
  std::string m_name;
  KernelFn m_fn = nullptr;
  const char **m_scal_names = nullptr;
  int m_n_scal = 0;
  int m_n_fields = 0;
};

} // namespace rgpot
