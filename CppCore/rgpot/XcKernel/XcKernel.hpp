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
 * (Fock o1 + fxc o2). TDA/RPA sigma assembly uses the singlet
 * spin-adapted o2 kernels (`xck_*_st_o2_p`) plus host Coulomb.
 * LDA TDA/RPA forms wv as w*rho*(v2rho2_0+v2rho2_1) and tiles
 * stage B at PySCF BLKSIZE=128. GGA stays on the generated C ABI.
 * Dispatch is by kernel name; scalar operand order is read from
 * <name>_scal_names / <name>_n_scal, not hard-coded.
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

/// Occupied/virtual MO blocks for TDA/RPA sigma.
/// Co is (nao, nocc), Cv is (nao, nvir), e_ia is (nocc, nvir) with
/// e_ia[i,a] = e_vir[a] - e_occ[i]. All row-major.
struct XcMo {
  std::int64_t nao = 0;
  std::int64_t nocc = 0;
  std::int64_t nvir = 0;
  const double *Co = nullptr;
  const double *Cv = nullptr;
  const double *e_ia = nullptr;
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

  /// Convenience: build rho / sigma / tau / lapl / grad_rho from an
  /// AO density (symmetric or transition). Host still evaluates Libxc.
  [[nodiscard]] static XcFields fieldsFromDensity(const XcGrid &grid,
                                                  const double *P);

  /// dm = einsum('qo,ov,pv->pq', Co*occ, z, Cv). Matches lib.einsum path
  /// ('qo,xov->vxq' then 'vxq,pv->xpq') used by TDA.gen_vind.
  static void transitionDm(const XcMo &mo, const double *z, double occ,
                           double *dm);

  /// dm = einsum X plus einsum('po,ov,qv->pq', Co*occ, y, Cv)
  /// ('po,xov->vxp' then 'vxp,qv->xpq').
  static void rpaTransitionDm(const XcMo &mo, const double *x, const double *y,
                              double occ, double *dm);

  /// ov = einsum('pv,pq,qo->ov', Cv, V, Co) == Co^T @ (V^T @ Cv).
  /// Matches lib.einsum path ('pv,xpq->vxq' then 'vxq,qo->xov').
  static void projectOv(const XcMo &mo, const double *Vao, double *ov);

  /// (A z)_ia = e_ia z_ia + (Co^T @ v1^T @ Cv)_ia. v1 is AO (nao*nao).
  static void tdaSigma(const XcMo &mo, const double *z, const double *v1,
                       double *sigma);

  /// [[A,B],[-B,-A]](X,Y). xy and out are 2*nocc*nvir (X then Y).
  static void rpaSigma(const XcMo &mo, const double *xy, const double *v1,
                       double *sigma);

  /// XC fxc on one AO density via this kernel. Ground-state scal holds
  /// weights, Libxc arrays, and (GGA) grad_rho_a_*. Perturbed rho/grad
  /// names (`rho_a_p1`, `grad_rho_a_p1_*`) are built from dm.
  /// Accumulates into vxc (nao*nao, +=).
  int applyFxc(const XcGrid &grid,
               const std::map<std::string, const double *> &ground,
               const double *dm, double *vxc) const;

  /// TDA sigma with host Coulomb: v1 = vj + 0.5 * applyFxc(dm(z)).
  /// vj is the host J matrix on the transition DM (nao*nao).
  int tdaSigma(const XcGrid &grid,
               const std::map<std::string, const double *> &ground,
               const XcMo &mo, const double *z, const double *vj,
               double *sigma) const;

  /// RPA sigma with host Coulomb on dm(X,Y). xy/out are 2*nocc*nvir.
  int rpaSigma(const XcGrid &grid,
               const std::map<std::string, const double *> &ground,
               const XcMo &mo, const double *xy, const double *vj,
               double *sigma) const;

private:
  std::string m_name;
  KernelFn m_fn = nullptr;
  const char **m_scal_names = nullptr;
  int m_n_scal = 0;
  int m_n_fields = 0;
};

} // namespace rgpot
