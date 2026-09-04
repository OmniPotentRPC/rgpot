// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/XcKernel/XcKernel.hpp"

#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef RGPOT_HAS_XCKERNEL
#include "xckernel.h"

namespace {

struct KernelAbi {
  rgpot::XcKernel::KernelFn fn;
  const char **scal_names;
  const int *n_scal;
  const int *n_fields;
};

#define XCK_KERNEL(name)                                                       \
  {                                                                            \
#name, {                                                                   \
      name, name##_scal_names, &name##_n_scal, &name##_n_fields                \
    }                                                                          \
  }                                                                            \
  ,

const std::unordered_map<std::string, KernelAbi> kTable = {
#include "rgpot/XcKernel/kernel_table.inc"
};

#undef XCK_KERNEL

} // namespace
#endif

namespace rgpot {

XcKernel::XcKernel(std::string name) : m_name(std::move(name)) {
#ifdef RGPOT_HAS_XCKERNEL
  auto it = kTable.find(m_name);
  if (it == kTable.end()) {
    throw std::invalid_argument("unknown first-slice XcKernel: " + m_name);
  }
  m_fn = it->second.fn;
  m_scal_names = it->second.scal_names;
  m_n_scal = *it->second.n_scal;
  m_n_fields = *it->second.n_fields;
#else
  throw std::runtime_error(
      "XcKernel requires meson -Dwith_xckernel=true (generated C ABI)");
#endif
}

std::vector<std::string> XcKernel::scalNames() const {
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(m_n_scal));
  for (int i = 0; i < m_n_scal; ++i) {
    out.emplace_back(m_scal_names[i]);
  }
  return out;
}

std::vector<std::string> XcKernel::catalog() {
  std::vector<std::string> names;
#ifdef RGPOT_HAS_XCKERNEL
#define XCK_KERNEL(name) names.emplace_back(#name);
#include "rgpot/XcKernel/kernel_table.inc"
#undef XCK_KERNEL
#endif
  return names;
}

int XcKernel::contract(const XcGrid &grid,
                       const std::map<std::string, const double *> &scal,
                       double *out) const {
  if (m_fn == nullptr || out == nullptr) {
    return 1;
  }
  if (grid.npts <= 0 || grid.nbf <= 0 || grid.chi == nullptr ||
      grid.dchi == nullptr) {
    return 2;
  }
  std::vector<const double *> ptrs(static_cast<std::size_t>(m_n_scal), nullptr);
  for (int i = 0; i < m_n_scal; ++i) {
    auto it = scal.find(m_scal_names[i]);
    if (it == scal.end() || it->second == nullptr) {
      return 3;
    }
    ptrs[static_cast<std::size_t>(i)] = it->second;
  }
  return m_fn(grid.npts, grid.nbf, grid.chi, grid.dchi, grid.lapl_chi,
              grid.hess_chi, ptrs.data(), out);
}

XcFields XcKernel::fieldsFromDensity(const XcGrid &grid, const double *P) {
  XcFields f;
  const auto ng = static_cast<std::size_t>(grid.npts);
  const auto nbf = static_cast<std::size_t>(grid.nbf);
  f.rho.assign(ng, 0.0);
  f.sigma.assign(ng, 0.0);
  f.tau.assign(ng, 0.0);
  f.lapl.assign(ng, 0.0);
  f.grad_rho.assign(3 * ng, 0.0);
  if (P == nullptr || grid.chi == nullptr || grid.dchi == nullptr) {
    return f;
  }
  const double *chi = grid.chi;
  const double *dchi = grid.dchi;
  const double *lapl_chi = grid.lapl_chi;
  for (std::size_t u = 0; u < nbf; ++u) {
    for (std::size_t v = 0; v < nbf; ++v) {
      const double Puv = P[u * nbf + v];
      const double *chi_u = chi + u * ng;
      const double *chi_v = chi + v * ng;
      const double *dx_u = dchi + (0 * nbf + u) * ng;
      const double *dy_u = dchi + (1 * nbf + u) * ng;
      const double *dz_u = dchi + (2 * nbf + u) * ng;
      const double *dx_v = dchi + (0 * nbf + v) * ng;
      const double *dy_v = dchi + (1 * nbf + v) * ng;
      const double *dz_v = dchi + (2 * nbf + v) * ng;
      for (std::size_t g = 0; g < ng; ++g) {
        f.rho[g] += Puv * chi_u[g] * chi_v[g];
        f.grad_rho[g] += Puv * (dx_u[g] * chi_v[g] + chi_u[g] * dx_v[g]);
        f.grad_rho[ng + g] += Puv * (dy_u[g] * chi_v[g] + chi_u[g] * dy_v[g]);
        f.grad_rho[2 * ng + g] +=
            Puv * (dz_u[g] * chi_v[g] + chi_u[g] * dz_v[g]);
        f.tau[g] +=
            0.5 * Puv *
            (dx_u[g] * dx_v[g] + dy_u[g] * dy_v[g] + dz_u[g] * dz_v[g]);
        if (lapl_chi != nullptr) {
          f.lapl[g] += Puv * (lapl_chi[u * ng + g] * chi_v[g] +
                              2.0 * (dx_u[g] * dx_v[g] + dy_u[g] * dy_v[g] +
                                     dz_u[g] * dz_v[g]) +
                              chi_u[g] * lapl_chi[v * ng + g]);
        }
      }
    }
  }
  for (std::size_t g = 0; g < ng; ++g) {
    const double gx = f.grad_rho[g];
    const double gy = f.grad_rho[ng + g];
    const double gz = f.grad_rho[2 * ng + g];
    f.sigma[g] = gx * gx + gy * gy + gz * gz;
  }
  return f;
}

namespace {

#ifdef RGPOT_HAS_XCKERNEL
// Perturbed fields in the PySCF eval_rho (hermi=0) parenthesization:
// c0 = ao @ dm, rho = einsum('gi,gi->g', ao, c0). GGA grads add the
// matching dchi@dm term. Stage B for st_o2_p is the generated double
// loop (see meson xckernel_st). GGA vxc is hermitized like nr_rks_fxc.
void ao_dot_dm(const double *ao, const double *dm, std::size_t nbf,
               std::size_t ng, double *out) {
  for (std::size_t v = 0; v < nbf; ++v) {
    for (std::size_t g = 0; g < ng; ++g) {
      double acc = 0.0;
      for (std::size_t u = 0; u < nbf; ++u) {
        acc += ao[u * ng + g] * dm[u * nbf + v];
      }
      out[v * ng + g] = acc;
    }
  }
}

void contract_rho(const double *ao, const double *c0, std::size_t nbf,
                  std::size_t ng, double *rho) {
  for (std::size_t v = 0; v < nbf; ++v) {
    const double *ao_v = ao + v * ng;
    const double *c0_v = c0 + v * ng;
    for (std::size_t g = 0; g < ng; ++g) {
      rho[g] += ao_v[g] * c0_v[g];
    }
  }
}

void hermitize(double *m, std::size_t n) {
  for (std::size_t u = 0; u < n; ++u) {
    for (std::size_t v = u + 1; v < n; ++v) {
      const double s = 0.5 * (m[u * n + v] + m[v * n + u]);
      m[u * n + v] = s;
      m[v * n + u] = s;
    }
  }
}

int apply_fxc_d(const XcKernel &k, const XcGrid &grid,
                const std::map<std::string, const double *> &ground,
                const double *dm, double *vxc) {
  if (dm == nullptr || vxc == nullptr) {
    return 4;
  }
  const auto ng = static_cast<std::size_t>(grid.npts);
  const auto nbf = static_cast<std::size_t>(grid.nbf);
  if (grid.chi == nullptr || grid.npts <= 0 || grid.nbf <= 0) {
    return 2;
  }

  const double *chi = grid.chi;
  const double *dchi = grid.dchi;
  std::vector<double> c0(nbf * ng, 0.0);
  ao_dot_dm(chi, dm, nbf, ng, c0.data());
  std::vector<double> rho_p1(ng, 0.0);
  contract_rho(chi, c0.data(), nbf, ng, rho_p1.data());

  std::vector<double> gxd(ng, 0.0);
  std::vector<double> gyd(ng, 0.0);
  std::vector<double> gzd(ng, 0.0);
  if (dchi != nullptr) {
    contract_rho(dchi + 0 * nbf * ng, c0.data(), nbf, ng, gxd.data());
    contract_rho(dchi + 1 * nbf * ng, c0.data(), nbf, ng, gyd.data());
    contract_rho(dchi + 2 * nbf * ng, c0.data(), nbf, ng, gzd.data());
    std::vector<double> c1(nbf * ng, 0.0);
    ao_dot_dm(dchi + 0 * nbf * ng, dm, nbf, ng, c1.data());
    contract_rho(chi, c1.data(), nbf, ng, gxd.data());
    ao_dot_dm(dchi + 1 * nbf * ng, dm, nbf, ng, c1.data());
    contract_rho(chi, c1.data(), nbf, ng, gyd.data());
    ao_dot_dm(dchi + 2 * nbf * ng, dm, nbf, ng, c1.data());
    contract_rho(chi, c1.data(), nbf, ng, gzd.data());
  }

  std::map<std::string, const double *> scal = ground;
  bool gga = false;
  for (const auto &name : k.scalNames()) {
    if (name == "rho_a_p1" || name == "rho_p1") {
      scal[name] = rho_p1.data();
    } else if (name == "grad_rho_a_p1_x" || name == "grad_rho_p1_x") {
      scal[name] = gxd.data();
      gga = true;
    } else if (name == "grad_rho_a_p1_y" || name == "grad_rho_p1_y") {
      scal[name] = gyd.data();
      gga = true;
    } else if (name == "grad_rho_a_p1_z" || name == "grad_rho_p1_z") {
      scal[name] = gzd.data();
      gga = true;
    }
  }
  const int rc = k.contract(grid, scal, vxc);
  if (rc == 0 && gga) {
    hermitize(vxc, nbf);
  }
  return rc;
}
#endif

} // namespace

void XcKernel::transitionDm(const XcMo &mo, const double *z, double occ,
                            double *dm) {
  const auto nao = static_cast<std::size_t>(mo.nao);
  const auto nocc = static_cast<std::size_t>(mo.nocc);
  const auto nvir = static_cast<std::size_t>(mo.nvir);
  const std::size_t n2 = nao * nao;
  if (dm == nullptr) {
    return;
  }
  for (std::size_t k = 0; k < n2; ++k) {
    dm[k] = 0.0;
  }
  if (z == nullptr || mo.Co == nullptr || mo.Cv == nullptr) {
    return;
  }
  std::vector<double> tmp(nao * nocc, 0.0);
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t i = 0; i < nocc; ++i) {
      double acc = 0.0;
      for (std::size_t a = 0; a < nvir; ++a) {
        acc += mo.Cv[p * nvir + a] * z[i * nvir + a];
      }
      tmp[p * nocc + i] = acc;
    }
  }
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t i = 0; i < nocc; ++i) {
        acc += tmp[p * nocc + i] * mo.Co[q * nocc + i];
      }
      dm[p * nao + q] = occ * acc;
    }
  }
}

void XcKernel::rpaTransitionDm(const XcMo &mo, const double *x, const double *y,
                               double occ, double *dm) {
  transitionDm(mo, x, occ, dm);
  const auto nao = static_cast<std::size_t>(mo.nao);
  const auto nocc = static_cast<std::size_t>(mo.nocc);
  const auto nvir = static_cast<std::size_t>(mo.nvir);
  if (y == nullptr || mo.Co == nullptr || mo.Cv == nullptr || dm == nullptr) {
    return;
  }
  std::vector<double> tmp(nao * nvir, 0.0);
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t i = 0; i < nocc; ++i) {
        acc += mo.Co[p * nocc + i] * y[i * nvir + a];
      }
      tmp[p * nvir + a] = acc;
    }
  }
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t a = 0; a < nvir; ++a) {
        acc += tmp[p * nvir + a] * mo.Cv[q * nvir + a];
      }
      dm[p * nao + q] += occ * acc;
    }
  }
}

void XcKernel::projectOv(const XcMo &mo, const double *Vao, double *ov) {
  const auto nao = static_cast<std::size_t>(mo.nao);
  const auto nocc = static_cast<std::size_t>(mo.nocc);
  const auto nvir = static_cast<std::size_t>(mo.nvir);
  const std::size_t nov = nocc * nvir;
  if (ov == nullptr) {
    return;
  }
  for (std::size_t k = 0; k < nov; ++k) {
    ov[k] = 0.0;
  }
  if (Vao == nullptr || mo.Co == nullptr || mo.Cv == nullptr) {
    return;
  }
  // einsum('pq,qo,pv->ov', V, Co, Cv): tmp = V @ Co, ov = tmp^T @ Cv.
  std::vector<double> tmp(nao * nocc, 0.0);
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t i = 0; i < nocc; ++i) {
      double acc = 0.0;
      for (std::size_t q = 0; q < nao; ++q) {
        acc += Vao[p * nao + q] * mo.Co[q * nocc + i];
      }
      tmp[p * nocc + i] = acc;
    }
  }
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t p = 0; p < nao; ++p) {
        acc += tmp[p * nocc + i] * mo.Cv[p * nvir + a];
      }
      ov[i * nvir + a] = acc;
    }
  }
}

void XcKernel::tdaSigma(const XcMo &mo, const double *z, const double *v1,
                        double *sigma) {
  projectOv(mo, v1, sigma);
  const auto nocc = static_cast<std::size_t>(mo.nocc);
  const auto nvir = static_cast<std::size_t>(mo.nvir);
  if (z == nullptr || mo.e_ia == nullptr || sigma == nullptr) {
    return;
  }
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvir; ++a) {
      const std::size_t ia = i * nvir + a;
      sigma[ia] += mo.e_ia[ia] * z[ia];
    }
  }
}

void XcKernel::rpaSigma(const XcMo &mo, const double *xy, const double *v1,
                        double *sigma) {
  const auto nao = static_cast<std::size_t>(mo.nao);
  const auto nocc = static_cast<std::size_t>(mo.nocc);
  const auto nvir = static_cast<std::size_t>(mo.nvir);
  const std::size_t nov = nocc * nvir;
  tdaSigma(mo, xy, v1, sigma);
  if (xy == nullptr || v1 == nullptr || sigma == nullptr ||
      mo.Co == nullptr || mo.Cv == nullptr || mo.e_ia == nullptr) {
    return;
  }
  const double *y = xy + nov;
  double *bot = sigma + nov;
  std::vector<double> tmp(nao * nvir, 0.0);
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t q = 0; q < nao; ++q) {
        acc += v1[p * nao + q] * mo.Cv[q * nvir + a];
      }
      tmp[p * nvir + a] = acc;
    }
  }
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t p = 0; p < nao; ++p) {
        acc += mo.Co[p * nocc + i] * tmp[p * nvir + a];
      }
      const std::size_t ia = i * nvir + a;
      bot[ia] = -(mo.e_ia[ia] * y[ia] + acc);
    }
  }
}

int XcKernel::applyFxc(const XcGrid &grid,
                       const std::map<std::string, const double *> &ground,
                       const double *dm, double *vxc) const {
#ifdef RGPOT_HAS_XCKERNEL
  return apply_fxc_d(*this, grid, ground, dm, vxc);
#else
  (void)grid;
  (void)ground;
  (void)dm;
  (void)vxc;
  return 1;
#endif
}

int XcKernel::tdaSigma(const XcGrid &grid,
                       const std::map<std::string, const double *> &ground,
                       const XcMo &mo, const double *z, const double *vj,
                       double *sigma) const {
  if (z == nullptr || vj == nullptr || sigma == nullptr || mo.nao <= 0) {
    return 4;
  }
#ifdef RGPOT_HAS_XCKERNEL
  const auto nao = static_cast<std::size_t>(mo.nao);
  std::vector<double> dm(nao * nao, 0.0);
  std::vector<double> vxc(nao * nao, 0.0);
  std::vector<double> v1(nao * nao, 0.0);
  transitionDm(mo, z, 2.0, dm.data());
  const int rc = apply_fxc_d(*this, grid, ground, dm.data(), vxc.data());
  if (rc != 0) {
    return rc;
  }
  for (std::size_t k = 0; k < nao * nao; ++k) {
    v1[k] = vj[k] + 0.5 * vxc[k];
  }
  tdaSigma(mo, z, v1.data(), sigma);
  return 0;
#else
  (void)grid;
  (void)ground;
  return 1;
#endif
}

int XcKernel::rpaSigma(const XcGrid &grid,
                       const std::map<std::string, const double *> &ground,
                       const XcMo &mo, const double *xy, const double *vj,
                       double *sigma) const {
  if (xy == nullptr || vj == nullptr || sigma == nullptr || mo.nao <= 0) {
    return 4;
  }
#ifdef RGPOT_HAS_XCKERNEL
  const auto nao = static_cast<std::size_t>(mo.nao);
  const auto nov =
      static_cast<std::size_t>(mo.nocc) * static_cast<std::size_t>(mo.nvir);
  std::vector<double> dm(nao * nao, 0.0);
  std::vector<double> vxc(nao * nao, 0.0);
  std::vector<double> v1(nao * nao, 0.0);
  rpaTransitionDm(mo, xy, xy + nov, 2.0, dm.data());
  const int rc = apply_fxc_d(*this, grid, ground, dm.data(), vxc.data());
  if (rc != 0) {
    return rc;
  }
  for (std::size_t k = 0; k < nao * nao; ++k) {
    v1[k] = vj[k] + 0.5 * vxc[k];
  }
  rpaSigma(mo, xy, v1.data(), sigma);
  return 0;
#else
  (void)grid;
  (void)ground;
  return 1;
#endif
}

} // namespace rgpot
