// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/XcKernel/XcKernel.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef RGPOT_HAS_XCKERNEL
#include "xckernel.h"

#include <algorithm>
#include <cstring>

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
// out = dm.T @ ao, i.e. PySCF _dot_ao_dm(ao, dm) with ao (npts, nbf) = chi.T.
void gemm_dmt_ao(const double *dm, const double *ao, std::size_t nbf,
                 std::size_t npts, double *out) {
  for (std::size_t u = 0; u < nbf; ++u) {
    for (std::size_t g = 0; g < npts; ++g) {
      double acc = 0.0;
      for (std::size_t v = 0; v < nbf; ++v) {
        acc += dm[v * nbf + u] * ao[v * npts + g];
      }
      out[u * npts + g] = acc;
    }
  }
}

// out = dm @ ao. PySCF hermi=0 second gradient term is ao @ dm.T.
void gemm_dm_ao(const double *dm, const double *ao, std::size_t nbf,
                std::size_t npts, double *out) {
  for (std::size_t u = 0; u < nbf; ++u) {
    for (std::size_t g = 0; g < npts; ++g) {
      double acc = 0.0;
      for (std::size_t v = 0; v < nbf; ++v) {
        acc += dm[u * nbf + v] * ao[v * npts + g];
      }
      out[u * npts + g] = acc;
    }
  }
}

void contract_rho(const double *left, const double *right, std::size_t nbf,
                  std::size_t npts, double *rho) {
  for (std::size_t g = 0; g < npts; ++g) {
    double acc = 0.0;
    for (std::size_t u = 0; u < nbf; ++u) {
      acc += left[u * npts + g] * right[u * npts + g];
    }
    rho[g] = acc;
  }
}

void add_contract_rho(const double *left, const double *right, std::size_t nbf,
                      std::size_t npts, double *rho) {
  for (std::size_t g = 0; g < npts; ++g) {
    double acc = 0.0;
    for (std::size_t u = 0; u < nbf; ++u) {
      acc += left[u * npts + g] * right[u * npts + g];
    }
    rho[g] += acc;
  }
}

// PySCF dft.gen_grid.BLKSIZE. nr_rks_fxc / _dot_ao_ao tile the grid
// this wide; a single fused sum_g U(u,g) c(g) V(v,g) over all points
// is 1-8 ulp farther from gen_vind on the sto-3g TDA/RPA pins.
constexpr std::size_t kPyscfBlk = 128;

void blocked_stage_b(std::int64_t npts, std::int64_t nbf, const double *U,
                     const double *c, const double *V, double *out) {
  const auto ng = static_cast<std::size_t>(npts);
  const auto nb = static_cast<std::size_t>(nbf);
  std::vector<double> aow(nb * kPyscfBlk, 0.0);
  for (std::size_t g0 = 0; g0 < ng; g0 += kPyscfBlk) {
    const std::size_t nblk = std::min(kPyscfBlk, ng - g0);
    for (std::size_t u = 0; u < nb; ++u) {
      const double *Ug = U + u * ng + g0;
      double *aw = aow.data() + u * nblk;
      for (std::size_t t = 0; t < nblk; ++t) {
        aw[t] = Ug[t] * c[g0 + t];
      }
    }
    for (std::size_t u = 0; u < nb; ++u) {
      for (std::size_t v = 0; v < nb; ++v) {
        double s = 0.0;
        const double *aw = aow.data() + u * nblk;
        const double *Vg = V + v * ng + g0;
        for (std::size_t t = 0; t < nblk; ++t) {
          s += aw[t] * Vg[t];
        }
        out[u * nb + v] += s;
      }
    }
  }
}

int fill_scal_ptrs(const XcKernel &k,
                   const std::map<std::string, const double *> &scal,
                   std::vector<const double *> *ptrs) {
  const auto names = k.scalNames();
  ptrs->assign(static_cast<std::size_t>(k.nScal()), nullptr);
  for (int i = 0; i < k.nScal(); ++i) {
    auto it = scal.find(names[static_cast<std::size_t>(i)]);
    if (it == scal.end() || it->second == nullptr) {
      return 3;
    }
    (*ptrs)[static_cast<std::size_t>(i)] = it->second;
  }
  return 0;
}

int contract_st_blocked(const XcKernel &k, const XcGrid &grid,
                        const std::map<std::string, const double *> &scal,
                        double *out) {
  std::vector<const double *> ptrs;
  const int rc = fill_scal_ptrs(k, scal, &ptrs);
  if (rc != 0) {
    return rc;
  }
  const auto npts = grid.npts;
  const auto nbf = grid.nbf;
  const auto n2 = static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  std::memset(out, 0, n2 * sizeof(double));
  std::vector<double> c(static_cast<std::size_t>(npts), 0.0);
  const double *const *fields = ptrs.data();
  const double *const *xc = ptrs.data() + k.nFields();
  const double *chi = grid.chi;
  if (k.name() == "xck_lda_st_o2_p") {
    // PySCF nr_rks_fxc_st wv is w * rho * (v2rho2_0 + v2rho2_1).
    // Two monomials (w*rho*v20 + w*rho*v21) land 1 ulp off gen_vind.
    const double *w = fields[0];
    const double *rho = fields[1];
    const double *v20 = xc[0];
    const double *v21 = xc[1];
    for (std::int64_t g = 0; g < npts; ++g) {
      c[static_cast<std::size_t>(g)] = w[g] * rho[g] * (v20[g] + v21[g]);
    }
    blocked_stage_b(npts, nbf, chi, c.data(), chi, out);
    return 0;
  }
  return k.contract(grid, scal, out);
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
  // PySCF eval_rho hermi=0: c0 = ao @ dm = (dm.T @ chi).T, rho = contract(ao, c0).
  std::vector<double> c0(nbf * ng, 0.0);
  gemm_dmt_ao(dm, chi, nbf, ng, c0.data());
  std::vector<double> rho_p1(ng, 0.0);
  contract_rho(chi, c0.data(), nbf, ng, rho_p1.data());

  std::vector<double> gxd(ng, 0.0);
  std::vector<double> gyd(ng, 0.0);
  std::vector<double> gzd(ng, 0.0);
  if (dchi != nullptr) {
    std::vector<double> c1(nbf * ng, 0.0);
    gemm_dm_ao(dm, chi, nbf, ng, c1.data());
    const double *dx = dchi;
    const double *dy = dchi + nbf * ng;
    const double *dz = dchi + 2 * nbf * ng;
    contract_rho(c0.data(), dx, nbf, ng, gxd.data());
    contract_rho(c0.data(), dy, nbf, ng, gyd.data());
    contract_rho(c0.data(), dz, nbf, ng, gzd.data());
    add_contract_rho(c1.data(), dx, nbf, ng, gxd.data());
    add_contract_rho(c1.data(), dy, nbf, ng, gyd.data());
    add_contract_rho(c1.data(), dz, nbf, ng, gzd.data());
  }

  std::map<std::string, const double *> scal = ground;
  for (const auto &name : k.scalNames()) {
    if (name == "rho_a_p1" || name == "rho_p1") {
      scal[name] = rho_p1.data();
    } else if (name == "grad_rho_a_p1_x" || name == "grad_rho_p1_x") {
      scal[name] = gxd.data();
    } else if (name == "grad_rho_a_p1_y" || name == "grad_rho_p1_y") {
      scal[name] = gyd.data();
    } else if (name == "grad_rho_a_p1_z" || name == "grad_rho_p1_z") {
      scal[name] = gzd.data();
    }
  }
  return contract_st_blocked(k, grid, scal, vxc);
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
  // lib.einsum path: 'qo,ov->vq' then 'vq,pv->pq' (TDA.gen_vind).
  std::vector<double> tmp(nvir * nao, 0.0);
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t i = 0; i < nocc; ++i) {
        acc += (occ * mo.Co[q * nocc + i]) * z[i * nvir + a];
      }
      tmp[a * nao + q] = acc;
    }
  }
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t a = 0; a < nvir; ++a) {
        acc += tmp[a * nao + q] * mo.Cv[p * nvir + a];
      }
      dm[p * nao + q] = acc;
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
  // lib.einsum path: 'po,ov->vp' then 'vp,qv->pq' (gen_tdhf_operation Y).
  std::vector<double> tmp(nvir * nao, 0.0);
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t p = 0; p < nao; ++p) {
      double acc = 0.0;
      for (std::size_t i = 0; i < nocc; ++i) {
        acc += (occ * mo.Co[p * nocc + i]) * y[i * nvir + a];
      }
      tmp[a * nao + p] = acc;
    }
  }
  for (std::size_t p = 0; p < nao; ++p) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t a = 0; a < nvir; ++a) {
        acc += tmp[a * nao + p] * mo.Cv[q * nvir + a];
      }
      dm[p * nao + q] += acc;
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
  // lib.einsum path: 'pv,pq->vq' then 'vq,qo->ov' (TDA.gen_vind).
  std::vector<double> tmp(nvir * nao, 0.0);
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t q = 0; q < nao; ++q) {
      double acc = 0.0;
      for (std::size_t p = 0; p < nao; ++p) {
        acc += mo.Cv[p * nvir + a] * Vao[p * nao + q];
      }
      tmp[a * nao + q] = acc;
    }
  }
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t q = 0; q < nao; ++q) {
        acc += tmp[a * nao + q] * mo.Co[q * nocc + i];
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
      sigma[ia] = sigma[ia] + mo.e_ia[ia] * z[ia];
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
  // lib.einsum path: 'qv,pq->vp' then 'vp,po->ov' (gen_tdhf_operation bot).
  std::vector<double> tmp(nvir * nao, 0.0);
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t p = 0; p < nao; ++p) {
      double acc = 0.0;
      for (std::size_t q = 0; q < nao; ++q) {
        acc += mo.Cv[q * nvir + a] * v1[p * nao + q];
      }
      tmp[a * nao + p] = acc;
    }
  }
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvir; ++a) {
      double acc = 0.0;
      for (std::size_t p = 0; p < nao; ++p) {
        acc += tmp[a * nao + p] * mo.Co[p * nocc + i];
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
