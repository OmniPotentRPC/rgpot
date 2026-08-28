// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/XcKernel/XcKernel.hpp"

#include <sstream>

#ifdef RGPOT_HAS_XCKERNEL
#include "xckernel.h"
#endif

namespace rgpot {
namespace {

#ifdef RGPOT_HAS_XCKERNEL
struct KernelEntry {
  const char *name;
  XcKernel::KernelFn fn;
  const char **scal_names;
  const int *n_scal;
};

#define XCK_ENTRY(sym)                                                         \
  {#sym, &sym, sym##_scal_names, &sym##_n_scal}

// First-slice C ABI: lda/gga/mgga_tau, r/ua/ub o1+o2 and st o2 p/m.
// Energy helpers (o0) and GIAO are not dispatched here.
const KernelEntry kTable[] = {
    XCK_ENTRY(xck_lda_r_o1),
    XCK_ENTRY(xck_lda_r_o2),
    XCK_ENTRY(xck_lda_ua_o1),
    XCK_ENTRY(xck_lda_ua_o2),
    XCK_ENTRY(xck_lda_ub_o1),
    XCK_ENTRY(xck_lda_ub_o2),
    XCK_ENTRY(xck_lda_st_o2_p),
    XCK_ENTRY(xck_lda_st_o2_m),
    XCK_ENTRY(xck_gga_r_o1),
    XCK_ENTRY(xck_gga_r_o2),
    XCK_ENTRY(xck_gga_ua_o1),
    XCK_ENTRY(xck_gga_ua_o2),
    XCK_ENTRY(xck_gga_ub_o1),
    XCK_ENTRY(xck_gga_ub_o2),
    XCK_ENTRY(xck_gga_st_o2_p),
    XCK_ENTRY(xck_gga_st_o2_m),
    XCK_ENTRY(xck_mgga_tau_r_o1),
    XCK_ENTRY(xck_mgga_tau_r_o2),
    XCK_ENTRY(xck_mgga_tau_ua_o1),
    XCK_ENTRY(xck_mgga_tau_ua_o2),
    XCK_ENTRY(xck_mgga_tau_ub_o1),
    XCK_ENTRY(xck_mgga_tau_ub_o2),
    XCK_ENTRY(xck_mgga_tau_st_o2_p),
    XCK_ENTRY(xck_mgga_tau_st_o2_m),
};

#undef XCK_ENTRY

const KernelEntry *find_entry(const std::string &name) {
  for (const auto &e : kTable) {
    if (name == e.name) {
      return &e;
    }
  }
  return nullptr;
}
#endif

} // namespace

XcKernel::XcKernel(std::string name) : name_(std::move(name)) {
#ifndef RGPOT_HAS_XCKERNEL
  throw XcKernelError(
      "XcKernel requires meson -Dwith_xckernel=true (libxckernel first slice)");
#else
  const KernelEntry *e = find_entry(name_);
  if (e == nullptr) {
    throw XcKernelError("unknown first-slice kernel '" + name_ + "'");
  }
  fn_ = e->fn;
  const int n = *e->n_scal;
  scal_names_.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    scal_names_.emplace_back(e->scal_names[i]);
  }
#endif
}

void XcKernel::accumulate(
    std::int64_t npts, std::int64_t nbf, const double *chi, const double *dchi,
    const double *lapl_chi, const double *hess_chi,
    const std::unordered_map<std::string, const double *> &scal,
    double *out) const {
#ifndef RGPOT_HAS_XCKERNEL
  (void)npts;
  (void)nbf;
  (void)chi;
  (void)dchi;
  (void)lapl_chi;
  (void)hess_chi;
  (void)scal;
  (void)out;
  throw XcKernelError(
      "XcKernel requires meson -Dwith_xckernel=true (libxckernel first slice)");
#else
  if (fn_ == nullptr || out == nullptr || chi == nullptr || dchi == nullptr) {
    throw XcKernelError(name_ + ": null pointer");
  }
  if (npts <= 0 || nbf <= 0) {
    throw XcKernelError(name_ + ": npts and nbf must be positive");
  }
  std::vector<const double *> missing_probe;
  missing_probe.reserve(scal_names_.size());
  std::vector<std::string> missing;
  for (const auto &key : scal_names_) {
    auto it = scal.find(key);
    if (it == scal.end() || it->second == nullptr) {
      missing.push_back(key);
    } else {
      missing_probe.push_back(it->second);
    }
  }
  if (!missing.empty()) {
    std::ostringstream oss;
    oss << name_ << ": missing operands [";
    for (std::size_t i = 0; i < missing.size(); ++i) {
      if (i) {
        oss << ", ";
      }
      oss << missing[i];
    }
    oss << "]";
    throw XcKernelError(oss.str());
  }
  const int rc =
      fn_(npts, nbf, chi, dchi, lapl_chi, hess_chi, missing_probe.data(), out);
  if (rc != 0) {
    throw XcKernelError(name_ + " returned " + std::to_string(rc));
  }
#endif
}

std::vector<double> XcKernel::contract(
    std::int64_t npts, std::int64_t nbf, const double *chi, const double *dchi,
    const double *lapl_chi, const double *hess_chi,
    const std::unordered_map<std::string, const double *> &scal) const {
  std::vector<double> out(static_cast<std::size_t>(nbf * nbf), 0.0);
  accumulate(npts, nbf, chi, dchi, lapl_chi, hess_chi, scal, out.data());
  return out;
}

std::vector<std::string> XcKernel::first_slice_names() {
#ifdef RGPOT_HAS_XCKERNEL
  std::vector<std::string> names;
  names.reserve(sizeof(kTable) / sizeof(kTable[0]));
  for (const auto &e : kTable) {
    names.emplace_back(e.name);
  }
  return names;
#else
  return {};
#endif
}

bool XcKernel::available() noexcept {
#ifdef RGPOT_HAS_XCKERNEL
  return true;
#else
  return false;
#endif
}

} // namespace rgpot
