// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "npy_io.hpp"
#include "scale_bound.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"

using rgpot::XcGrid;
using rgpot::XcKernel;
using rgpot::XcMo;
using rgpot::testing::deviation_in_eps;
using rgpot::testio::NpyArray;
using rgpot::testio::load_npy;
using rgpot::testio::load_npz;

namespace {

constexpr const char *kData = "CppCore/tests/data/xckernel";
// Bars. C vs NumPy is |C - ref| <= k * eps * max|ref| per array
// (scale_bound.hpp). The C contraction is a blocked dgemm/dsyr2k and the
// reference a dgemm of a different blocking, so the two differ by a few
// rounding errors of the largest term, never by a fixed decimal.
// Measured worst case on x86_64 over every fixture below: 2.2 eps*scale;
// k = 4 leaves room for another architecture's blocking and FMA use.
constexpr double kNumpyEps = 4.0;
// PySCF bars: Fock 1e-15, fxc 1e-13, TDA/RPA sigma 1e-17.
constexpr double kFockVsPyscf = 1e-15;
constexpr double kFxcVsPyscf = 1e-13;
constexpr double kTdaRpaVsPyscf = 1e-17;
// Measured libnwchemc TDA roots vs pin-operator TDA.kernel on this
// sto-3g case (7.44e-7 Ha). Not a loosened sigma-contraction bar.
constexpr double kTdaNwchemcVsPin = 1e-6;

void require_file(const std::string &path) {
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(std::filesystem::file_size(path) > 0);
}

std::string slurp(const std::string &path) {
  require_file(path);
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

double max_abs(const std::vector<double> &a, const std::vector<double> &b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

double max_rel(const std::vector<double> &got, const std::vector<double> &ref) {
  double num = max_abs(got, ref);
  double den = 0.0;
  for (double v : ref) {
    den = std::max(den, std::abs(v));
  }
  if (den < 1.0) {
    den = 1.0;
  }
  return num / den;
}

XcGrid grid_from_npz(const std::map<std::string, NpyArray> &op, std::int64_t nbf,
                     std::int64_t npts, std::vector<double> *dchi_c,
                     bool dchi_nbf_first) {
  XcGrid g;
  g.nbf = nbf;
  g.npts = npts;
  g.chi = op.at("chi").data.data();
  const auto &dchi = op.at("dchi").data;
  if (dchi_nbf_first) {
    // stored (nbf, 3, npts) -> C ABI (3, nbf, npts)
    dchi_c->assign(static_cast<std::size_t>(3 * nbf * npts), 0.0);
    for (std::int64_t u = 0; u < nbf; ++u) {
      for (int ax = 0; ax < 3; ++ax) {
        for (std::int64_t p = 0; p < npts; ++p) {
          (*dchi_c)[static_cast<std::size_t>((ax * nbf + u) * npts + p)] =
              dchi[static_cast<std::size_t>((u * 3 + ax) * npts + p)];
        }
      }
    }
    g.dchi = dchi_c->data();
  } else {
    g.dchi = dchi.data();
  }
  if (op.count("lapl_chi")) {
    g.lapl_chi = op.at("lapl_chi").data.data();
  }
  if (op.count("hess_chi")) {
    g.hess_chi = op.at("hess_chi").data.data();
  }
  return g;
}

std::map<std::string, const double *>
scal_from_npz(const XcKernel &k, const std::map<std::string, NpyArray> &op) {
  std::map<std::string, const double *> scal;
  for (const auto &name : k.scalNames()) {
    REQUIRE(op.count(name) == 1);
    scal[name] = op.at(name).data.data();
  }
  return scal;
}

std::map<std::string, const double *>
ground_from_npz(const XcKernel &k, const std::map<std::string, NpyArray> &op) {
  std::map<std::string, const double *> scal;
  for (const auto &name : k.scalNames()) {
    if (name.find("_p1") != std::string::npos) {
      continue;
    }
    REQUIRE(op.count(name) == 1);
    scal[name] = op.at(name).data.data();
  }
  return scal;
}

std::vector<double> run_kernel(const std::string &name,
                               const std::map<std::string, NpyArray> &op,
                               std::int64_t nbf, std::int64_t npts,
                               bool dchi_nbf_first) {
  XcKernel k(name);
  std::vector<double> dchi_c;
  XcGrid g = grid_from_npz(op, nbf, npts, &dchi_c, dchi_nbf_first);
  auto scal = scal_from_npz(k, op);
  std::vector<double> out(static_cast<std::size_t>(nbf * nbf), 0.0);
  REQUIRE(k.contract(g, scal, out.data()) == 0);
  return out;
}

} // namespace

TEST_CASE("XcKernel is not a Potential and has no PotType", "[xckernel][api]") {
  STATIC_REQUIRE(!std::is_base_of_v<rgpot::Potential<XcKernel>, XcKernel>);
  const std::string types = slurp("CppCore/rgpot/pot_types.hpp");
  REQUIRE(types.find("XcKernel") == std::string::npos);
  REQUIRE(types.find("XCKERNEL") == std::string::npos);
  auto names = XcKernel::catalog();
  REQUIRE(names.size() == 32);
  const std::string schema = slurp("CppCore/rgpot/rpc/Potentials.capnp");
  REQUIRE(schema.find("xckernel") == std::string::npos);
  REQUIRE(schema.find("XcKernel") == std::string::npos);
}

TEST_CASE("XcKernel sources do not mention D3 or D4", "[xckernel][api]") {
  const char *files[] = {
      "CppCore/rgpot/XcKernel/XcKernel.hpp",
      "CppCore/rgpot/XcKernel/XcKernel.cc",
      "CppCore/rgpot/XcKernel/meson.build",
      "CppCore/rgpot/XcKernel/kernel_table.inc",
  };
  for (const char *p : files) {
    const std::string txt = slurp(p);
    REQUIRE(txt.find("D3") == std::string::npos);
    REQUIRE(txt.find("D4") == std::string::npos);
  }
}

TEST_CASE("golden fixtures exist (fail closed)", "[xckernel][golden]") {
  const char *required[] = {
      "CppCore/tests/data/xckernel/MANIFEST.json",
      "CppCore/tests/data/xckernel/randgrid_s1_nbf4_ng200_operands.npz",
      "CppCore/tests/data/xckernel/lda_r_o1_scal.npz",
      "CppCore/tests/data/xckernel/gga_r_o1_scal.npz",
      "CppCore/tests/data/xckernel/mgga_tau_r_o1_scal.npz",
      "CppCore/tests/data/xckernel/lda_r_o1_fock_ref.npy",
      "CppCore/tests/data/xckernel/gga_r_o1_fock_ref.npy",
      "CppCore/tests/data/xckernel/mgga_tau_r_o1_fock_ref.npy",
      "CppCore/tests/data/xckernel/mol_h2o_sto3g_lvl3_operands.npz",
      "CppCore/tests/data/xckernel/gga_r_o2_fxc_ref.npy",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_lda_r_o1_operands.npz",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_lda_r_o1_ref.npy",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_gga_r_o1_operands.npz",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_gga_r_o1_ref.npy",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_gga_r_o2_operands.npz",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_gga_r_o2_ref.npy",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_mgga_tau_r_o1_operands.npz",
      "CppCore/tests/data/xckernel/c_vs_numpy/xck_mgga_tau_r_o1_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/meta.json",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/dm0.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/dm1.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/lda_fock_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_fock_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/mgga_tau_fock_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_fxc_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/lda_fock_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_fock_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/mgga_tau_fock_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_fxc_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_sigma_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_sigma_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_lda_sigma_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_gga_sigma_ref.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/lda_mo.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_mo.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/lda_st_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/gga_st_operands.npz",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_z.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_z.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_j.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_j.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_lda_xy.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_gga_xy.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_lda_j.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_gga_j.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_nwchemc_roots.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_nwchemc_roots.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_pyscf_roots.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_pyscf_roots.npy",
  };
  for (const char *p : required) {
    require_file(p);
  }
  const std::string man = slurp("CppCore/tests/data/xckernel/MANIFEST.json");
  REQUIRE(man.find("sha256") != std::string::npos);
  REQUIRE(man.find("d6a9d57") != std::string::npos);
}

TEST_CASE("random-grid Fock pins (s2jz)", "[xckernel][golden][fock]") {
  require_file(std::string(kData) + "/randgrid_s1_nbf4_ng200_operands.npz");
  auto geom = load_npz(std::string(kData) +
                       "/randgrid_s1_nbf4_ng200_operands.npz");
  struct Case {
    const char *kernel;
    const char *scal;
    const char *ref;
  };
  const Case cases[] = {
      {"xck_lda_r_o1", "lda_r_o1_scal.npz", "lda_r_o1_fock_ref.npy"},
      {"xck_gga_r_o1", "gga_r_o1_scal.npz", "gga_r_o1_fock_ref.npy"},
      {"xck_mgga_tau_r_o1", "mgga_tau_r_o1_scal.npz",
       "mgga_tau_r_o1_fock_ref.npy"},
  };
  for (const auto &c : cases) {
    auto scal = load_npz(std::string(kData) + "/" + c.scal);
    auto op = geom;
    op.insert(scal.begin(), scal.end());
    auto ref = load_npy(std::string(kData) + "/" + c.ref);
    auto got = run_kernel(c.kernel, op, 4, 200, true);
    INFO("deviation " << deviation_in_eps(got, ref.data) << " eps*scale");
    REQUIRE(deviation_in_eps(got, ref.data) <= kNumpyEps);
  }
}

TEST_CASE("H2O GGA fxc pin (s2jz)", "[xckernel][golden][fxc]") {
  auto op = load_npz(std::string(kData) + "/mol_h2o_sto3g_lvl3_operands.npz");
  auto ref = load_npy(std::string(kData) + "/gga_r_o2_fxc_ref.npy");
  const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
  const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
  auto got = run_kernel("xck_gga_r_o2", op, nbf, npts, false);
  INFO("deviation " << deviation_in_eps(got, ref.data) << " eps*scale");
  REQUIRE(deviation_in_eps(got, ref.data) <= kNumpyEps);
}

TEST_CASE("C backend vs NumPy pin in ulp (2520)", "[xckernel][golden][cnp]") {
  struct Case {
    const char *kernel;
  };
  const Case cases[] = {
      {"xck_lda_r_o1"},
      {"xck_gga_r_o1"},
      {"xck_gga_r_o2"},
      {"xck_mgga_tau_r_o1"},
  };
  for (const auto &c : cases) {
    const std::string base =
        std::string(kData) + "/c_vs_numpy/" + c.kernel;
    require_file(base + "_operands.npz");
    require_file(base + "_ref.npy");
    auto op = load_npz(base + "_operands.npz");
    auto ref = load_npy(base + "_ref.npy");
    auto got = run_kernel(c.kernel, op, 4, 60, false);
    INFO("deviation " << deviation_in_eps(got, ref.data) << " eps*scale");
    REQUIRE(deviation_in_eps(got, ref.data) <= kNumpyEps);
  }
}

// Regression pins hold the C backend's own output (dump_xckernel_goldens).
// They detect a change in the C evaluation; they do not validate it, which
// the NumPy and PySCF pins above do.
TEST_CASE("C backend regression pins", "[xckernel][golden][regression]") {
  const std::string root = std::string(kData) + "/c_regression";
  auto geom = load_npz(std::string(kData) +
                       "/randgrid_s1_nbf4_ng200_operands.npz");
  struct Fock {
    const char *kernel;
    const char *scal;
    const char *pin;
  };
  const Fock focks[] = {
      {"xck_lda_r_o1", "lda_r_o1_scal.npz", "randgrid_lda_r_o1_fock_c.npy"},
      {"xck_gga_r_o1", "gga_r_o1_scal.npz", "randgrid_gga_r_o1_fock_c.npy"},
      {"xck_mgga_tau_r_o1", "mgga_tau_r_o1_scal.npz",
       "randgrid_mgga_tau_r_o1_fock_c.npy"},
  };
  for (const auto &c : focks) {
    auto scal = load_npz(std::string(kData) + "/" + c.scal);
    auto op = geom;
    op.insert(scal.begin(), scal.end());
    auto pin = load_npy(root + "/" + c.pin);
    auto got = run_kernel(c.kernel, op, 4, 200, true);
    REQUIRE(deviation_in_eps(got, pin.data) <= kNumpyEps);
  }
  for (const char *k :
       {"xck_lda_r_o1", "xck_gga_r_o1", "xck_gga_r_o2", "xck_mgga_tau_r_o1"}) {
    auto op =
        load_npz(std::string(kData) + "/c_vs_numpy/" + k + "_operands.npz");
    auto pin = load_npy(root + "/c_vs_numpy_" + k + "_c.npy");
    auto got = run_kernel(k, op, 4, 60, false);
    REQUIRE(deviation_in_eps(got, pin.data) <= kNumpyEps);
  }
  auto mol = load_npz(std::string(kData) + "/mol_h2o_sto3g_lvl3_operands.npz");
  auto pin = load_npy(root + "/gga_r_o2_fxc_c.npy");
  const auto nbf = static_cast<std::int64_t>(mol.at("chi").shape[0]);
  const auto npts = static_cast<std::int64_t>(mol.at("chi").shape[1]);
  auto got = run_kernel("xck_gga_r_o2", mol, nbf, npts, false);
  REQUIRE(deviation_in_eps(got, pin.data) <= kNumpyEps);
}

TEST_CASE("PySCF Fock and GGA fxc pins (4e7y)", "[xckernel][golden][pyscf]") {
  const std::string root = std::string(kData) + "/pyscf_h2o_sto3g";
  require_file(root + "/meta.json");
  auto meta = slurp(root + "/meta.json");
  REQUIRE(meta.find("sto-3g") != std::string::npos);

  struct Case {
    const char *kernel;
    const char *operands;
    const char *ref;
    double tol;
    bool dchi_nbf_first;
  };
  const Case cases[] = {
      {"xck_lda_r_o1", "lda_fock_operands.npz", "lda_fock_ref.npy",
       kFockVsPyscf, false},
      {"xck_gga_r_o1", "gga_fock_operands.npz", "gga_fock_ref.npy",
       kFockVsPyscf, false},
      {"xck_mgga_tau_r_o1", "mgga_tau_fock_operands.npz",
       "mgga_tau_fock_ref.npy", kFockVsPyscf, false},
      {"xck_gga_r_o2", "gga_fxc_operands.npz", "gga_fxc_ref.npy", kFxcVsPyscf,
       false},
  };
  for (const auto &c : cases) {
    require_file(root + "/" + c.operands);
    require_file(root + "/" + c.ref);
    auto op = load_npz(root + "/" + c.operands);
    auto ref = load_npy(root + "/" + c.ref);
    const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    auto got = run_kernel(c.kernel, op, nbf, npts, c.dchi_nbf_first);
    INFO(c.kernel << " rel=" << max_rel(got, ref.data));
    REQUIRE(max_rel(got, ref.data) <= c.tol);
  }
  require_file(root + "/tda_lda_sigma_ref.npy");
  require_file(root + "/tda_gga_sigma_ref.npy");
  require_file(root + "/rpa_lda_sigma_ref.npy");
  require_file(root + "/rpa_gga_sigma_ref.npy");
}

TEST_CASE("PySCF TDA/RPA sigma pins at 1e-17 (4e7y)",
          "[xckernel][golden][tda][rpa]") {
  const std::string root = std::string(kData) + "/pyscf_h2o_sto3g";
  struct Case {
    const char *fam;
    const char *kernel;
  };
  const Case cases[] = {
      {"lda", "xck_lda_st_o2_p"},
      {"gga", "xck_gga_st_o2_p"},
  };
  for (const auto &c : cases) {
    auto mo_npz = load_npz(root + "/" + std::string(c.fam) + "_mo.npz");
    auto op = load_npz(root + "/" + std::string(c.fam) + "_st_operands.npz");
    auto zs = load_npy(root + "/tda_" + std::string(c.fam) + "_z.npy");
    auto tda_ref =
        load_npy(root + "/tda_" + std::string(c.fam) + "_sigma_ref.npy");
    auto tda_j = load_npy(root + "/tda_" + std::string(c.fam) + "_j.npy");
    auto xys = load_npy(root + "/rpa_" + std::string(c.fam) + "_xy.npy");
    auto rpa_ref =
        load_npy(root + "/rpa_" + std::string(c.fam) + "_sigma_ref.npy");
    auto rpa_j = load_npy(root + "/rpa_" + std::string(c.fam) + "_j.npy");
    REQUIRE(zs.shape.size() == 3);
    const auto nz = static_cast<std::int64_t>(zs.shape[0]);
    const auto nocc = static_cast<std::int64_t>(zs.shape[1]);
    const auto nvir = static_cast<std::int64_t>(zs.shape[2]);
    const auto nao = static_cast<std::int64_t>(mo_npz.at("Co").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    XcKernel k(c.kernel);
    std::vector<double> dchi_c;
    XcGrid g = grid_from_npz(op, nao, npts, &dchi_c, false);
    auto ground = ground_from_npz(k, op);
    XcMo mo;
    mo.nao = nao;
    mo.nocc = nocc;
    mo.nvir = nvir;
    mo.Co = mo_npz.at("Co").data.data();
    mo.Cv = mo_npz.at("Cv").data.data();
    mo.e_ia = mo_npz.at("e_ia").data.data();

    std::vector<double> got_tda(static_cast<std::size_t>(nz * nocc * nvir), 0.0);
    for (std::int64_t iz = 0; iz < nz; ++iz) {
      const double *z =
          zs.data.data() + static_cast<std::size_t>(iz * nocc * nvir);
      const double *vj =
          tda_j.data.data() + static_cast<std::size_t>(iz * nao * nao);
      double *sig = got_tda.data() + static_cast<std::size_t>(iz * nocc * nvir);
      REQUIRE(k.tdaSigma(g, ground, mo, z, vj, sig) == 0);
    }
    const double tda_rel = max_rel(got_tda, tda_ref.data);
    const double tda_abs = max_abs(got_tda, tda_ref.data);

    std::vector<double> got_rpa(static_cast<std::size_t>(nz * 2 * nocc * nvir),
                                0.0);
    for (std::int64_t iz = 0; iz < nz; ++iz) {
      const double *xy =
          xys.data.data() + static_cast<std::size_t>(iz * 2 * nocc * nvir);
      const double *vj =
          rpa_j.data.data() + static_cast<std::size_t>(iz * nao * nao);
      double *sig =
          got_rpa.data() + static_cast<std::size_t>(iz * 2 * nocc * nvir);
      REQUIRE(k.rpaSigma(g, ground, mo, xy, vj, sig) == 0);
    }
    const double rpa_rel = max_rel(got_rpa, rpa_ref.data);
    const double rpa_abs = max_abs(got_rpa, rpa_ref.data);
    UNSCOPED_INFO(c.fam << " TDA rel=" << tda_rel << " abs=" << tda_abs);
    UNSCOPED_INFO(c.fam << " RPA rel=" << rpa_rel << " abs=" << rpa_abs);
    CHECK(tda_rel <= kTdaRpaVsPyscf);
    CHECK(rpa_rel <= kTdaRpaVsPyscf);
  }
}

TEST_CASE("TDA pin operator vs libnwchemc roots",
          "[xckernel][golden][tda][nwchemc]") {
  const std::string root = std::string(kData) + "/pyscf_h2o_sto3g";
  for (const char *fam : {"lda", "gga"}) {
    auto nw = load_npy(root + "/tda_" + std::string(fam) + "_nwchemc_roots.npy");
    auto pin = load_npy(root + "/tda_" + std::string(fam) + "_pyscf_roots.npy");
    REQUIRE(nw.shape.size() == 1);
    REQUIRE(pin.shape.size() == 1);
    REQUIRE(nw.shape[0] == pin.shape[0]);
    REQUIRE(nw.shape[0] == 5);
    const double rel = max_rel(nw.data, pin.data);
    const double absd = max_abs(nw.data, pin.data);
    UNSCOPED_INFO(fam << " nwchemc vs pin-operator rel=" << rel
                      << " abs=" << absd);
    CHECK(rel <= kTdaNwchemcVsPin);
  }
}


namespace {
struct LaplacianGridFixture {
  static constexpr std::int64_t points = 3;
  static constexpr std::int64_t basis = 2;
  std::vector<double> chi{0.5, 0.75, 1.0, -0.25, 0.25, 0.5};
  std::vector<double> grad{0.25, -0.5, 0.75, 0.5, 0.25, -0.25,
                           -0.5, 0.25, 0.5, 0.25, -0.75, 0.5,
                           0.5, 0.75, -0.25, -0.25, 0.5, 0.25};
  std::vector<double> lapl{0.25, -0.5, 0.75, -0.5, 0.25, 0.125};
  std::vector<double> weights{0.5, 0.25, 1.0};

  XcGrid grid() const {
    return {points, basis, chi.data(), grad.data(), lapl.data(), nullptr};
  }

  double energy(const std::vector<double> &density) const {
    double total = 0.0;
    for (std::int64_t g = 0; g < points; ++g) {
      double rho = 0.0, laplacian = 0.0;
      double gradient[3] = {};
      for (std::int64_t u = 0; u < basis; ++u) {
        for (std::int64_t v = 0; v < basis; ++v) {
          const double P = density[u * basis + v];
          const double a = chi[u * points + g];
          const double b = chi[v * points + g];
          rho += P * a * b;
          laplacian += P * (lapl[u * points + g] * b +
                            a * lapl[v * points + g]);
          for (std::int64_t axis = 0; axis < 3; ++axis) {
            const double da = grad[(axis * basis + u) * points + g];
            const double db = grad[(axis * basis + v) * points + g];
            gradient[axis] += P * (da * b + a * db);
            laplacian += 2.0 * P * da * db;
          }
        }
      }
      double sigma = 0.0;
      for (double value : gradient) sigma += value * value;
      total += weights[g] * (0.5 * rho * rho + 0.25 * sigma +
                             0.75 * laplacian * laplacian +
                             0.125 * rho * laplacian);
    }
    return total;
  }

  std::vector<double> fock(const std::vector<double> &density) const {
    const auto fields = XcKernel::fieldsFromDensity(grid(), density.data());
    std::vector<double> vrho(points), vlapl(points), vsigma(points, 0.25);
    for (std::int64_t g = 0; g < points; ++g) {
      vrho[g] = fields.rho[g] + 0.125 * fields.lapl[g];
      vlapl[g] = 1.5 * fields.lapl[g] + 0.125 * fields.rho[g];
    }
    const std::map<std::string, const double *> scal{
        {"w", weights.data()}, {"vrho", vrho.data()},
        {"vsigma", vsigma.data()}, {"vlapl", vlapl.data()},
        {"grad_rho_x", fields.grad_rho.data()},
        {"grad_rho_y", fields.grad_rho.data() + points},
        {"grad_rho_z", fields.grad_rho.data() + 2 * points}};
    std::vector<double> result(basis * basis, 0.0);
    REQUIRE(XcKernel("xck_mgga_lapl_r_o1").contract(grid(), scal, result.data()) == 0);
    return result;
  }
};
} // namespace

TEST_CASE("Laplacian Fock and response differentiate a scalar functional",
          "[xckernel][laplacian]") {
  const LaplacianGridFixture fixture;
  const std::vector<double> density{0.5, 0.25, -0.125, 0.75};
  const std::vector<double> direction{0.25, -0.5, 0.125, 0.25};
  constexpr double step = 1.0 / 1024.0;
  const auto fock = fixture.fock(density);
  for (std::size_t i = 0; i < density.size(); ++i) {
    auto plus = density, minus = density;
    plus[i] += step;
    minus[i] -= step;
    REQUIRE(fock[i] == (fixture.energy(plus) - fixture.energy(minus)) / (2.0 * step));
  }

  const auto fields = XcKernel::fieldsFromDensity(fixture.grid(), density.data());
  const auto perturbation = XcKernel::fieldsFromDensity(fixture.grid(), direction.data());
  const auto n = fixture.points;
  const std::vector<double> one(n, 1.0), zero(n, 0.0), sigma(n, 0.25),
      lapl(n, 1.5), mixed(n, 0.125);
  const std::map<std::string, const double *> scal{
      {"w", fixture.weights.data()}, {"vsigma", sigma.data()},
      {"v2rho2", one.data()}, {"v2lapl2", lapl.data()},
      {"v2rholapl", mixed.data()}, {"v2sigma2", zero.data()},
      {"v2rhosigma", zero.data()}, {"v2sigmalapl", zero.data()},
      {"rho_p1", perturbation.rho.data()},
      {"lapl_rho_p1", perturbation.lapl.data()},
      {"grad_rho_x", fields.grad_rho.data()},
      {"grad_rho_y", fields.grad_rho.data() + n},
      {"grad_rho_z", fields.grad_rho.data() + 2 * n},
      {"grad_rho_p1_x", perturbation.grad_rho.data()},
      {"grad_rho_p1_y", perturbation.grad_rho.data() + n},
      {"grad_rho_p1_z", perturbation.grad_rho.data() + 2 * n}};
  std::vector<double> response(4, 0.0);
  REQUIRE(XcKernel("xck_mgga_lapl_r_o2").contract(fixture.grid(), scal, response.data()) == 0);
  auto plus = density, minus = density;
  for (std::size_t i = 0; i < density.size(); ++i) {
    plus[i] += step * direction[i];
    minus[i] -= step * direction[i];
  }
  const auto upper = fixture.fock(plus), lower = fixture.fock(minus);
  for (std::size_t i = 0; i < response.size(); ++i)
    REQUIRE(response[i] == (upper[i] - lower[i]) / (2.0 * step));
}

TEST_CASE("Every Laplacian host kernel rejects missing AO Laplacians",
          "[xckernel][laplacian][api]") {
  const LaplacianGridFixture fixture;
  auto incomplete = fixture.grid();
  incomplete.lapl_chi = nullptr;
  for (const char *suffix : {"r_o1", "ua_o1", "ub_o1", "r_o2", "ua_o2",
                             "ub_o2", "st_o2_p", "st_o2_m"}) {
    const XcKernel kernel(std::string("xck_mgga_lapl_") + suffix);
    REQUIRE(kernel.nScal() > kernel.nFields());
    const std::vector<double> sentinel{1.0, 2.0, 3.0, 4.0};
    auto output = sentinel;
    REQUIRE(kernel.contract(incomplete, {}, output.data()) == 2);
    REQUIRE(output == sentinel);
  }
}

TEST_CASE("UKS Fock pins for the ua and ub kernels",
          "[xckernel][golden][uks][fock]") {
  const std::string root = std::string(kData) + "/pyscf_uks_h2o_cation_sto3g";
  for (const char *fam : {"lda", "gga"}) {
    auto op = load_npz(root + "/" + fam + "_operands.npz");
    const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    for (const char *tag : {"ua", "ub"}) {
      auto ref = load_npy(root + "/" + fam + "_" + tag + "_fock_ref.npy");
      auto got = run_kernel(std::string("xck_") + fam + "_" + tag + "_o1", op,
                            nbf, npts, false);
      INFO(fam << " " << tag << " rel=" << max_rel(got, ref.data));
      REQUIRE(max_rel(got, ref.data) <= kFockVsPyscf);
    }
  }
}

TEST_CASE("UKS fxc pins for the ua and ub kernels",
          "[xckernel][golden][uks][fxc]") {
  const std::string root = std::string(kData) + "/pyscf_uks_h2o_cation_sto3g";
  for (const char *fam : {"lda", "gga"}) {
    auto op = load_npz(root + "/" + fam + "_operands.npz");
    const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    const auto n2 = static_cast<std::size_t>(nbf * nbf);
    // dm1 is the (alpha, beta) pair of perturbed densities.
    REQUIRE(op.at("dm1").data.size() == 2 * n2);
    const double *dm_a = op.at("dm1").data.data();
    const double *dm_b = dm_a + n2;
    for (const char *tag : {"ua", "ub"}) {
      XcKernel k(std::string("xck_") + fam + "_" + tag + "_o2");
      std::vector<double> dchi_c;
      XcGrid g = grid_from_npz(op, nbf, npts, &dchi_c, false);
      auto ground = ground_from_npz(k, op);
      std::vector<double> got(n2, 1.0e3); // overwritten, not accumulated
      REQUIRE(k.applyFxcUnrestricted(g, ground, dm_a, dm_b, got.data()) == 0);
      auto ref = load_npy(root + "/" + fam + "_" + tag + "_fxc_ref.npy");
      INFO(fam << " " << tag << " rel=" << max_rel(got, ref.data));
      REQUIRE(max_rel(got, ref.data) <= kFxcVsPyscf);
    }
  }
}

TEST_CASE("UKS fxc rejects a missing density",
          "[xckernel][golden][uks][fxc]") {
  const std::string root = std::string(kData) + "/pyscf_uks_h2o_cation_sto3g";
  auto op = load_npz(root + "/lda_operands.npz");
  const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
  const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
  XcKernel k("xck_lda_ua_o2");
  std::vector<double> dchi_c;
  XcGrid g = grid_from_npz(op, nbf, npts, &dchi_c, false);
  auto ground = ground_from_npz(k, op);
  std::vector<double> out(static_cast<std::size_t>(nbf * nbf), 0.0);
  REQUIRE(k.applyFxcUnrestricted(g, ground, nullptr, nullptr, out.data()) !=
          0);
}

TEST_CASE("Closed-shell spin-adapted fxc pins, singlet and triplet",
          "[xckernel][golden][st][fxc]") {
  const std::string root = std::string(kData) + "/pyscf_h2o_sto3g";
  for (const char *fam : {"lda", "gga"}) {
    auto op = load_npz(root + "/" + fam + "_st_corr_operands.npz");
    const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    std::vector<double> singlet_minus_triplet;
    for (const char *tag : {"p", "m"}) {
      XcKernel k(std::string("xck_") + fam + "_st_o2_" + tag);
      std::vector<double> dchi_c;
      XcGrid g = grid_from_npz(op, nbf, npts, &dchi_c, false);
      auto ground = ground_from_npz(k, op);
      std::vector<double> got(static_cast<std::size_t>(nbf * nbf), 0.0);
      REQUIRE(k.applyFxc(g, ground, op.at("dm1").data.data(), got.data()) ==
              0);
      auto ref = load_npy(root + "/" + fam + "_st_" + tag + "_fxc_ref.npy");
      INFO(fam << " " << tag << " rel=" << max_rel(got, ref.data));
      REQUIRE(max_rel(got, ref.data) <= kFxcVsPyscf);
    }
    // The pinned kernels differ: with correlation the singlet and triplet
    // responses are distinct matrices.
    auto p = load_npy(root + "/" + fam + "_st_p_fxc_ref.npy");
    auto m = load_npy(root + "/" + fam + "_st_m_fxc_ref.npy");
    REQUIRE(max_abs(p.data, m.data) > 1e-2);
  }
}
