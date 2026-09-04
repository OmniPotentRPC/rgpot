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
#include "rgpot/Potential.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"

using rgpot::XcGrid;
using rgpot::XcKernel;
using rgpot::XcMo;
using rgpot::testio::NpyArray;
using rgpot::testio::load_npy;
using rgpot::testio::load_npz;

namespace {

constexpr const char *kData = "CppCore/tests/data/xckernel";
// Paper/README bars. Exclusive: C vs NumPy 1e-16, Fock vs PySCF 1e-15,
// fxc vs PySCF 1e-13, TDA/RPA vs PySCF 1e-17. Do not invent looser values.
constexpr double kCVsNumpy = 1e-16;
constexpr double kFockVsPyscf = 1e-15;
constexpr double kFxcVsPyscf = 1e-13;
constexpr double kTdaRpaVsPyscf = 1e-17;

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

bool is_pert_scal(const std::string &name) {
  return name.find("_p1") != std::string::npos;
}

std::map<std::string, const double *>
ground_scal_from_npz(const XcKernel &k,
                     const std::map<std::string, NpyArray> &op) {
  std::map<std::string, const double *> scal;
  for (const auto &name : k.scalNames()) {
    if (is_pert_scal(name)) {
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
  REQUIRE(names.size() == 24);
  const std::string schema = slurp("CppCore/rgpot/rpc/Potentials.capnp");
  REQUIRE(schema.find("xckernel") == std::string::npos);
  REQUIRE(schema.find("XcKernel") == std::string::npos);
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
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_lda_vj.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/tda_gga_vj.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_lda_xy.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_gga_xy.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_lda_vj.npy",
      "CppCore/tests/data/xckernel/pyscf_h2o_sto3g/rpa_gga_vj.npy",
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
    REQUIRE(max_rel(got, ref.data) <= kCVsNumpy);
  }
}

TEST_CASE("H2O GGA fxc pin (s2jz)", "[xckernel][golden][fxc]") {
  auto op = load_npz(std::string(kData) + "/mol_h2o_sto3g_lvl3_operands.npz");
  auto ref = load_npy(std::string(kData) + "/gga_r_o2_fxc_ref.npy");
  const auto nbf = static_cast<std::int64_t>(op.at("chi").shape[0]);
  const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
  auto got = run_kernel("xck_gga_r_o2", op, nbf, npts, false);
  REQUIRE(max_rel(got, ref.data) <= kFxcVsPyscf);
}

TEST_CASE("C backend vs NumPy pin at 1e-16 (2520)", "[xckernel][golden][cnp]") {
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
    REQUIRE(max_rel(got, ref.data) <= kCVsNumpy);
  }
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
}

TEST_CASE("PySCF TDA/RPA sigma at 1e-17", "[xckernel][golden][tda]") {
  const std::string root = std::string(kData) + "/pyscf_h2o_sto3g";

  struct Case {
    const char *kernel;
    const char *fam;
  };
  const Case cases[] = {
      {"xck_lda_st_o2_p", "lda"},
      {"xck_gga_st_o2_p", "gga"},
  };
  for (const auto &c : cases) {
    auto mo_np = load_npz(root + "/" + std::string(c.fam) + "_mo.npz");
    auto op = load_npz(root + "/" + std::string(c.fam) + "_st_operands.npz");
    auto zref = load_npy(root + "/tda_" + std::string(c.fam) + "_z.npy");
    auto tda_ref =
        load_npy(root + "/tda_" + std::string(c.fam) + "_sigma_ref.npy");
    auto tda_vj = load_npy(root + "/tda_" + std::string(c.fam) + "_vj.npy");
    auto xyref = load_npy(root + "/rpa_" + std::string(c.fam) + "_xy.npy");
    auto rpa_ref =
        load_npy(root + "/rpa_" + std::string(c.fam) + "_sigma_ref.npy");
    auto rpa_vj = load_npy(root + "/rpa_" + std::string(c.fam) + "_vj.npy");
    REQUIRE(zref.shape.size() == 3);
    REQUIRE(tda_ref.shape == zref.shape);
    REQUIRE(xyref.shape.size() == 4);
    REQUIRE(rpa_ref.shape == xyref.shape);

    const auto nocc = static_cast<std::int64_t>(zref.shape[1]);
    const auto nvir = static_cast<std::int64_t>(zref.shape[2]);
    const auto nz = static_cast<std::int64_t>(zref.shape[0]);
    const auto nao = static_cast<std::int64_t>(mo_np.at("Co").shape[0]);
    REQUIRE(tda_vj.shape.size() == 3);
    REQUIRE(static_cast<std::int64_t>(tda_vj.shape[1]) == nao);

    XcMo mo;
    mo.nao = nao;
    mo.nocc = nocc;
    mo.nvir = nvir;
    mo.Co = mo_np.at("Co").data.data();
    mo.Cv = mo_np.at("Cv").data.data();
    mo.e_ia = mo_np.at("e_ia").data.data();

    XcKernel k(c.kernel);
    std::vector<double> dchi_c;
    const auto npts = static_cast<std::int64_t>(op.at("chi").shape[1]);
    XcGrid g = grid_from_npz(op, nao, npts, &dchi_c, false);
    auto ground = ground_scal_from_npz(k, op);

    const auto nov = static_cast<std::size_t>(nocc * nvir);
    const auto n2 = static_cast<std::size_t>(nao * nao);
    std::vector<double> tda_got(static_cast<std::size_t>(nz) * nov, 0.0);
    for (std::int64_t iz = 0; iz < nz; ++iz) {
      const double *z = zref.data.data() + static_cast<std::size_t>(iz) * nov;
      const double *vj = tda_vj.data.data() + static_cast<std::size_t>(iz) * n2;
      REQUIRE(k.tdaSigma(g, ground, mo, z, vj,
                         tda_got.data() + static_cast<std::size_t>(iz) * nov) ==
              0);
    }
    const double tda_rel = max_rel(tda_got, tda_ref.data);
    const double tda_abs = max_abs(tda_got, tda_ref.data);
    UNSCOPED_INFO(c.fam << " TDA rel=" << tda_rel << " abs=" << tda_abs);

    std::vector<double> rpa_got(static_cast<std::size_t>(nz) * 2 * nov, 0.0);
    for (std::int64_t iz = 0; iz < nz; ++iz) {
      const double *xy =
          xyref.data.data() + static_cast<std::size_t>(iz) * 2 * nov;
      const double *vj = rpa_vj.data.data() + static_cast<std::size_t>(iz) * n2;
      REQUIRE(k.rpaSigma(g, ground, mo, xy, vj,
                         rpa_got.data() + static_cast<std::size_t>(iz) * 2 *
                                              nov) == 0);
    }
    const double rpa_rel = max_rel(rpa_got, rpa_ref.data);
    const double rpa_abs = max_abs(rpa_got, rpa_ref.data);
    UNSCOPED_INFO(c.fam << " TDA rel=" << tda_rel << " abs=" << tda_abs);
    UNSCOPED_INFO(c.fam << " RPA rel=" << rpa_rel << " abs=" << rpa_abs);
    CHECK(tda_rel <= kTdaRpaVsPyscf);
    CHECK(rpa_rel <= kTdaRpaVsPyscf);
  }
}
