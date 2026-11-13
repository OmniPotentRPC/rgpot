// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/OrcaPot/OrcaPot.hpp"
#include "rgpot/OrcaPot/orca_c_abi.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using rgpot::types::AtomMatrix;

namespace {

std::string readWhole(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

rgpot::OrcaConfig tightConfig() {
  rgpot::OrcaConfig cfg;
  cfg.method = "HF";
  cfg.basis = "STO-3G";
  cfg.keywords = "TightSCF";
  cfg.charge = 0;
  cfg.multiplicity = 1;
  cfg.nprocs = 2;
  cfg.maxcore_mb = 400;
  return cfg;
}

rgpot::OrcaAtoms hydrogen() {
  static const double pos[6] = {0.0, 0.0, 0.0, 0.74, 0.0, 0.0};
  static const int z[2] = {1, 1};
  rgpot::OrcaAtoms atoms;
  atoms.n = 2;
  atoms.positions = pos;
  atoms.atomic_numbers = z;
  return atoms;
}

AtomMatrix hydrogenMatrix() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {0.74, 0.0, 0.0}};
}

std::array<std::array<double, 3>, 3> box() {
  return {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
}

#ifndef _WIN32
struct SavedEnv {
  std::string name;
  bool present = false;
  std::string value;
  explicit SavedEnv(const char *key) : name(key) {
    const char *cur = std::getenv(key);
    present = cur != nullptr;
    if (present) {
      value = cur;
    }
  }
  ~SavedEnv() {
    if (present) {
      setenv(name.c_str(), value.c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

void clearFakeFlags() {
  unsetenv("RGPOT_ORCA_FAKE_FAIL");
  unsetenv("RGPOT_ORCA_FAKE_NO_ENGRAD");
  unsetenv("RGPOT_ORCA_FAKE_BAD_ENGRAD");
  unsetenv("RGPOT_ORCA_FAKE_SCF_FAIL");
  unsetenv("RGPOT_ORCA_FAKE_MOREAD_FAIL");
}

struct FakeOrca {
  SavedEnv path{"PATH"};
  SavedEnv rgpot{"RGPOT_ORCA"};
  SavedEnv orca{"ORCA"};
  SavedEnv log{"RGPOT_ORCA_FAKE_LOG"};
  SavedEnv engrad{"RGPOT_ORCA_FAKE_ENGRAD"};
  SavedEnv fail{"RGPOT_ORCA_FAKE_FAIL"};
  SavedEnv noEngrad{"RGPOT_ORCA_FAKE_NO_ENGRAD"};
  SavedEnv bad{"RGPOT_ORCA_FAKE_BAD_ENGRAD"};
  SavedEnv scf{"RGPOT_ORCA_FAKE_SCF_FAIL"};
  SavedEnv moread{"RGPOT_ORCA_FAKE_MOREAD_FAIL"};
  std::filesystem::path dir;
  std::filesystem::path calls;

  FakeOrca() {
    clearFakeFlags();
    dir = std::filesystem::temp_directory_path() /
          ("rgpot-fake-orca-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const auto src =
        std::filesystem::absolute("CppCore/tests/data/orca/fake_orca.sh");
    const auto dest = dir / "orca";
    std::filesystem::copy_file(
        src, dest, std::filesystem::copy_options::overwrite_existing);
    REQUIRE(::chmod(dest.c_str(), 0755) == 0);
    calls = dir / "calls.log";
    const std::string old = path.value;
    setenv("PATH", (dir.string() + ":" + old).c_str(), 1);
    unsetenv("RGPOT_ORCA");
    unsetenv("ORCA");
    setenv("RGPOT_ORCA_FAKE_LOG", calls.c_str(), 1);
    const auto fixture =
        std::filesystem::absolute("CppCore/tests/data/orca/h2.engrad");
    setenv("RGPOT_ORCA_FAKE_ENGRAD", fixture.c_str(), 1);
  }

  ~FakeOrca() {
    clearFakeFlags();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  std::string callsText() const {
    std::ifstream in(calls);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }
};

int countToken(const std::string &text, const std::string &token) {
  int n = 0;
  std::size_t at = 0;
  while ((at = text.find(token, at)) != std::string::npos) {
    ++n;
    at += token.size();
  }
  return n;
}
#endif

} // namespace

TEST_CASE("ORCA engrad parser reads a hand-written fixture", "[orca]") {
  const std::string text = readWhole("CppCore/tests/data/orca/h2.engrad");
  const rgpot::OrcaEngrad parsed = rgpot::parseOrcaEngrad(text);
  REQUIRE(parsed.n_atoms == 2);
  REQUIRE(parsed.energy_eh == Approx(-1.5));
  REQUIRE(parsed.gradient_eh_bohr.size() == 6);
  REQUIRE(parsed.gradient_eh_bohr[0] == Approx(0.25));
  REQUIRE(parsed.gradient_eh_bohr[3] == Approx(-0.25));
  REQUIRE(parsed.gradient_eh_bohr[1] == Approx(0.0).margin(1e-15));
  // Force = -gradient, Eh/bohr -> eV/Angstrom. The factors are the
  // CODATA 2018 constants in units.hpp.
  REQUIRE(
      rgpot::units::NEG_GRAD_TO_FORCE ==
      Approx(-rgpot::units::HARTREE_TO_EV / rgpot::units::BOHR_TO_ANGSTROM));
}

TEST_CASE("ORCA engrad parser rejects a short gradient", "[orca]") {
  const char *text = "# Number of atoms\n1\n"
                     "# The current total energy in Eh\n-0.5\n"
                     "# The current gradient in Eh/bohr\n0.1\n0.2\n";
  REQUIRE_THROWS_WITH(rgpot::parseOrcaEngrad(text),
                      ContainsSubstring("malformed ORCA engrad"));
  REQUIRE_THROWS_WITH(rgpot::parseOrcaEngrad("not an engrad\n"),
                      ContainsSubstring("missing atom count"));
}

TEST_CASE("ORCA input writer matches the golden files", "[orca]") {
  const rgpot::OrcaConfig cfg = tightConfig();
  const rgpot::OrcaAtoms atoms = hydrogen();
  const std::string plain = rgpot::renderOrcaInput(cfg, atoms, false, "");
  REQUIRE(plain == readWhole("CppCore/tests/data/orca/h2.inp"));
  const std::string reused =
      rgpot::renderOrcaInput(cfg, atoms, true, "job.prev.gbw");
  REQUIRE(reused == readWhole("CppCore/tests/data/orca/h2_moread.inp"));

  rgpot::OrcaConfig block = cfg;
  block.nprocs = 0;
  block.maxcore_mb = 0;
  block.keywords = "%scf\n  MaxIter 30\nend";
  const std::string raw = rgpot::renderOrcaInput(block, atoms, false, "");
  REQUIRE(raw.find("%scf\n  MaxIter 30\nend\n") != std::string::npos);
  REQUIRE(raw.find("! HF STO-3G EnGrad\n") == 0);
}

TEST_CASE("ORCA reports a missing binary", "[orca]") {
  rgpot::OrcaConfig cfg;
  cfg.executable = "/no/such/rgpot-orca-binary";
  rgpot::OrcaPot pot(cfg);
  REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), std::vector<int>{1, 1}, box()),
                      ContainsSubstring("ORCA binary not found"));
  REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), std::vector<int>{1, 1}, box()),
                      ContainsSubstring("/no/such/rgpot-orca-binary"));
}

#ifndef _WIN32
TEST_CASE("ORCA subprocess, MORead and the C ABI", "[orca]") {
  FakeOrca fake;
  rgpot::OrcaConfig cfg = tightConfig();
  rgpot::OrcaPot pot(cfg);
  REQUIRE(pot.get_type() == rgpot::PotType::Orca);
  const std::vector<int> z{1, 1};
  auto [energy, forces, variance] = pot(hydrogenMatrix(), z, box());
  (void)variance;
  const double expectE = -1.5 * rgpot::units::HARTREE_TO_EV;
  const double expectF = 0.25 * rgpot::units::NEG_GRAD_TO_FORCE;
  REQUIRE(energy == Approx(expectE));
  REQUIRE(forces(0, 0) == Approx(expectF));
  REQUIRE(forces(1, 0) == Approx(-expectF));
  REQUIRE(forces(0, 1) == Approx(0.0).margin(1e-12));
  REQUIRE_FALSE(pot.reusedWavefunction());
  const std::string first = fake.callsText();
  REQUIRE(first.find("! MORead") == std::string::npos);
  REQUIRE(first.find("TightSCF") != std::string::npos);
  REQUIRE(first.find("nprocs 2") != std::string::npos);
  REQUIRE(first.find("%maxcore 400") != std::string::npos);

  auto [energy2, forces2, variance2] = pot(hydrogenMatrix(), z, box());
  (void)forces2;
  (void)variance2;
  REQUIRE(energy2 == Approx(expectE));
  REQUIRE(pot.reusedWavefunction());
  const std::string both = fake.callsText();
  REQUIRE(countToken(both, "! MORead") == 1);
  REQUIRE(both.find("%moinp \"job.prev.gbw\"") != std::string::npos);

  RgpotOrcaConfig c{};
  c.charge = 0;
  c.multiplicity = 1;
  c.method = "HF";
  c.basis = "STO-3G";
  c.keywords = "";
  c.executable = nullptr;
  c.reuse_gbw = 1;
  c.nprocs = 0;
  c.maxcore_mb = 0;
  char err[256] = {};
  RgpotOrcaPot *handle = rgpot_orca_create(&c, err, sizeof(err));
  INFO(err);
  REQUIRE(handle != nullptr);
  REQUIRE(rgpot_orca_abi_version() == RGPOT_ORCA_ABI_VERSION);
  REQUIRE(rgpot_orca_available() == 1);
  const double pos[6] = {0, 0, 0, 0.74, 0, 0};
  const int atoms[2] = {1, 1};
  const double cell[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double f[6] = {};
  double e = 0;
  REQUIRE(rgpot_orca_calculate(handle, 2, pos, atoms, f, &e, nullptr, cell) ==
          0);
  REQUIRE(e == Approx(expectE));
  REQUIRE(rgpot_orca_reused_wavefunction(handle) == 0);
  REQUIRE(rgpot_orca_calculate(handle, 2, pos, atoms, f, &e, nullptr, cell) ==
          0);
  REQUIRE(rgpot_orca_reused_wavefunction(handle) == 1);
  REQUIRE(std::string(rgpot_orca_last_error(handle)).empty());
  rgpot_orca_destroy(handle);
}

TEST_CASE("ORCA MORead falls back to a fresh guess", "[orca]") {
  FakeOrca fake;
  rgpot::OrcaPot pot(tightConfig());
  const std::vector<int> z{1, 1};
  auto first = pot(hydrogenMatrix(), z, box());
  (void)first;
  setenv("RGPOT_ORCA_FAKE_MOREAD_FAIL", "1", 1);
  auto [energy, forces, variance] = pot(hydrogenMatrix(), z, box());
  (void)forces;
  (void)variance;
  REQUIRE(energy == Approx(-1.5 * rgpot::units::HARTREE_TO_EV));
  REQUIRE_FALSE(pot.reusedWavefunction());
  const std::string log = fake.callsText();
  REQUIRE(countToken(log, "=== ") == 3);
  REQUIRE(countToken(log, "! MORead") == 1);
}

TEST_CASE("ORCA reuse can be turned off", "[orca]") {
  FakeOrca fake;
  rgpot::OrcaConfig cfg = tightConfig();
  cfg.reuse_gbw = false;
  rgpot::OrcaPot pot(cfg);
  const std::vector<int> z{1, 1};
  (void)pot(hydrogenMatrix(), z, box());
  (void)pot(hydrogenMatrix(), z, box());
  REQUIRE_FALSE(pot.reusedWavefunction());
  REQUIRE(fake.callsText().find("! MORead") == std::string::npos);
}

TEST_CASE("ORCA surfaces a non-zero exit, a missing engrad and SCF", "[orca]") {
  FakeOrca fake;
  const std::vector<int> z{1, 1};
  {
    setenv("RGPOT_ORCA_FAKE_FAIL", "3", 1);
    rgpot::OrcaPot pot(tightConfig());
    REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), z, box()),
                        ContainsSubstring("exited with status 3"));
    REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), z, box()),
                        ContainsSubstring("fake orca forced failure"));
    unsetenv("RGPOT_ORCA_FAKE_FAIL");
  }
  {
    setenv("RGPOT_ORCA_FAKE_NO_ENGRAD", "1", 1);
    rgpot::OrcaPot pot(tightConfig());
    REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), z, box()),
                        ContainsSubstring("did not write an .engrad file"));
    unsetenv("RGPOT_ORCA_FAKE_NO_ENGRAD");
  }
  {
    setenv("RGPOT_ORCA_FAKE_BAD_ENGRAD", "1", 1);
    rgpot::OrcaPot pot(tightConfig());
    REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), z, box()),
                        ContainsSubstring("malformed ORCA engrad"));
    unsetenv("RGPOT_ORCA_FAKE_BAD_ENGRAD");
  }
  {
    setenv("RGPOT_ORCA_FAKE_SCF_FAIL", "1", 1);
    rgpot::OrcaPot pot(tightConfig());
    REQUIRE_THROWS_WITH(pot(hydrogenMatrix(), z, box()),
                        ContainsSubstring("SCF did not converge"));
    unsetenv("RGPOT_ORCA_FAKE_SCF_FAIL");
  }
}
#endif

TEST_CASE("ORCA water when the binary is installed", "[orca]") {
  std::string found;
  try {
    found = rgpot::OrcaPot::resolveExecutable("");
  } catch (const std::runtime_error &) {
    SKIP("orca is not installed");
  }
  INFO(found);
  rgpot::OrcaConfig cfg;
  cfg.executable = found;
  cfg.method = "HF";
  cfg.basis = "STO-3G";
  cfg.reuse_gbw = true;
  rgpot::OrcaPot pot(cfg);
  AtomMatrix water{{0.000000, 0.000000, 0.117300},
                   {0.000000, 0.757200, -0.469200},
                   {0.000000, -0.757200, -0.469200}};
  const std::vector<int> z{8, 1, 1};
  auto [e1, f1, v1] = pot(water, z, box());
  (void)v1;
  REQUIRE(std::isfinite(e1));
  REQUIRE(e1 < -1000.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE(std::isfinite(
          f1(static_cast<std::size_t>(i), static_cast<std::size_t>(j))));
    }
  }
  auto [e2, f2, v2] = pot(water, z, box());
  (void)f2;
  (void)v2;
  REQUIRE(e2 == Approx(e1).epsilon(1e-5));
}
