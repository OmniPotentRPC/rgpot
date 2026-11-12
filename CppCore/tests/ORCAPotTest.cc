// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using rgpot::types::AtomMatrix;

namespace {

std::string env_or_empty(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

class EnvRestore {
public:
  EnvRestore()
      : path_(env_or_empty("PATH")), orca_(env_or_empty("ORCA_EXE")),
        had_orca_(std::getenv("ORCA_EXE") != nullptr) {}

  ~EnvRestore() {
    set("PATH", path_);
    if (had_orca_) {
      set("ORCA_EXE", orca_);
    } else {
      ::unsetenv("ORCA_EXE");
    }
    ::unsetenv("ORCA_FAKE_LOG");
    ::unsetenv("ORCA_FAKE_FAIL_MOREAD");
    ::unsetenv("ORCA_FAKE_NATOMS");
    ::unsetenv("ORCA_FAKE_BAD_ENGRAD");
    ::unsetenv("ORCA_FAKE_ENERGY");
  }

  EnvRestore(const EnvRestore &) = delete;
  EnvRestore &operator=(const EnvRestore &) = delete;

private:
  static void set(const char *name, const std::string &value) {
    ::setenv(name, value.c_str(), 1);
  }

  std::string path_;
  std::string orca_;
  bool had_orca_;
};

constexpr char kFakeOrca[] = R"PY(#!/usr/bin/python3
import os
import re
import sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("fake orca: missing input file\n")
        return 2
    inp = Path(sys.argv[1])
    text = inp.read_text(encoding="utf-8")
    log_path = os.environ.get("ORCA_FAKE_LOG", "")

    def log(extra: str = "") -> None:
        if not log_path:
            return
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            if extra:
                fh.write(extra)
                if not extra.endswith("\n"):
                    fh.write("\n")

    moread = "MORead" in text
    note = ""
    if moread:
        match = re.search(r'%moinp\s+"([^"]+)"', text)
        if match is None:
            log("MOINP_LINE_MISSING\n")
            sys.stderr.write("fake orca: MORead without %moinp\n")
            return 2
        mo = Path(match.group(1))
        if not mo.is_absolute():
            mo = Path.cwd() / mo
        out_gbw = inp if inp.is_absolute() else Path.cwd() / inp
        out_gbw = out_gbw.with_suffix(".gbw")
        if not mo.is_file():
            log("MOINP_MISSING\n")
            sys.stderr.write("fake orca: moinp missing\n")
            return 3
        if mo.resolve() == out_gbw.resolve():
            log("MOINP_SAME_AS_GBW\n")
            sys.stderr.write("fake orca: moinp is the output gbw\n")
            return 4
        if not mo.read_bytes().startswith(b"GBW1"):
            log("MOINP_NOT_GBW\n")
            sys.stderr.write("fake orca: moinp is not a gbw\n")
            return 5
        note = "MOINP_OK\n"
    if moread and os.environ.get("ORCA_FAKE_FAIL_MOREAD") == "1":
        log(note + "MOREAD_REJECTED\n")
        sys.stderr.write("fake orca: MORead rejected\n")
        return 1
    atoms = 0
    in_xyz = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("* xyz") or stripped.startswith("*xyz"):
            in_xyz = True
            continue
        if in_xyz:
            if stripped.startswith("*"):
                break
            if stripped:
                atoms += 1
    override = os.environ.get("ORCA_FAKE_NATOMS", "")
    if override:
        atoms = int(override)
    energy = os.environ.get("ORCA_FAKE_ENERGY", "-1.5")
    if os.environ.get("ORCA_FAKE_BAD_ENGRAD") == "1":
        log(note)
        inp.with_suffix(".engrad").write_text("not an engrad\n", encoding="utf-8")
        return 0
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f" {atoms}",
        "#",
        "# The current total energy in Eh",
        "#",
        f" {energy}",
        "#",
        "# The current gradient in Eh/bohr",
        "#",
    ]
    for i in range(atoms):
        scale = i + 1
        lines.append(f" {0.1 * scale}")
        lines.append(f" {-0.2 * scale}")
        lines.append(f" {0.3 * scale}")
    lines += [
        "#",
        "# The atomic numbers and current coordinates in Bohr",
        "#",
    ]
    target = inp if inp.is_absolute() else Path.cwd() / inp
    target.with_suffix(".engrad").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gbw = target.with_suffix(".gbw")
    gen = 1
    if gbw.is_file():
        prev = gbw.read_bytes()
        if prev.startswith(b"GBW1:"):
            try:
                gen = int(prev.split(b":", 1)[1]) + 1
            except ValueError:
                gen = 1
    gbw.write_bytes(f"GBW1:{gen}\n".encode())
    log(note)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
)PY";

struct FakeOrca {
  EnvRestore env;
  std::filesystem::path dir;
  std::filesystem::path script;
  std::filesystem::path log;

  FakeOrca() {
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "rgpot-fake-orca-XXXXXX")
            .string();
    if (::mkdtemp(tmpl.data()) == nullptr) {
      throw std::runtime_error("cannot create fake orca directory");
    }
    dir = tmpl;
    script = dir / "orca";
    log = dir / "calls.log";
    {
      std::ofstream out(script);
      out << kFakeOrca;
    }
    if (::chmod(script.c_str(), 0755) != 0) {
      throw std::runtime_error("cannot chmod fake orca");
    }
    const std::string path = script.parent_path().string() + ":" + env_or_empty("PATH");
    ::setenv("PATH", path.c_str(), 1);
    ::unsetenv("ORCA_EXE");
    ::setenv("ORCA_FAKE_LOG", log.c_str(), 1);
  }

  ~FakeOrca() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  std::string calls() const {
    std::ifstream in(log);
    return std::string(std::istreambuf_iterator<char>(in), {});
  }
};

AtomMatrix twoAtoms() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {1.25, 0.0, 0.0}};
}

std::array<std::array<double, 3>, 3> vacuum() {
  return {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

int sections(const std::string &log) {
  int count = 0;
  for (std::size_t at = 0; (at = log.find("----\n", at)) != std::string::npos;
       at += 5) {
    ++count;
  }
  return count;
}

void expect_canned(double energy, const AtomMatrix &forces, int n_atoms) {
  REQUIRE(energy == Approx(-1.5 * rgpot::units::HARTREE_TO_EV));
  for (int atom = 0; atom < n_atoms; ++atom) {
    const double scale = static_cast<double>(atom + 1);
    REQUIRE(forces(atom, 0) ==
            Approx(0.1 * scale * rgpot::units::NEG_GRAD_TO_FORCE));
    REQUIRE(forces(atom, 1) ==
            Approx(-0.2 * scale * rgpot::units::NEG_GRAD_TO_FORCE));
    REQUIRE(forces(atom, 2) ==
            Approx(0.3 * scale * rgpot::units::NEG_GRAD_TO_FORCE));
  }
}

} // namespace

TEST_CASE("ORCAPot writes the input and converts the engrad", "[orca]") {
  FakeOrca fake;
  rgpot::ORCAConfig cfg;
  cfg.method = "HF";
  cfg.basis = "def2-SVP";
  cfg.charge = -1;
  cfg.multiplicity = 3;
  cfg.keywords = "TightSCF";
  rgpot::ORCAPot pot(cfg);
  const std::vector<int> z{1, 8};
  auto [energy, forces, variance] = pot(twoAtoms(), z, vacuum());
  REQUIRE(variance == 0.0);
  expect_canned(energy, forces, 2);

  const std::string log = fake.calls();
  REQUIRE(sections(log) == 1);
  REQUIRE(log.find("! HF def2-SVP TightSCF EnGrad\n") != std::string::npos);
  REQUIRE(log.find("MORead") == std::string::npos);
  REQUIRE(log.find("* xyz -1 3\n") != std::string::npos);
  REQUIRE(log.find("H ") != std::string::npos);
  REQUIRE(log.find("O ") != std::string::npos);
  REQUIRE(log.find("1.25") != std::string::npos);
}

TEST_CASE("ORCAPot reuses the wavefunction and falls back once", "[orca]") {
  FakeOrca fake;
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  const std::vector<int> z{1, 8};
  auto [first, forces, variance] = pot(twoAtoms(), z, vacuum());
  (void)forces;
  (void)variance;
  expect_canned(first, forces, 2);

  auto [second, forces2, variance2] = pot(twoAtoms(), z, vacuum());
  (void)variance2;
  expect_canned(second, forces2, 2);
  std::string log = fake.calls();
  REQUIRE(sections(log) == 2);
  REQUIRE(log.find("! HF def2-SVP EnGrad MORead\n") != std::string::npos);
  REQUIRE(log.find("%moinp \"rgpot.moinp.gbw\"") != std::string::npos);
  REQUIRE(log.find("MOINP_OK") != std::string::npos);
  REQUIRE(log.find("MOINP_MISSING") == std::string::npos);
  REQUIRE(log.find("MOINP_SAME_AS_GBW") == std::string::npos);

  ::setenv("ORCA_FAKE_FAIL_MOREAD", "1", 1);
  auto [third, forces3, variance3] = pot(twoAtoms(), z, vacuum());
  (void)variance3;
  expect_canned(third, forces3, 2);
  log = fake.calls();
  REQUIRE(sections(log) == 4);
  REQUIRE(log.find("MOREAD_REJECTED") != std::string::npos);
  const auto rejected = log.rfind("MOREAD_REJECTED");
  const auto cold = log.find("! HF def2-SVP EnGrad\n", rejected);
  REQUIRE(cold != std::string::npos);
  REQUIRE(log.find("MORead", cold) == std::string::npos);
}

TEST_CASE("ORCAPot drops the wavefunction when the charge changes", "[orca]") {
  FakeOrca fake;
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  const std::vector<int> z{1, 8};
  (void)pot(twoAtoms(), z, vacuum());
  pot.setChargeSpin(-1, 2);
  (void)pot(twoAtoms(), z, vacuum());
  const std::string log = fake.calls();
  REQUIRE(sections(log) == 2);
  REQUIRE(log.find("* xyz -1 2\n") != std::string::npos);
  REQUIRE(log.find("MORead") == std::string::npos);
}

TEST_CASE("ORCAPot rejects a mismatched engrad", "[orca]") {
  FakeOrca fake;
  ::setenv("ORCA_FAKE_NATOMS", "1", 1);
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  const std::vector<int> z{1, 8};
  REQUIRE_THROWS_AS(pot(twoAtoms(), z, vacuum()), std::runtime_error);
}

TEST_CASE("ORCAPot reports a missing executable", "[orca]") {
  EnvRestore env;
  std::string empty =
      (std::filesystem::temp_directory_path() / "rgpot-no-orca-XXXXXX").string();
  REQUIRE(::mkdtemp(empty.data()) != nullptr);
  ::setenv("PATH", empty.c_str(), 1);
  ::unsetenv("ORCA_EXE");
  rgpot::ORCAConfig cfg;
  rgpot::ORCAPot pot(cfg);
  REQUIRE_FALSE(rgpot::ORCAPot::available());
  const std::vector<int> z{1, 8};
  try {
    (void)pot(twoAtoms(), z, vacuum());
    FAIL("expected a missing-executable error");
  } catch (const std::runtime_error &ex) {
    const std::string msg = ex.what();
    REQUIRE(msg.find("ORCA executable not found") != std::string::npos);
    REQUIRE(msg.find("ORCA_EXE") != std::string::npos);
    REQUIRE(msg.find("PATH") != std::string::npos);
  }
  std::filesystem::remove_all(empty);

  cfg.executable = "/no/such/orca-binary";
  rgpot::ORCAPot explicit_pot(cfg);
  try {
    (void)explicit_pot(twoAtoms(), z, vacuum());
    FAIL("expected a missing configured executable");
  } catch (const std::runtime_error &ex) {
    REQUIRE(std::string(ex.what()).find("/no/such/orca-binary") !=
            std::string::npos);
  }

  ::setenv("ORCA_EXE", "/no/such/orca-from-env", 1);
  cfg.executable.clear();
  rgpot::ORCAPot env_pot(cfg);
  try {
    (void)env_pot(twoAtoms(), z, vacuum());
    FAIL("expected a missing ORCA_EXE");
  } catch (const std::runtime_error &ex) {
    const std::string msg = ex.what();
    REQUIRE(msg.find("ORCA_EXE") != std::string::npos);
    REQUIRE(msg.find("/no/such/orca-from-env") != std::string::npos);
  }
}

TEST_CASE("ORCAPot accepts the executable from the config or ORCA_EXE",
          "[orca]") {
  FakeOrca fake;
  ::setenv("PATH", "/no/such/bin", 1);
  rgpot::ORCAConfig cfg;
  cfg.executable = fake.script.string();
  rgpot::ORCAPot from_config(cfg);
  const std::vector<int> z{1, 8};
  auto [energy, forces, variance] = from_config(twoAtoms(), z, vacuum());
  (void)forces;
  (void)variance;
  expect_canned(energy, forces, 2);

  ::unsetenv("ORCA_FAKE_LOG");
  const std::filesystem::path log2 = fake.dir / "exe.log";
  ::setenv("ORCA_FAKE_LOG", log2.c_str(), 1);
  cfg.executable.clear();
  ::setenv("ORCA_EXE", fake.script.c_str(), 1);
  rgpot::ORCAPot from_env(cfg);
  auto [energy2, forces2, variance2] = from_env(twoAtoms(), z, vacuum());
  (void)forces2;
  (void)variance2;
  expect_canned(energy2, forces2, 2);
  REQUIRE(std::filesystem::file_size(log2) > 0);
}

TEST_CASE("ORCA H2O HF/def2-SVP gradient matches a finite difference",
          "[orca][integration]") {
  if (std::getenv("ORCA_FAKE_LOG") != nullptr) {
    FAIL("a unit test left ORCA_FAKE_LOG set");
  }
  if (!rgpot::ORCAPot::available()) {
    SKIP("ORCA executable not found");
  }
  rgpot::ORCAConfig cfg;
  cfg.method = "HF";
  cfg.basis = "def2-SVP";
  cfg.charge = 0;
  cfg.multiplicity = 1;
  rgpot::ORCAPot pot(cfg);
  AtomMatrix pos{{0.0, 0.0, 0.1173}, {0.0, 0.7572, -0.4692}, {0.0, -0.7572, -0.4692}};
  const std::vector<int> z{8, 1, 1};
  auto [energy, forces, variance] = pot(pos, z, vacuum());
  (void)variance;
  REQUIRE(energy < -1000.0);

  constexpr double delta = 1.0e-4;
  double worst = 0.0;
  for (int atom = 0; atom < 3; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      AtomMatrix plus = pos;
      AtomMatrix minus = pos;
      plus(atom, axis) += delta;
      minus(atom, axis) -= delta;
      const double e_plus = std::get<0>(pot(plus, z, vacuum()));
      const double e_minus = std::get<0>(pot(minus, z, vacuum()));
      const double numerical = -(e_plus - e_minus) / (2.0 * delta);
      const double err = std::abs(numerical - forces(atom, axis));
      if (err > worst) {
        worst = err;
      }
    }
  }
  INFO("worst |analytical - finite difference| eV/Angstrom " << worst);
  REQUIRE(worst < 5.0e-3);
}
