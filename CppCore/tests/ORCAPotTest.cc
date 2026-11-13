// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#ifdef RGPOT_ORCA_TEST_RPC
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>

#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <capnp/serialize.h>

#include "rgpot/engine_c_abi.h"
#include "rgpot/rpc/Potentials.capnp.h"
#endif

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
    ::unsetenv("ORCA_FAKE_REPLAY");
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
            fh.write(f"CWD {Path.cwd()}\n")

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
    target = inp if inp.is_absolute() else Path.cwd() / inp
    replay = os.environ.get("ORCA_FAKE_REPLAY", "")
    if replay:
        src = Path(replay)
        engrad_text = (src / "replay.engrad").read_text(encoding="utf-8")
        target.with_suffix(".engrad").write_text(engrad_text, encoding="utf-8")
        # ASE's ORCA calculator reads the process stdout as the .out file.
        out_text = (src / "replay.stdout").read_text(encoding="utf-8")
        sys.stdout.write(out_text)
        if not out_text.endswith("\n"):
            sys.stdout.write("\n")
    else:
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
        target.with_suffix(".engrad").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
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

TEST_CASE("ORCA conversion uses the CODATA 2018 Hartree and Bohr", "[orca]") {
  // NIST CODATA 2018 recommended values. Eh is 4.3597447222071e-18 J and
  // the elementary charge is 1.602176634e-19 C, so Eh/eV is
  // 27.211386245988. The Bohr radius is 0.529177210903 Å.
  constexpr double eh_joule = 4.3597447222071e-18;
  constexpr double elementary_charge = 1.602176634e-19;
  constexpr double published_hartree_ev = 27.211386245988;
  REQUIRE(rgpot::units::HARTREE_TO_EV == published_hartree_ev);
  REQUIRE(std::abs(eh_joule / elementary_charge - published_hartree_ev) <
          5e-13);
  REQUIRE(rgpot::units::BOHR_TO_ANGSTROM == 0.529177210903);
  REQUIRE(rgpot::units::NEG_GRAD_TO_FORCE ==
          -rgpot::units::HARTREE_TO_EV / rgpot::units::BOHR_TO_ANGSTROM);
}

TEST_CASE("ORCAPot rejects a cell that folds a pair", "[orca]") {
  FakeOrca fake;
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  AtomMatrix pos{{0.0, 0.0, 0.0}, {4.5, 0.0, 0.0}};
  const std::vector<int> z{1, 1};
  // Shear so a transposed lattice inverse would report a different
  // fractional coordinate than the row-vector cell used by ForceInput.
  const std::array<std::array<double, 3>, 3> cell{{
      {{5.0, 1.0, 0.0}},
      {{0.0, 5.0, 0.0}},
      {{0.0, 0.0, 5.0}},
  }};
  try {
    (void)pot(pos, z, cell);
    FAIL("a folding cell was accepted");
  } catch (const std::invalid_argument &ex) {
    const std::string msg = ex.what();
    REQUIRE(msg.find("clusters only") != std::string::npos);
    REQUIRE(msg.find("periodic boundary conditions") != std::string::npos);
  }
  REQUIRE(fake.calls().find("----") == std::string::npos);

  AtomMatrix inside{{0.0, 0.0, 0.0}, {1.0, 0.2, 0.0}};
  auto [energy, forces, variance] = pot(inside, z, cell);
  (void)forces;
  (void)variance;
  expect_canned(energy, forces, 2);

  const std::array<std::array<double, 3>, 3> none{};
  auto [open_energy, open_forces, open_variance] = pot(twoAtoms(), z, none);
  (void)open_forces;
  (void)open_variance;
  expect_canned(open_energy, open_forces, 2);
}

TEST_CASE("ORCAPot keeps one work directory per instance", "[orca]") {
  FakeOrca fake;
  const std::vector<int> z{1, 8};
  std::string log;
  {
    rgpot::ORCAPot first(rgpot::ORCAConfig{});
    rgpot::ORCAPot second(rgpot::ORCAConfig{});
    (void)first(twoAtoms(), z, vacuum());
    (void)second(twoAtoms(), z, vacuum());
    log = fake.calls();
  }
  std::vector<std::string> dirs;
  for (std::size_t at = 0; (at = log.find("CWD ", at)) != std::string::npos;
       at += 4) {
    const auto end = log.find('\n', at);
    dirs.push_back(log.substr(at + 4, end - (at + 4)));
  }
  REQUIRE(dirs.size() == 2);
  REQUIRE(dirs[0] != dirs[1]);
  REQUIRE_FALSE(std::filesystem::exists(dirs[0]));
  REQUIRE_FALSE(std::filesystem::exists(dirs[1]));
}

namespace {

constexpr char kReplay[] = "CppCore/tests/data/orca";

std::string replay_dir() {
  return std::filesystem::absolute(kReplay).string();
}
constexpr double kRecordedHa = -75.5;
constexpr double kRecordedGrad[9] = {0.020, -0.010, 0.005, -0.015, 0.012,
                                     -0.003, -0.005, -0.002, -0.002};

AtomMatrix recordedWater() {
  return AtomMatrix{{0.0, 0.0, 0.1173},
                    {0.0, 0.7572, -0.4692},
                    {0.0, -0.7572, -0.4692}};
}

void expect_recorded(double energy, const AtomMatrix &forces) {
  REQUIRE(energy == Approx(kRecordedHa * rgpot::units::HARTREE_TO_EV));
  for (int atom = 0; atom < 3; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(forces(atom, axis) ==
              Approx(kRecordedGrad[atom * 3 + axis] *
                     rgpot::units::NEG_GRAD_TO_FORCE));
    }
  }
}

} // namespace

TEST_CASE("ORCAPot replays a recorded engrad in CODATA units", "[orca]") {
  const std::filesystem::path replay = replay_dir();
  REQUIRE(std::filesystem::is_regular_file(replay / "replay.engrad"));
  REQUIRE(std::filesystem::is_regular_file(replay / "replay.stdout"));
  std::ifstream recorded(replay / "replay.engrad");
  const std::string text((std::istreambuf_iterator<char>(recorded)), {});
  REQUIRE(text.find("-75.500000000000") != std::string::npos);
  REQUIRE(text.find("Eh/bohr") != std::string::npos);

  FakeOrca fake;
  ::setenv("ORCA_FAKE_REPLAY", replay.c_str(), 1);
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  const std::vector<int> z{8, 1, 1};
  auto [energy, forces, variance] = pot(recordedWater(), z, vacuum());
  REQUIRE(variance == 0.0);
  expect_recorded(energy, forces);
  REQUIRE(fake.calls().find("! HF def2-SVP EnGrad\n") != std::string::npos);
}

TEST_CASE("recorded ORCA output matches ASE's ORCA calculator", "[orca][ase]") {
  FakeOrca fake;
  ::setenv("ORCA_FAKE_REPLAY", replay_dir().c_str(), 1);
  rgpot::ORCAPot pot(rgpot::ORCAConfig{});
  const std::vector<int> z{8, 1, 1};
  auto [energy, forces, variance] = pot(recordedWater(), z, vacuum());
  (void)variance;
  expect_recorded(energy, forces);

  const std::filesystem::path script = fake.dir / "ase_orca.py";
  const std::filesystem::path directory = fake.dir / "ase-run";
  std::filesystem::create_directory(directory);
  {
    std::ofstream out(script);
    out << R"PY(import sys
try:
    from ase import Atoms
    from ase.calculators.orca import ORCA, OrcaProfile
except Exception as exc:
    sys.stderr.write("SKIP %s\n" % exc)
    raise SystemExit(0)
atoms = Atoms(
    "OHH",
    positions=[
        (0.0, 0.0, 0.1173),
        (0.0, 0.7572, -0.4692),
        (0.0, -0.7572, -0.4692),
    ],
    pbc=False,
)
atoms.calc = ORCA(
    profile=OrcaProfile(sys.argv[1]),
    directory=sys.argv[2],
    charge=0,
    mult=1,
    orcasimpleinput="HF def2-SVP",
    orcablocks="",
)
value = atoms.get_potential_energy()
got = atoms.get_forces()
with open(sys.argv[3], "w", encoding="utf-8") as fh:
    fh.write("%.16e\n" % value)
    for row in got:
        fh.write("%.16e %.16e %.16e\n" % (row[0], row[1], row[2]))
)PY";
  }
  const auto result_path = fake.dir / "ase-result.txt";
  const std::string command =
      "python3 " + script.string() + " " + fake.script.string() + " " +
      directory.string() + " " + result_path.string() + " >" +
      (fake.dir / "ase.out").string() + " 2>" +
      (fake.dir / "ase.err").string();
  FILE *pipe = ::popen(command.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::string output;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    output += buf;
  }
  const int status = ::pclose(pipe);
  std::ifstream err(fake.dir / "ase.err");
  const std::string errors((std::istreambuf_iterator<char>(err)), {});
  if (errors.find("SKIP ") != std::string::npos) {
    SKIP(errors);
  }
  INFO(errors);
  INFO(output);
  REQUIRE(status == 0);
  std::ifstream result(result_path);
  const std::string numbers((std::istreambuf_iterator<char>(result)), {});
  INFO(numbers);
  std::istringstream in(numbers);
  double ase_energy = 0.0;
  REQUIRE(in >> ase_energy);
  double ase_forces[9];
  for (double &value : ase_forces) {
    REQUIRE(in >> value);
  }
  // ASE derives Hartree and Bohr from the CODATA 2018 constants. rgpot
  // uses NIST's published Eh/eV and a0 conversions. The two agree well
  // past a part in a million, and a missed Bohr factor would not.
  REQUIRE(energy == Approx(ase_energy).epsilon(1e-6));
  for (int atom = 0; atom < 3; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(forces(atom, axis) ==
              Approx(ase_forces[atom * 3 + axis]).epsilon(1e-6));
    }
  }
}

#ifdef RGPOT_ORCA_TEST_RPC

namespace {

int free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
  socklen_t len = sizeof(addr);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

struct Potserv {
  pid_t pid = -1;
  std::filesystem::path log;

  explicit Potserv(const std::filesystem::path &log_path, int port,
                   const char *kind)
      : log(log_path) {
    const char *bin = std::getenv("RGPOT_POTSERV");
    REQUIRE(bin != nullptr);
    pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      FILE *out = std::fopen(log.c_str(), "w");
      if (out != nullptr) {
        ::dup2(fileno(out), STDOUT_FILENO);
        ::dup2(fileno(out), STDERR_FILENO);
        std::fclose(out);
      }
      const std::string port_text = std::to_string(port);
      ::execl(bin, bin, port_text.c_str(), kind, static_cast<char *>(nullptr));
      _exit(127);
    }
  }

  ~Potserv() {
    if (pid > 0) {
      ::kill(pid, SIGTERM);
      ::waitpid(pid, nullptr, 0);
    }
  }

  Potserv(const Potserv &) = delete;
  Potserv &operator=(const Potserv &) = delete;

  void wait_until_listening() const {
    for (int i = 0; i < 50; ++i) {
      std::ifstream in(log);
      const std::string text((std::istreambuf_iterator<char>(in)), {});
      if (text.find("Server running") != std::string::npos) {
        return;
      }
      int status = 0;
      const pid_t done = ::waitpid(pid, &status, WNOHANG);
      if (done == pid) {
        FAIL("potserv exited before listening:\n" << text);
      }
      ::usleep(100000);
    }
    std::ifstream in(log);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    FAIL("potserv did not start:\n" << text);
  }
};

void fill_water(::ForceInput::Builder input) {
  auto pos = input.initPos(9);
  const double xyz[9] = {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692,
                         0.0, -0.7572, -0.4692};
  for (unsigned i = 0; i < 9; ++i) {
    pos.set(i, xyz[i]);
  }
  auto numbers = input.initAtmnrs(3);
  numbers.set(0, 8);
  numbers.set(1, 1);
  numbers.set(2, 1);
  auto box = input.initBox(9);
  for (unsigned i = 0; i < 9; ++i) {
    box.set(i, i % 4 == 0 ? 100.0 : 0.0);
  }
  input.setLengthUnit("angstrom");
  input.setEnergyUnit("eV");
}

} // namespace

TEST_CASE("the ORCA engine reads a recorded engrad", "[orca][engine]") {
  const char *library = std::getenv("RGPOT_ENGINE_ORCA");
  if (library == nullptr) {
    SKIP("RGPOT_ENGINE_ORCA is not set");
  }
  void *handle = ::dlopen(library, RTLD_NOW | RTLD_LOCAL);
  const char *why = handle == nullptr ? ::dlerror() : nullptr;
  const std::string load_error = why == nullptr ? std::string() : std::string(why);
  INFO(load_error);
  REQUIRE(handle != nullptr);
  auto create = reinterpret_cast<RgpotEnginePot *(*)(const void *, size_t, char *,
                                                     size_t)>(
      ::dlsym(handle, "rgpot_engine_create"));
  auto destroy =
      reinterpret_cast<void (*)(RgpotEnginePot *)>(::dlsym(handle, "rgpot_engine_destroy"));
  auto force = reinterpret_cast<int (*)(
      RgpotEnginePot *, long, const double *, const int *, double *, double *,
      double *, const double *, rgpot_engine_coord_transform, void *)>(
      ::dlsym(handle, "rgpot_engine_force"));
  auto caps = reinterpret_cast<int (*)(const RgpotEnginePot *, RgpotEngineCaps *)>(
      ::dlsym(handle, "rgpot_engine_caps"));
  auto set_charge = reinterpret_cast<int (*)(RgpotEnginePot *, int, int)>(
      ::dlsym(handle, "rgpot_engine_set_charge_spin"));
  REQUIRE(create != nullptr);
  REQUIRE(destroy != nullptr);
  REQUIRE(force != nullptr);
  REQUIRE(caps != nullptr);
  REQUIRE(set_charge != nullptr);

  FakeOrca fake;
  ::setenv("ORCA_FAKE_REPLAY", replay_dir().c_str(), 1);
  capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::OrcaParams>();
  params.setMethod("HF");
  params.setBasis("def2-SVP");
  params.setExecutable(fake.script.string());
  const auto words = capnp::messageToFlatArray(msg);
  const auto bytes = words.asBytes();
  char err[512] = {};
  RgpotEnginePot *engine =
      create(bytes.begin(), bytes.size(), err, sizeof(err));
  INFO(err);
  REQUIRE(engine != nullptr);
  RgpotEngineCaps got{};
  got.size = sizeof(got);
  REQUIRE(caps(engine, &got) == 0);
  REQUIRE(got.periodic == 0);
  REQUIRE(got.reentrancy == 1);
  REQUIRE(set_charge(engine, -1, 2) == 0);

  const auto pos = recordedWater();
  const int z[3] = {8, 1, 1};
  const double box[9] = {100, 0, 0, 0, 100, 0, 0, 0, 100};
  double forces[9] = {};
  double energy = 0.0;
  const int rc = force(engine, 3, pos.data(), z, forces, &energy, nullptr, box,
                       nullptr, nullptr);
  INFO(err);
  REQUIRE(rc == 0);
  REQUIRE(energy == Approx(kRecordedHa * rgpot::units::HARTREE_TO_EV));
  destroy(engine);
}

TEST_CASE("potserv ORCA replays a recorded engrad and rejects PBC",
          "[orca][rpc]") {
  const char *bin = std::getenv("RGPOT_POTSERV");
  if (bin == nullptr) {
    SKIP("RGPOT_POTSERV is not set");
  }
  FakeOrca fake;
  ::setenv("ORCA_EXE", fake.script.c_str(), 1);
  ::setenv("ORCA_FAKE_REPLAY", replay_dir().c_str(), 1);
  const int port = free_port();
  const auto server_log = fake.dir / "potserv.log";
  Potserv server(server_log, port, "ORCA:HF:def2-SVP:0:1");
  server.wait_until_listening();

  capnp::EzRpcClient client("127.0.0.1", static_cast<uint16_t>(port));
  auto potential = client.getMain<Potential>();
  auto &wait = client.getWaitScope();
  {
    auto request = potential.calculateRequest();
    fill_water(request.initFip());
    auto response = request.send().wait(wait);
    const double energy = response.getResult().getEnergy();
    REQUIRE(energy == Approx(kRecordedHa * rgpot::units::HARTREE_TO_EV));
    REQUIRE(response.getResult().getForces().size() == 9);
  }
  {
    auto request = potential.configureRequest();
    auto config = request.initConfig();
    auto params = config.initOrca();
    params.setMethod("HF");
    params.setBasis("def2-SVP");
    params.setKeywords("TightSCF");
    params.setExecutable(fake.script.string());
    auto response = request.send().wait(wait);
    REQUIRE(response.getOk());
  }
  {
    auto request = potential.calculateRequest();
    fill_water(request.initFip());
    auto response = request.send().wait(wait);
    REQUIRE(response.getResult().getEnergy() ==
            Approx(kRecordedHa * rgpot::units::HARTREE_TO_EV));
  }
  REQUIRE(fake.calls().find("TightSCF") != std::string::npos);

  auto request = potential.calculateRequest();
  auto input = request.initFip();
  auto pos = input.initPos(6);
  pos.set(0, 0.0);
  pos.set(1, 0.0);
  pos.set(2, 0.0);
  pos.set(3, 4.5);
  pos.set(4, 0.0);
  pos.set(5, 0.0);
  auto numbers = input.initAtmnrs(2);
  numbers.set(0, 1);
  numbers.set(1, 1);
  auto box = input.initBox(9);
  const double cell[9] = {5, 1, 0, 0, 5, 0, 0, 0, 5};
  for (unsigned i = 0; i < 9; ++i) {
    box.set(i, cell[i]);
  }
  try {
    (void)request.send().wait(wait);
    FAIL("potserv accepted a periodic cell");
  } catch (const kj::Exception &ex) {
    const std::string msg = ex.getDescription().cStr();
    REQUIRE(msg.find("periodic boundary conditions") != std::string::npos);
  }
}

#endif
