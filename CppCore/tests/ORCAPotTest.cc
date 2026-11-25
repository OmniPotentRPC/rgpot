// MIT License
// Copyright 2023--present rgpot developers
//
// ORCAPot checks. CI has no ORCA, so a fake `orca` shell script, written
// into a temporary directory at run time, stands in for it. The script
// checks the input it is given (keyword line, %maxcore, coordinates, the
// MORead and %moinp lines), keeps a copy of every input, and writes a known
// .engrad file and a .gbw whose text names the call. One mode sleeps in its
// own process group with a child that ignores SIGTERM, so a timeout has to
// kill the group; another exits with a non-zero status. Four modes print
// synthetic SCF report lines modelled on ORCA's wording; they are written
// by the script, not taken from an ORCA run. The tests that need a real
// ORCA skip when `orca` is not on PATH.

#include <array>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestTempDir.hpp"
#include "rgpot/ORCAPot/ORCAFiles.hpp"
#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/units.hpp"
#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

namespace {

// Energy of the fake engrad in Hartree, and the gradient of component k
// (counting from 1) in Hartree/Bohr is kGradientStep * k.
constexpr double kFakeEnergy = -1.25;
constexpr double kGradientStep = 0.01;
constexpr double kUnitTolerance = 1.0e-9;

const char *const kFakeOrca = R"SH(#!/usr/bin/env bash
set -euo pipefail
inp=$1
base=${inp%.inp}
state=$FAKE_ORCA_STATE
n=0
if [[ -f "$state/count" ]]; then n=$(<"$state/count"); fi
n=$((n + 1))
echo "$n" > "$state/count"
cp "$inp" "$state/call$n.inp"
pwd > "$state/call$n.cwd"
fail() { echo "fake orca: $*" >&2; exit 3; }

first=$(head -n 1 "$inp")
last=$(tail -n 1 "$inp")
[[ $first =~ ^!\ .*\ EnGrad ]] || fail "no EnGrad on the keyword line"
grep -Eq '^%maxcore [0-9]+$' "$inp" || fail "no %maxcore line"
grep -Eq '^\* xyz -?[0-9]+ [0-9]+$' "$inp" || fail "no xyz header"
[[ $last == "*" ]] || fail "the xyz block is not closed"

if [[ $first == *" MORead"* ]]; then
  moread=1
  moinp=$(sed -n 's/^%moinp "\(.*\)"$/\1/p' "$inp")
  [[ -n $moinp ]] || fail "MORead without a %moinp line"
  [[ $moinp != "$base.gbw" ]] || fail "%moinp names the file ORCA writes"
  [[ -f $moinp ]] || fail "the %moinp file $moinp is missing"
  [[ $(<"$moinp") == "gbw$((n - 1))" ]] || fail "the guess is not the last gbw"
else
  moread=0
  if grep -q '^%moinp' "$inp"; then fail "%moinp without MORead"; fi
fi
case "${FAKE_ORCA_EXPECT_MOREAD:-any}" in
  1) [[ $moread == 1 ]] || fail "MORead was expected" ;;
  0) [[ $moread == 0 ]] || fail "MORead was not expected" ;;
esac
case "${FAKE_ORCA_EXPECT_CONVFORCED:-any}" in
  1) grep -qx '%scf ConvForced true end' "$inp" || fail "no %scf ConvForced line" ;;
esac

if [[ -n ${FAKE_ORCA_EXPECT_XYZ:-} ]]; then
  awk 'NR == FNR { el[NR] = $1; for (k = 1; k <= 3; k++) v[NR, k] = $(k + 1); n = NR; next }
       /^\* xyz/ { f = 1; next }
       f && /^\*$/ { f = 0 }
       f { i++; if ($1 != el[i]) bad = 1
           for (k = 1; k <= 3; k++) { d = $(k + 1) - v[i, k]; if (d < 0) d = -d; if (d > 1e-9) bad = 1 } }
       END { exit (bad || i != n) }' "$FAKE_ORCA_EXPECT_XYZ" "$inp" \
    || fail "the coordinates differ from the expected ones"
fi

if [[ -n ${FAKE_ORCA_FORBIDDEN_FDS:-} ]]; then
  for fd in $FAKE_ORCA_FORBIDDEN_FDS; do
    if ( : >&"$fd" ) 2>/dev/null; then
      fail "inherited open fd $fd"
    fi
  done
fi

pgid_of() { awk -F') ' '{ split($2, f, " "); print f[3] }' "/proc/$1/stat"; }
mode=${FAKE_ORCA_MODE:-ok}
case "$mode" in
  fail)
    echo $$ > "$state/pid"
    echo "ORCA finished by error termination in SCF"
    echo "STDERR-HEAD-MARKER" >&2
    for ((i = 0; i < 400; i++)); do echo "filler line $i of the ORCA error stream" >&2; done
    echo "STDERR-TAIL-MARKER" >&2
    exit 1 ;;
  hang)
    echo $$ > "$state/pid"
    pgid_of $$ > "$state/pgid"
    trap '' TERM
    (
      trap '' TERM
      echo "$BASHPID" > "$state/child.pid"
      pgid_of "$BASHPID" > "$state/child.pgid"
      exec sleep 300
    ) &
    for _ in $(seq 1 200); do
      [[ -s "$state/child.pgid" ]] && break
      sleep 0.01
    done
    sleep 300
    exit 0 ;;
  hang_leader_exits)
    echo $$ > "$state/pid"
    pgid_of $$ > "$state/pgid"
    (
      trap '' TERM
      echo "$BASHPID" > "$state/child.pid"
      pgid_of "$BASHPID" > "$state/child.pgid"
      exec sleep 300
    ) &
    for _ in $(seq 1 200); do
      [[ -s "$state/child.pgid" ]] && break
      sleep 0.01
    done
    trap 'echo 1 > "$state/leader_exited"; exit 0' TERM
    sleep 300 &
    wait $!
    exit 0 ;;
  killed) kill -9 $$ ;;
  noengrad) exit 0 ;;
  scf_forced_stop)
    echo "fake orca: synthetic output"
    echo "SCF NOT CONVERGED AFTER 125 CYCLES"
    exit 1 ;;
  scf_failed_engrad)
    echo "fake orca: synthetic output"
    echo "SCF NOT CONVERGED AFTER 125 CYCLES" ;;
  scf_near)
    echo "fake orca: synthetic output"
    echo "FINAL SINGLE POINT ENERGY -1.250000000000 (SCF not fully converged!)" ;;
  scf_converged)
    echo "fake orca: synthetic output"
    echo "SCF CONVERGED AFTER  14 CYCLES"
    echo "FINAL SINGLE POINT ENERGY -1.250000000000" ;;
esac

if [[ -z ${FAKE_ORCA_NO_GBW:-} ]]; then echo "gbw$n" > "$base.gbw"; fi

energy=-1.25
if [[ $mode == nan ]]; then energy=nan; fi
awk -v e="$energy" -v mode="$mode" '
  BEGIN { split("H He Li Be B C N O F Ne", sym, " ") }
  /^\* xyz/ { f = 1; next }
  f && /^\*$/ { f = 0 }
  f { n++; for (i = 1; i <= 10; i++) if (sym[i] == $1) z[n] = i
      for (k = 1; k <= 3; k++) c[n, k] = $(k + 1) / 0.529177210903 }
  END {
    print "#\n# Number of atoms\n#\n", n
    if (mode == "truncated") exit
    print "#\n# The current total energy in Eh\n#"
    printf "%.12f\n", e
    print "#\n# The current gradient in Eh/bohr\n#"
    for (k = 1; k <= 3 * n; k++) printf "%.12f\n", 0.01 * k
    print "#\n# The atomic numbers and current coordinates in Bohr\n#"
    for (i = 1; i <= n; i++) {
      zi = z[i]; if (mode == "wrongz") zi = zi + 1
      printf "%d %.12f %.12f %.12f\n", zi, c[i, 1], c[i, 2], c[i, 3]
    }
  }' "$inp" > "$base.engrad"
)SH";

/// Sets an environment variable for one scope and puts the old value back.
class EnvVar {
public:
  EnvVar(const char *name, const std::string &value) : m_name(name) {
    if (const char *old = std::getenv(name)) {
      m_had = true;
      m_old = old;
    }
    ::setenv(name, value.c_str(), 1);
  }
  ~EnvVar() {
    if (m_had) {
      ::setenv(m_name.c_str(), m_old.c_str(), 1);
    } else {
      ::unsetenv(m_name.c_str());
    }
  }
  EnvVar(const EnvVar &) = delete;
  EnvVar &operator=(const EnvVar &) = delete;

private:
  std::string m_name;
  std::string m_old;
  bool m_had = false;
};

bool processGone(pid_t pid) {
  if (pid <= 0) {
    return false;
  }
  for (int i = 0; i < 250; ++i) {
    if (::kill(pid, 0) < 0 && errno == ESRCH) {
      return true;
    }
    timespec ts{};
    ts.tv_nsec = 20'000'000;
    ::nanosleep(&ts, nullptr);
  }
  return false;
}

bool isZombie(pid_t pid) {
  std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
  if (!in) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  const auto close = line.rfind(')');
  return close != std::string::npos && close + 2 < line.size() &&
         line[close + 2] == 'Z';
}

bool groupGone(pid_t pgid) {
  if (pgid <= 1) {
    return false;
  }
  for (int i = 0; i < 250; ++i) {
    if (::kill(-pgid, 0) < 0 && errno == ESRCH) {
      return true;
    }
    timespec ts{};
    ts.tv_nsec = 20'000'000;
    ::nanosleep(&ts, nullptr);
  }
  return false;
}

class FdGuard {
public:
  explicit FdGuard(int fd = -1) : m_fd(fd) {}
  ~FdGuard() { close(); }
  FdGuard(const FdGuard &) = delete;
  FdGuard &operator=(const FdGuard &) = delete;
  int get() const noexcept { return m_fd; }
  void close() noexcept {
    if (m_fd >= 0) {
      ::close(m_fd);
      m_fd = -1;
    }
  }

private:
  int m_fd = -1;
};

class CommaDecimalPunct : public std::numpunct<char> {
protected:
  char do_decimal_point() const override { return ','; }
};

class CxxLocaleGuard {
public:
  explicit CxxLocaleGuard(const std::locale &next)
      : m_saved(std::locale::global(next)) {}
  ~CxxLocaleGuard() { std::locale::global(m_saved); }
  CxxLocaleGuard(const CxxLocaleGuard &) = delete;
  CxxLocaleGuard &operator=(const CxxLocaleGuard &) = delete;

private:
  std::locale m_saved;
};

class CLocaleGuard {
public:
  CLocaleGuard() {
    if (const char *cur = std::setlocale(LC_ALL, nullptr)) {
      m_saved = cur;
    }
  }
  ~CLocaleGuard() {
    if (!m_saved.empty()) {
      std::setlocale(LC_ALL, m_saved.c_str());
    }
  }
  CLocaleGuard(const CLocaleGuard &) = delete;
  CLocaleGuard &operator=(const CLocaleGuard &) = delete;

private:
  std::string m_saved;
};

std::string slurp(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::ostringstream all;
  all << in.rdbuf();
  return all.str();
}

pid_t readPid(const fs::path &path) {
  return static_cast<pid_t>(std::stoi(slurp(path)));
}

/// A directory holding the fake `orca` and the state it records into.
struct Fake {
  rgpot::tests::TestTempDir dir{"rgpot-orca-test"};
  fs::path bin;
  fs::path state;
  fs::path scratch;
  std::string orca;

  Fake() {
    bin = dir.path() / "bin";
    state = dir.path() / "state";
    scratch = dir.path() / "scratch";
    fs::create_directories(bin);
    fs::create_directories(state);
    orca = (bin / "orca").string();
    {
      std::ofstream script(orca, std::ios::binary);
      script << kFakeOrca;
    }
    fs::permissions(orca,
                    fs::perms::owner_all | fs::perms::group_read |
                        fs::perms::group_exec | fs::perms::others_read |
                        fs::perms::others_exec);
  }

  int calls() const {
    const fs::path count = state / "count";
    if (!fs::exists(count)) {
      return 0;
    }
    return std::stoi(slurp(count));
  }

  std::string input(int call) const {
    return slurp(state / ("call" + std::to_string(call) + ".inp"));
  }

  rgpot::ORCAConfig config() const {
    rgpot::ORCAConfig c;
    c.executable = orca;
    c.method = "PBE";
    c.basis = "def2-SVP";
    c.keywords = "TightSCF";
    c.nprocs = 2;
    c.maxcore_mb = 2000;
    c.scratch_dir = scratch.string();
    return c;
  }
};

// Water, with exact decimals so the coordinate rows read back unchanged.
const std::vector<int> kWaterZ = {8, 1, 1};
const std::vector<double> kWater = {0.0, 0.0,   0.1173, 0.0, 0.7572,
                                    -0.4692, 0.0, -0.7572, -0.4692};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
};

Result evaluate(const rgpot::ORCAPot &pot, const std::vector<double> &pos,
                const std::vector<int> &z, const double *box = nullptr) {
  Result r;
  r.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out);
  r.energy = out.energy;
  return r;
}

void writeExpectedXyz(const Fake &fake, const std::vector<int> &z,
                      const std::vector<double> &pos, const char *name) {
  std::ofstream out(fake.state / name);
  for (std::size_t i = 0; i < z.size(); ++i) {
    out << rgpot::orca::elementSymbol(z[i]) << ' ' << pos[3 * i] << ' '
        << pos[3 * i + 1] << ' ' << pos[3 * i + 2] << '\n';
  }
}

} // namespace

TEST_CASE("ORCAPot input carries the method, resources and geometry", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar mode("FAKE_ORCA_MODE", "ok");
  writeExpectedXyz(fake, kWaterZ, kWater, "expected.xyz");
  EnvVar expect("FAKE_ORCA_EXPECT_XYZ", (fake.state / "expected.xyz").string());

  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);

  REQUIRE(fake.calls() == 1);
  const std::string text = fake.input(1);
  REQUIRE(text ==
          "! PBE def2-SVP TightSCF EnGrad\n"
          "%maxcore 2000\n"
          "%pal nprocs 2 end\n"
          "%scf ConvForced true end\n"
          "* xyz 0 1\n"
          "O  0.000000000000 0.000000000000 0.117300000000\n"
          "H  0.000000000000 0.757200000000 -0.469200000000\n"
          "H  0.000000000000 -0.757200000000 -0.469200000000\n"
          "*\n");
  const std::string cwd = slurp(fake.state / "call1.cwd");
  REQUIRE_THAT(cwd, ContainsSubstring(pot.runDirectory()));
}

TEST_CASE("ORCAPot leaves out %pal for one process and sets charge and spin",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAConfig cfg = fake.config();
  cfg.nprocs = 1;
  cfg.basis.clear();
  cfg.keywords.clear();
  cfg.method = "r2SCAN-3c";
  cfg.charge = 1;
  cfg.multiplicity = 2;
  rgpot::ORCAPot pot(cfg);
  evaluate(pot, kWater, kWaterZ);
  const std::string text = fake.input(1);
  REQUIRE_THAT(text, ContainsSubstring("! r2SCAN-3c EnGrad\n"));
  REQUIRE_THAT(text, !ContainsSubstring("%pal"));
  REQUIRE_THAT(text, ContainsSubstring("* xyz 1 2\n"));
}

TEST_CASE("ORCAPot converts the energy and the gradient with units.hpp",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  const Result r = evaluate(pot, kWater, kWaterZ);

  REQUIRE_THAT(r.energy,
               WithinRel(kFakeEnergy * rgpot::units::HARTREE_TO_EV,
                         kUnitTolerance));
  REQUIRE(r.forces.size() == 9);
  for (std::size_t k = 0; k < r.forces.size(); ++k) {
    // The force is the negative gradient, Hartree/Bohr to eV/Angstrom.
    const double gradient = kGradientStep * static_cast<double>(k + 1);
    REQUIRE_THAT(r.forces[k],
                 WithinRel(-gradient * rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM,
                           kUnitTolerance));
  }
}

TEST_CASE("ORCAPot works through the Potential call operator", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  rgpot::types::AtomMatrix pos = rgpot::types::AtomMatrix::Zero(3, 3);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      pos(i, k) = kWater[3 * i + k];
    }
  }
  const std::array<std::array<double, 3>, 3> open_box{};
  const auto [energy, forces, variance] = pot(pos, kWaterZ, open_box);
  REQUIRE_THAT(energy, WithinRel(kFakeEnergy * rgpot::units::HARTREE_TO_EV,
                                 kUnitTolerance));
  REQUIRE(forces.rows() == 3);
  REQUIRE(variance == 0.0);
  REQUIRE(pot.getType() == rgpot::PotType::ORCA);
}

TEST_CASE("ORCAPot reads the previous .gbw through MORead", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());

  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, kWater, kWaterZ);
  }
  REQUIRE_FALSE(pot.reusedWavefunction());
  REQUIRE_THAT(fake.input(1), !ContainsSubstring("MORead"));

  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "1");
    evaluate(pot, kWater, kWaterZ);
    evaluate(pot, kWater, kWaterZ);
  }
  REQUIRE(pot.reusedWavefunction());
  for (int call : {2, 3}) {
    const std::string text = fake.input(call);
    REQUIRE_THAT(text, ContainsSubstring("! PBE def2-SVP TightSCF EnGrad MORead\n"));
    REQUIRE_THAT(text, ContainsSubstring("%moinp \"rgpot_orca_guess.gbw\"\n"));
  }
  // The guess is a copy under a second name, beside ORCA's own output.
  REQUIRE(fs::exists(fs::path(pot.runDirectory()) / "rgpot_orca_guess.gbw"));
  REQUIRE(fs::exists(fs::path(pot.runDirectory()) / "rgpot_orca.gbw"));
  REQUIRE(slurp(fs::path(pot.runDirectory()) / "rgpot_orca_guess.gbw") ==
          "gbw3\n");
}

TEST_CASE("ORCAPot with restart off never writes MORead", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
  rgpot::ORCAConfig cfg = fake.config();
  cfg.restart = false;
  rgpot::ORCAPot pot(cfg);
  evaluate(pot, kWater, kWaterZ);
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fake.calls() == 2);
  REQUIRE_FALSE(pot.reusedWavefunction());
  REQUIRE_FALSE(fs::exists(fs::path(pot.runDirectory()) / "rgpot_orca_guess.gbw"));
}

TEST_CASE("ORCAPot starts afresh when no gbw exists", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());

  {
    EnvVar no_gbw("FAKE_ORCA_NO_GBW", "1");
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, kWater, kWaterZ);
    evaluate(pot, kWater, kWaterZ);
  }
  REQUIRE_FALSE(pot.reusedWavefunction());
  REQUIRE_THAT(fake.input(2), !ContainsSubstring("MORead"));
  REQUIRE_THAT(fake.input(2), !ContainsSubstring("%moinp"));

  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, kWater, kWaterZ); // makes the first gbw
  }
  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "1");
    evaluate(pot, kWater, kWaterZ);
  }
  REQUIRE(pot.reusedWavefunction());
}

TEST_CASE("ORCAPot drops the guess on reset, deletion and a new composition",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  const fs::path guess = fs::path(pot.runDirectory()) / "rgpot_orca_guess.gbw";

  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fs::exists(guess));

  pot.reset();
  REQUIRE_FALSE(fs::exists(guess));
  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, kWater, kWaterZ);
  }
  REQUIRE(fs::exists(guess));

  fs::remove(guess);
  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, kWater, kWaterZ);
  }

  // Ammonia has water's ten electrons but other atoms: the orbitals of the
  // last call do not carry over, and the call after that restarts again.
  const std::vector<int> ammonia_z = {7, 1, 1, 1};
  const std::vector<double> ammonia = {0.0,  0.0,    0.1165, 0.0,
                                       0.9377, -0.3816, 0.8121, -0.4689,
                                       -0.3816, -0.8121, -0.4689, -0.3816};
  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
    evaluate(pot, ammonia, ammonia_z);
  }
  {
    EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "1");
    evaluate(pot, ammonia, ammonia_z);
  }
  REQUIRE(pot.reusedWavefunction());
}

TEST_CASE("ORCAPot refuses a periodic cell and runs for none", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());

  const std::array<double, 9> cell = {10.0, 0.0, 0.0, 0.0, 10.0,
                                      0.0,  0.0, 0.0, 10.0};
  REQUIRE_THROWS_MATCHES(
      evaluate(pot, kWater, kWaterZ, cell.data()), std::invalid_argument,
      Catch::Matchers::MessageMatches(ContainsSubstring("periodic")));
  REQUIRE(fake.calls() == 0);

  const std::array<double, 9> open = {};
  evaluate(pot, kWater, kWaterZ, open.data());
  evaluate(pot, kWater, kWaterZ, nullptr);
  REQUIRE(fake.calls() == 2);
}

TEST_CASE("ORCAPot rejects an impossible spin state", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAConfig cfg = fake.config();
  cfg.multiplicity = 2; // ten electrons cannot hold one unpaired spin
  rgpot::ORCAPot pot(cfg);
  REQUIRE_THROWS_AS(evaluate(pot, kWater, kWaterZ), std::invalid_argument);
  REQUIRE(fake.calls() == 0);

  const std::vector<int> unknown = {8, 1, 119};
  rgpot::ORCAPot plain(fake.config());
  REQUIRE_THROWS_AS(evaluate(plain, kWater, unknown), std::invalid_argument);
}

TEST_CASE("ORCAPot reports a failed run with the tail of stderr", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);

  EnvVar mode("FAKE_ORCA_MODE", "fail");
  try {
    evaluate(pot, kWater, kWaterZ);
    FAIL("a non-zero exit status must throw");
  } catch (const rgpot::ORCAError &e) {
    const std::string message = e.what();
    REQUIRE_THAT(message, ContainsSubstring("status 1"));
    REQUIRE_THAT(message, ContainsSubstring("STDERR-TAIL-MARKER"));
    REQUIRE_THAT(message, !ContainsSubstring("STDERR-HEAD-MARKER"));
    REQUIRE_THAT(message,
                 ContainsSubstring("ORCA finished by error termination in SCF"));
  }
  const pid_t failed = readPid(fake.state / "pid");
  REQUIRE(processGone(failed));
  REQUIRE_FALSE(isZombie(failed));
  // ORCAError is a std::runtime_error, the type the other frontends throw.
  REQUIRE_THROWS_AS(evaluate(pot, kWater, kWaterZ), std::runtime_error);

  // A failure forgets the orbitals: the next call starts fresh.
  EnvVar ok("FAKE_ORCA_MODE", "ok");
  EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
  evaluate(pot, kWater, kWaterZ);
  REQUIRE_FALSE(pot.reusedWavefunction());
}

TEST_CASE("ORCAPot does not take an old engrad for a new result", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fs::exists(fs::path(pot.runDirectory()) / "rgpot_orca.engrad"));

  EnvVar mode("FAKE_ORCA_MODE", "noengrad");
  REQUIRE_THROWS_MATCHES(
      evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
      Catch::Matchers::MessageMatches(ContainsSubstring("without writing")));
}

TEST_CASE("ORCAPot fails when a stale engrad cannot be removed", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fake.calls() == 1);

  const fs::path engrad = fs::path(pot.runDirectory()) / "rgpot_orca.engrad";
  fs::remove(engrad);
  // A non-empty directory cannot be unlinked by fs::remove, even as root.
  fs::create_directories(engrad / "blocker");
  {
    std::ofstream keep(engrad / "blocker" / "keep");
    keep << "keep\n";
  }

  REQUIRE_THROWS_MATCHES(
      evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
      Catch::Matchers::MessageMatches(ContainsSubstring("remove stale")));
  REQUIRE(fake.calls() == 1);

  fs::remove_all(engrad);
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fake.calls() == 2);
  REQUIRE_FALSE(pot.reusedWavefunction());
}

TEST_CASE("ORCAPot rejects a NaN energy, a short or a mismatched engrad",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  {
    EnvVar mode("FAKE_ORCA_MODE", "nan");
    REQUIRE_THROWS_MATCHES(
        evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
        Catch::Matchers::MessageMatches(ContainsSubstring("non-finite")));
  }
  {
    EnvVar mode("FAKE_ORCA_MODE", "truncated");
    REQUIRE_THROWS_MATCHES(
        evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
        Catch::Matchers::MessageMatches(ContainsSubstring("malformed")));
  }
  {
    EnvVar mode("FAKE_ORCA_MODE", "wrongz");
    REQUIRE_THROWS_MATCHES(
        evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
        Catch::Matchers::MessageMatches(ContainsSubstring("atomic numbers")));
  }
}

TEST_CASE("ORCAPot asks ORCA to stop on an unconverged SCF", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar expect("FAKE_ORCA_EXPECT_CONVFORCED", "1");
  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fake.calls() == 2);
}

TEST_CASE("ORCAPot never returns the gradient of an unconverged SCF",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAConfig cfg = fake.config();
  cfg.restart = true;
  rgpot::ORCAPot pot(cfg);
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fs::exists(fs::path(pot.runDirectory()) / "rgpot_orca_guess.gbw"));

  // The forced stop exits non-zero; the other two write a full .engrad.
  for (const char *mode : {"scf_forced_stop", "scf_failed_engrad", "scf_near"}) {
    INFO(mode);
    EnvVar scf("FAKE_ORCA_MODE", mode);
    std::vector<double> forces(kWater.size(), 7.0);
    rgpot::ForceOut out{};
    out.F = forces.data();
    out.energy = 7.0;
    REQUIRE_THROWS_MATCHES(
        pot.forceImpl(rgpot::ForceInput{kWaterZ.size(), kWater.data(),
                                        kWaterZ.data(), nullptr},
                      &out),
        rgpot::ORCAError,
        Catch::Matchers::MessageMatches(
            ContainsSubstring("SCF did not converge")));
    REQUIRE(out.energy == 7.0);
    for (const double f : forces) {
      REQUIRE(f == 7.0);
    }
  }

  // The failed SCF drops the stored orbitals.
  EnvVar ok("FAKE_ORCA_MODE", "ok");
  EnvVar expect("FAKE_ORCA_EXPECT_MOREAD", "0");
  evaluate(pot, kWater, kWaterZ);
  REQUIRE_FALSE(pot.reusedWavefunction());
}

TEST_CASE("ORCAPot takes a converged SCF report as converged", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar mode("FAKE_ORCA_MODE", "scf_converged");
  rgpot::ORCAPot pot(fake.config());
  const Result r = evaluate(pot, kWater, kWaterZ);
  REQUIRE_THAT(r.energy,
               WithinRel(kFakeEnergy * rgpot::units::HARTREE_TO_EV,
                         kUnitTolerance));
}

TEST_CASE("ORCAPot stops a sleeping process group when the timeout passes",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar mode("FAKE_ORCA_MODE", "hang");
  rgpot::ORCAConfig cfg = fake.config();
  cfg.timeout_s = 1.0;
  rgpot::ORCAPot pot(cfg);
  const auto started = std::chrono::steady_clock::now();
  try {
    evaluate(pot, kWater, kWaterZ);
    FAIL("a run past timeout_s must throw");
  } catch (const rgpot::ORCAError &e) {
    const std::string message = e.what();
    REQUIRE_THAT(message, ContainsSubstring("timed out"));
    REQUIRE_THAT(message, ContainsSubstring("process group"));
    REQUIRE_THAT(message, ContainsSubstring("SIGKILL"));
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  // The stand-in sleeps for 300s and ignores SIGTERM. Stopping the group
  // has to finish well inside that, after the 1s budget and the 2s grace.
  REQUIRE(elapsed < 20.0);
  const pid_t shell = readPid(fake.state / "pid");
  const pid_t group = readPid(fake.state / "pgid");
  const pid_t child = readPid(fake.state / "child.pid");
  const pid_t child_group = readPid(fake.state / "child.pgid");
  REQUIRE(group == shell);
  REQUIRE(child != shell);
  REQUIRE(child_group == shell);
  REQUIRE(processGone(shell));
  REQUIRE(processGone(child));
  REQUIRE(groupGone(group));
  REQUIRE_FALSE(isZombie(shell));
  REQUIRE_FALSE(isZombie(child));
}

TEST_CASE("ORCAPot escalates to SIGKILL when the leader exits but a child survives",
          "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar mode("FAKE_ORCA_MODE", "hang_leader_exits");
  rgpot::ORCAConfig cfg = fake.config();
  cfg.timeout_s = 1.0;
  rgpot::ORCAPot pot(cfg);
  const auto started = std::chrono::steady_clock::now();
  try {
    evaluate(pot, kWater, kWaterZ);
    FAIL("a run past timeout_s must throw");
  } catch (const rgpot::ORCAError &e) {
    const std::string message = e.what();
    REQUIRE_THAT(message, ContainsSubstring("timed out"));
    REQUIRE_THAT(message, ContainsSubstring("process group"));
    REQUIRE_THAT(message, ContainsSubstring("SIGKILL"));
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  REQUIRE(elapsed < 20.0);
  const pid_t shell = readPid(fake.state / "pid");
  const pid_t group = readPid(fake.state / "pgid");
  const pid_t child = readPid(fake.state / "child.pid");
  const pid_t child_group = readPid(fake.state / "child.pgid");
  REQUIRE(group == shell);
  REQUIRE(child != shell);
  REQUIRE(child_group == shell);
  REQUIRE(processGone(shell));
  REQUIRE(processGone(child));
  REQUIRE(groupGone(group));
  REQUIRE(fs::exists(fake.state / "leader_exited"));
  REQUIRE_FALSE(isZombie(shell));
  REQUIRE_FALSE(isZombie(child));
}

TEST_CASE("ORCAPot closes extra non-CLOEXEC file descriptors before exec",
          "[orca]") {
  int fds[2] = {-1, -1};
  REQUIRE(::pipe(fds) == 0);
  FdGuard read_end(fds[0]);
  FdGuard write_end(fds[1]);
  REQUIRE(write_end.get() > 2);
  REQUIRE(::fcntl(write_end.get(), F_GETFD) == 0);
  REQUIRE(::fcntl(read_end.get(), F_SETFL, O_NONBLOCK) == 0);

  FdGuard high_write(::dup2(write_end.get(), 42));
  REQUIRE(high_write.get() == 42);
  REQUIRE(::fcntl(high_write.get(), F_GETFD) == 0);

  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar forbidden("FAKE_ORCA_FORBIDDEN_FDS",
                   std::to_string(write_end.get()) + " " +
                       std::to_string(high_write.get()));
  rgpot::ORCAPot pot(fake.config());
  evaluate(pot, kWater, kWaterZ);
  REQUIRE(fake.calls() == 1);

  char buf[16];
  errno = 0;
  REQUIRE(::read(read_end.get(), buf, sizeof(buf)) == -1);
  REQUIRE(errno == EAGAIN);
}

TEST_CASE("ORCAPot reports a signal", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  EnvVar mode("FAKE_ORCA_MODE", "killed");
  rgpot::ORCAPot pot(fake.config());
  REQUIRE_THROWS_MATCHES(
      evaluate(pot, kWater, kWaterZ), rgpot::ORCAError,
      Catch::Matchers::MessageMatches(ContainsSubstring("signal 9")));
}

TEST_CASE("ORCAPot finds the executable by name on PATH", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAConfig cfg = fake.config();
  cfg.executable.clear();

  const char *inherited = std::getenv("PATH");
  const std::string with_fake =
      fake.bin.string() + ":" + (inherited != nullptr ? inherited : "");
  {
    EnvVar path("PATH", with_fake);
    REQUIRE(rgpot::ORCAPot::probeAvailable());
    rgpot::ORCAPot pot(cfg);
    REQUIRE(pot.executablePath() == (fake.bin / "orca").string());
    evaluate(pot, kWater, kWaterZ);
  }
  {
    EnvVar path("PATH", fake.state.string());
    REQUIRE_FALSE(rgpot::ORCAPot::probeAvailable());
    REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), rgpot::ORCAError);
  }
  cfg.executable = (fake.bin / "no-such-orca").string();
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), rgpot::ORCAError);
}

TEST_CASE("ORCAPot accepts a symlink to an executable", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  const fs::path links = fake.dir.path() / "links";
  fs::create_directories(links);
  const fs::path link_orca = links / "orca";
  const fs::path link_broken = links / "broken";
  const fs::path link_dir = links / "dirlink";
  fs::create_symlink(fake.orca, link_orca);
  fs::create_symlink(fake.dir.path() / "does-not-exist", link_broken);
  fs::create_symlink(fake.state, link_dir);

  REQUIRE(rgpot::ORCAPot::probeAvailable(link_orca.string()));
  REQUIRE_FALSE(rgpot::ORCAPot::probeAvailable(link_broken.string()));
  REQUIRE_FALSE(rgpot::ORCAPot::probeAvailable(link_dir.string()));

  rgpot::ORCAConfig cfg = fake.config();
  cfg.executable = link_orca.string();
  rgpot::ORCAPot direct(cfg);
  evaluate(direct, kWater, kWaterZ);
  REQUIRE(fake.calls() == 1);

  cfg.executable.clear();
  const char *inherited = std::getenv("PATH");
  const std::string with_links =
      links.string() + ":" + (inherited != nullptr ? inherited : "");
  EnvVar path("PATH", with_links);
  REQUIRE(rgpot::ORCAPot::probeAvailable());
  rgpot::ORCAPot via_path(cfg);
  REQUIRE(via_path.executablePath() == link_orca.string());
  evaluate(via_path, kWater, kWaterZ);
  REQUIRE(fake.calls() == 2);

  cfg.executable = link_broken.string();
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), rgpot::ORCAError);
  cfg.executable = link_dir.string();
  REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), rgpot::ORCAError);
}

TEST_CASE("ORCAPot rejects a config that cannot run", "[orca]") {
  Fake fake;
  const auto bad = [&](auto edit) {
    rgpot::ORCAConfig cfg = fake.config();
    edit(cfg);
    REQUIRE_THROWS_AS(rgpot::ORCAPot(cfg), std::invalid_argument);
  };
  bad([](rgpot::ORCAConfig &c) { c.method.clear(); });
  bad([](rgpot::ORCAConfig &c) { c.keywords = "TightSCF\n%pal nprocs 64 end"; });
  bad([](rgpot::ORCAConfig &c) { c.basis = "def2-SVP\r"; });
  bad([](rgpot::ORCAConfig &c) { c.nprocs = 0; });
  bad([](rgpot::ORCAConfig &c) { c.maxcore_mb = 0; });
  bad([](rgpot::ORCAConfig &c) { c.multiplicity = 0; });
  bad([](rgpot::ORCAConfig &c) { c.timeout_s = -0.1; });
  bad([](rgpot::ORCAConfig &c) {
    c.timeout_s = std::numeric_limits<double>::quiet_NaN();
  });
  bad([](rgpot::ORCAConfig &c) {
    c.timeout_s = std::numeric_limits<double>::infinity();
  });
  REQUIRE(rgpot::ORCAConfig{}.timeout_s == rgpot::ORCAConfig::kDefaultTimeout);
  REQUIRE(rgpot::ORCAConfig::kDefaultTimeout == 3600.0);
  rgpot::ORCAConfig unlimited = fake.config();
  unlimited.timeout_s = 0.0;
  REQUIRE_NOTHROW(rgpot::ORCAPot(unlimited));
}

TEST_CASE("ORCAPot advertises a per-instance, open-boundary kernel", "[orca]") {
  Fake fake;
  rgpot::ORCAPot pot(fake.config());
  const rgpot::PotCaps caps = pot.caps();
  REQUIRE(caps.reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE_FALSE(caps.periodic);
  REQUIRE_FALSE(caps.stress);

  rgpot::ORCAConfig other = fake.config();
  other.method = "B3LYP";
  rgpot::ORCAPot different(other);
  REQUIRE(pot.paramsKey() != different.paramsKey());
  rgpot::ORCAConfig more_cores = fake.config();
  more_cores.nprocs = 8;
  REQUIRE(rgpot::ORCAPot(more_cores).paramsKey() == pot.paramsKey());
  rgpot::ORCAConfig longer = fake.config();
  longer.timeout_s = 10.0;
  REQUIRE(rgpot::ORCAPot(longer).paramsKey() == pot.paramsKey());
}

TEST_CASE("ORCAPot keys results on the executable, charge and spin", "[orca]") {
  Fake fake;
  const uint64_t key = rgpot::ORCAPot(fake.config()).paramsKey();
  REQUIRE(rgpot::ORCAPot(fake.config()).paramsKey() == key);

  // A copy elsewhere is another installation.
  const fs::path other_bin = fake.dir.path() / "other-bin";
  fs::create_directories(other_bin);
  fs::copy_file(fake.orca, other_bin / "orca");
  fs::last_write_time(other_bin / "orca", fs::last_write_time(fake.orca));
  rgpot::ORCAConfig elsewhere = fake.config();
  elsewhere.executable = (other_bin / "orca").string();
  REQUIRE(rgpot::ORCAPot(elsewhere).paramsKey() != key);

  // A symlink runs the same file, so it shares the key.
  const fs::path link = fake.dir.path() / "orca-link";
  fs::create_symlink(fake.orca, link);
  rgpot::ORCAConfig linked = fake.config();
  linked.executable = link.string();
  rgpot::ORCAPot via_link(linked);
  REQUIRE(via_link.executablePath() == link.string());
  REQUIRE(via_link.paramsKey() == key);

  rgpot::ORCAConfig cation = fake.config();
  cation.charge = 1;
  cation.multiplicity = 2;
  REQUIRE(rgpot::ORCAPot(cation).paramsKey() != key);
  rgpot::ORCAConfig triplet = fake.config();
  triplet.multiplicity = 3;
  REQUIRE(rgpot::ORCAPot(triplet).paramsKey() != key);
  rgpot::ORCAConfig anion = fake.config();
  anion.charge = -1;
  anion.multiplicity = 2;
  REQUIRE(rgpot::ORCAPot(anion).paramsKey() !=
          rgpot::ORCAPot(cation).paramsKey());
}

TEST_CASE("ORCAPot keys results on the executable modification time",
          "[orca]") {
  Fake fake;
  const uint64_t before = rgpot::ORCAPot(fake.config()).paramsKey();
  fs::last_write_time(fake.orca,
                      fs::last_write_time(fake.orca) + std::chrono::seconds(5));
  REQUIRE(rgpot::ORCAPot(fake.config()).paramsKey() != before);
}

#ifdef RGPOT_HAS_CACHE
TEST_CASE("ORCAPot cache entries do not cross installations or spin states",
          "[orca][cache]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::tests::TestTempDir db("rgpot-orca-cache");
  rgpot::cache::PotentialCache cache((db.path() / "db").string());
  rgpot::types::AtomMatrix pos = rgpot::types::AtomMatrix::Zero(3, 3);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      pos(i, k) = kWater[3 * i + k];
    }
  }
  const std::array<std::array<double, 3>, 3> open_box{};
  const auto run = [&](const rgpot::ORCAConfig &cfg) {
    rgpot::ORCAPot pot(cfg);
    pot.setCache(&cache);
    (void)pot(pos, kWaterZ, open_box);
  };

  run(fake.config());
  REQUIRE(fake.calls() == 1);
  run(fake.config());
  REQUIRE(fake.calls() == 1);
  fs::last_write_time(fake.orca,
                      fs::last_write_time(fake.orca) + std::chrono::seconds(5));
  run(fake.config());
  REQUIRE(fake.calls() == 2);
  rgpot::ORCAConfig cation = fake.config();
  cation.charge = 1;
  cation.multiplicity = 2;
  run(cation);
  REQUIRE(fake.calls() == 3);
  rgpot::ORCAConfig triplet = fake.config();
  triplet.multiplicity = 3;
  run(triplet);
  REQUIRE(fake.calls() == 4);
}
#endif

TEST_CASE("ORCAPot instances keep separate working directories", "[orca]") {
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  std::string kept;
  {
    rgpot::ORCAPot a(fake.config());
    rgpot::ORCAPot b(fake.config());
    REQUIRE(a.runDirectory() != b.runDirectory());
    REQUIRE(fs::path(a.runDirectory()).parent_path() == fake.scratch);
    evaluate(a, kWater, kWaterZ);
    REQUIRE(fs::exists(fs::path(a.runDirectory()) / "rgpot_orca.inp"));
    REQUIRE_FALSE(fs::exists(fs::path(b.runDirectory()) / "rgpot_orca.inp"));
    kept = a.runDirectory();
  }
  REQUIRE_FALSE(fs::exists(kept));

  rgpot::ORCAConfig cfg = fake.config();
  cfg.keep_files = true;
  {
    rgpot::ORCAPot pot(cfg);
    evaluate(pot, kWater, kWaterZ);
    kept = pot.runDirectory();
  }
  REQUIRE(fs::exists(fs::path(kept) / "rgpot_orca.engrad"));
}

TEST_CASE("parseEngrad reads the layout ORCA writes", "[orca]") {
  const std::string text =
      "#\n# Number of atoms\n#\n 2\n#\n# The current total energy in Eh\n#\n"
      "   -2.500000000000\n#\n# The current gradient in Eh/bohr\n#\n"
      "  0.1\n  0.2\n  0.3\n -0.1\n -0.2\n -0.3\n"
      "#\n# The atomic numbers and current coordinates in Bohr\n#\n"
      "   1    0.0 0.0 0.0\n   1    0.0 0.0 1.4\n";
  const auto engrad = rgpot::orca::parseEngrad(text, 2);
  REQUIRE(engrad.energy_hartree == -2.5);
  REQUIRE(engrad.gradient_hartree_per_bohr.size() == 6);
  REQUIRE(engrad.gradient_hartree_per_bohr[5] == -0.3);
  REQUIRE(engrad.atomic_numbers == std::vector<int>{1, 1});
  REQUIRE_THROWS_AS(rgpot::orca::parseEngrad(text, 3),
                    std::runtime_error);
  REQUIRE_THROWS_AS(rgpot::orca::parseEngrad("", 2),
                    std::runtime_error);
  REQUIRE_THROWS_AS(rgpot::orca::parseEngrad(text + " 7\n", 2),
                    std::runtime_error);
}

TEST_CASE("ORCAPot input and engrad ignore the C++ global locale", "[orca]") {
  CxxLocaleGuard guard(std::locale(std::locale::classic(), new CommaDecimalPunct));
  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  const Result r = evaluate(pot, kWater, kWaterZ);
  REQUIRE_THAT(r.energy,
               WithinRel(kFakeEnergy * rgpot::units::HARTREE_TO_EV,
                         kUnitTolerance));
  REQUIRE_THAT(fake.input(1), ContainsSubstring("0.117300000000"));
  REQUIRE_THAT(fake.input(1), !ContainsSubstring(","));

  const std::string comma_engrad =
      "1\n-2,5\n0,1\n0,2\n0,3\n1 0,0 0,0 0,0\n";
  REQUIRE_THROWS_AS(rgpot::orca::parseEngrad(comma_engrad, 1),
                    std::runtime_error);
}

TEST_CASE("ORCAPot input and engrad ignore a comma-decimal C locale",
          "[orca]") {
  CLocaleGuard guard;
  bool found = false;
  for (const char *cand :
       {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8",
        "es_ES.UTF-8", "es_ES.utf8", "it_IT.UTF-8", "pt_BR.UTF-8"}) {
    if (std::setlocale(LC_ALL, cand) != nullptr) {
      const std::lconv *conv = std::localeconv();
      if (conv != nullptr && conv->decimal_point != nullptr &&
          conv->decimal_point[0] == ',') {
        found = true;
        break;
      }
    }
  }
  if (!found) {
    SKIP("no comma-decimal C locale is installed");
  }

  Fake fake;
  EnvVar state("FAKE_ORCA_STATE", fake.state.string());
  rgpot::ORCAPot pot(fake.config());
  const Result r = evaluate(pot, kWater, kWaterZ);
  REQUIRE_THAT(r.energy,
               WithinRel(kFakeEnergy * rgpot::units::HARTREE_TO_EV,
                         kUnitTolerance));
  REQUIRE_THAT(fake.input(1), ContainsSubstring("0.117300000000"));
  REQUIRE_THAT(fake.input(1), !ContainsSubstring(","));

  const std::string comma_engrad =
      "1\n-2,5\n0,1\n0,2\n0,3\n1 0,0 0,0 0,0\n";
  REQUIRE_THROWS_AS(rgpot::orca::parseEngrad(comma_engrad, 1),
                    std::runtime_error);
}

TEST_CASE("ORCAPot against a real ORCA", "[orca][real]") {
  if (!rgpot::ORCAPot::probeAvailable()) {
    SKIP("orca is not on PATH");
  }
  rgpot::tests::TestTempDir scratch("rgpot-orca-real");
  rgpot::ORCAConfig cfg;
  cfg.method = "PBE";
  cfg.basis = "def2-SVP";
  cfg.keywords = "TightSCF";
  cfg.scratch_dir = scratch.string();
  rgpot::ORCAPot pot(cfg);

  const Result first = evaluate(pot, kWater, kWaterZ);
  REQUIRE(std::isfinite(first.energy));
  REQUIRE(first.energy < 0.0);
  double net[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      net[k] += first.forces[3 * i + k];
    }
  }
  // Translational invariance: the forces on an isolated molecule sum to
  // zero.
  for (const double component : net) {
    REQUIRE_THAT(component, Catch::Matchers::WithinAbs(0.0, 1.0e-4));
  }

  const Result second = evaluate(pot, kWater, kWaterZ);
  REQUIRE(pot.reusedWavefunction());
  REQUIRE_THAT(second.energy, WithinRel(first.energy, 1.0e-6));
}
