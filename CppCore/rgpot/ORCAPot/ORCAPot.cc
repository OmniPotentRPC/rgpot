// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAPot.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rgpot {
namespace {

constexpr const char *kJob = "rgpot";

const char *element_symbol(int z) {
  static constexpr const char *kSymbols[] = {
      nullptr, "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
      "Mg",    "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
      "Cr",    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
      "Kr",    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
      "Cd",    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd",    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
      "Hf",    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
      "Po",    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
      "Cm",    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
      "Hs",    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
  static_assert(std::size(kSymbols) == 119);
  if (z <= 0 || static_cast<std::size_t>(z) >= std::size(kSymbols)) {
    throw std::invalid_argument("ORCAPot: atomic number " + std::to_string(z) +
                                " has no element symbol");
  }
  return kSymbols[z];
}

void validate(const ORCAConfig &config) {
  if (config.method.empty()) {
    throw std::invalid_argument("ORCAPot: method is empty");
  }
  if (config.multiplicity < 1) {
    throw std::invalid_argument("ORCAPot: multiplicity must be >= 1");
  }
}

bool executable_file(const std::string &path) {
  return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

std::string search_path(const char *name) {
  const char *raw = std::getenv("PATH");
  if (raw == nullptr || raw[0] == '\0') {
    return {};
  }
  std::string_view path(raw);
  while (!path.empty()) {
    const auto colon = path.find(':');
    std::string dir(path.substr(0, colon));
    if (dir.empty()) {
      dir = ".";
    }
    const std::string candidate = dir + "/" + name;
    if (executable_file(candidate)) {
      return candidate;
    }
    if (colon == std::string_view::npos) {
      break;
    }
    path.remove_prefix(colon + 1);
  }
  return {};
}

void append_tail(std::string *text, const char *data, std::size_t n) {
  constexpr std::size_t kKeep = 4096;
  text->append(data, n);
  if (text->size() > kKeep) {
    text->erase(0, text->size() - kKeep);
  }
}

struct ChildResult {
  int code = -1;
  std::string stderr_text;
};

ChildResult run_process(const std::string &exe, const std::string &workdir,
                        const char *inp_name) {
  int pipefd[2];
  if (::pipe(pipefd) != 0) {
    throw std::runtime_error("ORCAPot: cannot create a pipe for ORCA stderr");
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    throw std::runtime_error("ORCAPot: fork failed");
  }
  if (pid == 0) {
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    const int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDOUT_FILENO);
      ::close(devnull);
    }
    if (::chdir(workdir.c_str()) != 0) {
      _exit(127);
    }
    ::execl(exe.c_str(), exe.c_str(), inp_name, static_cast<char *>(nullptr));
    _exit(127);
  }
  ::close(pipefd[1]);
  ChildResult result;
  char buf[4096];
  for (;;) {
    const ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
    if (n > 0) {
      append_tail(&result.stderr_text, buf, static_cast<std::size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
  ::close(pipefd[0]);
  int status = 0;
  for (;;) {
    const pid_t waited = ::waitpid(pid, &status, 0);
    if (waited < 0 && errno == EINTR) {
      continue;
    }
    if (waited < 0) {
      throw std::runtime_error("ORCAPot: waitpid failed");
    }
    break;
  }
  if (WIFEXITED(status)) {
    result.code = WEXITSTATUS(status);
  }
  return result;
}

bool parse_number(const std::string &line, double *out) {
  const char *begin = line.c_str();
  char *end = nullptr;
  *out = std::strtod(begin, &end);
  if (end == begin) {
    return false;
  }
  while (*end != '\0' &&
         std::isspace(static_cast<unsigned char>(*end)) != 0) {
    ++end;
  }
  return *end == '\0';
}

void parse_engrad(const std::filesystem::path &path, std::size_t n_atoms,
                  double *energy, std::vector<double> *gradient) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("ORCAPot: ORCA did not write " + path.string());
  }
  enum class Section { None, Atoms, Energy, Gradient };
  Section section = Section::None;
  int reported = -1;
  bool have_energy = false;
  gradient->clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("Number of atoms") != std::string::npos) {
      section = Section::Atoms;
      continue;
    }
    if (line.find("total energy") != std::string::npos) {
      section = Section::Energy;
      continue;
    }
    if (line.find("current gradient") != std::string::npos) {
      section = Section::Gradient;
      continue;
    }
    if (line.find("atomic numbers") != std::string::npos) {
      section = Section::None;
      continue;
    }
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    double value = 0.0;
    if (!parse_number(line.substr(first), &value)) {
      continue;
    }
    if (section == Section::Atoms && reported < 0) {
      reported = static_cast<int>(value);
    } else if (section == Section::Energy && !have_energy) {
      *energy = value;
      have_energy = true;
    } else if (section == Section::Gradient &&
               reported > 0 &&
               gradient->size() < static_cast<std::size_t>(reported) * 3u) {
      gradient->push_back(value);
    }
  }
  if (reported < 0 || !have_energy) {
    throw std::runtime_error("ORCAPot: cannot read energy from " +
                             path.string());
  }
  if (static_cast<std::size_t>(reported) != n_atoms) {
    throw std::runtime_error(
        "ORCAPot: .engrad atom count " + std::to_string(reported) +
        " does not match the geometry (" + std::to_string(n_atoms) + ")");
  }
  if (gradient->size() != n_atoms * 3u) {
    throw std::runtime_error("ORCAPot: .engrad gradient length " +
                             std::to_string(gradient->size()) +
                             " is not 3 times the atom count");
  }
}

double cell_determinant(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) -
         h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

// Rows of `box` are the lattice vectors. A pair folds when its cartesian
// displacement, written in that basis, has a fractional component outside
// [-1/2, 1/2]. A singular cell has no images.
bool folds_periodic_image(const double *box, const double *pos,
                          std::size_t n_atoms) {
  const double det = cell_determinant(box);
  if (!std::isfinite(det) || std::abs(det) <= 1e-12) {
    return false;
  }
  const double invdet = 1.0 / det;
  const double h0 = box[0], h1 = box[1], h2 = box[2];
  const double h3 = box[3], h4 = box[4], h5 = box[5];
  const double h6 = box[6], h7 = box[7], h8 = box[8];
  const double inv[9] = {
      (h4 * h8 - h5 * h7) * invdet, (h2 * h7 - h1 * h8) * invdet,
      (h1 * h5 - h2 * h4) * invdet, (h5 * h6 - h3 * h8) * invdet,
      (h0 * h8 - h2 * h6) * invdet, (h2 * h3 - h0 * h5) * invdet,
      (h3 * h7 - h4 * h6) * invdet, (h1 * h6 - h0 * h7) * invdet,
      (h0 * h4 - h1 * h3) * invdet};
  constexpr double kHalf = 0.5 + 1e-8;
  for (std::size_t j = 1; j < n_atoms; ++j) {
    const double *pj = pos + j * 3;
    for (std::size_t i = 0; i < j; ++i) {
      const double *pi = pos + i * 3;
      const double dx = pj[0] - pi[0];
      const double dy = pj[1] - pi[1];
      const double dz = pj[2] - pi[2];
      const double sx = dx * inv[0] + dy * inv[3] + dz * inv[6];
      const double sy = dx * inv[1] + dy * inv[4] + dz * inv[7];
      const double sz = dx * inv[2] + dy * inv[5] + dz * inv[8];
      if (std::abs(sx) > kHalf || std::abs(sy) > kHalf ||
          std::abs(sz) > kHalf) {
        return true;
      }
    }
  }
  return false;
}

std::string failure_text(int code, const std::string &stderr_text,
                         const std::string &detail) {
  std::ostringstream msg;
  msg << "ORCA failed";
  if (code >= 0) {
    msg << " (exit " << code << ")";
  }
  if (!detail.empty()) {
    msg << ": " << detail;
  }
  if (!stderr_text.empty()) {
    msg << ": " << stderr_text;
  }
  return msg.str();
}

} // namespace

std::string ORCAPot::locate(const std::string &executable) {
  if (!executable.empty()) {
    if (!executable_file(executable)) {
      throw std::runtime_error("ORCA executable not found: " + executable);
    }
    return executable;
  }
  if (const char *env = std::getenv("ORCA_EXE")) {
    if (env[0] != '\0') {
      if (!executable_file(env)) {
        throw std::runtime_error(
            "ORCA_EXE is set but is not an executable file: " +
            std::string(env));
      }
      return env;
    }
  }
  if (const std::string found = search_path("orca"); !found.empty()) {
    return found;
  }
  throw std::runtime_error(
      "ORCA executable not found. Set the executable in the config, set "
      "ORCA_EXE, or put orca on PATH.");
}

bool ORCAPot::available(const std::string &executable) {
  try {
    (void)locate(executable);
    return true;
  } catch (const std::runtime_error &) {
    return false;
  }
}

void ORCAPot::rehash() {
  Fnv1a hash;
  hash.u64(1);
  hash.str(config_.method);
  hash.str(config_.basis);
  hash.i64(config_.charge);
  hash.i64(config_.multiplicity);
  hash.str(config_.keywords);
  hash.str(config_.executable);
  params_key_.store(hash.h, std::memory_order_relaxed);
}

void ORCAPot::dropWavefunction() const {
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(work_) / (std::string(kJob) + ".gbw"),
                          ec);
  std::filesystem::remove(
      std::filesystem::path(work_) / (std::string(kJob) + ".moinp.gbw"), ec);
}

ORCAPot::ORCAPot(const Config &config)
    : Potential(PotType::ORCA), config_(config) {
  validate(config_);
  std::string tmpl =
      (std::filesystem::temp_directory_path() / "rgpot-orca-XXXXXX").string();
  if (::mkdtemp(tmpl.data()) == nullptr) {
    throw std::runtime_error("ORCAPot: cannot create a work directory");
  }
  work_ = std::move(tmpl);
  rehash();
}

ORCAPot::~ORCAPot() {
  std::error_code ec;
  std::filesystem::remove_all(work_, ec);
}

void ORCAPot::applyParams(const Config &config) {
  validate(config);
  std::lock_guard<std::mutex> lock(mu_);
  const bool same = config.method == config_.method &&
                    config.basis == config_.basis &&
                    config.charge == config_.charge &&
                    config.multiplicity == config_.multiplicity &&
                    config.keywords == config_.keywords;
  config_ = config;
  rehash();
  if (!same) {
    dropWavefunction();
  }
}

void ORCAPot::setChargeSpin(int charge, int multiplicity) {
  if (multiplicity < 1) {
    throw std::invalid_argument("ORCAPot: multiplicity must be >= 1");
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (charge != config_.charge || multiplicity != config_.multiplicity) {
    dropWavefunction();
  }
  config_.charge = charge;
  config_.multiplicity = multiplicity;
  rehash();
}

void ORCAPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (in.nAtoms == 0) {
    throw std::invalid_argument("ORCAPot: nAtoms must be positive");
  }
  if (in.pos == nullptr || in.atmnrs == nullptr || out == nullptr ||
      out->F == nullptr) {
    throw std::invalid_argument("ORCAPot: null positions, atomic numbers, or forces");
  }
  std::vector<const char *> symbols(in.nAtoms);
  for (std::size_t i = 0; i < in.nAtoms; ++i) {
    symbols[i] = element_symbol(in.atmnrs[i]);
  }
  if (in.box != nullptr &&
      folds_periodic_image(in.box, in.pos, in.nAtoms)) {
    throw std::invalid_argument(
        "ORCAPot handles clusters only and does not support periodic "
        "boundary conditions");
  }

  std::lock_guard<std::mutex> lock(mu_);
  const std::string exe = locate(config_.executable);
  const std::filesystem::path work(work_);
  const std::filesystem::path inp = work / (std::string(kJob) + ".inp");
  const std::filesystem::path engrad = work / (std::string(kJob) + ".engrad");
  const std::filesystem::path gbw = work / (std::string(kJob) + ".gbw");
  const auto moinp_name = std::string(kJob) + ".moinp.gbw";
  const std::filesystem::path moinp = work / moinp_name;

  const bool had_gbw = std::filesystem::is_regular_file(gbw);
  bool copied = false;
  std::string last_error;
  for (int attempt = 0; attempt < (had_gbw ? 2 : 1); ++attempt) {
    const bool use_moread = had_gbw && attempt == 0 && !copied;
    if (had_gbw && attempt == 0) {
      std::error_code ec;
      std::filesystem::copy_file(
          gbw, moinp, std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        continue;
      }
      copied = true;
    }
    {
      std::ofstream file(inp);
      if (!file) {
        throw std::runtime_error("ORCAPot: cannot write " + inp.string());
      }
      std::string simple = config_.keywords;
      std::string blocks;
      const auto nl = simple.find('\n');
      if (nl != std::string::npos) {
        blocks = simple.substr(nl + 1);
        simple.resize(nl);
      }
      file << "! " << config_.method;
      if (!config_.basis.empty()) {
        file << ' ' << config_.basis;
      }
      if (!simple.empty()) {
        file << ' ' << simple;
      }
      file << " EnGrad";
      if (use_moread) {
        file << " MORead";
      }
      file << '\n';
      if (!blocks.empty()) {
        file << blocks;
        if (blocks.back() != '\n') {
          file << '\n';
        }
      }
      if (use_moread) {
        file << "%moinp \"" << moinp_name << "\"\n";
      }
      file << "* xyz " << config_.charge << ' ' << config_.multiplicity << '\n';
      file << std::setprecision(16);
      for (std::size_t atom = 0; atom < in.nAtoms; ++atom) {
        const double *xyz = in.pos + atom * 3;
        file << symbols[atom] << ' ' << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2]
             << '\n';
      }
      file << "*\n";
      if (!file) {
        throw std::runtime_error("ORCAPot: cannot write " + inp.string());
      }
    }

    std::error_code ec;
    std::filesystem::remove(engrad, ec);
    const ChildResult child = run_process(exe, work_, (std::string(kJob) + ".inp").c_str());
    if (child.code == 0) {
      try {
        double energy_ha = 0.0;
        std::vector<double> gradient;
        parse_engrad(engrad, in.nAtoms, &energy_ha, &gradient);
        out->energy = energy_ha * units::HARTREE_TO_EV;
        out->variance = 0.0;
        out->has_stress = 0;
        for (std::size_t i = 0; i < gradient.size(); ++i) {
          out->F[i] = gradient[i] * units::NEG_GRAD_TO_FORCE;
        }
        return;
      } catch (const std::exception &ex) {
        last_error = failure_text(child.code, child.stderr_text, ex.what());
      }
    } else {
      last_error = failure_text(child.code, child.stderr_text, "");
    }

    if (!use_moread) {
      if (!had_gbw) {
        std::filesystem::remove(gbw, ec);
      } else if (copied) {
        std::filesystem::copy_file(
            moinp, gbw, std::filesystem::copy_options::overwrite_existing, ec);
      }
      throw std::runtime_error(last_error);
    }
    std::filesystem::copy_file(
        moinp, gbw, std::filesystem::copy_options::overwrite_existing, ec);
  }
  throw std::runtime_error(last_error.empty() ? "ORCA failed" : last_error);
}

} // namespace rgpot
