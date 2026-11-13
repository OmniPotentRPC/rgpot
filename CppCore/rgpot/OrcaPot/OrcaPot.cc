// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/OrcaPot/OrcaPot.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace rgpot {

namespace {

// CODATA 2018 (Tiesinga, Mohr, Newell and Taylor, Rev. Mod. Phys. 93,
// 025010, 2021). The numeric values live in units.hpp: hartree in eV,
// and the bohr radius in angstrom. A force is the negated gradient.
constexpr double kEhToEv = units::HARTREE_TO_EV;
constexpr double kGradEhBohrToForceEvA = units::NEG_GRAD_TO_FORCE;

constexpr const char *kSymbols[] = {
    nullptr, "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg",    "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr",    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr",    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd",    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd",    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf",    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po",    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm",    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs",    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};

constexpr const char *kJob = "job";
constexpr const char *kMoinp = "job.prev.gbw";

std::string_view trim(std::string_view s) {
  while (!s.empty() &&
         std::isspace(static_cast<unsigned char>(s.front())) != 0) {
    s.remove_prefix(1);
  }
  while (!s.empty() &&
         std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.remove_suffix(1);
  }
  return s;
}

std::string trimCopy(std::string_view s) { return std::string(trim(s)); }

[[noreturn]] void malformed(const std::string &why) {
  throw std::runtime_error("malformed ORCA engrad: " + why);
}

const char *symbolFor(int z) {
  const std::size_t nsym = sizeof(kSymbols) / sizeof(kSymbols[0]);
  if (z < 1 || static_cast<std::size_t>(z) >= nsym || kSymbols[z] == nullptr) {
    throw std::invalid_argument("ORCA input: no symbol for atomic number " +
                                std::to_string(z));
  }
  return kSymbols[z];
}

bool isExecutable(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return false;
  }
#ifndef _WIN32
  return ::access(path.c_str(), X_OK) == 0;
#else
  (void)path;
  return true;
#endif
}

std::string missingBinary(const std::string &spec) {
  return "ORCA binary not found: '" + spec +
         "'. Set OrcaConfig::executable or the RGPOT_ORCA environment "
         "variable.";
}

long processId() {
#ifndef _WIN32
  return static_cast<long>(::getpid());
#else
  return static_cast<long>(::GetCurrentProcessId());
#endif
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string tailOf(const std::string &text) {
  if (text.empty()) {
    return "(no ORCA output)";
  }
  constexpr std::size_t kKeep = 40;
  std::vector<std::string> lines;
  std::string line;
  std::istringstream in(text);
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(line);
  }
  const std::size_t start = lines.size() > kKeep ? lines.size() - kKeep : 0;
  std::string out;
  for (std::size_t i = start; i < lines.size(); ++i) {
    if (!out.empty()) {
      out.push_back('\n');
    }
    out += lines[i];
  }
  constexpr std::size_t kCap = 4000;
  if (out.size() > kCap) {
    out = out.substr(out.size() - kCap);
  }
  if (out.empty()) {
    return "(no ORCA output)";
  }
  return out;
}

bool scfFailed(const std::string &text) {
  return text.find("SCF NOT CONVERGED") != std::string::npos ||
         text.find("SCF DID NOT CONVERGE") != std::string::npos ||
         text.find("The SCF has not converged") != std::string::npos ||
         text.find("error termination in SCF") != std::string::npos;
}

enum class Section { None, Natoms, Energy, Grad, Coords };

Section sectionOf(std::string_view comment) {
  if (comment.find("Number of atoms") != std::string_view::npos) {
    return Section::Natoms;
  }
  if (comment.find("total energy") != std::string_view::npos) {
    return Section::Energy;
  }
  if (comment.find("gradient") != std::string_view::npos) {
    return Section::Grad;
  }
  if (comment.find("atomic numbers") != std::string_view::npos) {
    return Section::Coords;
  }
  return Section::None;
}

bool rawKeywordBlock(const std::string &keywords) {
  const std::string t = trimCopy(keywords);
  if (t.empty()) {
    return false;
  }
  return t.find('\n') != std::string::npos || t.front() == '!' ||
         t.front() == '%';
}

std::vector<std::string> keywordTokens(const std::string &keywords) {
  std::vector<std::string> out;
  std::istringstream in(keywords);
  std::string tok;
  while (in >> tok) {
    out.push_back(tok);
  }
  return out;
}

struct ProcResult {
  int status = -1;
  int start_errno = 0;
  bool signaled = false;
  int signal = 0;
  std::string output;
};

#ifndef _WIN32
ProcResult runProcess(const std::string &exe, const std::filesystem::path &cwd,
                      const std::string &inputName,
                      const std::filesystem::path &output) {
  std::vector<char> arg0(exe.begin(), exe.end());
  arg0.push_back('\0');
  std::vector<char> arg1(inputName.begin(), inputName.end());
  arg1.push_back('\0');
  std::string cwdStr = cwd.string();
  std::string outStr = output.string();

  int errpipe[2];
  if (::pipe(errpipe) != 0) {
    throw std::runtime_error("ORCA: pipe failed");
  }
  ::fcntl(errpipe[0], F_SETFD, FD_CLOEXEC);
  ::fcntl(errpipe[1], F_SETFD, FD_CLOEXEC);

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(errpipe[0]);
    ::close(errpipe[1]);
    throw std::runtime_error("ORCA: fork failed");
  }
  if (pid == 0) {
    ::close(errpipe[0]);
    auto report = [&](int err) {
      const ssize_t n = ::write(errpipe[1], &err, sizeof(err));
      (void)n;
      _exit(127);
    };
    if (::chdir(cwdStr.c_str()) != 0) {
      report(errno);
    }
    const int fd = ::open(outStr.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      report(errno);
    }
    if (::dup2(fd, STDOUT_FILENO) < 0 || ::dup2(fd, STDERR_FILENO) < 0) {
      report(errno);
    }
    if (fd > 2) {
      ::close(fd);
    }
    char *argv[] = {arg0.data(), arg1.data(), nullptr};
    ::execv(arg0.data(), argv);
    report(errno);
  }

  ::close(errpipe[1]);
  int err = 0;
  const ssize_t n = ::read(errpipe[0], &err, sizeof(err));
  ::close(errpipe[0]);

  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) {
    throw std::runtime_error("ORCA: waitpid failed");
  }

  ProcResult proc;
  proc.output = readFile(output);
  if (n == static_cast<ssize_t>(sizeof(err))) {
    proc.start_errno = err;
    proc.status = 127;
    return proc;
  }
  if (WIFEXITED(status)) {
    proc.status = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    proc.signaled = true;
    proc.signal = WTERMSIG(status);
  }
  return proc;
}
#else
ProcResult runProcess(const std::string &exe, const std::filesystem::path &cwd,
                      const std::string &inputName,
                      const std::filesystem::path &output) {
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;
  HANDLE out = ::CreateFileA(output.string().c_str(), GENERIC_WRITE, 0, &sa,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  ProcResult proc;
  if (out == INVALID_HANDLE_VALUE) {
    proc.start_errno = static_cast<int>(::GetLastError());
    proc.status = 127;
    return proc;
  }
  STARTUPINFOA si;
  std::memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = out;
  si.hStdError = out;
  std::string cmd = "\"" + exe + "\" \"" + inputName + "\"";
  std::vector<char> cmdline(cmd.begin(), cmd.end());
  cmdline.push_back('\0');
  PROCESS_INFORMATION pi;
  std::memset(&pi, 0, sizeof(pi));
  const BOOL ok =
      ::CreateProcessA(exe.c_str(), cmdline.data(), nullptr, nullptr, TRUE, 0,
                       nullptr, cwd.string().c_str(), &si, &pi);
  ::CloseHandle(out);
  if (!ok) {
    proc.start_errno = static_cast<int>(::GetLastError());
    proc.status = 127;
    proc.output = readFile(output);
    return proc;
  }
  ::WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  ::GetExitCodeProcess(pi.hProcess, &code);
  ::CloseHandle(pi.hProcess);
  ::CloseHandle(pi.hThread);
  proc.status = static_cast<int>(code);
  proc.output = readFile(output);
  return proc;
}
#endif

std::string failureMessage(const ProcResult &proc, const std::string &exe) {
  const std::string tail = tailOf(proc.output);
  if (proc.start_errno != 0) {
#ifndef _WIN32
    if (proc.start_errno == ENOENT) {
      return missingBinary(exe);
    }
    return "ORCA failed to start: " +
           std::string(std::strerror(proc.start_errno)) + ".\n" + tail;
#else
    if (proc.start_errno == ERROR_FILE_NOT_FOUND ||
        proc.start_errno == ERROR_PATH_NOT_FOUND) {
      return missingBinary(exe);
    }
    return "ORCA failed to start (Windows error " +
           std::to_string(proc.start_errno) + ").\n" + tail;
#endif
  }
  if (scfFailed(proc.output)) {
    std::string msg = "ORCA SCF did not converge";
    if (proc.signaled) {
      msg += " (killed by signal " + std::to_string(proc.signal) + ")";
    } else if (proc.status != 0) {
      msg += " (exit status " + std::to_string(proc.status) + ")";
    }
    return msg + ".\n" + tail;
  }
  if (proc.signaled) {
    return "ORCA process was killed by signal " + std::to_string(proc.signal) +
           ".\n" + tail;
  }
  if (proc.status != 0) {
    return "ORCA exited with status " + std::to_string(proc.status) + ".\n" +
           tail;
  }
  return {};
}

} // namespace

std::string renderOrcaInput(const OrcaConfig &cfg, const OrcaAtoms &atoms,
                            bool moread, std::string_view moinp) {
  if (atoms.n == 0 || atoms.positions == nullptr ||
      atoms.atomic_numbers == nullptr) {
    throw std::invalid_argument("ORCA input: empty geometry");
  }
  if (cfg.multiplicity < 1) {
    throw std::invalid_argument("ORCA input: multiplicity must be at least 1");
  }

  std::vector<std::string> tokens;
  const std::string method = trimCopy(cfg.method);
  const std::string basis = trimCopy(cfg.basis);
  if (!method.empty()) {
    tokens.push_back(method);
  }
  if (!basis.empty()) {
    tokens.push_back(basis);
  }
  const bool block = rawKeywordBlock(cfg.keywords);
  if (!block) {
    for (const std::string &tok : keywordTokens(cfg.keywords)) {
      tokens.push_back(tok);
    }
  }
  bool hasEnGrad = false;
  for (const std::string &tok : tokens) {
    if (tok == "EnGrad") {
      hasEnGrad = true;
    }
  }
  if (!hasEnGrad) {
    tokens.push_back("EnGrad");
  }

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << '!';
  for (const std::string &tok : tokens) {
    out << ' ' << tok;
  }
  out << '\n';
  if (moread) {
    out << "! MORead\n";
    out << "%moinp \"" << moinp << "\"\n";
  }
  if (cfg.nprocs > 0) {
    out << "%pal\n  nprocs " << cfg.nprocs << "\nend\n";
  }
  if (cfg.maxcore_mb > 0) {
    out << "%maxcore " << cfg.maxcore_mb << '\n';
  }
  if (block) {
    std::string raw = trimCopy(cfg.keywords);
    out << raw;
    if (raw.empty() || raw.back() != '\n') {
      out << '\n';
    }
  }
  out << "* xyz " << cfg.charge << ' ' << cfg.multiplicity << '\n';
  out << std::fixed << std::setprecision(10);
  for (std::size_t i = 0; i < atoms.n; ++i) {
    const int z = atoms.atomic_numbers[i];
    const double *p = atoms.positions + 3 * i;
    out << symbolFor(z);
    out << ' ' << std::setw(16) << p[0] << ' ' << std::setw(16) << p[1] << ' '
        << std::setw(16) << p[2] << '\n';
  }
  out << "*\n";
  return out.str();
}

OrcaEngrad parseOrcaEngrad(std::string_view text) {
  OrcaEngrad parsed;
  bool sawAtoms = false;
  bool sawEnergy = false;
  Section sec = Section::None;
  Section labeled = Section::None;

  std::size_t begin = 0;
  while (begin <= text.size()) {
    std::size_t end = text.find('\n', begin);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    std::string_view line = text.substr(begin, end - begin);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const std::string_view trimmed = trim(line);
    if (!trimmed.empty()) {
      if (trimmed.front() == '#') {
        const Section next = sectionOf(trimmed);
        if (next != Section::None) {
          sec = next;
          labeled = next;
        }
      } else if (sec == Section::Natoms && !sawAtoms) {
        std::istringstream in{std::string(trimmed)};
        int n = 0;
        if (!(in >> n) || n < 1) {
          malformed("atom count is not a positive integer");
        }
        parsed.n_atoms = n;
        sawAtoms = true;
      } else if (sec == Section::Energy && !sawEnergy) {
        std::istringstream in{std::string(trimmed)};
        double e = 0.0;
        if (!(in >> e)) {
          malformed("energy is not a number");
        }
        parsed.energy_eh = e;
        sawEnergy = true;
      } else if (sec == Section::Grad) {
        std::istringstream in{std::string(trimmed)};
        double v = 0.0;
        bool any = false;
        while (in >> v) {
          parsed.gradient_eh_bohr.push_back(v);
          any = true;
        }
        if (!any) {
          malformed("gradient line is not numeric");
        }
      }
    }
    if (end == text.size()) {
      break;
    }
    begin = end + 1;
  }

  if (!sawAtoms || labeled == Section::None) {
    malformed("missing atom count");
  }
  if (!sawEnergy) {
    malformed("missing energy");
  }
  const std::size_t need = static_cast<std::size_t>(parsed.n_atoms) * 3;
  if (parsed.gradient_eh_bohr.size() != need) {
    malformed("expected " + std::to_string(need) +
              " gradient components, found " +
              std::to_string(parsed.gradient_eh_bohr.size()));
  }
  return parsed;
}

std::string OrcaPot::resolveExecutable(const std::string &configured) {
  std::string spec = trimCopy(configured);
  if (spec.empty()) {
    if (const char *env = std::getenv("RGPOT_ORCA")) {
      if (env[0] != '\0') {
        spec = env;
      }
    }
  }
  if (spec.empty()) {
    if (const char *env = std::getenv("ORCA")) {
      if (env[0] != '\0') {
        spec = env;
      }
    }
  }
  if (spec.empty()) {
    spec = "orca";
  }
  const bool hasSep = spec.find('/') != std::string::npos ||
                      spec.find('\\') != std::string::npos;
  if (hasSep) {
    if (!isExecutable(spec)) {
      throw std::runtime_error(missingBinary(spec));
    }
    return std::filesystem::absolute(spec).string();
  }
  const char *pathEnv = std::getenv("PATH");
  if (pathEnv != nullptr) {
#ifdef _WIN32
    const char sep = ';';
#else
    const char sep = ':';
#endif
    const std::string path = pathEnv;
    std::size_t start = 0;
    while (start <= path.size()) {
      std::size_t cut = path.find(sep, start);
      if (cut == std::string::npos) {
        cut = path.size();
      }
      if (cut > start) {
        const auto cand =
            std::filesystem::path(path.substr(start, cut - start)) / spec;
        if (isExecutable(cand)) {
          return std::filesystem::absolute(cand).string();
        }
      }
      if (cut == path.size()) {
        break;
      }
      start = cut + 1;
    }
  }
  throw std::runtime_error(missingBinary(spec));
}

OrcaPot::OrcaPot() : OrcaPot(OrcaConfig{}) {}

void OrcaPot::validate(const OrcaConfig &cfg) const {
  if (cfg.multiplicity < 1) {
    throw std::invalid_argument("ORCA: multiplicity must be at least 1");
  }
  if (cfg.nprocs < 0) {
    throw std::invalid_argument("ORCA: nprocs must be zero or positive");
  }
  if (cfg.maxcore_mb < 0) {
    throw std::invalid_argument("ORCA: maxcore_mb must be zero or positive");
  }
}

uint64_t OrcaPot::hashConfig(const OrcaConfig &cfg) const {
  Fnv1a fp;
  fp.str("orca-engrad-1");
  fp.i64(cfg.charge);
  fp.i64(cfg.multiplicity);
  fp.str(cfg.method);
  fp.str(cfg.basis);
  fp.str(cfg.keywords);
  fp.str(cfg.executable);
  fp.b8(cfg.reuse_gbw);
  fp.i64(cfg.nprocs);
  fp.i64(cfg.maxcore_mb);
  return fp.h;
}

OrcaPot::OrcaPot(const OrcaConfig &config)
    : Potential(PotType::Orca), m_cfg(config), m_paramsKey(hashConfig(config)) {
  validate(m_cfg);
  static std::atomic<uint64_t> seq{0};
  const auto id = seq.fetch_add(1, std::memory_order_relaxed);
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("rgpot-orca-" + std::to_string(processId()) + "-" + std::to_string(id));
  std::filesystem::create_directories(dir);
  m_scratch = dir.string();
}

OrcaPot::~OrcaPot() {
  if (m_scratch.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::remove_all(m_scratch, ec);
}

void OrcaPot::reset() {
  std::lock_guard<std::mutex> lock(m_mu);
  m_have_gbw = false;
  m_last_reused = false;
  std::error_code ec;
  std::filesystem::remove(
      std::filesystem::path(m_scratch) / (std::string(kJob) + ".gbw"), ec);
  std::filesystem::remove(std::filesystem::path(m_scratch) / kMoinp, ec);
}

void OrcaPot::setChargeSpin(int charge, int multiplicity) {
  if (multiplicity < 1) {
    throw std::invalid_argument("ORCA: multiplicity must be at least 1");
  }
  std::lock_guard<std::mutex> lock(m_mu);
  m_cfg.charge = charge;
  m_cfg.multiplicity = multiplicity;
  m_paramsKey = hashConfig(m_cfg);
  m_have_gbw = false;
  m_last_reused = false;
  std::error_code ec;
  std::filesystem::remove(
      std::filesystem::path(m_scratch) / (std::string(kJob) + ".gbw"), ec);
  std::filesystem::remove(std::filesystem::path(m_scratch) / kMoinp, ec);
}

bool OrcaPot::reusedWavefunction() const {
  std::lock_guard<std::mutex> lock(m_mu);
  return m_last_reused;
}

void OrcaPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("ORCA: null force output");
  }
  if (in.nAtoms == 0 || in.pos == nullptr || in.atmnrs == nullptr) {
    throw std::invalid_argument("ORCA: empty geometry");
  }
  std::lock_guard<std::mutex> lock(m_mu);

  const std::filesystem::path scratch(m_scratch);
  const std::filesystem::path inp = scratch / (std::string(kJob) + ".inp");
  const std::filesystem::path outp = scratch / (std::string(kJob) + ".out");
  const std::filesystem::path engrad =
      scratch / (std::string(kJob) + ".engrad");
  const std::filesystem::path gbw = scratch / (std::string(kJob) + ".gbw");
  const std::filesystem::path prev = scratch / kMoinp;

  const std::string exe = resolveExecutable(m_cfg.executable);
  m_last_reused = false;

  auto once = [&](bool moread) -> OrcaEngrad {
    std::error_code ec;
    std::filesystem::remove(engrad, ec);
    std::filesystem::remove(outp, ec);
    OrcaAtoms atoms;
    atoms.n = in.nAtoms;
    atoms.positions = in.pos;
    atoms.atomic_numbers = in.atmnrs;
    const std::string text = renderOrcaInput(m_cfg, atoms, moread, kMoinp);
    {
      std::ofstream file(inp);
      if (!file) {
        throw std::runtime_error("ORCA: could not write " + inp.string());
      }
      file << text;
      if (!file) {
        throw std::runtime_error("ORCA: could not write " + inp.string());
      }
    }
    const ProcResult proc =
        runProcess(exe, scratch, inp.filename().string(), outp);
    if (proc.start_errno != 0 || proc.signaled || proc.status != 0 ||
        scfFailed(proc.output)) {
      throw std::runtime_error(failureMessage(proc, exe));
    }
    if (!std::filesystem::is_regular_file(engrad)) {
      throw std::runtime_error("ORCA did not write an .engrad file (" +
                               engrad.filename().string() + ").\n" +
                               tailOf(proc.output));
    }
    return parseOrcaEngrad(readFile(engrad));
  };

  const bool tryMoread =
      m_cfg.reuse_gbw && m_have_gbw && std::filesystem::is_regular_file(gbw);
  if (tryMoread) {
    std::filesystem::copy_file(
        gbw, prev, std::filesystem::copy_options::overwrite_existing);
  }

  OrcaEngrad parsed;
  bool usedMoread = false;
  try {
    parsed = once(tryMoread);
    usedMoread = tryMoread;
  } catch (const std::runtime_error &first) {
    if (!tryMoread) {
      throw;
    }
    const std::string msg = first.what();
    if (msg.find("ORCA binary not found") != std::string::npos) {
      throw;
    }
    m_have_gbw = false;
    std::error_code ec;
    std::filesystem::remove(gbw, ec);
    std::filesystem::remove(prev, ec);
    parsed = once(false);
    usedMoread = false;
  }

  if (static_cast<std::size_t>(parsed.n_atoms) != in.nAtoms) {
    throw std::runtime_error(
        "malformed ORCA engrad: atom count does not match the input");
  }
  std::error_code ec;
  const auto gbwSize = std::filesystem::file_size(gbw, ec);
  m_have_gbw = m_cfg.reuse_gbw && !ec && gbwSize > 0;
  m_last_reused = usedMoread;

  out->energy = parsed.energy_eh * kEhToEv;
  out->variance = 0.0;
  for (std::size_t i = 0; i < parsed.gradient_eh_bohr.size(); ++i) {
    out->F[i] = parsed.gradient_eh_bohr[i] * kGradEhBohrToForceEvA;
  }
}

} // namespace rgpot
