// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAPot.hpp"

#include "rgpot/ORCAPot/ORCAFiles.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <numeric>
#include <sstream>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace rgpot {

namespace fs = std::filesystem;

namespace {

constexpr const char *kBase = "rgpot_orca";
constexpr const char *kGuessName = "rgpot_orca_guess.gbw";
// How much of ORCA's stderr and output file an error message carries.
constexpr std::size_t kTailBytes = 2048;
// After the timeout, SIGTERM is sent to the process group. A rank that can
// still exit is given this long before SIGKILL.
constexpr double kTermGraceSeconds = 2.0;
// A process stuck in uninterruptible sleep can outlive SIGKILL. The wait
// stops here and the caller is told, instead of blocking.
constexpr double kReapGraceSeconds = 5.0;

std::string runFile(const std::string &dir, const char *suffix) {
  return (fs::path(dir) / (std::string(kBase) + suffix)).string();
}

/// The whole of @p path, or empty when it cannot be read.
std::string readText(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream all;
  if (file) {
    all << file.rdbuf();
  }
  return all.str();
}

/// True when ORCA output @p text reports an SCF that did not converge:
/// `SCF NOT CONVERGED` for one that failed, `SCF not fully converged` for
/// one that stopped near convergence. Case is ignored.
bool scfNotConverged(const std::string &text) {
  std::string lower(text.size(), '\0');
  std::transform(text.begin(), text.end(), lower.begin(), [](char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  });
  return lower.find("scf not converged") != std::string::npos ||
         lower.find("scf not fully converged") != std::string::npos;
}

bool isExecutableFile(const fs::path &path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISREG(st.st_mode) && ::access(path.c_str(), X_OK) == 0;
}

/// Absolute path of the executable @p requested names, or empty. A name with
/// a directory part is taken as it stands, a bare name is looked up on PATH.
std::string findExecutable(const std::string &requested) {
  if (requested.find('/') != std::string::npos) {
    std::error_code ec;
    const fs::path path = fs::absolute(requested, ec);
    return (!ec && isExecutableFile(path)) ? path.string() : std::string();
  }
  const std::string name = requested.empty() ? "orca" : requested;
  const char *search = std::getenv("PATH");
  if (search == nullptr) {
    return {};
  }
  std::istringstream entries(search);
  std::string dir;
  while (std::getline(entries, dir, ':')) {
    if (dir.empty()) {
      continue;
    }
    std::error_code ec;
    const fs::path path = fs::absolute(fs::path(dir) / name, ec);
    if (!ec && isExecutableFile(path)) {
      return path.string();
    }
  }
  return {};
}

void rejectPeriodicBox(const double *box) {
  if (box == nullptr) {
    return;
  }
  for (int k = 0; k < 9; ++k) {
    if (box[k] != 0.0) {
      throw std::invalid_argument(
          "ORCAPot: ORCA has no periodic boundary conditions; pass an "
          "all-zero box for an isolated system");
    }
  }
}

void checkSpin(const ORCAConfig &config, const std::vector<int> &numbers) {
  const long protons = std::accumulate(numbers.begin(), numbers.end(), 0L);
  const long electrons = protons - config.charge;
  const long unpaired = static_cast<long>(config.multiplicity) - 1;
  if (electrons < 0 || electrons < unpaired ||
      (electrons - unpaired) % 2 != 0) {
    throw std::invalid_argument(
        "ORCAPot: charge " + std::to_string(config.charge) +
        " and multiplicity " + std::to_string(config.multiplicity) +
        " are impossible for " + std::to_string(electrons) + " electrons");
  }
}

int openAbove(int fd) {
  // The child maps these onto 0, 1 and 2, so keep them clear of those.
  if (fd > 2) {
    return fd;
  }
  const int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
  ::close(fd);
  return moved;
}

struct ChildStatus {
  bool exited = false;
  int exit_code = 0;
  int signal = 0;
  std::string problem; // the process could not be started or waited for
};

enum class WaitKind { Done, Timeout, Failed };

void sleepSeconds(double seconds) {
  if (!(seconds > 0.0)) {
    return;
  }
  struct timespec req;
  req.tv_sec = static_cast<time_t>(seconds);
  const double frac = seconds - static_cast<double>(req.tv_sec);
  req.tv_nsec = static_cast<long>(frac * 1.0e9);
  if (req.tv_nsec < 0) {
    req.tv_nsec = 0;
  }
  if (req.tv_nsec > 999999999L) {
    req.tv_nsec = 999999999L;
  }
  while (::nanosleep(&req, &req) < 0 && errno == EINTR) {
  }
}

/// @p budget below zero waits without a limit. A still-running child is
/// Timeout once the budget has passed. EINTR retries the wait and the sleep.
WaitKind waitChild(pid_t pid, int *wait_status, double budget,
                   std::string *problem) {
  using clock = std::chrono::steady_clock;
  const bool forever = budget < 0.0;
  const auto deadline =
      forever ? clock::time_point::max()
              : clock::now() + std::chrono::duration<double>(budget);
  for (;;) {
    pid_t got;
    do {
      got = ::waitpid(pid, wait_status, WNOHANG);
    } while (got < 0 && errno == EINTR);
    if (got == pid) {
      return WaitKind::Done;
    }
    if (got < 0) {
      *problem = std::string("waitpid failed: ") + std::strerror(errno);
      return WaitKind::Failed;
    }
    const auto now = clock::now();
    if (!forever && now >= deadline) {
      return WaitKind::Timeout;
    }
    double pause = 0.05;
    if (!forever) {
      const double left = std::chrono::duration<double>(deadline - now).count();
      if (left <= 1.0e-4) {
        return WaitKind::Timeout;
      }
      if (left < pause) {
        pause = left;
      }
    }
    sleepSeconds(pause);
  }
}

/// Signals the child's process group. A group that is already gone falls
/// back to the child itself, which covers a setpgid that did not take.
void signalGroup(pid_t pid, int signo) {
  if (pid <= 1) {
    return;
  }
  int rc;
  do {
    rc = ::kill(-pid, signo);
  } while (rc < 0 && errno == EINTR);
  if (rc == 0 || errno != ESRCH) {
    return;
  }
  do {
    rc = ::kill(pid, signo);
  } while (rc < 0 && errno == EINTR);
}

void killGroupOnly(pid_t pgid, int signo) {
  if (pgid <= 1) {
    return;
  }
  int rc;
  do {
    rc = ::kill(-pgid, signo);
  } while (rc < 0 && errno == EINTR);
}

bool groupAlive(pid_t pgid) {
  if (pgid <= 1) {
    return false;
  }
  int rc;
  do {
    rc = ::kill(-pgid, 0);
  } while (rc < 0 && errno == EINTR);
  return rc == 0 || errno != ESRCH;
}

/// Waits until both the leader @p pid is reaped and its process group has no
/// remaining processes, or until @p budget seconds pass.
WaitKind waitGroup(pid_t pid, int *wait_status, double budget,
                   bool *leader_reaped, std::string *problem) {
  using clock = std::chrono::steady_clock;
  const auto deadline =
      clock::now() + std::chrono::duration<double>(budget);
  for (;;) {
    if (!*leader_reaped) {
      pid_t got;
      do {
        got = ::waitpid(pid, wait_status, WNOHANG);
      } while (got < 0 && errno == EINTR);
      if (got == pid) {
        *leader_reaped = true;
      } else if (got < 0) {
        if (errno == ECHILD) {
          *leader_reaped = true;
        } else {
          *problem = std::string("waitpid failed: ") + std::strerror(errno);
          return WaitKind::Failed;
        }
      }
    }
    if (*leader_reaped && !groupAlive(pid)) {
      return WaitKind::Done;
    }
    const auto now = clock::now();
    if (now >= deadline) {
      return WaitKind::Timeout;
    }
    double pause = 0.05;
    const double left = std::chrono::duration<double>(deadline - now).count();
    if (left <= 1.0e-4) {
      return WaitKind::Timeout;
    }
    if (left < pause) {
      pause = left;
    }
    sleepSeconds(pause);
  }
}

void closeExtraFds(long max_fd) noexcept {
#if (defined(__GLIBC__) &&                                                     \
     (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34)) &&         \
     defined(_GNU_SOURCE)) ||                                                \
    (defined(__FreeBSD__) && __FreeBSD__ >= 13)
  if (::close_range(3U, ~0U, 0U) == 0) {
    return;
  }
#endif
#if defined(__linux__) && defined(SYS_close_range)
  if (::syscall(SYS_close_range, 3U, ~0U, 0U) == 0) {
    return;
  }
#endif
  const long limit = max_fd > 0 ? std::min(max_fd, 1048576L) : 1024L;
  for (int fd = 3; fd < static_cast<int>(limit); ++fd) {
    ::close(fd);
  }
}

void storeWait(ChildStatus *status, int wait_status) {
  if (WIFEXITED(wait_status)) {
    status->exited = true;
    status->exit_code = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    status->signal = WTERMSIG(wait_status);
  } else {
    status->problem = "waitpid returned an unexpected status";
  }
}

std::string timeoutText(double timeout_s, bool sent_kill) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << "orca timed out after " << timeout_s
       << " s; the process group was sent SIGTERM";
  if (sent_kill) {
    text << " and then SIGKILL";
  }
  return text.str();
}

/// Runs `executable input` in @p cwd with the standard streams on files. The
/// child is the leader of a new process group. Only async-signal-safe calls
/// run between fork and exec, so a threaded caller is safe. @p timeout_s is
/// the wait in seconds; 0 waits without a limit. Past the limit the group is
/// sent SIGTERM and then, if it is still alive, SIGKILL, and the child is
/// reaped.
ChildStatus runChild(const std::string &executable, const std::string &input,
                     const std::string &cwd, const std::string &out_path,
                     const std::string &err_path, double timeout_s) {
  ChildStatus status;
  const int in_fd = openAbove(::open("/dev/null", O_RDONLY | O_CLOEXEC));
  const int out_fd = openAbove(::open(
      out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
  const int err_fd = openAbove(::open(
      err_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
  if (in_fd < 0 || out_fd < 0 || err_fd < 0) {
    status.problem = std::string("could not open the standard streams: ") +
                     std::strerror(errno);
    for (const int fd : {in_fd, out_fd, err_fd}) {
      if (fd >= 0) {
        ::close(fd);
      }
    }
    return status;
  }
  const long max_fd = ::sysconf(_SC_OPEN_MAX);
  char *argv[] = {const_cast<char *>(executable.c_str()),
                  const_cast<char *>(input.c_str()), nullptr};
  const pid_t pid = ::fork();
  if (pid < 0) {
    status.problem = std::string("fork failed: ") + std::strerror(errno);
  } else if (pid == 0) {
    int grouped;
    do {
      grouped = ::setpgid(0, 0);
    } while (grouped != 0 && errno == EINTR);
    if (grouped != 0 || ::dup2(in_fd, 0) < 0 || ::dup2(out_fd, 1) < 0 ||
        ::dup2(err_fd, 2) < 0 || ::chdir(cwd.c_str()) != 0) {
      ::_exit(126);
    }
    closeExtraFds(max_fd);
    ::execv(executable.c_str(), argv);
    ::_exit(127);
  }
  ::close(in_fd);
  ::close(out_fd);
  ::close(err_fd);
  if (pid < 0) {
    return status;
  }
  // The child calls setpgid too. EACCES means it has already exec'd, ESRCH
  // means it has already exited; either way there is nothing to repair.
  int grouped;
  do {
    grouped = ::setpgid(pid, pid);
  } while (grouped != 0 && errno == EINTR);
  (void)grouped;

  int wait_status = 0;
  const double budget = timeout_s == 0.0 ? -1.0 : timeout_s;
  const WaitKind first = waitChild(pid, &wait_status, budget, &status.problem);
  if (first == WaitKind::Failed) {
    signalGroup(pid, SIGKILL);
    bool leader_reaped = false;
    std::string ignored;
    waitGroup(pid, &wait_status, kReapGraceSeconds, &leader_reaped, &ignored);
    return status;
  }
  if (first == WaitKind::Timeout) {
    signalGroup(pid, SIGTERM);
    bool leader_reaped = false;
    std::string ignored;
    const WaitKind grace = waitGroup(pid, &wait_status, kTermGraceSeconds,
                                     &leader_reaped, &ignored);
    if (grace != WaitKind::Done) {
      if (leader_reaped) {
        killGroupOnly(pid, SIGKILL);
      } else {
        signalGroup(pid, SIGKILL);
      }
      std::string reap_problem;
      const WaitKind dead = waitGroup(pid, &wait_status, kReapGraceSeconds,
                                      &leader_reaped, &reap_problem);
      status.problem = timeoutText(timeout_s, true);
      if (dead != WaitKind::Done) {
        status.problem += "; the process could not be reaped";
        if (!reap_problem.empty()) {
          status.problem += ": ";
          status.problem += reap_problem;
        }
      }
      return status;
    }
    status.problem = timeoutText(timeout_s, false);
    return status;
  }
  storeWait(&status, wait_status);
  return status;
}

} // namespace

ORCAPot::ORCAPot(const ORCAConfig &config)
    : Potential(PotType::ORCA), m_config(config) {
  if (config.method.empty()) {
    throw std::invalid_argument("ORCAPot: method is required");
  }
  if (config.multiplicity < 1) {
    throw std::invalid_argument("ORCAPot: multiplicity must be at least 1");
  }
  if (config.nprocs < 1) {
    throw std::invalid_argument("ORCAPot: nprocs must be at least 1");
  }
  if (config.maxcore_mb < 1) {
    throw std::invalid_argument("ORCAPot: maxcore_mb must be at least 1");
  }
  if (!std::isfinite(config.timeout_s) || config.timeout_s < 0.0) {
    throw std::invalid_argument(
        "ORCAPot: timeout_s must be a finite number of seconds, or 0 to wait "
        "without a limit");
  }
  // The keyword fields are checked here too, so a bad config fails before a
  // directory is made.
  orca::InputSpec probe;
  probe.method = config.method;
  probe.basis = config.basis;
  probe.keywords = config.keywords;
  (void)orca::renderInput(probe, 0, nullptr, nullptr);

  m_executable = findExecutable(config.executable);
  if (m_executable.empty()) {
    throw ORCAError(
        "ORCAPot: no ORCA executable found for '" +
        (config.executable.empty() ? std::string("orca") : config.executable) +
        "'; set ORCAConfig::executable or put orca on PATH");
  }

  std::error_code ec;
  fs::path parent = config.scratch_dir.empty()
                        ? fs::temp_directory_path(ec)
                        : fs::path(config.scratch_dir);
  if (!ec) {
    parent = fs::absolute(parent, ec);
  }
  if (!ec) {
    fs::create_directories(parent, ec);
  }
  if (ec) {
    throw ORCAError("ORCAPot: scratch directory " + parent.string() + ": " +
                    ec.message());
  }
  std::string pattern = (parent / "rgpot-orca-XXXXXX").string();
  if (::mkdtemp(pattern.data()) == nullptr) {
    throw ORCAError("ORCAPot: could not create a directory under " +
                    parent.string() + ": " + std::strerror(errno));
  }
  m_run_dir = pattern;
}

ORCAPot::~ORCAPot() {
  if (!m_config.keep_files && !m_run_dir.empty()) {
    std::error_code ignored;
    fs::remove_all(m_run_dir, ignored);
  }
}

bool ORCAPot::probeAvailable(const std::string &executable) {
  return !findExecutable(executable).empty();
}

uint64_t ORCAPot::paramsKey() const noexcept {
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.str(m_config.method);
  fp.str(m_config.basis);
  fp.str(m_config.keywords);
  fp.i64(m_config.charge);
  fp.i64(m_config.multiplicity);
  fp.b8(m_config.restart);
  return fp.h;
}

void ORCAPot::reset() {
  std::error_code ignored;
  fs::remove(fs::path(m_run_dir) / kGuessName, ignored);
  m_has_guess = false;
  m_guess_numbers.clear();
}

void ORCAPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  rejectPeriodicBox(in.box);
  const std::size_t n = in.nAtoms;
  const std::vector<int> numbers(in.atmnrs, in.atmnrs + n);
  checkSpin(m_config, numbers);

  const std::string input_path = runFile(m_run_dir, ".inp");
  const std::string output_path = runFile(m_run_dir, ".out");
  const std::string stderr_path = runFile(m_run_dir, ".err");
  const std::string engrad_path = runFile(m_run_dir, ".engrad");
  const std::string gbw_path = runFile(m_run_dir, ".gbw");
  const fs::path guess_path = fs::path(m_run_dir) / kGuessName;

  std::error_code ec;
  const bool use_guess = m_config.restart && m_has_guess &&
                         m_guess_numbers == numbers &&
                         fs::is_regular_file(guess_path, ec) && !ec;
  if (!use_guess) {
    m_has_guess = false;
  }

  orca::InputSpec spec;
  spec.method = m_config.method;
  spec.basis = m_config.basis;
  spec.keywords = m_config.keywords;
  spec.charge = m_config.charge;
  spec.multiplicity = m_config.multiplicity;
  spec.nprocs = m_config.nprocs;
  spec.maxcore_mb = m_config.maxcore_mb;
  if (use_guess) {
    spec.moinp = kGuessName;
  }
  const std::string text =
      orca::renderInput(spec, n, in.pos, in.atmnrs);

  const auto fail = [&](const std::string &what) {
    m_has_guess = false;
    return ORCAError("ORCAPot: " + what + "\n  input: " + input_path +
                     "\n  stderr tail:\n" +
                     orca::fileTail(stderr_path, kTailBytes) +
                     "\n  output tail:\n" +
                     orca::fileTail(output_path, kTailBytes));
  };

  // A result of an earlier call must never pass for this one.
  const auto removeStale = [&](const std::string &path) {
    std::error_code remove_ec;
    fs::remove(path, remove_ec);
    if (remove_ec && remove_ec != std::errc::no_such_file_or_directory) {
      m_has_guess = false;
      throw ORCAError("ORCAPot: could not remove stale " + path + ": " +
                      remove_ec.message());
    }
  };
  removeStale(engrad_path);
  removeStale(gbw_path);
  {
    std::ofstream file(input_path, std::ios::binary | std::ios::trunc);
    file << text;
    file.flush();
    if (!file) {
      throw ORCAError("ORCAPot: could not write " + input_path);
    }
  }

  const ChildStatus child =
      runChild(m_executable, std::string(kBase) + ".inp", m_run_dir,
               output_path, stderr_path, m_config.timeout_s);
  if (!child.problem.empty()) {
    throw fail(child.problem);
  }
  if (!child.exited) {
    throw fail("orca was killed by signal " + std::to_string(child.signal));
  }
  // Checked before the exit status: ConvForced makes ORCA stop with an
  // error, and the SCF is the cause the caller needs to see.
  if (scfNotConverged(readText(output_path))) {
    throw fail("the SCF did not converge, so the gradient is not used");
  }
  if (child.exit_code != 0) {
    throw fail("orca exited with status " + std::to_string(child.exit_code) +
               (child.exit_code == 127 ? " (the executable could not be run)"
                                       : ""));
  }

  std::string engrad_text;
  {
    std::ifstream file(engrad_path, std::ios::binary);
    if (!file) {
      throw fail("orca finished without writing " + engrad_path);
    }
    std::ostringstream all;
    all << file.rdbuf();
    engrad_text = all.str();
  }
  orca::Engrad result;
  try {
    result = orca::parseEngrad(engrad_text, n);
  } catch (const std::runtime_error &error) {
    throw fail(error.what());
  }
  if (result.atomic_numbers != numbers) {
    throw fail("the .engrad atomic numbers differ from the input");
  }
  bool finite = std::isfinite(result.energy_hartree);
  for (const double g : result.gradient_hartree_per_bohr) {
    finite = finite && std::isfinite(g);
  }
  if (!finite) {
    throw fail("non-finite energy or gradient in the .engrad file");
  }

  out->energy = result.energy_hartree * units::HARTREE_TO_EV;
  for (std::size_t k = 0; k < 3 * n; ++k) {
    out->F[k] = result.gradient_hartree_per_bohr[k] * units::NEG_GRAD_TO_FORCE;
  }
  out->variance = 0.0;
  out->has_stress = 0;

  m_last_reused = use_guess;
  if (!m_config.restart) {
    return;
  }
  m_has_guess = false;
  if (fs::is_regular_file(gbw_path, ec) && !ec) {
    fs::copy_file(gbw_path, guess_path, fs::copy_options::overwrite_existing,
                  ec);
    if (ec) {
      throw ORCAError("ORCAPot: could not keep the orbitals of this call: " +
                      ec.message());
    }
    m_has_guess = true;
    m_guess_numbers = numbers;
  }
}

} // namespace rgpot
