// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/DxtbPot/DxtbPot.hpp"

#include "rgpot/DxtbPot/DxtbWorkerSource.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/engine/EngineError.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef RGPOT_DXTB_PYTHON
#define RGPOT_DXTB_PYTHON "python3"
#endif

extern char **environ;

namespace rgpot {
namespace {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::uint32_t kConfigure = 1;
constexpr std::uint32_t kForce = 2;
constexpr std::uint32_t kHvp = 3;
constexpr std::uint32_t kThreads = 4;
constexpr std::uint32_t kShutdown = 5;
constexpr std::size_t kHeaderSize = 20;
/// Largest status or error reply the host reads. A force or HVP reply may be
/// larger, but never larger than the payload its atom count implies.
constexpr std::size_t kMaxMessageBytes = std::size_t{1} << 20;
/// How long shutdown waits for the worker to acknowledge, close its end and
/// exit before the host kills it.
constexpr std::chrono::milliseconds kShutdownGrace{3000};
/// How long the availability probe may spend importing Torch and dxtb.
constexpr std::chrono::seconds kProbeTimeout{120};

void appendU32(std::vector<unsigned char> &out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<unsigned char>((value >> shift) & 0xffu));
  }
}

void appendI32(std::vector<unsigned char> &out, std::int32_t value) {
  appendU32(out, static_cast<std::uint32_t>(value));
}

void appendF64(std::vector<unsigned char> &out, double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<unsigned char>((bits >> shift) & 0xffu));
  }
}

std::uint32_t readU32(const unsigned char *data) {
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(data[shift / 8]) << shift;
  }
  return value;
}

double readF64(const unsigned char *data) {
  std::uint64_t bits = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    bits |= static_cast<std::uint64_t>(data[shift / 8]) << shift;
  }
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string systemMessage(const char *operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

/// The peer closed its end. During an intentional shutdown this is a clean
/// close; on a live exchange it is reported as the worker exiting.
struct PeerClosed {};

bool peerClosedErrno(int err) {
  return err == EPIPE || err == ECONNRESET;
}

void setSocketTimeout(int fd, std::chrono::milliseconds timeout) {
  const auto ms = std::max<std::chrono::milliseconds::rep>(timeout.count(), 1);
  timeval tv{};
  tv.tv_sec = static_cast<decltype(tv.tv_sec)>(ms / 1000);
  tv.tv_usec = static_cast<decltype(tv.tv_usec)>((ms % 1000) * 1000);
  (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

std::chrono::milliseconds
remaining(std::chrono::steady_clock::time_point deadline) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
}

/// DxtbConfig::python when set, then RGPOT_DXTB_PYTHON, then the build
/// default.
std::string dxtbPython(const std::string &configured) {
  if (!configured.empty()) {
    return configured;
  }
  const char *environment = std::getenv("RGPOT_DXTB_PYTHON");
  if (environment && environment[0] != '\0') {
    return environment;
  }
  return RGPOT_DXTB_PYTHON;
}

uint64_t paramsKeyFor(const DxtbConfig &config, uint64_t kernelVersion) {
  Fnv1a hash;
  hash.str("rgpot::DxtbPot");
  hash.u64(kernelVersion);
  hash.i64(config.method == DxtbMethod::GFN1 ? 1 : 2);
  hash.f64(config.charge);
  hash.i64(config.uhf);
  hash.f64(config.accuracy);
  hash.f64(config.electronic_temperature);
  hash.i64(config.max_iterations);
  return hash.h;
}

} // namespace

class DxtbPot::Worker {
public:
  explicit Worker(const DxtbConfig &config) : config_(config) {
    start();
    try {
      configure(config_);
    } catch (...) {
      stop(false);
      throw;
    }
  }

  ~Worker() { stop(true); }

  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;

  void stop(bool graceful) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + kShutdownGrace;
    if (fd_ >= 0) {
      if (graceful) {
        setSocketTimeout(fd_, kShutdownGrace);
        try {
          // A peer that vanishes while this request is written or read has
          // already closed. That is the end of shutdown, not a failed call.
          (void)request(kShutdown, 0, {}, kMaxMessageBytes, true);
        } catch (...) {
        }
        // Stay readable until the worker closes. Closing first turns the
        // worker's last write into EPIPE.
        ::shutdown(fd_, SHUT_WR);
        unsigned char extra[256];
        while (remaining(deadline).count() > 0) {
          setSocketTimeout(fd_, remaining(deadline));
          const ssize_t n = ::recv(fd_, extra, sizeof(extra), 0);
          if (n < 0 && errno == EINTR) {
            continue;
          }
          if (n <= 0) {
            break;
          }
        }
      }
      close(fd_);
      fd_ = -1;
    }
    if (pid_ > 0) {
      reap(deadline);
    }
    pid_ = -1;
  }

  void force(const ForceInput &in, ForceOut *out) {
    const std::uint32_t nAtoms = checkedAtoms(in);
    const std::size_t expected = (1 + 3 * in.nAtoms) * sizeof(double);
    const auto response =
        request(kForce, nAtoms, geometry(in, nullptr), expected);
    if (response.size() != expected) {
      throw DxtbError(-1, "dxtb worker returned an invalid force payload");
    }
    out->energy = readF64(response.data());
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      out->F[i] = readF64(response.data() + sizeof(double) * (i + 1));
    }
    out->variance = 0.0;
  }

  void hvp(const ForceInput &in, const double *direction, double *product) {
    const std::uint32_t nAtoms = checkedAtoms(in);
    const std::size_t expected = 3 * in.nAtoms * sizeof(double);
    const auto response =
        request(kHvp, nAtoms, geometry(in, direction), expected);
    if (response.size() != expected) {
      throw DxtbError(-1, "dxtb worker returned an invalid HVP payload");
    }
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      product[i] = readF64(response.data() + sizeof(double) * i);
    }
  }

  void setChargeSpin(int charge, int multiplicity) {
    if (multiplicity < 1) {
      throw engine::Unsupported("dxtb spin multiplicity must be >= 1; got " +
                                std::to_string(multiplicity));
    }
    DxtbConfig next = config_;
    next.charge = static_cast<double>(charge);
    next.uhf = multiplicity - 1;
    configure(next);
    config_ = next;
  }

  void setNumThreads(int intra_op, int inter_op) {
    std::vector<unsigned char> payload;
    appendI32(payload, intra_op);
    appendI32(payload, inter_op);
    (void)request(kThreads, 0, payload);
  }

private:
  DxtbConfig config_;
  int fd_ = -1;
  pid_t pid_ = -1;

  /// Wait for the worker until @p deadline, then kill it. A worker that
  /// ignores shutdown must not hang the host's destructor.
  void reap(std::chrono::steady_clock::time_point deadline) noexcept {
    int status = 0;
    while (true) {
      const pid_t done = waitpid(pid_, &status, WNOHANG);
      if (done == pid_ || (done < 0 && errno != EINTR)) {
        return;
      }
      if (remaining(deadline).count() <= 0) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    (void)kill(pid_, SIGKILL);
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
  }

  static std::uint32_t checkedAtoms(const ForceInput &in) {
    if (in.nAtoms == 0 ||
        in.nAtoms > static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max())) {
      throw DxtbError(1, "invalid dxtb atom count");
    }
    return static_cast<std::uint32_t>(in.nAtoms);
  }

  static void makeSocketPair(int sockets[2]) {
#ifdef SOCK_CLOEXEC
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
      throw DxtbError(-1, systemMessage("socketpair"));
    }
#else
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
      throw DxtbError(-1, systemMessage("socketpair"));
    }
    for (int i = 0; i < 2; ++i) {
      if (fcntl(sockets[i], F_SETFD, FD_CLOEXEC) != 0) {
        const std::string message = systemMessage("fcntl(FD_CLOEXEC)");
        close(sockets[0]);
        close(sockets[1]);
        throw DxtbError(-1, message);
      }
    }
#endif
  }

  void start() {
    int sockets[2] = {-1, -1};
    makeSocketPair(sockets);
#ifndef MSG_NOSIGNAL
    int enabled = 1;
    if (setsockopt(sockets[0], SOL_SOCKET, SO_NOSIGPIPE, &enabled,
                   sizeof(enabled)) != 0) {
      close(sockets[0]);
      close(sockets[1]);
      throw DxtbError(-1, systemMessage("setsockopt(SO_NOSIGPIPE)"));
    }
#endif
    const std::string python = dxtbPython(config_.python);
    const std::string source =
        std::string(detail::kDxtbRuntimeCheck) + detail::kDxtbWorkerSource;
    pid_ = fork();
    if (pid_ < 0) {
      const std::string message = systemMessage("fork");
      close(sockets[0]);
      close(sockets[1]);
      throw DxtbError(-1, message);
    }
    if (pid_ == 0) {
      close(sockets[0]);
      if (dup2(sockets[1], STDIN_FILENO) < 0 ||
          dup2(sockets[1], STDOUT_FILENO) < 0) {
        _exit(126);
      }
      // dup2 onto itself keeps FD_CLOEXEC, so clear it on both descriptors.
      if (fcntl(STDIN_FILENO, F_SETFD, 0) != 0 ||
          fcntl(STDOUT_FILENO, F_SETFD, 0) != 0) {
        _exit(126);
      }
      if (sockets[1] != STDIN_FILENO && sockets[1] != STDOUT_FILENO) {
        close(sockets[1]);
      }
      execlp(python.c_str(), python.c_str(), "-u", "-c", source.c_str(),
             static_cast<char *>(nullptr));
      _exit(127);
    }
    close(sockets[1]);
    fd_ = sockets[0];
  }

  void configure(const DxtbConfig &config) {
    std::vector<unsigned char> payload;
    appendU32(payload, config.method == DxtbMethod::GFN1 ? 1u : 2u);
    appendI32(payload, config.uhf);
    appendI32(payload, config.max_iterations);
    appendF64(payload, config.charge);
    appendF64(payload, config.accuracy);
    appendF64(payload, config.electronic_temperature);
    appendI32(payload, 0);
    appendI32(payload, 0);
    (void)request(kConfigure, 0, payload);
  }

  static std::vector<unsigned char> geometry(const ForceInput &in,
                                             const double *direction) {
    std::vector<unsigned char> payload;
    payload.reserve(in.nAtoms * (sizeof(std::int32_t) + 3 * sizeof(double) +
                                 (direction ? 3 * sizeof(double) : 0)));
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      appendI32(payload, static_cast<std::int32_t>(in.atmnrs[i]));
    }
    for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
      appendF64(payload, in.pos[i]);
    }
    if (direction) {
      for (std::size_t i = 0; i < 3 * in.nAtoms; ++i) {
        appendF64(payload, direction[i]);
      }
    }
    return payload;
  }

  void writeAll(const unsigned char *data, std::size_t size) {
    while (size > 0) {
#ifdef MSG_NOSIGNAL
      const ssize_t done = send(fd_, data, size, MSG_NOSIGNAL);
#else
      const ssize_t done = send(fd_, data, size, 0);
#endif
      if (done < 0 && errno == EINTR) {
        continue;
      }
      if (done < 0 && peerClosedErrno(errno)) {
        throw PeerClosed();
      }
      if (done <= 0) {
        throw DxtbError(-1, systemMessage("write to dxtb worker"));
      }
      data += done;
      size -= static_cast<std::size_t>(done);
    }
  }

  void readAll(unsigned char *data, std::size_t size) {
    while (size > 0) {
      const ssize_t done = recv(fd_, data, size, 0);
      if (done < 0 && errno == EINTR) {
        continue;
      }
      if (done == 0 || (done < 0 && peerClosedErrno(errno))) {
        throw PeerClosed();
      }
      if (done < 0) {
        throw DxtbError(-1, systemMessage("read from dxtb worker"));
      }
      data += done;
      size -= static_cast<std::size_t>(done);
    }
  }

  std::vector<unsigned char>
  request(std::uint32_t operation, std::uint32_t nAtoms,
          const std::vector<unsigned char> &payload,
          std::size_t maxResponse = kMaxMessageBytes,
          bool peer_close_is_clean = false) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw DxtbError(1, "dxtb request is too large");
    }
    if (fd_ < 0) {
      throw DxtbError(-1, "dxtb worker exited");
    }
    try {
      std::vector<unsigned char> header;
      header.reserve(kHeaderSize);
      header.insert(header.end(), {'D', 'X', 'T', 'B'});
      appendU32(header, kProtocolVersion);
      appendU32(header, operation);
      appendU32(header, nAtoms);
      appendU32(header, static_cast<std::uint32_t>(payload.size()));
      writeAll(header.data(), header.size());
      if (!payload.empty()) {
        writeAll(payload.data(), payload.size());
      }

      std::vector<unsigned char> responseHeader(kHeaderSize);
      readAll(responseHeader.data(), responseHeader.size());
      if (std::string_view(reinterpret_cast<const char *>(responseHeader.data()),
                           4) != "DXTR" ||
          readU32(responseHeader.data() + 4) != kProtocolVersion) {
        throw DxtbError(-1, "invalid response from dxtb worker");
      }
      const int status =
          static_cast<std::int32_t>(readU32(responseHeader.data() + 8));
      const std::uint32_t size = readU32(responseHeader.data() + 16);
      const std::size_t limit = std::max(maxResponse, kMaxMessageBytes);
      if (size > limit) {
        // The unread reply leaves the stream out of step, so the worker
        // cannot serve another request.
        stop(false);
        throw DxtbError(-1, "dxtb worker reply of " + std::to_string(size) +
                                " bytes exceeds the " +
                                std::to_string(limit) + " byte limit");
      }
      std::vector<unsigned char> response(size);
      if (size > 0) {
        readAll(response.data(), response.size());
      }
      if (status != 0) {
        throw DxtbError(
            status, std::string(reinterpret_cast<const char *>(response.data()),
                                response.size()));
      }
      return response;
    } catch (const PeerClosed &) {
      if (peer_close_is_clean) {
        return {};
      }
      throw DxtbError(-1, "dxtb worker exited");
    }
  }
};

namespace {

enum class Probe { Usable, Unusable, Undecided };

Probe probeRuntime(const std::string &python) {
  const std::string source =
      std::string(detail::kDxtbRuntimeCheck) +
      "\nraise SystemExit(0 if runtime_problem() is None else 3)\n";
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    return Probe::Undecided;
  }
  (void)posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                         O_RDONLY, 0);
  (void)posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                         O_WRONLY, 0);
  (void)posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null",
                                         O_WRONLY, 0);
  std::vector<char> program(python.begin(), python.end());
  program.push_back('\0');
  std::vector<char> script(source.begin(), source.end());
  script.push_back('\0');
  char flag[] = "-c";
  char *argv[] = {program.data(), flag, script.data(), nullptr};
  pid_t pid = -1;
  const int spawned =
      posix_spawnp(&pid, program.data(), &actions, nullptr, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawned != 0) {
    return Probe::Unusable;
  }
  const auto deadline = std::chrono::steady_clock::now() + kProbeTimeout;
  int status = 0;
  while (true) {
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? Probe::Usable
                                                           : Probe::Unusable;
    }
    if (done < 0 && errno != EINTR) {
      return Probe::Undecided;
    }
    if (remaining(deadline).count() <= 0) {
      (void)kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      return Probe::Undecided;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace

bool DxtbPot::runtimeAvailable() {
  static std::mutex mutex;
  static std::map<std::string, bool> verdicts;
  const std::string python = dxtbPython({});
  std::lock_guard<std::mutex> lock(mutex);
  if (const auto found = verdicts.find(python); found != verdicts.end()) {
    return found->second;
  }
  const Probe probe = probeRuntime(python);
  // A probe that could not finish says nothing about the runtime: report it
  // usable, leave it uncached, and let creation give the verdict.
  if (probe == Probe::Undecided) {
    return true;
  }
  verdicts.emplace(python, probe == Probe::Usable);
  return probe == Probe::Usable;
}

DxtbPot::DxtbPot(const DxtbConfig &config)
    : Potential(PotType::UNKNOWN), config_(config),
      params_key_(paramsKeyFor(config_, kKernelVersion)),
      worker_(std::make_unique<Worker>(config_)) {}

DxtbPot::~DxtbPot() = default;

void DxtbPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  worker_->force(in, out);
}

void DxtbPot::hessianVectorImpl(const ForceInput &in, const double *direction,
                                double *product) const {
  worker_->hvp(in, direction, product);
}

void DxtbPot::setChargeSpin(int charge, int multiplicity) {
  worker_->setChargeSpin(charge, multiplicity);
  config_.charge = static_cast<double>(charge);
  config_.uhf = multiplicity - 1;
  params_key_ = paramsKeyFor(config_, kKernelVersion);
}

void DxtbPot::setNumThreads(int intra_op, int inter_op) {
  worker_->setNumThreads(intra_op, inter_op);
}

} // namespace rgpot
