// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/DxtbPot/DxtbPot.hpp"

#include "rgpot/DxtbPot/DxtbWorkerSource.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/engine/EngineError.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef RGPOT_DXTB_PYTHON
#define RGPOT_DXTB_PYTHON "python3"
#endif

namespace rgpot {
namespace {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::uint32_t kConfigure = 1;
constexpr std::uint32_t kForce = 2;
constexpr std::uint32_t kHvp = 3;
constexpr std::uint32_t kThreads = 4;
constexpr std::uint32_t kShutdown = 5;
constexpr std::size_t kHeaderSize = 20;

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
    if (fd_ >= 0) {
      if (graceful) {
        try {
          (void)request(kShutdown, 0, {});
        } catch (...) {
        }
      }
      close(fd_);
      fd_ = -1;
    }
    if (pid_ > 0) {
      int status = 0;
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
      }
    }
    pid_ = -1;
  }

  void force(const ForceInput &in, ForceOut *out) {
    const auto response =
        request(kForce, checkedAtoms(in), geometry(in, nullptr));
    const std::size_t expected = (1 + 3 * in.nAtoms) * sizeof(double);
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
    const auto response =
        request(kHvp, checkedAtoms(in), geometry(in, direction));
    const std::size_t expected = 3 * in.nAtoms * sizeof(double);
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

  /// BuiltinParams.library, then RGPOT_DXTB_PYTHON, then the build default.
  std::string pythonExecutable() const {
    if (!config_.python.empty()) {
      return config_.python;
    }
    const char *configured = std::getenv("RGPOT_DXTB_PYTHON");
    if (configured && configured[0] != '\0') {
      return configured;
    }
    return RGPOT_DXTB_PYTHON;
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
    const std::string python = pythonExecutable();
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
      execlp(python.c_str(), python.c_str(), "-u", "-c",
             detail::kDxtbWorkerSource, static_cast<char *>(nullptr));
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
      if (done < 0 && (errno == EPIPE || errno == ECONNRESET)) {
        throw DxtbError(-1, "dxtb worker exited");
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
      if (done == 0 || (done < 0 && errno == ECONNRESET)) {
        throw DxtbError(-1, "dxtb worker exited");
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
          const std::vector<unsigned char> &payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw DxtbError(1, "dxtb request is too large");
    }
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
  }
};

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
