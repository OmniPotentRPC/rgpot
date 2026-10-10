// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/IPIPot/IPIPot.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rgpot {
namespace {

constexpr int kHeaderLen = 12;
constexpr int32_t kMaxAtoms = 10'000'000;
constexpr int32_t kMaxExtra = 1 << 20;

using clock = std::chrono::steady_clock;

struct Deadline {
  clock::time_point end;

  static Deadline after(double seconds) {
    if (!(seconds >= 0.0) || !std::isfinite(seconds)) {
      throw std::invalid_argument("IPIPot: timeout_s must be finite and >= 0");
    }
    const auto delta = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(seconds));
    return Deadline{clock::now() + delta};
  }

  [[nodiscard]] bool expired() const { return clock::now() >= end; }

  [[nodiscard]] int remaining_ms() const {
    const auto left = end - clock::now();
    if (left <= clock::duration::zero()) {
      return 0;
    }
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(left).count();
    if (ms <= 0) {
      return 1;
    }
    if (ms > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(ms);
  }
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("IPIPot: " + message);
}

[[noreturn]] void fail_errno(const char *what) {
  fail(std::string(what) + " (errno " + std::to_string(errno) + ")");
}

std::array<char, kHeaderLen> header(std::string_view text) {
  std::array<char, kHeaderLen> out;
  out.fill(' ');
  const auto n = std::min(text.size(), static_cast<size_t>(kHeaderLen));
  std::copy_n(text.begin(), n, out.begin());
  return out;
}

const std::array<char, kHeaderLen> kStatus = header("STATUS");
const std::array<char, kHeaderLen> kPosdata = header("POSDATA");
const std::array<char, kHeaderLen> kGetforce = header("GETFORCE");
const std::array<char, kHeaderLen> kExit = header("EXIT");
const std::array<char, kHeaderLen> kInit = header("INIT");
const std::array<char, kHeaderLen> kReady = header("READY");
const std::array<char, kHeaderLen> kNeedinit = header("NEEDINIT");
const std::array<char, kHeaderLen> kForceready = header("FORCEREADY");

bool same_header(const char *got, const std::array<char, kHeaderLen> &want) {
  return std::memcmp(got, want.data(), kHeaderLen) == 0;
}

std::string header_text(const char *got) {
  std::string text(got, got + kHeaderLen);
  for (char &ch : text) {
    if (ch < 32 || ch > 126) {
      ch = '?';
    }
  }
  while (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }
  return text;
}

void wait_fd(int fd, short event, const Deadline &deadline, const char *what) {
  while (true) {
    if (deadline.expired()) {
      fail(std::string("timed out ") + what);
    }
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = event;
    const int rc = ::poll(&pfd, 1, deadline.remaining_ms());
    if (rc == 0) {
      fail(std::string("timed out ") + what);
    }
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail_errno(what);
    }
    if ((pfd.revents & (POLLERR | POLLNVAL)) != 0) {
      fail(what);
    }
    return;
  }
}

void write_full(int fd, const void *data, size_t n, const Deadline &deadline) {
  const auto *bytes = static_cast<const char *>(data);
  size_t sent = 0;
  while (sent < n) {
    wait_fd(fd, POLLOUT, deadline, "writing to the i-PI driver");
#ifdef MSG_NOSIGNAL
    constexpr int kSendFlags = MSG_NOSIGNAL;
#else
    constexpr int kSendFlags = 0;
#endif
    const ssize_t rc = ::send(fd, bytes + sent, n - sent, kSendFlags);
    if (rc < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == EPIPE || errno == ECONNRESET) {
        fail("i-PI driver disconnected");
      }
      fail_errno("writing to the i-PI driver");
    }
    if (rc == 0) {
      fail("i-PI driver disconnected");
    }
    sent += static_cast<size_t>(rc);
  }
}

void read_full(int fd, void *data, size_t n, const Deadline &deadline) {
  auto *bytes = static_cast<char *>(data);
  size_t got = 0;
  while (got < n) {
    wait_fd(fd, POLLIN, deadline, "reading from the i-PI driver");
    const ssize_t rc = ::recv(fd, bytes + got, n - got, 0);
    if (rc < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == ECONNRESET) {
        fail("i-PI driver disconnected");
      }
      fail_errno("reading from the i-PI driver");
    }
    if (rc == 0) {
      fail("i-PI driver disconnected");
    }
    got += static_cast<size_t>(rc);
  }
}

void read_header(int fd, char out[kHeaderLen], const Deadline &deadline) {
  read_full(fd, out, kHeaderLen, deadline);
}

void append_bytes(std::vector<char> &buf, const void *data, size_t n) {
  const auto *bytes = static_cast<const char *>(data);
  buf.insert(buf.end(), bytes, bytes + n);
}

void set_cloexec(int fd) {
  const int flags = ::fcntl(fd, F_GETFD);
  if (flags >= 0) {
    ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
}

void set_nonblock(int fd) {
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags >= 0) {
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
}

void disable_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
  const int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
  (void)fd;
#endif
}

int listen_tcp(const std::string &host, int port, std::string *endpoint) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  const std::string port_text = std::to_string(port);
  const int gai = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &res);
  if (gai != 0 || res == nullptr) {
    fail(std::string("cannot resolve ") + host + ": " + gai_strerror(gai));
  }
  int fd = -1;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (fd < 0) {
      continue;
    }
    set_cloexec(fd);
    set_nonblock(fd);
    disable_sigpipe(fd);
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, it->ai_addr, it->ai_addrlen) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) {
    fail_errno("binding the TCP socket");
  }
  if (::listen(fd, 1) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("listening on the TCP socket");
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &len) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("reading the bound TCP port");
  }
  *endpoint = "tcp:" + host + ":" + std::to_string(ntohs(bound.sin_port));
  return fd;
}

int listen_unix(const std::string &path, std::string *endpoint) {
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::invalid_argument("IPIPot: UNIX socket path is too long");
  }
  ::unlink(path.c_str());
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    fail_errno("opening a UNIX socket");
  }
  set_cloexec(fd);
  set_nonblock(fd);
  disable_sigpipe(fd);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size());
  addr.sun_path[path.size()] = '\0';
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("binding the UNIX socket");
  }
  if (::listen(fd, 1) != 0) {
    const int saved = errno;
    ::close(fd);
    ::unlink(path.c_str());
    errno = saved;
    fail_errno("listening on the UNIX socket");
  }
  *endpoint = "unix:" + path;
  return fd;
}

// Columns of h are the lattice vectors (i-PI). rgpot stores those vectors
// as rows, in Angstrom. The inverse is of the bohr matrix that is sent.
void pack_cell(const double *box_angstrom, double h[9], double ih[9]) {
  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = box_angstrom[i] * units::ANGSTROM_TO_BOHR;
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      h[row * 3 + col] = box_bohr[col * 3 + row];
    }
  }
  const double a = h[0], b = h[1], c = h[2];
  const double d = h[3], e = h[4], f = h[5];
  const double g = h[6], hh = h[7], i = h[8];
  const double det =
      a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
  if (!std::isfinite(det) || det == 0.0) {
    throw std::invalid_argument("IPIPot: cell matrix is singular");
  }
  const double s = 1.0 / det;
  ih[0] = (e * i - f * hh) * s;
  ih[1] = (c * hh - b * i) * s;
  ih[2] = (b * f - c * e) * s;
  ih[3] = (f * g - d * i) * s;
  ih[4] = (a * i - c * g) * s;
  ih[5] = (c * d - a * f) * s;
  ih[6] = (d * hh - e * g) * s;
  ih[7] = (b * g - a * hh) * s;
  ih[8] = (a * e - b * d) * s;
}

void parse_address(const std::string &address, bool *tcp, std::string *host,
                   int *port, std::string *unix_path) {
  if (address.rfind("unix:", 0) == 0) {
    const std::string rest = address.substr(5);
    if (rest.empty()) {
      throw std::invalid_argument("IPIPot: UNIX address is empty");
    }
    *tcp = false;
    if (rest.front() == '/') {
      *unix_path = rest;
    } else {
      if (rest.find('/') != std::string::npos) {
        throw std::invalid_argument(
            "IPIPot: UNIX name must not contain '/'; use unix:/absolute/path");
      }
      *unix_path = "/tmp/ipi_" + rest;
    }
    return;
  }
  if (address.rfind("tcp:", 0) != 0) {
    throw std::invalid_argument(
        "IPIPot: address must start with unix: or tcp:");
  }
  *tcp = true;
  const std::string rest = address.substr(4);
  const auto colon = rest.rfind(':');
  std::string port_text;
  if (colon == std::string::npos) {
    *host = "127.0.0.1";
    port_text = rest;
  } else {
    *host = rest.substr(0, colon);
    if (host->empty()) {
      *host = "127.0.0.1";
    }
    port_text = rest.substr(colon + 1);
  }
  if (port_text.empty()) {
    throw std::invalid_argument("IPIPot: TCP port is missing");
  }
  try {
    std::size_t used = 0;
    const int value = std::stoi(port_text, &used);
    if (used != port_text.size() || value < 0 || value > 65535) {
      throw std::invalid_argument("range");
    }
    *port = value;
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument("IPIPot: TCP port must be an integer 0..65535");
  } catch (const std::out_of_range &) {
    throw std::invalid_argument("IPIPot: TCP port must be an integer 0..65535");
  }
}

} // namespace

IPIPot::IPIPot(const IPIConfig &config)
    : Potential(PotType::IPI), m_config(config) {
  if (!(config.timeout_s >= 0.0) || !std::isfinite(config.timeout_s)) {
    throw std::invalid_argument("IPIPot: timeout_s must be finite and >= 0");
  }
  bindSocket();
  Fnv1a fp;
  fp.u64(1);
  fp.str(m_config.address);
  fp.f64(m_config.timeout_s);
  m_paramsKey = fp.h;
}

IPIPot::~IPIPot() {
  std::lock_guard<std::mutex> lock(m_mu);
  if (m_conn >= 0) {
    try {
      const Deadline deadline = Deadline::after(1.0);
      write_full(m_conn, kExit.data(), kExit.size(), deadline);
    } catch (...) {
    }
  }
  closeConn();
  closeListen();
}

void IPIPot::bindSocket() {
  bool tcp = false;
  std::string host;
  int port = 0;
  std::string path;
  parse_address(m_config.address, &tcp, &host, &port, &path);
  m_tcp = tcp;
  if (tcp) {
    m_listen = listen_tcp(host, port, &m_endpoint);
  } else {
    m_unix_path = path;
    m_listen = listen_unix(path, &m_endpoint);
  }
}

void IPIPot::closeConn() const {
  if (m_conn >= 0) {
    ::close(m_conn);
    m_conn = -1;
  }
}

void IPIPot::closeListen() {
  if (m_listen >= 0) {
    ::close(m_listen);
    m_listen = -1;
  }
  if (!m_unix_path.empty()) {
    ::unlink(m_unix_path.c_str());
    m_unix_path.clear();
  }
}

bool IPIPot::hasStress() const {
  std::lock_guard<std::mutex> lock(m_mu);
  return m_has_stress != 0;
}

bool IPIPot::copyStress(double out[9]) const {
  std::lock_guard<std::mutex> lock(m_mu);
  if (m_has_stress == 0 || out == nullptr) {
    return false;
  }
  std::copy(m_stress, m_stress + 9, out);
  return true;
}

void IPIPot::evaluateLocked(const double h[9], const double ih[9],
                            const double *pos_bohr, size_t n_atoms,
                            double *energy, double *forces,
                            double virial[9]) const {
  const Deadline deadline = Deadline::after(m_config.timeout_s);
  if (m_conn < 0) {
    while (m_conn < 0) {
      wait_fd(m_listen, POLLIN, deadline, "waiting for an i-PI driver");
      const int fd = ::accept(m_listen, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
          continue;
        }
        fail_errno("accepting an i-PI driver");
      }
      set_cloexec(fd);
      set_nonblock(fd);
      disable_sigpipe(fd);
      if (m_tcp) {
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      }
      m_conn = fd;
    }
  }
  bool ready = false;
  for (int attempt = 0; attempt < 8 && !ready; ++attempt) {
    write_full(m_conn, kStatus.data(), kStatus.size(), deadline);
    char reply[kHeaderLen];
    read_header(m_conn, reply, deadline);
    if (same_header(reply, kReady)) {
      ready = true;
    } else if (same_header(reply, kNeedinit)) {
      std::vector<char> init;
      init.reserve(kHeaderLen + 8);
      append_bytes(init, kInit.data(), kInit.size());
      const int32_t rid = 0;
      const int32_t len = 0;
      append_bytes(init, &rid, sizeof(rid));
      append_bytes(init, &len, sizeof(len));
      write_full(m_conn, init.data(), init.size(), deadline);
    } else {
      fail("unexpected driver header '" + header_text(reply) + "'");
    }
  }
  if (!ready) {
    fail("driver did not become READY");
  }

  const auto n32 = static_cast<int32_t>(n_atoms);
  std::vector<char> payload;
  payload.reserve(kHeaderLen * 2 + 8 * (18 + static_cast<size_t>(n32) * 3) + 4);
  append_bytes(payload, kPosdata.data(), kPosdata.size());
  append_bytes(payload, h, 9 * sizeof(double));
  append_bytes(payload, ih, 9 * sizeof(double));
  append_bytes(payload, &n32, sizeof(n32));
  append_bytes(payload, pos_bohr,
               static_cast<size_t>(n32) * 3 * sizeof(double));
  append_bytes(payload, kGetforce.data(), kGetforce.size());
  write_full(m_conn, payload.data(), payload.size(), deadline);

  char reply[kHeaderLen];
  read_header(m_conn, reply, deadline);
  if (!same_header(reply, kForceready)) {
    fail("unexpected driver header '" + header_text(reply) + "'");
  }
  read_full(m_conn, energy, sizeof(double), deadline);
  int32_t nat = 0;
  read_full(m_conn, &nat, sizeof(nat), deadline);
  if (nat != n32) {
    fail("driver atom count does not match the geometry");
  }
  read_full(m_conn, forces, static_cast<size_t>(nat) * 3 * sizeof(double),
            deadline);
  read_full(m_conn, virial, 9 * sizeof(double), deadline);
  int32_t extra = 0;
  read_full(m_conn, &extra, sizeof(extra), deadline);
  if (extra < 0 || extra > kMaxExtra) {
    fail("driver extra string length is invalid");
  }
  if (extra > 0) {
    std::vector<char> sink(static_cast<size_t>(extra));
    read_full(m_conn, sink.data(), sink.size(), deadline);
  }
}

void IPIPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr || out->F == nullptr) {
    throw std::invalid_argument("IPIPot: force output is missing");
  }
  if (in.pos == nullptr || in.box == nullptr) {
    throw std::invalid_argument("IPIPot: positions and cell are required");
  }
  if (in.nAtoms == 0) {
    throw std::invalid_argument("IPIPot: at least one atom is required");
  }
  if (in.nAtoms > static_cast<size_t>(kMaxAtoms)) {
    throw std::invalid_argument(
        "IPIPot: atom count exceeds the protocol limit");
  }
  const size_t n3 = in.nAtoms * 3;
  for (size_t i = 0; i < n3; ++i) {
    if (!std::isfinite(in.pos[i])) {
      throw std::invalid_argument("IPIPot: positions must be finite");
    }
  }
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(in.box[i])) {
      throw std::invalid_argument("IPIPot: cell must be finite");
    }
  }

  std::vector<double> pos_bohr(n3);
  for (size_t i = 0; i < n3; ++i) {
    pos_bohr[i] = in.pos[i] * units::ANGSTROM_TO_BOHR;
  }
  double h[9];
  double ih[9];
  pack_cell(in.box, h, ih);

  double energy_ha = 0.0;
  std::vector<double> forces_ha(n3);
  double virial_ha[9]{};
  {
    std::lock_guard<std::mutex> lock(m_mu);
    m_has_stress = 0;
    try {
      evaluateLocked(h, ih, pos_bohr.data(), in.nAtoms, &energy_ha,
                     forces_ha.data(), virial_ha);
    } catch (...) {
      closeConn();
      throw;
    }

    out->energy = energy_ha * units::HARTREE_TO_EV;
    out->variance = 0.0;
    for (size_t i = 0; i < n3; ++i) {
      out->F[i] = forces_ha[i] * units::HARTREE_BOHR_TO_EV_ANGSTROM;
    }
    const double volume = cellVolume(in.box);
    std::fill(out->stress, out->stress + 9, 0.0);
    if (volume > 0.0) {
      const double scale = -units::HARTREE_TO_EV / volume;
      for (int i = 0; i < 9; ++i) {
        out->stress[i] = virial_ha[i] * scale;
        m_stress[i] = out->stress[i];
      }
      out->has_stress = 1;
      m_has_stress = 1;
    } else {
      out->has_stress = 0;
    }
  }
}

} // namespace rgpot
