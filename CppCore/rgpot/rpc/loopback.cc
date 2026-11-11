// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/rpc/loopback.hpp"

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#endif

namespace rgpot::rpc {

int connect_abstract(int port) {
#if !defined(__linux__)
  (void)port;
  return -1;
#else
  if (port <= 0 || port > 65535)
    return -1;
  const std::string name = "rgpot-" + std::to_string(port);
  // sun_path[0] is the abstract marker. KJ's addrlen covers that byte plus
  // the name and not the extra NUL KJ writes past the length for toString().
  if (name.size() + 1 >= sizeof(sockaddr_un::sun_path))
    return -1;
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return -1;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  addr.sun_path[0] = '\0';
  std::memcpy(addr.sun_path + 1, name.data(), name.size());
  const socklen_t len = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + 1 + name.size());
  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), len) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
#endif
}

} // namespace rgpot::rpc
