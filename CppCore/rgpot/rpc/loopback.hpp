#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Loopback transport for a co-located potserv.
 *
 * The Cap'n Proto bytes are unchanged. On Linux a client whose host is
 * loopback dials the abstract socket `rgpot-<port>` first and uses TCP only
 * when nothing is listening there, so an older server keeps working. The
 * server binds that name beside its TCP port. Abstract sockets disappear
 * with the process, so a crashed server does not leave a filesystem node.
 */

#include <string>
#include <string_view>

namespace rgpot::rpc {

inline bool is_loopback_host(std::string_view host) {
  return host == "localhost" || host == "127.0.0.1" || host == "::1" ||
         host == "[::1]";
}

inline std::string abstract_address(int port) {
  return "unix-abstract:rgpot-" + std::to_string(port);
}

/// Connected abstract-namespace fd, or -1. The caller owns a non-negative fd.
int connect_abstract(int port);

} // namespace rgpot::rpc
