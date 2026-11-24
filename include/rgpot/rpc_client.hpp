#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file rpc_client.hpp
 * @author rgpot Developers
 * @date 2026-02-15
 * @brief Move-only RAII wrapper around the rgpot Cap'n Proto RPC client.
 *
 * Provides @c rgpot::RpcClient, a thin C++ handle that connects to a
 * remote rgpot server and proxies @c calculate() calls over Cap'n Proto
 * RPC.  The class is only available when the Rust core is compiled with
 * the @c rpc feature. The build defines the @c RGPOT_HAS_RPC preprocessor
 * macro (Meson adds it with @c with_rust_core); @c rgpot.h only tests it.
 *
 * # Example
 * @code
 * #ifdef RGPOT_HAS_RPC
 * rgpot::RpcClient client("localhost", 12345);
 *
 * double pos[] = {0,0,0, 1,0,0};
 * int    atm[] = {1, 1};
 * double box[] = {10,0,0, 0,10,0, 0,0,10};
 * rgpot::InputSpec input(2, pos, atm, box);
 *
 * auto result = client.calculate(input);
 * double energy = result.energy();
 * #endif
 * @endcode
 *
 * @ingroup rgpot_cpp
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rgpot.h"
#include "errors.hpp"
#include "types.hpp"

#ifdef RGPOT_HAS_RPC

namespace rgpot {

/**
 * @brief Neighbour query sent by @c RpcClient::neighbours().
 */
struct RpcNeighborQuery {
  double cutoff = 0.0;
  double skin = 0.0;
  uint8_t format = 0;
  uint8_t cut = 0;
  std::array<bool, 3> periodic{{true, true, true}};
  bool shifts = true;
  bool vectors = true;
  bool distances = true;
};

/**
 * @brief Neighbour list returned by @c RpcClient::neighbours().
 */
struct RpcNeighborList {
  std::vector<int32_t> first;
  std::vector<int32_t> second;
  std::vector<int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<uint8_t> mask;
};

/**
 * @class RpcClient
 * @brief Move-only RAII wrapper around @c rgpot_rpc_client_t.
 * @ingroup rgpot_cpp
 *
 * Owns an opaque RPC client handle allocated by the Rust core.  The
 * connection is established during construction and torn down on
 * destruction via @c rgpot_rpc_client_free().  The class is non-copyable;
 * move semantics transfer ownership of the connection.
 */
class RpcClient final {
public:
  /**
   * @brief Connect to a remote rgpot server.
   *
   * Calls @c rgpot_rpc_client_new() and throws on failure.
   *
   * @param host Hostname or IP address of the server.
   * @param port TCP port the server listens on.
   * @throws rgpot::Error if the connection cannot be established.
   */
  RpcClient(const std::string &host, uint16_t port) {
    handle_ = rgpot_rpc_client_new(host.c_str(), port);
    if (!handle_) {
      const char *msg = rgpot_last_error();
      throw Error(msg ? msg : "failed to create RPC client");
    }
  }

  /**
   * @brief Destructor — disconnects and frees the Rust-side client.
   *
   * Safe to call on a moved-from handle (internal pointer is @c nullptr).
   */
  ~RpcClient() {
    if (handle_) {
      rgpot_rpc_client_free(handle_);
    }
  }

  /**
   * @brief Move constructor — transfers connection ownership.
   * @param other Client to move from; left in a null state.
   */
  RpcClient(RpcClient &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  /**
   * @brief Move assignment — releases current connection, transfers
   *        ownership.
   * @param other Client to move from; left in a null state.
   * @return Reference to @c *this.
   */
  RpcClient &operator=(RpcClient &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        rgpot_rpc_client_free(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  RpcClient(const RpcClient &) = delete;
  RpcClient &operator=(const RpcClient &) = delete;

  /**
   * @brief Select the image or bead whose orbitals following calls should use.
   */
  void select_orbitals(std::int64_t key) {
    detail::check_status(rgpot_rpc_client_select_orbitals(handle_, key));
  }

  /**
   * @brief Perform a remote force/energy calculation.
   *
   * Serialises the input via Cap'n Proto, sends the request to the
   * server, and deserialises the response into a @c CalcResult.
   *
   * @param input The atomic configuration to evaluate.
   * @return A @c CalcResult containing energy, variance, and forces.
   * @throws rgpot::Error on RPC transport or server-side failure.
   */
  CalcResult calculate(const InputSpec &input) {
    CalcResult result;
    auto status =
        rgpot_rpc_calculate(handle_, &input.c_struct(), &result.c_struct());
    detail::check_status(status);
    return result;
  }

  /**
   * @brief Request the remote potential's neighbour list.
   *
   * @param input The atomic configuration to query.
   * @param query Cutoff, skin, shell/format, cut mode and requested columns.
   * @return A @c RpcNeighborList holding the returned arrays.
   * @throws rgpot::Error on RPC transport or server-side failure.
   */
  RpcNeighborList neighbours(const InputSpec &input,
                             const RpcNeighborQuery &query) {
    rgpot_rpc_neighbors_request_t req{};
    req.cutoff = query.cutoff;
    req.skin = query.skin;
    req.format = query.format;
    req.cut = query.cut;
    req.periodic[0] = query.periodic[0] ? 1 : 0;
    req.periodic[1] = query.periodic[1] ? 1 : 0;
    req.periodic[2] = query.periodic[2] ? 1 : 0;
    req.shifts = query.shifts ? 1 : 0;
    req.vectors = query.vectors ? 1 : 0;
    req.distances = query.distances ? 1 : 0;
    size_t count = 0;
    size_t needed = 0;
    detail::check_status(rgpot_rpc_neighbours(
        handle_, &input.c_struct(), &req, nullptr, &count, &needed));
    RpcNeighborList list;
    if (needed == 0) {
      return list;
    }
    list.first.resize(needed);
    list.second.resize(needed);
    list.mask.resize(needed);
    if (query.shifts) {
      list.shift.resize(needed * 3);
    }
    if (query.vectors) {
      list.vector.resize(needed * 3);
    }
    if (query.distances) {
      list.distance.resize(needed);
    }
    rgpot_rpc_neighbors_out_t out{};
    out.first = list.first.data();
    out.second = list.second.data();
    out.shift = query.shifts ? list.shift.data() : nullptr;
    out.vector = query.vectors ? list.vector.data() : nullptr;
    out.distance = query.distances ? list.distance.data() : nullptr;
    out.mask = list.mask.data();
    out.cap = needed;
    detail::check_status(rgpot_rpc_neighbours(
        handle_, &input.c_struct(), &req, &out, &count, &needed));
    return list;
  }

private:
  rgpot_rpc_client_t *handle_ = nullptr; //!< Owned opaque RPC client handle.
};

} // namespace rgpot

#endif // RGPOT_HAS_RPC
