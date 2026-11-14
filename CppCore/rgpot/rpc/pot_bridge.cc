// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the C API bridge for the RPC client.
 *
 * This file implements the C-compatible wrapper for the Cap'n Proto RPC client.
 * It manages the lifecycle of the EzRpcClient and handles error reporting
 * through a simple string-based interface.
 */

#include "pot_bridge.h"
#include "Potentials.capnp.h"
#include "rgpot/abi/Handshake.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"
#include <algorithm>
#include <cstring>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/common.h>
#include <memory>
#include <string>

/**
 * @class PotClient
 * @details
 * Container for the C++ objects required to manage an RPC session. It is
 * exposed to C as an opaque pointer.  The use of @c std::unique_ptr ensures
 * that the RPC client is cleaned up correctly upon destruction.
 */
struct PotClient {
  std::unique_ptr<capnp::EzRpcClient>
      rpc_client;               //!< The Cap'n Proto RPC client.
  Potential::Client capability; //!< The RPC capability for potential calls.
  kj::WaitScope *wait_scope;    //!< Reference to the client wait scope.
  std::string last_error;       //!< Buffer for the most recent error message.
  bool server_checked = false;  //!< True once the server passed the handshake.

  /**
   * @brief Constructor for PotClient.
   * @param client Ownership of the initialized EzRpcClient.
   * @param cap The capability client for the Potential interface.
   */
  PotClient(std::unique_ptr<capnp::EzRpcClient> client, Potential::Client cap)
      : rpc_client(std::move(client)), capability(std::move(cap)),
        wait_scope(&rpc_client->getWaitScope()) {}
};

// Helper macros to enforce safety without clutter
#define CATCH_AND_REPORT(client, default_ret)                                  \
  catch (const std::exception &e) {                                            \
    if (client)                                                                \
      client->last_error = e.what();                                           \
    return default_ret;                                                        \
  }                                                                            \
  catch (...) {                                                                \
    if (client)                                                                \
      client->last_error = "Unknown C++ exception";                            \
    return default_ret;                                                        \
  }

extern "C" {

/**
 * @details
 * Constructs a string-based address from the @a host and @a port,
 * then initializes a new @c capnp::EzRpcClient. The main capability
 * is cast to the @c Potential interface defined in the schema.
 *
 * @note Returns @c nullptr if the connection cannot be established
 * or if the address is malformed.
 */
PotClient *pot_client_init(const char *host, int32_t port) {
  // We cannot use the client struct for storage yet, so we use a temporary
  if (!host)
    return nullptr;
  try {
    // "unix:/path" is the KJ address. A hostname still joins the port,
    // which is the TCP default. The messages on the socket are unchanged.
    std::string address;
#if !defined(_WIN32)
    if (std::strncmp(host, "unix:", 5) == 0) {
      address = host;
    } else
#endif
    {
      address = std::string(host) + ":" + std::to_string(port);
    }
    auto rpc_client = std::make_unique<capnp::EzRpcClient>(address);
    auto cap = rpc_client->getMain().castAs<Potential>();

    return new PotClient(std::move(rpc_client), std::move(cap));
  } catch (...) {
    // In init, we return nullptr. The caller must assume initialization failed.
    // TODO(rg): Advanced logging could go to a thread-local static
    return nullptr;
  }
}

/**
 * @details
 * Performs a standard @c delete on the client pointer. This triggers
 * the @c PotClient destructor, which cleans up the @c unique_ptr
 * and associated RPC state.
 */
void pot_client_free(PotClient *client) {
  if (client) {
    delete client;
  }
}

/**
 * @details
 * This function performs the following steps:
 * 1. Resets the @c last_error buffer.
 * 2. Initializes a calculation request.
 * 3. Maps the input pointers to @c kj::arrayPtr views to avoid
 * unnecessary copying before serialization.
 * 4. Waits for the RPC promise to resolve.
 * 5. Validates the size of the returned force array.
 * 6. Copies the results into the provided output buffers.
 *
 * @warning The server must return a force array matching @a natoms * 3.
 * If the sizes do not match, a non-zero error code is returned.
 */
int32_t pot_calculate(PotClient *client, int32_t natoms, const double *pos,
                      const int32_t *atmnrs, const double *box,
                      double *out_energy, double *out_forces) {

  // Safety Checks
  if (!client)
    return -1;
  client->last_error.clear();

  try {
    auto &waitScope = *client->wait_scope;
    if (!client->server_checked) {
      const std::string why =
          rgpot::abi::check_server(client->capability, waitScope);
      if (!why.empty()) {
        client->last_error = "server refused: " + why;
        return -3;
      }
      client->server_checked = true;
    }
    auto req = client->capability.calculateRequest();
    auto fip = req.initFip();

    const auto npos = static_cast<unsigned int>(natoms * 3);
    auto pos_builder = fip.initPos(npos);
    if (double *dest = rgpot::types::adapt::capnp::contiguous_builder<double>(pos_builder)) {
      std::memcpy(dest, pos, sizeof(double) * npos);
    } else {
      for (unsigned int i = 0; i < npos; ++i) {
        pos_builder.set(i, pos[i]);
      }
    }
    auto type_builder = fip.initAtmnrs(static_cast<unsigned int>(natoms));
    if (std::int32_t *dest =
            rgpot::types::adapt::capnp::contiguous_builder<std::int32_t>(type_builder)) {
      std::memcpy(dest, atmnrs, sizeof(std::int32_t) * static_cast<std::size_t>(natoms));
    } else {
      for (int32_t i = 0; i < natoms; ++i) {
        type_builder.set(static_cast<unsigned int>(i), atmnrs[i]);
      }
    }
    fip.setBox(kj::arrayPtr(box, 9));

    // ---------------------------------------------------------

    // Execute RPC
    auto promise = req.send();
    auto response = promise.wait(waitScope);
    auto result = response.getResult();

    *out_energy = result.getEnergy();

    // ---------------------------------------------------------
    // RETRIEVAL - Bulk Copy Back
    // ---------------------------------------------------------

    auto res_forces = result.getForces();

    // Strict runtime check to ensure server didn't return garbage size
    if (res_forces.size() != static_cast<size_t>(natoms * 3)) {
      client->last_error = "Server returned force array of incorrect size";
      return -2;
    }

    for (size_t i = 0; i < res_forces.size(); ++i) {
      out_forces[i] = res_forces[i];
    }

    return 0;
  }
  CATCH_AND_REPORT(client, -1)
}

/**
 * @details
 * Checks if a valid client handle is provided and if the @c last_error
 * buffer contains data. If no error is recorded, an empty string is returned.
 */
const char *pot_get_last_error(PotClient *client) {
  if (!client || client->last_error.empty()) {
    return "";
  }
  return client->last_error.c_str();
}

} // extern "C"
