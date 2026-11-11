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
#include "rgpot/rpc/loopback.hpp"
#include <algorithm>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <capnp/rpc-twoparty.h>
#include <capnp/rpc.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <kj/async-io.h>
#include <kj/common.h>
#include <memory>
#include <string>

#include <poll.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define RGPOT_PAUSE() _mm_pause()
#else
#define RGPOT_PAUSE()                                                          \
  do {                                                                         \
  } while (0)
#endif

/// After a write, poll until the peer's reply is queued or the budget runs
/// out. The event loop's epoll_wait then returns without a scheduler sleep
/// when the reply landed during the poll. The bytes stay in the socket for
/// the normal read. Past the budget (a large system) we block in epoll as
/// before, so the spin does not stretch a slow call.
class SpinAfterWrite final : public kj::AsyncIoStream {
public:
  SpinAfterWrite(kj::Own<kj::AsyncIoStream> inner, int fd,
                 const std::shared_ptr<int> &budget_us)
      : inner(kj::mv(inner)), fd(fd), budget_us(budget_us) {}

  kj::Promise<size_t> tryRead(void *buffer, size_t minBytes,
                              size_t maxBytes) override {
    return inner->tryRead(buffer, minBytes, maxBytes);
  }

  kj::Promise<void> write(const void *buffer, size_t size) override {
    auto written = inner->write(buffer, size);
    if (budget() <= 0)
      return written;
    return written.then([this]() { spin(); });
  }

  kj::Promise<void> write(
      kj::ArrayPtr<const kj::ArrayPtr<const kj::byte>> pieces) override {
    auto written = inner->write(pieces);
    if (budget() <= 0)
      return written;
    return written.then([this]() { spin(); });
  }

  kj::Promise<void> whenWriteDisconnected() override {
    return inner->whenWriteDisconnected();
  }

  void shutdownWrite() override { inner->shutdownWrite(); }

private:
  int budget() const { return budget_us ? *budget_us : 0; }

  void spin() const {
    const int usec = budget();
    if (usec <= 0)
      return;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(usec);
    while (std::chrono::steady_clock::now() < deadline) {
      pollfd pfd{};
      pfd.fd = fd;
      pfd.events = POLLIN;
      if (::poll(&pfd, 1, 0) > 0)
        return;
      RGPOT_PAUSE();
    }
  }

  kj::Own<kj::AsyncIoStream> inner;
  int fd;
  std::shared_ptr<int> budget_us;
};

/// Loopback connection. The fd is owned by the wrapped stream. Members are
/// destroyed in reverse declaration order, so the capability dies before the
/// RPC system, the system before the vat, and the vat before the socket.
struct LocalConn {
  using Rpc = capnp::RpcSystem<capnp::rpc::twoparty::VatId>;

  kj::AsyncIoContext io;
  kj::Own<kj::AsyncIoStream> stream;
  kj::Own<capnp::TwoPartyVatNetwork> network;
  kj::Own<Rpc> rpc;
  // Client's default constructor is protected; the pointer stays null until
  // bootstrap() below.
  std::unique_ptr<Potential::Client> cap;

  explicit LocalConn(int fd, const std::shared_ptr<int> &budget_us)
      : io(kj::setupAsyncIo()),
        stream(kj::heap<SpinAfterWrite>(
            io.lowLevelProvider->wrapSocketFd(
                fd, kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP |
                        kj::LowLevelAsyncIoProvider::ALREADY_CLOEXEC),
            fd, budget_us)) {
    network = kj::heap<capnp::TwoPartyVatNetwork>(
        *stream, capnp::rpc::twoparty::Side::CLIENT);
    rpc = kj::heap<Rpc>(capnp::makeRpcClient(*network));
    ::capnp::word scratch[4];
    std::memset(scratch, 0, sizeof(scratch));
    capnp::MallocMessageBuilder message(scratch);
    auto hostId = message.getRoot<capnp::rpc::twoparty::VatId>();
    hostId.setSide(capnp::rpc::twoparty::Side::SERVER);
    cap = std::make_unique<Potential::Client>(
        rpc->bootstrap(hostId).castAs<Potential>());
  }
};

/**
 * @class PotClient
 * @details
 * Container for the C++ objects required to manage an RPC session. It is
 * exposed to C as an opaque pointer.  The use of @c std::unique_ptr ensures
 * that the RPC client is cleaned up correctly upon destruction.
 */
/// Microseconds. Unset, empty, or not a non-negative integer keeps 40.
int reply_poll_us_from_env() {
  const char *value = std::getenv("RGPOT_RPC_POLL_US");
  if (value == nullptr || *value == '\0')
    return 40;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0)
    return 40;
  if (parsed > 1000000L)
    return 1000000;
  return static_cast<int>(parsed);
}

struct PotClient {
  std::shared_ptr<int> reply_poll_us; //!< Post-write poll budget. 0 disables.
  std::unique_ptr<capnp::EzRpcClient>
      rpc_client;                   //!< TCP client. Empty on the loopback path.
  std::unique_ptr<LocalConn> local; //!< Abstract-socket client, when dialed.
  Potential::Client capability;     //!< The RPC capability for potential calls.
  kj::WaitScope *wait_scope;        //!< Reference to the client wait scope.
  std::string last_error;           //!< Buffer for the most recent error message.
  bool server_checked = false;      //!< True once the server passed the handshake.

  PotClient(int poll_us, std::unique_ptr<capnp::EzRpcClient> client,
            Potential::Client cap)
      : reply_poll_us(std::make_shared<int>(poll_us)),
        rpc_client(std::move(client)), capability(std::move(cap)),
        wait_scope(&rpc_client->getWaitScope()) {}

  PotClient(int poll_us, int fd)
      : reply_poll_us(std::make_shared<int>(poll_us)),
        local(std::make_unique<LocalConn>(fd, reply_poll_us)),
        capability(std::move(*local->cap)), wait_scope(&local->io.waitScope) {}
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
PotClient *pot_client_init_ex(const char *host, int32_t port, int32_t poll_us) {
  if (!host)
    return nullptr;
  const int budget =
      poll_us < 0 ? reply_poll_us_from_env()
                  : (poll_us > 1000000 ? 1000000 : static_cast<int>(poll_us));
  try {
    // Loopback skips the TCP stack when this build's server is listening.
    // connect() fails immediately if it is not, and the TCP client below is
    // still lazy: init succeeds and the first calculate reports the error.
    if (rgpot::rpc::is_loopback_host(host)) {
      const int fd = rgpot::rpc::connect_abstract(port);
      if (fd >= 0)
        return new PotClient(budget, fd);
    }
    std::string address = std::string(host) + ":" + std::to_string(port);
    auto rpc_client = std::make_unique<capnp::EzRpcClient>(address);
    auto cap = rpc_client->getMain().castAs<Potential>();

    return new PotClient(budget, std::move(rpc_client), std::move(cap));
  } catch (...) {
    // In init, we return nullptr. The caller must assume initialization failed.
    // TODO(rg): Advanced logging could go to a thread-local static
    return nullptr;
  }
}

PotClient *pot_client_init(const char *host, int32_t port) {
  return pot_client_init_ex(host, port, -1);
}

void pot_client_set_reply_poll_us(PotClient *client, int32_t poll_us) {
  if (!client || !client->reply_poll_us)
    return;
  if (poll_us < 0)
    poll_us = 0;
  if (poll_us > 1000000)
    poll_us = 1000000;
  *client->reply_poll_us = poll_us;
}

int32_t pot_client_reply_poll_us(const PotClient *client) {
  if (!client || !client->reply_poll_us)
    return 0;
  return *client->reply_poll_us;
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

    // kj::arrayPtr creates a "view" (pointer + size) with NO copying.
    // It is essentially a span.
    auto pos_view = kj::arrayPtr(pos, natoms * 3);
    auto atm_view = kj::arrayPtr(atmnrs, natoms);
    auto box_view = kj::arrayPtr(box, 9);

    // Cap'n Proto sees these views and performs a bulk memcpy
    // directly into the message builder's memory segment.
    fip.setPos(pos_view);
    fip.setAtmnrs(atm_view);
    fip.setBox(box_view);

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
