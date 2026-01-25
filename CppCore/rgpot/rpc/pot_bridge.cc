#include "pot_bridge.h"
#include "Potentials.capnp.h"
#include <algorithm>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/common.h>
#include <memory>
#include <string>

struct PotClient {
  std::unique_ptr<capnp::EzRpcClient> rpc_client;
  Potential::Client capability;
  kj::WaitScope *wait_scope;
  std::string last_error;

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

PotClient *pot_client_init(const char *host, int32_t port) {
  // We cannot use the client struct for storage yet, so we use a temporary
  if (!host)
    return nullptr;
  try {
    std::string address = std::string(host) + ":" + std::to_string(port);
    auto rpc_client = std::make_unique<capnp::EzRpcClient>(address);
    auto cap = rpc_client->getMain().castAs<Potential>();

    return new PotClient(std::move(rpc_client), std::move(cap));
  } catch (...) {
    // In init, we return nullptr. The caller must assume initialization failed.
    // TODO(rg): Advanced logging could go to a thread-local static
    return nullptr;
  }
}

void pot_client_free(PotClient *client) {
  if (client) {
    delete client;
  }
}

int32_t pot_calculate(PotClient *client, int32_t natoms, const double *pos,
                      const int32_t *atmnrs, const double *box,
                      double *out_energy, double *out_forces) {

  // Safety Checks
  if (!client)
    return -1;
  client->last_error.clear();

  try {
    auto &waitScope = *client->wait_scope;
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

const char *pot_get_last_error(PotClient *client) {
  if (!client || client->last_error.empty()) {
    return "";
  }
  return client->last_error.c_str();
}

} // extern "C"
