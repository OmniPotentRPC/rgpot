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
#include <algorithm>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/common.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct NeighborReplyCache {
  int32_t natoms = 0;
  // Content of the geometry, not the caller's pointers. A probe followed by
  // an in-place edit of those buffers must miss this cache.
  std::vector<double> pos;
  std::vector<int32_t> atmnrs;
  std::vector<double> box;
  PotNeighborsRequest request{};
  std::vector<int32_t> first;
  std::vector<int32_t> second;
  std::vector<int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<uint8_t> mask;
};

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
  bool server_checked = false;  //!< True once calculate passed the handshake.
  bool neighbors_checked = false; //!< True once neighbours passed its handshake.
  bool has_orbital_key = false; //!< Whether following requests name a key.
  int64_t orbital_key = 0;      //!< Image or bead key sent with each request.
  std::optional<NeighborReplyCache> neighbor_cache;

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

namespace {

template <typename T, typename Reader>
void copyCapnpList(Reader source, std::vector<T> &destination) {
  destination.resize(source.size());
  for (unsigned i = 0; i < source.size(); ++i)
    destination[i] = source[i];
}

bool sameNeighborRequest(const PotNeighborsRequest &left,
                         const PotNeighborsRequest &right) {
  return left.cutoff == right.cutoff && left.skin == right.skin &&
         left.format == right.format && left.cut == right.cut &&
         left.periodic[0] == right.periodic[0] &&
         left.periodic[1] == right.periodic[1] &&
         left.periodic[2] == right.periodic[2] && left.shifts == right.shifts &&
         left.vectors == right.vectors && left.distances == right.distances;
}

bool sameGeometry(const NeighborReplyCache &cached, int32_t natoms,
                  const double *pos, const int32_t *atmnrs, const double *box) {
  const auto npos = static_cast<size_t>(natoms) * 3;
  if (cached.natoms != natoms || cached.pos.size() != npos ||
      cached.atmnrs.size() != static_cast<size_t>(natoms) ||
      cached.box.size() != 9)
    return false;
  if (natoms > 0 &&
      (!std::equal(cached.pos.begin(), cached.pos.end(), pos) ||
       !std::equal(cached.atmnrs.begin(), cached.atmnrs.end(), atmnrs)))
    return false;
  return std::equal(cached.box.begin(), cached.box.end(), box);
}

bool matchesNeighborProbe(const NeighborReplyCache &cached, int32_t natoms,
                          const double *pos, const int32_t *atmnrs,
                          const double *box,
                          const PotNeighborsRequest &request) {
  return sameGeometry(cached, natoms, pos, atmnrs, box) &&
         sameNeighborRequest(cached.request, request);
}

int32_t copyNeighborReply(PotClient *client, int32_t natoms,
                          const PotNeighborsRequest &request,
                          const NeighborReplyCache &reply, PotNeighborsOut *out,
                          size_t *out_count, size_t *out_needed) {
  const size_t total = reply.first.size();
  if (reply.second.size() != total) {
    client->last_error = "Server returned mismatched pair arrays";
    return -2;
  }
  if (out_needed)
    *out_needed = total;
  if (!out || out->cap == 0)
    return 0;
  if (!out->first || !out->second || (request.shifts && !out->shift) ||
      (request.vectors && !out->vector) ||
      (request.distances && !out->distance) ||
      (request.format == 2 && !out->mask)) {
    client->last_error = "Required neighbour output pointer is null";
    return 2;
  }
  const size_t nwrite = std::min(out->cap, total);
  const size_t triple_total =
      total > std::numeric_limits<size_t>::max() / 3 ? 0 : total * 3;
  if ((request.shifts || request.vectors) && triple_total == 0 && total != 0) {
    client->last_error = "Neighbour list length overflows size checks";
    return -2;
  }
  if (request.shifts && reply.shift.size() < triple_total) {
    client->last_error = "Server returned undersized neighbour shift column";
    return -2;
  }
  if (request.vectors && reply.vector.size() < triple_total) {
    client->last_error = "Server returned undersized neighbour vector column";
    return -2;
  }
  if (request.distances && reply.distance.size() < total) {
    client->last_error = "Server returned undersized neighbour distance column";
    return -2;
  }
  if (out->mask && reply.mask.size() < total) {
    client->last_error = "Server returned undersized neighbour mask column";
    return -2;
  }
  for (size_t i = 0; i < nwrite; ++i) {
    const int32_t center = reply.first[i];
    const int32_t partner = reply.second[i];
    if (center < 0 || partner < 0 || center >= natoms || partner >= natoms) {
      client->last_error = "Server returned neighbour index out of bounds";
      return -2;
    }
    out->first[i] = center;
    out->second[i] = partner;
    if (request.shifts) {
      out->shift[3 * i] = reply.shift[3 * i];
      out->shift[3 * i + 1] = reply.shift[3 * i + 1];
      out->shift[3 * i + 2] = reply.shift[3 * i + 2];
    }
    if (request.vectors) {
      out->vector[3 * i] = reply.vector[3 * i];
      out->vector[3 * i + 1] = reply.vector[3 * i + 1];
      out->vector[3 * i + 2] = reply.vector[3 * i + 2];
    }
    if (request.distances)
      out->distance[i] = reply.distance[i];
    if (out->mask)
      out->mask[i] = reply.mask[i];
  }
  if (out_count)
    *out_count = nwrite;
  return nwrite < total ? 6 : 0;
}

} // namespace

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

/**
 * @details
 * Performs a standard @c delete on the client pointer. This triggers
 * the @c PotClient destructor, which cleans up the @c unique_ptr
 * and associated RPC state.
 */
void pot_client_free(PotClient *client) {
  if (!client) {
    return;
  }
  try {
    delete client;
  } catch (...) {
    // C ABI must not leak exceptions across extern "C". There is no status
    // channel on free, so swallow and return.
  }
}

int32_t pot_client_select_orbitals(PotClient *client, int64_t key) {
  if (!client)
    return -1;
  client->last_error.clear();
  client->has_orbital_key = true;
  client->orbital_key = key;
  client->server_checked = false;
  return 0;
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
  if (natoms < 0) {
    client->last_error = "natoms must be non-negative";
    return -1;
  }
  if (!pos || !atmnrs || !box || !out_energy || !out_forces) {
    client->last_error =
        "pos, atmnrs, box, out_energy and out_forces must be non-null";
    return -1;
  }
  constexpr int32_t kMaxAtoms = std::numeric_limits<int32_t>::max() / 3;
  if (natoms > kMaxAtoms) {
    client->last_error = "natoms is too large";
    return -1;
  }

  try {
    auto &waitScope = *client->wait_scope;
    if (!client->server_checked) {
      const auto expectation =
          rgpot::abi::force_input_expectation(client->has_orbital_key);
      const std::string why =
          rgpot::abi::check_server(client->capability, waitScope, expectation);
      if (!why.empty()) {
        if (client->has_orbital_key &&
            why.find("protocol minor") != std::string::npos)
          client->last_error =
              "ForceInput orbital key requires RPC protocol minor " +
              std::to_string(rgpot::abi::kOrbitalKeyIntroducedMinor) +
              ": server refused: " + why;
        else
          client->last_error = "server refused: " + why;
        return -3;
      }
      client->server_checked = true;
    }
    auto req = client->capability.calculateRequest();
    auto fip = req.initFip();

    // kj::arrayPtr creates a "view" (pointer + size) with NO copying.
    // It is essentially a span.
    auto pos_view = kj::arrayPtr(pos, static_cast<size_t>(natoms) * 3);
    auto atm_view = kj::arrayPtr(atmnrs, natoms);
    auto box_view = kj::arrayPtr(box, 9);

    // Cap'n Proto sees these views and performs a bulk memcpy
    // directly into the message builder's memory segment.
    fip.setPos(pos_view);
    fip.setAtmnrs(atm_view);
    fip.setBox(box_view);
    if (client->has_orbital_key) {
      fip.setHasOrbitalKey(true);
      fip.setOrbitalKey(client->orbital_key);
    }

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
    if (res_forces.size() != static_cast<size_t>(natoms) * 3) {
      client->last_error = "Server returned force array of incorrect size";
      return -2;
    }

    for (size_t i = 0; i < res_forces.size(); ++i) {
      out_forces[i] = res_forces[i];
    }

    return 0;
  } catch (const kj::Exception &error) {
    client->last_error = error.getDescription().cStr();
    return -1;
  }
  CATCH_AND_REPORT(client, -1)
}

int32_t pot_neighbours(PotClient *client, int32_t natoms, const double *pos,
                       const int32_t *atmnrs, const double *box,
                       const PotNeighborsRequest *request, PotNeighborsOut *out,
                       size_t *out_count, size_t *out_needed) {
  if (out_count)
    *out_count = 0;
  if (out_needed)
    *out_needed = 0;
  if (!client || !request || natoms < 0 || (natoms > 0 && (!pos || !atmnrs)) ||
      !box)
    return -1;
  client->last_error.clear();
  constexpr int32_t kMaxAtoms = std::numeric_limits<int32_t>::max() / 3;
  if (natoms > kMaxAtoms) {
    client->last_error = "natoms is too large";
    return -1;
  }
  if (out && out->cap > 0 && client->neighbor_cache) {
    if (matchesNeighborProbe(*client->neighbor_cache, natoms, pos, atmnrs, box,
                             *request)) {
      auto cached = std::move(*client->neighbor_cache);
      client->neighbor_cache.reset();
      return copyNeighborReply(client, natoms, *request, cached, out, out_count,
                               out_needed);
    }
    client->neighbor_cache.reset();
  }

  try {
    auto &waitScope = *client->wait_scope;
    if (!client->neighbors_checked) {
      const auto expectation = rgpot::abi::neighbours_expectation();
      const std::string why =
          rgpot::abi::check_server(client->capability, waitScope, expectation);
      if (!why.empty()) {
        if (why.find("protocol minor") != std::string::npos)
          client->last_error =
              "Potential.neighbours requires RPC protocol minor " +
              std::to_string(rgpot::abi::kNeighborsIntroducedMinor) +
              ": server refused: " + why;
        else if (why.find("not served") != std::string::npos)
          client->last_error =
              "Potential.neighbours requires Capabilities.Operation.neighbors: "
              "server refused: " +
              why;
        else
          client->last_error = "server refused: " + why;
        return -3;
      }
      client->neighbors_checked = true;
    }
    auto req = client->capability.neighboursRequest();
    auto fip = req.initFip();
    fip.setPos(kj::arrayPtr(pos, static_cast<size_t>(natoms) * 3));
    fip.setAtmnrs(kj::arrayPtr(atmnrs, static_cast<size_t>(natoms)));
    fip.setBox(kj::arrayPtr(box, 9));

    auto nreq = req.initRequest();
    nreq.setCutoff(request->cutoff);
    nreq.setSkin(request->skin);
    nreq.setFormat(request->format);
    nreq.setCut(request->cut);
    auto pbc = nreq.initPeriodic(3);
    for (unsigned k = 0; k < 3; ++k)
      pbc.set(k, request->periodic[k] != 0);
    nreq.setShifts(request->shifts != 0);
    nreq.setVectors(request->vectors != 0);
    nreq.setDistances(request->distances != 0);
    nreq.setMode(1);

    auto response = req.send().wait(waitScope);
    auto list = response.getNeighbors();
    if (list.getStatus() != 0) {
      client->last_error = list.getMessage().cStr();
      return list.getStatus();
    }
    NeighborReplyCache reply;
    reply.natoms = natoms;
    if (natoms > 0) {
      reply.pos.assign(pos, pos + static_cast<size_t>(natoms) * 3);
      reply.atmnrs.assign(atmnrs, atmnrs + static_cast<size_t>(natoms));
    }
    reply.box.assign(box, box + 9);
    reply.request = *request;
    const auto first = list.getFirst();
    const auto second = list.getSecond();
    const auto shift = list.getShift();
    const auto vector = list.getVector();
    const auto distance = list.getDistance();
    const auto mask = list.getMask();
    copyCapnpList(first, reply.first);
    copyCapnpList(second, reply.second);
    copyCapnpList(shift, reply.shift);
    copyCapnpList(vector, reply.vector);
    copyCapnpList(distance, reply.distance);
    copyCapnpList(mask, reply.mask);
    const int32_t status = copyNeighborReply(client, natoms, *request, reply,
                                             out, out_count, out_needed);
    if (status == 0 && (!out || out->cap == 0))
      client->neighbor_cache = std::move(reply);
    return status;
  } catch (const kj::Exception &error) {
    client->last_error = error.getDescription().cStr();
    return -1;
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
