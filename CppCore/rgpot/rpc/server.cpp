// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"

// Fully define CuH2PotImpl before use
class CuH2PotImpl final : public CuH2Pot::Server {
public:
  kj::Promise<void> calculate(CalculateContext context) override {
    // --- Use the adapter to convert FROM Cap'n Proto ---
    auto fip = context.getParams().getFip();
    const auto numAtoms = fip.getNatm();

    // Use the adapter functions for cleaner conversion
    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    std::vector<int> nativeAtomTypes =
        rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(
            fip.getAtmnrs());
    std::array<std::array<double, 3>, 3> nativeBoxMatrix =
        rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(fip.getBox());

    // --- Call the core library code ---
    auto cuh2pot = rgpot::CuH2Pot();
    auto [energy, forces] =
        cuh2pot(nativePositions, nativeAtomTypes, nativeBoxMatrix);

    // --- Use the adapter to populate TO Cap'n Proto ---
    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy);

    // Initialize and use the adapter to populate forces
    auto forcesList = pres.initForces(numAtoms * 3);
    rgpot::types::adapt::capnp::populateForcesToCapnp(forcesList, forces);

    return kj::READY_NOW;
  }
};

// Main server setup
int main(int argc, char *argv[]) {
  // Set up the Cap'n Proto RPC server on a specific address and port
  // Parse port from command line arguments, default to 12345
  int port = 12345;
  if (argc > 1) {
    try {
      port = std::stoi(argv[1]);
    } catch (const std::exception &e) {
      std::cerr << "Invalid port argument, using default port 12345."
                << std::endl;
    }
  }

  // Set up the Cap'n Proto RPC server on a specific address and port
  capnp::EzRpcServer server(kj::heap<CuH2PotImpl>(), "localhost", port);

  // Keep the server running indefinitely
  auto &waitScope = server.getWaitScope();
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}
