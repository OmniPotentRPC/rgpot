// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "Potentials.capnp.h"
#include <capnp/ez-rpc.h>
#include <capnp/message.h>

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

// Fully define CuH2PotImpl before use
class CuH2PotImpl final : public CuH2Pot::Server {
public:
  kj::Promise<void> calculate(CalculateContext context) override {
    // Extract params directly from Cap'n Proto
    // (zero-copy) is a suggestion, not now
    auto fip = context.getParams().getFip();

    // Get number of atoms (rows in positions)
    const auto numAtoms = fip.getNatm();

    // Read directly
    // Read position data from Cap'n Proto and copy to std::vector
    rgpot::types::AtomMatrix nativePositions(numAtoms, 3);
    auto posReader = fip.getPos();
    for (size_t i = 0; i < numAtoms * 3; ++i) {
      nativePositions.data()[i] = posReader[i];
    }

    // Read atom numbers
    auto atmnrsReader = fip.getAtmnrs();
    std::vector<int> nativeAtomTypes(numAtoms);
    for (size_t i = 0; i < numAtoms; ++i) {
      nativeAtomTypes[i] = atmnrsReader[i];
    }

    // Read box matrix
    auto boxReader = fip.getBox();
    std::array<std::array<double, 3>, 3> nativeBoxMatrix;
    for (size_t i = 0; i < 3; ++i) {
      nativeBoxMatrix[i] = {boxReader[i], boxReader[i + 1], boxReader[i + 2]};
    }

    // Call CuH2Pot with the AtomMatrix and other parameters
    auto cuh2pot = rgpot::CuH2Pot();
    auto [energy, forces] =
        cuh2pot(nativePositions, nativeAtomTypes, nativeBoxMatrix);

    // Set up the result in Cap'n Proto
    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy);

    // Initialize and set forces
    auto forcesList = pres.initForces(numAtoms * 3);
    for (size_t i = 0; i < numAtoms * 3; ++i) {
      forcesList.set(i, forces.data()[i]);
    }

    return kj::READY_NOW;
  }
};

// Main server setup
int main(int argc, char *argv[]) {
  CuH2PotImpl cuh2potImpl;

  // Set up the Cap'n Proto RPC server on a specific address and port
  capnp::EzRpcServer server(kj::heap<CuH2PotImpl>(), "localhost", 12345);

  // Keep the server running indefinitely
  auto &waitScope = server.getWaitScope();
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}
