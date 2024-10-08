#include "Potentials.capnp.h" // Cap'n Proto schema header
#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <iostream>

// Fully define CuH2PotImpl before use
class CuH2PotImpl final : public CuH2Pot::Server {
public:
  kj::Promise<void> calculate(CalculateContext context) override {
    // Extract params directly from Cap'n Proto
    auto atomMatrixReader = context.getParams().getPositions();
    auto atomTypesReader = context.getParams().getAtomTypes();
    auto boxMatrixReader = context.getParams().getBoxMatrix();

    // Get number of atoms (rows in positions)
    const auto numAtoms = atomMatrixReader.getPositions().size();

    // Use positions directly from Cap'n Proto (zero-copy)
    auto positionsReader = atomMatrixReader.getPositions();

    // Access atom types directly from Cap'n Proto (zero-copy)
    auto atomTypes = atomTypesReader.getAtomTypes();

    // Access box matrix directly (zero-copy)
    auto boxMatrix = boxMatrixReader.getBox();

    // Prepare input in the format expected by CuH2Pot
    rgpot::types::AtomMatrix nativePositions(numAtoms, 3);
    for (size_t i = 0; i < numAtoms; ++i) {
      nativePositions(i, 0) = positionsReader[i].getX();
      nativePositions(i, 1) = positionsReader[i].getY();
      nativePositions(i, 2) = positionsReader[i].getZ();
    }

    std::vector<int> nativeAtomTypes(atomTypes.size());
    for (size_t i = 0; i < atomTypes.size(); ++i) {
      nativeAtomTypes[i] = atomTypes[i];
    }

    std::array<std::array<double, 3>, 3> nativeBoxMatrix;
    for (size_t i = 0; i < 3; ++i) {
      nativeBoxMatrix[i] = {boxMatrix[i].getX(), boxMatrix[i].getY(),
                            boxMatrix[i].getZ()};
    }

    // Call CuH2Pot with the AtomMatrix and other parameters
    auto cuh2pot = rgpot::CuH2Pot();
    auto [energy, forces] =
        cuh2pot(nativePositions, nativeAtomTypes, nativeBoxMatrix);

    // Set up the result in Cap'n Proto
    auto result = context.getResults();
    ::capnp::MallocMessageBuilder message;
    PotentialResult::Builder pres = message.initRoot<PotentialResult>();
    pres.setEnergy(energy);
    pres.initForces(numAtoms);

    for (int i = 0; i < forces.rows(); ++i) {
      auto force = ::capnp::MallocMessageBuilder().initRoot<ForceVector>();
      force.setX(forces(i, 0));
      force.setY(forces(i, 1));
      force.setZ(forces(i, 2));
    }
    result.setResult(pres);

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
