// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"

class GenericPotImpl final : public Potential::Server {
private:
  std::unique_ptr<rgpot::PotentialBase> m_potential;

public:
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)) {}

  kj::Promise<void> calculate(CalculateContext context) override {
    auto fip = context.getParams().getFip();
    const auto numAtoms = fip.getNatm();

    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    std::vector<int> nativeAtomTypes =
        rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(
            fip.getAtmnrs());
    std::array<std::array<double, 3>, 3> nativeBoxMatrix =
        rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(fip.getBox());

    // Call via the virtual operator() on PotentialBase
    auto [energy, forces] =
        (*m_potential)(nativePositions, nativeAtomTypes, nativeBoxMatrix);

    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy);

    auto forcesList = pres.initForces(numAtoms * 3);
    rgpot::types::adapt::capnp::populateForcesToCapnp(forcesList, forces);

    return kj::READY_NOW;
  }
};

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <PotentialType>" << std::endl;
    std::cerr << "  Available PotentialTypes: CuH2, LJ" << std::endl;
    return 1;
  }

  int port = 12345;
  try {
    port = std::stoi(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid port argument '" << argv[1]
              << "'. Using default 12345." << std::endl;
  }

  std::string pot_type = argv[2];
  std::unique_ptr<rgpot::PotentialBase> potential_to_use;

  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::CuH2Pot>();
  } else if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
  } else {
    std::cerr << "Error: Unknown potential type '" << pot_type << "'"
              << std::endl;
    return 1;
  }

  capnp::EzRpcServer server(
      kj::heap<GenericPotImpl>(std::move(potential_to_use)), "localhost", port);

  auto &waitScope = server.getWaitScope();
  std::cout << "Server running on port " << port << " with " << pot_type
            << " potential." << std::endl;
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}
