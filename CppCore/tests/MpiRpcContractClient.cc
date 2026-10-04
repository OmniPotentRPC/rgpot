// MIT License
#include "rgpot/abi/Compat.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/units.hpp"
#include <capnp/ez-rpc.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
}

int main(int argc, char **argv) {
  try {
    require(argc == 4, "need address, backend, and serial or mpi mode");
    const std::string backend = argv[2];
    const bool multi = std::string(argv[3]) == "mpi";
    capnp::EzRpcClient client(argv[1]);
    auto potential = client.getMain<Potential>();
    auto &wait = client.getWaitScope();
    {
      auto request = potential.getCapabilitiesRequest();
      auto response = request.send().wait(wait);
      const std::string why =
          rgpot::abi::check_capabilities(response.getCapabilities());
      require(why.empty(), why.c_str());
    }
    auto configure = [&](int charge, const char *title, bool accepted) {
      auto request = potential.configureRequest();
      auto configuration = request.initConfig();
      if (backend == "NWChem") {
        auto parameters = configuration.initNwchem();
        parameters.setCharge(charge);
        parameters.setTitle(title);
      } else {
        auto parameters = configuration.initCpmd();
        parameters.setCharge(charge);
        parameters.setTitle(title);
      }
      auto response = request.send().wait(wait);
      require(response.getOk() == accepted, "collective configuration result changed");
      if (!accepted)
        require(std::string(response.getMessage().cStr()).find("rank 1") != std::string::npos,
                "worker configuration rejection did not reach client");
    };
    auto calculate = [&](int charge, bool fail) {
      const std::vector<double> positions = {fail ? 13.0 : 0.1, -0.2, 0.3,
                                             -0.4, 0.5, -0.6};
      auto request = potential.calculateRequest();
      auto input = request.initFip();
      auto pos = input.initPos(6);
      for (unsigned i = 0; i < 6; ++i)
        pos.set(i, positions[i]);
      auto numbers = input.initAtmnrs(2);
      numbers.set(0, 1);
      numbers.set(1, 8);
      auto box = input.initBox(9);
      for (unsigned i = 0; i < 9; ++i)
        box.set(i, i % 4 == 0 ? 100.0 : 0.0);
      input.setLengthUnit("angstrom");
      input.setEnergyUnit("eV");
      try {
        auto response = request.send().wait(wait);
        require(!fail, "worker force failure was reported as a success");
        auto result = response.getResult();
        double expected_energy = charge + 0.09;
        for (double coordinate : positions)
          expected_energy += 0.5 * coordinate * coordinate;
        expected_energy *= rgpot::units::HARTREE_TO_EV;
        require(std::abs(result.getEnergy() - expected_energy) < 1e-11,
                "collective RPC energy differs from independent oracle");
        auto forces = result.getForces();
        require(forces.size() == 6, "collective RPC force count changed");
        for (unsigned i = 0; i < 6; ++i) {
          const double expected = (0.001 * (i + 1) + 0.01 * positions[i]) *
                                  rgpot::units::NEG_GRAD_TO_FORCE;
          require(std::abs(forces[i] - expected) < 1e-12,
                  "collective RPC force differs from independent oracle");
        }
      } catch (const kj::Exception &error) {
        require(fail, "successful RPC calculation raised an exception");
        const std::string message = error.getDescription().cStr();
        require(message.find("rank 1") != std::string::npos &&
                    message.find("fixture worker force failure") != std::string::npos,
                "RPC error does not contain the worker failure");
      }
    };
    configure(2, "accepted", true);
    calculate(2, false);
    if (multi) {
      calculate(2, true);
      calculate(2, false);
      configure(4, "reject-worker", false);
    }
    configure(3, "recovered", true);
    calculate(3, false);
    std::printf("%s %s RPC contract passed\n", backend.c_str(), multi ? "mpi" : "serial");
    return 0;
  } catch (const kj::Exception &error) {
    std::fprintf(stderr, "%s\n", error.getDescription().cStr());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
  }
  return 1;
}
