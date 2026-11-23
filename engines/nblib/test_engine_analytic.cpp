// MIT License
// Copyright 2026 Rohit Goswami

// OpenMM and nblib against each other and against the closed forms of the
// harmonic chain, through the one session ABI. Both engines run the same
// harness; every sampled mean must lie within the standard error of its
// closed form and the two engines must agree within the pooled standard
// error, for the kinetic energy, the potential energy and the mean constraint
// force.

#include "engines/common/md_session_chain_nblib.hpp"
#include "engines/common/md_session_chain_openmm.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  using namespace rgpot::analytic;
  try {
    require(argc == 3,
            "usage: test_engine_analytic <nblib-engine> <openmm-engine>");
    const Chain chain;
    const std::vector<double> box{50.0, 0.0, 0.0, 0.0, 50.0, 0.0,
                                  0.0,  0.0, 50.0};
    char error[1024] = {};

    const std::string topology = chainNblibTopology(chain);
    Session nblib(rgpot_md_session_create("nblib", argv[1], topology.data(),
                                          topology.size(), nullptr, error,
                                          sizeof(error)));
    require(nblib.handle != nullptr, std::string("nblib: ") + error);
    const std::string xml = chainOpenMMXml(chain);
    Session openmm(rgpot_md_session_create("openmm", argv[2], xml.data(),
                                           xml.size(), "Reference", error,
                                           sizeof(error)));
    require(openmm.handle != nullptr, std::string("openmm: ") + error);

    Harness a(nblib.handle, chain);
    Harness b(openmm.handle, chain);
    a.prepare(box);
    b.prepare(box);
    a.checkReplay();
    b.checkReplay();
    const Sampled fromNblib = a.sample(6000, 100, 2207);
    const Sampled fromOpenmm = b.sample(6000, 100, 3301);
    checkAgainstAnalytic(fromNblib, chain, "nblib");
    checkAgainstAnalytic(fromOpenmm, chain, "OpenMM");
    checkAgreement(fromNblib, fromOpenmm, "nblib vs OpenMM");
    std::cout << "nblib and OpenMM agree with each other and with dF/dc = "
              << number(chain.meanConstraintForce())
              << " eV/A: <lambda> = " << number(fromNblib.constraint.value)
              << " +- " << number(fromNblib.constraint.error) << " and "
              << number(fromOpenmm.constraint.value) << " +- "
              << number(fromOpenmm.constraint.error) << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << '\n';
    return EXIT_FAILURE;
  }
}
