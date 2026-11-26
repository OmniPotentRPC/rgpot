// MIT License
// Copyright 2026 Rohit Goswami

// The analytic checks of the MD session ABI on the nblib engine, the same
// harness the OpenMM engine runs: equipartition on a hyperplane, the mean
// constraint force against dF/dc of the harmonic well, seeded replay and
// Maxwell-Boltzmann reseeding.

#include "engines/common/md_session_chain_nblib.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  using namespace rgpot::analytic;
  try {
    require(argc == 2, "usage: test_nblib_analytic <nblib-engine>");
    const Chain chain;
    const std::string topology = chainNblibTopology(chain);
    char error[1024] = {};
    Session session(rgpot_md_session_create("nblib", argv[1], topology.data(),
                                            topology.size(), nullptr, error,
                                            sizeof(error)));
    require(session.handle != nullptr, std::string("nblib: ") + error);
    Harness harness(session.handle, chain);
    const std::vector<double> box{50.0, 0.0, 0.0, 0.0, 50.0, 0.0,
                                  0.0,  0.0, 50.0};
    harness.prepare(box);
    harness.checkReplay();
    harness.checkReseed();
    const Sampled sampled = harness.sample(6000, 100, 1103);
    checkAgainstAnalytic(sampled, chain, "nblib");
    std::cout << "nblib: <K> = " << number(sampled.kinetic.value) << " +- "
              << number(sampled.kinetic.error)
              << " eV, <U> = " << number(sampled.potential.value) << " +- "
              << number(sampled.potential.error)
              << " eV, <lambda> = " << number(sampled.constraint.value)
              << " +- " << number(sampled.constraint.error)
              << " eV/A (dF/dc = " << number(chain.meanConstraintForce())
              << ")\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << '\n';
    return EXIT_FAILURE;
  }
}
