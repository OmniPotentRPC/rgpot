// MIT License
// Copyright 2023--present rgpot developers
//
// Minimal driver around eOn's QSC class (client/potentials/QSC/QSC.cpp at
// eOn commit 3b2586f). It links the unmodified eOn source and prints the
// energy and forces of every case it reads from stdin, with 17 significant
// digits. gen_eon_pure.py builds and runs it; see that script for the exact
// compile line and the input grammar.
//
// stdin grammar (whitespace separated):
//   nparams
//   nparams x { Z n m epsilon c a }   // set_qsc_parameter overrides
//   cutoff verlet_skin
//   ncases
//   ncases x { name N bx by bz ; N x { Z x y z } }   // orthorhombic box
//
// stdout: per case "case <name> <N>", "energy <E>", N x "force fx fy fz".

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "client/potentials/QSC/QSC.h"

namespace {

struct Param {
  int z;
  double n, m, eps, c, a;
};

} // namespace

int main() {
  std::ios::sync_with_stdio(false);
  int nparams = 0;
  std::cin >> nparams;
  std::vector<Param> params(static_cast<size_t>(nparams));
  for (auto &p : params) {
    std::cin >> p.z >> p.n >> p.m >> p.eps >> p.c >> p.a;
  }
  double cutoff = 0.0;
  double skin = 0.0;
  std::cin >> cutoff >> skin;

  int ncases = 0;
  std::cin >> ncases;
  for (int k = 0; k < ncases; ++k) {
    std::string name;
    long n = 0;
    double bx = 0.0, by = 0.0, bz = 0.0;
    std::cin >> name >> n >> bx >> by >> bz;
    std::vector<int> z(static_cast<size_t>(n));
    std::vector<double> pos(static_cast<size_t>(3 * n));
    for (long i = 0; i < n; ++i) {
      std::cin >> z[static_cast<size_t>(i)] >> pos[3 * i] >> pos[3 * i + 1] >>
          pos[3 * i + 2];
    }
    const double box[9] = {bx, 0.0, 0.0, 0.0, by, 0.0, 0.0, 0.0, bz};

    // A fresh potential per case: eOn's Verlet list is stateful.
    QSC pot;
    for (const auto &p : params) {
      pot.set_qsc_parameter(p.z, p.n, p.m, p.eps, p.c, p.a);
    }
    pot.set_cutoff(cutoff);
    pot.set_verlet_skin(skin);

    std::vector<double> forces(static_cast<size_t>(3 * n), 0.0);
    double energy = 0.0;
    pot.force(n, pos.data(), z.data(), forces.data(), &energy, nullptr, box);
    std::printf("case %s %ld\n", name.c_str(), n);
    std::printf("energy %.17g\n", energy);
    for (long i = 0; i < n; ++i) {
      std::printf("force %.17g %.17g %.17g\n", forces[3 * i],
                  forces[3 * i + 1], forces[3 * i + 2]);
    }
  }
  return 0;
}
