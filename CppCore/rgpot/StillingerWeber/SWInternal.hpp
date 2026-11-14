#pragma once

#include "rgpot/nlist/NeighbourList.hpp"

#include <vector>

namespace rgpot {
namespace swi {

struct Bond {
  int j;
  double x, y, z, r, gexp, dgdr;
};

// Stored cells of at least 256 atoms. Bonds are written into counted slices,
// and each force update is stored in the scatter-order row of its atom.
// Kept out of StillingerWeberSi::evaluate so the one-thread walk stays put.
void runWide(nlist::Neighbours &cells, int n, int nt, double cut2, double scale, double *energy,
             double *acc);

} // namespace swi
} // namespace rgpot
