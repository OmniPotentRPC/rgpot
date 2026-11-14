#pragma once

#include "rgpot/nlist/NeighbourList.hpp"

#include <vector>

namespace rgpot {
namespace swi {

struct Bond {
  int j;
  double x, y, z, r, gexp, dgdr;
};

// Counted slices and slot prefixes for a stored cell of at least 1024 atoms.
// Kept out of StillingerWeberSi::evaluate so the one-thread walk stays put.
void prepareWide(nlist::Neighbours &cells, int n, int nt, double cut2, std::vector<Bond> &bonds,
                 std::vector<int> &bond_at, std::vector<int> &bond_n, std::vector<int> &energy_at,
                 std::vector<int> &energy_n, std::vector<int> &energy_used,
                 std::vector<int> &force_at, std::vector<int> &force_n,
                 std::vector<int> &force_used);

} // namespace swi
} // namespace rgpot
