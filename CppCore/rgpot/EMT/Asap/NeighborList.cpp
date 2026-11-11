#include "NeighborList.h"
#include "Atoms.h"
#include "SuperCell.h"
#include "Vec.h"

#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

using std::cerr;
using std::endl;
using std::flush;
extern int verbose;

namespace {

rgpot::nlist::Mode modeOf(int mode) {
  if (mode == 2) {
    return rgpot::nlist::Mode::Stream;
  }
  if (mode == 0) {
    return rgpot::nlist::Mode::Auto;
  }
  return rgpot::nlist::Mode::Stored;
}

void boxOf(const SuperCell *cell, double box[9], std::array<bool, 3> &periodic) {
  for (int a = 0; a < 3; ++a) {
    const Vec v = cell->vectors[a];
    box[3 * a] = v[0];
    box[3 * a + 1] = v[1];
    box[3 * a + 2] = v[2];
    periodic[static_cast<std::size_t>(a)] = cell->periodic[a];
  }
}

} // namespace

NeighborList::NeighborList(Atoms *a, double rCut, double driftfactor, int mode)
    : atoms(a), mode_(mode), forceRebuild(true), cells(new rgpot::nlist::Neighbours) {
  if (verbose > 2)
    cerr << "NeighborList::NeighborList beginning" << endl;
  Allocate();
  this->rCut = rCut;
  rCut2 = rCut * rCut;
  const double drift = driftfactor * rCut;
  drift2 = drift * drift;
  positions = atoms->GetPositionsPtr();
  atoms->NormalizePositions();
  MakeList();
  if (verbose > 2)
    cerr << "NeighborList::NeighborList completed." << endl;
  CheckAndUpdateNeighborList();
}

NeighborList::~NeighborList() { delete cells; }

void NeighborList::Allocate() {
  nAtoms = atoms->GetNumberOfRealAtoms();
  nSize = nAtoms;
  atoms->SetNumberOfImages(0);
}

void NeighborList::UpdateSuperCell(const SuperCell *superCell) { (void)superCell; }

void NeighborList::setMode(int mode) {
  if (mode_ == mode) {
    return;
  }
  mode_ = mode;
  forceRebuild = true;
}

int NeighborList::slots() const {
  if (cells == nullptr || cells->streaming()) {
    return 0;
  }
  return static_cast<int>(cells->slots());
}

void NeighborList::bindCells() {
  positions = atoms->GetPositionsPtr();
  double box[9] = {};
  std::array<bool, 3> periodic{{true, true, true}};
  boxOf(atoms->GetSuperCell(), box, periodic);
  const double drift = std::sqrt(drift2);
  const bool stream = modeOf(mode_) == rgpot::nlist::Mode::Stream;
  const double skin = stream ? 0.0 : 2.0 * drift;
  const double *xyz = nAtoms > 0 ? reinterpret_cast<const double *>(positions) : nullptr;
  rgpot::nlist::updateCells(*cells, xyz, static_cast<std::size_t>(std::max(nAtoms, 0)), box,
                            rCut, skin, rgpot::nlist::Image::Round, modeOf(mode_), periodic);
}

void NeighborList::MakeList() {
  if (verbose >= 1)
    cerr << " NeighborList-Update ";
  Allocate();
  bindCells();
  referencePositions.resize(static_cast<std::size_t>(nAtoms));
  if (nAtoms > 0) {
    std::memcpy(&referencePositions[0], &positions[0],
                static_cast<std::size_t>(nAtoms) * 3 * sizeof(double));
  }
  maxlistlen = 0;
  if (nAtoms < 2) {
    forceRebuild = false;
    return;
  }
  // One direction of every candidate the walk can return. GetNeighbors
  // drops those outside the cutoff, so this length is an upper bound.
  for (int atom = 0; atom < nAtoms; ++atom) {
    int count = 0;
    cells->forAtom(atom, [&](std::int32_t, std::int32_t j, double, double, double, double r2) {
      if (j > atom && r2 > 0.0) {
        ++count;
      }
    });
    if (count > maxlistlen) {
      maxlistlen = count;
    }
  }
  forceRebuild = false;
  if (verbose > 1)
    cerr << "NeighborList::MakeList completed\n";
}

bool NeighborList::CheckAndUpdateNeighborList() {
  positions = atoms->GetPositionsPtr();
  bool updateRequired = forceRebuild || static_cast<int>(referencePositions.size()) != nAtoms;
  // A stream trusts the bins, so any movement rebuilds them. A stored
  // list survives until an atom leaves the drift.
  const double limit2 = (cells != nullptr && cells->streaming()) ? 0.0 : drift2;
  if (!updateRequired) {
    for (int n = 0; n < nAtoms; n++) {
      if (Length2(positions[n] - referencePositions[static_cast<std::size_t>(n)]) > limit2) {
        updateRequired = true;
        break;
      }
    }
  }
  if (updateRequired) {
    atoms->NormalizePositions();
    positions = atoms->GetPositionsPtr();
    MakeList();
  }
  return false;
}

int NeighborList::GetNeighbors(int a1, int *neighbors, Vec *diffs, double *diffs2, int &size,
                               double r) const {
  if (a1 < 0 || a1 >= nAtoms || cells == nullptr) {
    size = 0;
    return 0;
  }
  double rC2 = rCut2;
  if (r > 0.0)
    rC2 = r * r;
  int nNeighbors = 0;
  bool overrun = false;
  auto take = [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
    if (!(j > a1) || !(r2 > 0.0) || !(r2 < rC2)) {
      return;
    }
    if (nNeighbors >= size) {
      overrun = true;
      return;
    }
    // r_j - r_i. The layer reports r_i - r_j.
    diffs[nNeighbors] = Vec(-dx, -dy, -dz);
    diffs2[nNeighbors] = r2;
    neighbors[nNeighbors] = static_cast<int>(j);
    nNeighbors++;
  };
  if (cells->streaming()) {
    cells->forAtom(a1, take);
  } else {
    rgpot::nlist::visitDirected(*cells, a1, static_cast<std::int64_t>(a1) + 1, take);
  }
  if (overrun) {
    cerr << "NBLIST OVERRUN" << endl << flush;
    size = -1;
    return 0;
  }
  size -= nNeighbors;
  return nNeighbors;
}

const int *NeighborList::GetList(int a, int &n) {
  (void)a;
  n = 0;
  return nullptr;
}
