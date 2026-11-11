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

bool NeighborList::streaming() const noexcept {
  return cells != nullptr && cells->streaming();
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
  maxlistlen = 1;
  order_.clear();
  center_ = -2;
  plane_ = -2;
  place_.assign(static_cast<std::size_t>(std::max(nAtoms, 0)), -1);
  cached_.clear();
  if (nAtoms < 2) {
    forceRebuild = false;
    return;
  }
  half_off_.clear();
  half_.clear();
  stream_pairs_.clear();
  if (!cells->streaming()) {
    half_off_.assign(static_cast<std::size_t>(nAtoms) + 1, 0);
    const SuperCell *box = atoms->GetSuperCell();
    const double L0 = box->vectors[0][0];
    const double L1 = box->vectors[1][1];
    const double L2 = box->vectors[2][2];
    const double cut2 = rCut2;
    cells->forHalf(0, static_cast<std::int64_t>(nAtoms),
                   [&](std::int32_t i, std::int32_t, double, double, double, double r2) {
                     if (!(r2 > 0.0) || !(r2 < cut2)) {
                       return;
                     }
                     ++half_off_[static_cast<std::size_t>(i) + 1];
                   });
    for (int a = 0; a < nAtoms; ++a) {
      const int count = half_off_[static_cast<std::size_t>(a) + 1];
      if (count > maxlistlen) {
        maxlistlen = count;
      }
      half_off_[static_cast<std::size_t>(a) + 1] += half_off_[static_cast<std::size_t>(a)];
    }
    half_.resize(static_cast<std::size_t>(half_off_[static_cast<std::size_t>(nAtoms)]));
    std::vector<int> cursor(half_off_.begin(), half_off_.end() - 1);
    cells->forHalf(0, static_cast<std::int64_t>(nAtoms),
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
                     if (!(r2 > 0.0) || !(r2 < cut2)) {
                       return;
                     }
                     const double rx = positions[i][0] - positions[j][0];
                     const double ry = positions[i][1] - positions[j][1];
                     const double rz = positions[i][2] - positions[j][2];
                     HalfBond bond;
                     bond.j = static_cast<int>(j);
                     bond.sx = L0 != 0.0 ? static_cast<int>(std::lround((rx - dx) / L0)) : 0;
                     bond.sy = L1 != 0.0 ? static_cast<int>(std::lround((ry - dy) / L1)) : 0;
                     bond.sz = L2 != 0.0 ? static_cast<int>(std::lround((rz - dz) / L2)) : 0;
                     half_[static_cast<std::size_t>(cursor[static_cast<std::size_t>(i)]++)] = bond;
                   });
  } else {
    // The half-list of a metal near this cutoff is far below this. The
    // batch loop reads the bound before the first window is filled.
    maxlistlen = 1024;
    order_.reserve(static_cast<std::size_t>(nAtoms));
    const int ncells = cells->gridCells();
    for (int c = 0; c < ncells; ++c) {
      const std::int32_t *ids = nullptr;
      int n = 0;
      cells->cellAtoms(c, ids, n);
      for (int k = 0; k < n; ++k) {
        order_.push_back(static_cast<int>(ids[k]));
      }
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

void NeighborList::dropCell(int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= cached_.size()) {
    return;
  }
  auto &rows = cached_[static_cast<std::size_t>(cell)];
  if (rows == nullptr) {
    return;
  }
  for (int atom : rows->atoms) {
    if (atom >= 0 && static_cast<std::size_t>(atom) < place_.size()) {
      place_[static_cast<std::size_t>(atom)] = -1;
    }
  }
  rows.reset();
}

void NeighborList::fillCell(int cell) {
  if (cell < 0 || cells == nullptr) {
    return;
  }
  const auto ncells = static_cast<std::size_t>(cells->gridCells());
  if (cached_.size() != ncells) {
    cached_.clear();
    cached_.resize(ncells);
  }
  if (cached_[static_cast<std::size_t>(cell)] != nullptr) {
    return;
  }
  const std::int32_t *ids = nullptr;
  int n = 0;
  cells->cellAtoms(cell, ids, n);
  auto rows = std::make_unique<CellRows>();
  rows->atoms.assign(ids, ids + n);
  rows->off.assign(static_cast<std::size_t>(n) + 1, 0);
  if (lookup_.size() != static_cast<std::size_t>(std::max(nAtoms, 0))) {
    lookup_.assign(static_cast<std::size_t>(std::max(nAtoms, 0)), -1);
  }
  for (int k = 0; k < n; ++k) {
    lookup_[static_cast<std::size_t>(ids[k])] = k;
  }
  struct Item {
    int j;
    double x, y, z, r2;
  };
  std::vector<std::vector<Item>> bonds(static_cast<std::size_t>(std::max(n, 0)));
  cells->forCell(cell, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                           double r2) {
    if (!(r2 > 0.0) || i < 0 || static_cast<std::size_t>(i) >= lookup_.size()) {
      return;
    }
    const int slot = lookup_[static_cast<std::size_t>(i)];
    if (slot < 0) {
      return;
    }
    bonds[static_cast<std::size_t>(slot)].push_back(Item{static_cast<int>(j), -dx, -dy, -dz, r2});
  });
  for (int k = 0; k < n; ++k) {
    auto &row = bonds[static_cast<std::size_t>(k)];
    rows->off[static_cast<std::size_t>(k)] = static_cast<int>(rows->nbr.size());
    for (const Item &it : row) {
      rows->nbr.push_back(it.j);
      rows->xyz.push_back(it.x);
      rows->xyz.push_back(it.y);
      rows->xyz.push_back(it.z);
      rows->r2.push_back(it.r2);
    }
    lookup_[static_cast<std::size_t>(ids[k])] = -1;
    place_[static_cast<std::size_t>(ids[k])] = k;
  }
  rows->off[static_cast<std::size_t>(n)] = static_cast<int>(rows->nbr.size());
  cached_[static_cast<std::size_t>(cell)] = std::move(rows);
}

void NeighborList::fillPlane(int ix) {
  const int ny = cells->gridY();
  const int nz = cells->gridZ();
  if (ny < 1 || nz < 1 || ix < 0) {
    return;
  }
  const int c0 = ix * ny * nz;
  const int c1 = c0 + ny * nz;
  const auto ncells = static_cast<std::size_t>(cells->gridCells());
  if (cached_.size() != ncells) {
    cached_.clear();
    cached_.resize(ncells);
  }
  if (lookup_.size() != static_cast<std::size_t>(std::max(nAtoms, 0))) {
    lookup_.assign(static_cast<std::size_t>(std::max(nAtoms, 0)), -1);
  }
  std::vector<int> atoms;
  for (int c = c0; c < c1; ++c) {
    const std::int32_t *ids = nullptr;
    int n = 0;
    cells->cellAtoms(c, ids, n);
    for (int k = 0; k < n; ++k) {
      lookup_[static_cast<std::size_t>(ids[k])] = static_cast<int>(atoms.size());
      atoms.push_back(static_cast<int>(ids[k]));
    }
  }
  struct Item {
    int j;
    double x, y, z, r2;
  };
  std::vector<std::vector<Item>> bonds(atoms.size());
  cells->forCellRange(c0, c1, [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                                  double r2) {
    if (!(r2 > 0.0) || i < 0 || static_cast<std::size_t>(i) >= lookup_.size()) {
      return;
    }
    const int slot = lookup_[static_cast<std::size_t>(i)];
    if (slot < 0) {
      return;
    }
    bonds[static_cast<std::size_t>(slot)].push_back(
        Item{static_cast<int>(j), -dx, -dy, -dz, r2});
  });
  for (int c = c0; c < c1; ++c) {
    const std::int32_t *ids = nullptr;
    int n = 0;
    cells->cellAtoms(c, ids, n);
    auto rows = std::make_unique<CellRows>();
    rows->atoms.assign(ids, ids + n);
    rows->off.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int li = 0; li < n; ++li) {
      const int atom = static_cast<int>(ids[li]);
      const int slot = lookup_[static_cast<std::size_t>(atom)];
      rows->off[static_cast<std::size_t>(li)] = static_cast<int>(rows->nbr.size());
      if (slot >= 0) {
        for (const Item &it : bonds[static_cast<std::size_t>(slot)]) {
          rows->nbr.push_back(it.j);
          rows->xyz.push_back(it.x);
          rows->xyz.push_back(it.y);
          rows->xyz.push_back(it.z);
          rows->r2.push_back(it.r2);
        }
      }
      place_[static_cast<std::size_t>(atom)] = li;
    }
    rows->off[static_cast<std::size_t>(n)] = static_cast<int>(rows->nbr.size());
    cached_[static_cast<std::size_t>(c)] = std::move(rows);
  }
  for (int atom : atoms) {
    lookup_[static_cast<std::size_t>(atom)] = -1;
  }
}

void NeighborList::slide(int cell) {
  if (cell < 0 || cells == nullptr) {
    return;
  }
  const int ny = cells->gridY();
  const int nz = cells->gridZ();
  const int nx = cells->gridX();
  const int span = ny * nz;
  if (span < 1 || nx < 1) {
    return;
  }
  const int ix = cell / span;
  if (ix == plane_) {
    return;
  }
  auto wrap = [&](int x) {
    if (x < 0 || x >= nx) {
      if (!cells->axisPeriodic(0)) {
        return -1;
      }
      x %= nx;
      if (x < 0) {
        x += nx;
      }
    }
    return x;
  };
  int want[3];
  int nwant = 0;
  for (int d = -1; d <= 1; ++d) {
    const int p = wrap(ix + d);
    if (p < 0) {
      continue;
    }
    bool seen = false;
    for (int k = 0; k < nwant; ++k) {
      if (want[k] == p) {
        seen = true;
      }
    }
    if (!seen) {
      want[nwant++] = p;
    }
  }
  if (cached_.size() == static_cast<std::size_t>(cells->gridCells())) {
    for (int p = 0; p < nx; ++p) {
      bool still = false;
      for (int k = 0; k < nwant; ++k) {
        if (want[k] == p) {
          still = true;
        }
      }
      if (still) {
        continue;
      }
      for (int c = p * span; c < (p + 1) * span; ++c) {
        dropCell(c);
      }
    }
  }
  for (int k = 0; k < nwant; ++k) {
    const int c = want[k] * span;
    if (static_cast<std::size_t>(c) < cached_.size() && cached_[static_cast<std::size_t>(c)] != nullptr) {
      continue;
    }
    fillPlane(want[k]);
  }
  plane_ = ix;
  center_ = cell;
}

int NeighborList::GetNeighbors(int a1, int *neighbors, Vec *diffs, double *diffs2, int &size,
                               double r) {
  if (a1 < 0 || a1 >= nAtoms || cells == nullptr) {
    size = 0;
    return 0;
  }
  double rC2 = rCut2;
  if (r > 0.0)
    rC2 = r * r;
  int nNeighbors = 0;
  bool overrun = false;
  auto take = [&](int j, double x, double y, double z, double r2) {
    if (!(j > a1) || !(r2 > 0.0) || !(r2 < rC2)) {
      return;
    }
    if (nNeighbors >= size) {
      overrun = true;
      return;
    }
    diffs[nNeighbors] = Vec(x, y, z);
    diffs2[nNeighbors] = r2;
    neighbors[nNeighbors] = j;
    nNeighbors++;
  };
  if (cells->streaming()) {
    const int cell = cells->cellOf(a1);
    if (place_.size() != static_cast<std::size_t>(nAtoms) ||
        (cell >= 0 && (place_[static_cast<std::size_t>(a1)] < 0))) {
      slide(cell);
    }
    const int slot = (a1 >= 0 && static_cast<std::size_t>(a1) < place_.size())
                         ? place_[static_cast<std::size_t>(a1)]
                         : -1;
    if (slot >= 0 && cell >= 0 && static_cast<std::size_t>(cell) < cached_.size() &&
        cached_[static_cast<std::size_t>(cell)] != nullptr) {
      const auto &rows = *cached_[static_cast<std::size_t>(cell)];
      const int b = rows.off[static_cast<std::size_t>(slot)];
      const int e = rows.off[static_cast<std::size_t>(slot) + 1];
      for (int p = b; p < e; ++p) {
        take(rows.nbr[static_cast<std::size_t>(p)], rows.xyz[static_cast<std::size_t>(3 * p)],
             rows.xyz[static_cast<std::size_t>(3 * p + 1)],
             rows.xyz[static_cast<std::size_t>(3 * p + 2)], rows.r2[static_cast<std::size_t>(p)]);
      }
    }
  } else if (static_cast<int>(half_off_.size()) == nAtoms + 1) {
    const SuperCell *box = atoms->GetSuperCell();
    const double L0 = box->vectors[0][0];
    const double L1 = box->vectors[1][1];
    const double L2 = box->vectors[2][2];
    const Vec pos1 = positions[a1];
    const int b = half_off_[static_cast<std::size_t>(a1)];
    const int e = half_off_[static_cast<std::size_t>(a1) + 1];
    for (int p = b; p < e; ++p) {
      const HalfBond &bond = half_[static_cast<std::size_t>(p)];
      const Vec d = positions[bond.j] - pos1;
      const double x = d[0] + static_cast<double>(bond.sx) * L0;
      const double y = d[1] + static_cast<double>(bond.sy) * L1;
      const double z = d[2] + static_cast<double>(bond.sz) * L2;
      take(bond.j, x, y, z, x * x + y * y + z * z);
    }
  } else {
    cells->forHalf(static_cast<std::int64_t>(a1), static_cast<std::int64_t>(a1) + 1,
                   [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
                     take(static_cast<int>(j), -dx, -dy, -dz, r2);
                   });
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
