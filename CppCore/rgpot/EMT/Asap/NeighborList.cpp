#include "NeighborList.h"
#include "Atoms.h"
#include "GhostAtoms.h"
#include "GhostPotential.h"
#include "SuperCell.h"
// #include "Timing.h"
#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <map>
#include <string.h>
#include <utility>
#include <vector>

using std::cerr;
using std::endl;
using std::flush;
extern int verbose;

NeighborList::NeighborList(Atoms *a, double rCut, double driftfactor,
                           int defaultNeighborEstimate)
    : atoms(a) {
  // Driftfactor defaults to 0.05, but is set to 0.0 by CNA and friends
  if (verbose > 2)
    cerr << "NeighborList::NeighborList beginning" << endl;
  indices = 0;
  Allocate(); // Initializes nAtoms and nSize
  neighborList = 0;
  rCut2 = rCut * rCut;
  double drift = driftfactor * rCut;
  drift2 = drift * drift;
  rMax = rCut + 2.0 * drift;
  positions = atoms->GetPositionsPtr();
  // following is estimated number of neighbors. The lists will only store
  // half-neighbors, but a safety factor of 2 is a good thing. We make sure to
  // have at least 50 neighbors per atom though; for example a surface
  // involving a vacuum region would lower the apparent density
  double density = nSize / atoms->GetSuperCell()->GetVolume();
  double PI = 4.0 * atan(1.0);
  int estNbrsPerAtom =
      static_cast<int>(density * 4 * PI * rCut * rCut * rCut / 3.);
  if (estNbrsPerAtom < defaultNeighborEstimate)
    estNbrsPerAtom = defaultNeighborEstimate;

  nMax = estNbrsPerAtom * nSize;

  if (verbose > 2)
    cerr << "NeighborList::NeighborList calling MakeList()" << endl;
  atoms->NormalizePositions();
  MakeList();
  if (verbose > 2)
    cerr << "NeighborList::NeighborList calling UpdateImagePositions()" << endl;
  UpdateImagePositions();
  if (verbose > 2)
    cerr << "NeighborList::NeighborList completed." << endl;
  CheckAndUpdateNeighborList();
}

NeighborList::~NeighborList() {
  if (indices != 0)
    delete[] indices;
  if (neighborList != 0)
    delete[] neighborList;
}

void NeighborList::Allocate() {
  nAtoms = atoms->GetNumberOfRealAtoms();
  nSize = nAtoms;
  // Do we have any ghosts?
  GhostAtoms *ghostAtoms;
  //  try {
  //	ghostAtoms = dynamic_cast<GhostAtoms *>(atoms); //use below line for
  // windows

  //  }catch(...)
  //  {
  ghostAtoms = NULL;
  //  }

  if (ghostAtoms != 0)
    nSize += ghostAtoms->GetNumberOfGhosts(); // Yes!
  if (indices != 0)
    delete[] indices;
  indices = new int[nAtoms + 1];
}

/// \bug Changing the supercell while scaling the atomic positions has
/// little effect on the interatomic distances, but that little effect
/// should be included in the decision to update the neighborlist.
/// That is not done, so currently this function doesn't do anything.
void NeighborList::UpdateSuperCell(const SuperCell *superCell) {}

void NeighborList::MakeList() {
  // USETIMER("NeighborList::MakeList");
  if (verbose >= 1)
    cerr << " NeighborList-Update ";

  if (verbose > 2) {
    Vec realmax;
    Vec realmin;
    Vec ghmax;
    Vec ghmin;
    const Vec *pos = atoms->GetCartesianPositions();
    realmin = realmax = pos[0];
    for (int i = 1; i < nAtoms; i++)
      for (int j = 0; j < 3; j++) {
        if (realmax[j] < pos[i][j])
          realmax[j] = pos[i][j];
        if (realmin[j] > pos[i][j])
          realmin[j] = pos[i][j];
      }
    if (nSize > nAtoms) {
      ghmin = ghmax = pos[nAtoms];
      for (int i = nAtoms; i < nSize; i++)
        for (int j = 0; j < 3; j++) {
          if (ghmax[j] < pos[i][j])
            ghmax[j] = pos[i][j];
          if (ghmin[j] > pos[i][j])
            ghmin[j] = pos[i][j];
        }
    }
    cerr << "  Real atoms [" << nAtoms << "]:  " << realmin << " - " << realmax
         << endl;
    if (nSize > nAtoms)
      cerr << "  Ghost atoms [" << nSize - nAtoms << "]: " << ghmin << " - "
           << ghmax << endl;
  }

  // The shared neighbour query replaces the private cell list. The half
  // rule matches the old list: a direct pair is stored on the lower atom
  // index, and a periodic image is stored on the side whose translation
  // index is at most 13 (the 27-slot centre).
  const SuperCell *cell = atoms->GetSuperCell();
  // The old cell list rounded positions through scaled space and back, and
  // walked cells in (z, y, x) order. The force sum follows that order, so
  // both stay. The pairs themselves come from the shared query.
  cell->TransformToScaledSpace(positions, nSize);
  int bins[3];
  std::vector<int> bin(static_cast<std::size_t>(nSize) * 3, 0);
  for (int axis = 0; axis < 3; ++axis) {
    bins[axis] = static_cast<int>(cell->heights[axis] / rMax);
    if (bins[axis] < 1)
      bins[axis] = 1;
  }
  for (int atom = 0; atom < nSize; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      int k = 0;
      if (cell->periodic[axis]) {
        double scaled = positions[atom][axis];
        k = static_cast<int>(scaled * bins[axis]);
        if (k == bins[axis])
          k--;
        if (k < 0)
          k = 0;
      }
      bin[static_cast<std::size_t>(atom) * 3 + static_cast<std::size_t>(axis)] = k;
    }
  }
  cell->TransformToRealSpace(positions, nSize);
  double box[9];
  for (int axis = 0; axis < 3; ++axis) {
    for (int k = 0; k < 3; ++k)
      box[axis * 3 + k] = cell->vectors[axis][k];
  }
  rgpot::nlist::Query query;
  query.cutoff = rMax;
  query.cut = rgpot::nlist::Cut::Strict;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.periodic = {cell->periodic[0], cell->periodic[1], cell->periodic[2]};
  query.fields.shifts = true;
  query.fields.vectors = true;
  query.fields.distances = false;
  const Vec *cart = atoms->GetCartesianPositions();
  rgpot::nlist::Table table;
  if (nAtoms > 0) {
    rgpot::nlist::build(table, cart[0].x, static_cast<std::size_t>(nAtoms), box, query);
  }
  const double rMax2 = rMax * rMax;
  struct Slot {
    int id;
    int j0;
    int j1;
    int j2;
    int real_id;
  };
  std::vector<std::vector<Slot>> owned(static_cast<std::size_t>(nAtoms));
  images.resize(0);
  std::map<std::pair<int, int>, int> imageOf;
  auto coord = [&](int atom, int axis) {
    return bin[static_cast<std::size_t>(atom) * 3 + static_cast<std::size_t>(axis)];
  };
  for (std::size_t p = 0; p < table.size(); ++p) {
    const int i = table.first[p];
    const int j = table.second[p];
    const int sa = table.shift[3 * p];
    const int sb = table.shift[3 * p + 1];
    const int sc = table.shift[3 * p + 2];
    const double vx = table.vector[3 * p];
    const double vy = table.vector[3 * p + 1];
    const double vz = table.vector[3 * p + 2];
    if (!(vx * vx + vy * vy + vz * vz < rMax2))
      continue;
    if (sa == 0 && sb == 0 && sc == 0) {
      if (j > i)
        owned[static_cast<std::size_t>(i)].push_back(
            Slot{j, coord(j, 0), coord(j, 1), coord(j, 2), j});
      continue;
    }
    if (sa < -1 || sa > 1 || sb < -1 || sb > 1 || sc < -1 || sc > 1)
      continue;
    const int nTranslation = 13 + sa + 3 * sb + 9 * sc;
    if (nTranslation > 13)
      continue;
    const std::pair<int, int> key(j, nTranslation);
    int image = 0;
    const auto found = imageOf.find(key);
    if (found == imageOf.end()) {
      image = nSize + static_cast<int>(images.size());
      images.push_back(Image(j, nTranslation));
      imageOf.emplace(key, image);
    } else {
      image = found->second;
    }
    owned[static_cast<std::size_t>(i)].push_back(
        Slot{image, coord(j, 0) + sa * bins[0], coord(j, 1) + sb * bins[1],
             coord(j, 2) + sc * bins[2], j});
  }
  int npairs = 0;
  for (int atom = 0; atom < nAtoms; ++atom)
    npairs += static_cast<int>(owned[static_cast<std::size_t>(atom)].size());
  if (neighborList != 0)
    delete[] neighborList;
  nMax = npairs > 0 ? npairs : 1;
  neighborList = new int[nMax];
  int cursor = 0;
  indices[0] = 0;
  for (int atom = 0; atom < nAtoms; ++atom) {
    auto &row = owned[static_cast<std::size_t>(atom)];
    std::sort(row.begin(), row.end(), [](const Slot &a, const Slot &b) {
      if (a.j2 != b.j2)
        return a.j2 < b.j2;
      if (a.j1 != b.j1)
        return a.j1 < b.j1;
      if (a.j0 != b.j0)
        return a.j0 < b.j0;
      return a.real_id < b.real_id;
    });
    for (const Slot &neighbor : row)
      neighborList[cursor++] = neighbor.id;
    indices[atom + 1] = cursor;
  }
  nMax = cursor;

  atoms->SetNumberOfImages(images.size());
  // this could have changed the adress of positions. Get it again:
  positions = atoms->GetPositionsPtr();

  // update reference positions for neighbor list:
  referencePositions.resize(nAtoms);
  memcpy(&referencePositions[0], &positions[0], nAtoms * 3 * sizeof(double));

  // Now find the maximal list length
  maxlistlen = 0;
  for (int i = 0; i < nAtoms; i++)
    if (indices[i + 1] - indices[i] > maxlistlen)
      maxlistlen = indices[i + 1] - indices[i];
  if (verbose > 1)
    cerr << "NeighborList::MakeList completed\n";
}

bool NeighborList::CheckAndUpdateNeighborList() {
  // USETIMER("NeighborList::CheckAndUpdateNeighborList");
  // if (verbose > 2)
  //   cerr << "NeighborList::CheckAndUpdate" << endl;

  bool updateRequired = false;
  for (int n = 0; n < nAtoms; n++)
    if (Length2(positions[n] - referencePositions[n]) > drift2) {
      updateRequired = true;
      break;
    }

  // try to get GhostAtoms class:
  GhostAtoms *ghostAtoms;
  //	try {
  //		ghostAtoms = dynamic_cast<GhostAtoms *>(atoms);
  //
  //  }catch(...)
  //  {
  ghostAtoms = NULL;
  //  }

  // if any atom on any processor has moved more than "drift" then migrate all
  // atoms and make new ghost atoms:
  bool reallocationRequired = false;
  if (ghostAtoms != 0) {
    updateRequired =
        ghostAtoms->GetGhostPotential()->CheckAndUpdateAtoms(updateRequired);
    // We might need to reallocate some stuff, if the number of atoms or
    // the number of ghosts has changed. Also the potential might want to
    // do some reallocation (that's why we return "reallocationRequired"
    // from this method.
    if (atoms->GetNumberOfRealAtoms() != nAtoms ||
        ghostAtoms->GetNumberOfGhosts() != nSize - nAtoms) {
      reallocationRequired = true;
      Allocate();
    }
  } else if (updateRequired) {
    // ghostAtoms->GetGhostPotential()->CheckAndUpdateAtoms has called
    // atoms->NormalizePositions, but for simulations without ghosts it must
    // also be done.
    atoms->NormalizePositions();
  }
  // update ghost positions:
  if (ghostAtoms != 0)
    ghostAtoms->GetGhostPotential()->UpdateGhostPositions();

  if (updateRequired)
    MakeList();
  UpdateImagePositions();
  return reallocationRequired;
}

void NeighborList::UpdateImagePositions() {
  const Vec *translations = atoms->GetSuperCell()->translations;
  for (unsigned int n = 0; n < images.size(); n++)
    positions[nSize + n] =
        positions[images[n].number] + translations[images[n].nTranslation];
}

int NeighborList::GetNeighbors(int a1, int *neighbors, Vec *diffs,
                               double *diffs2, int &size, double r) const {
  int i0 = indices[a1];
  int i1 = indices[a1 + 1];
  if (i1 - i0 > size) {
    cerr << "NBLIST OVERRUN" << endl << flush;
    assert(i1 - i0 > maxlistlen);
    size = -1;
    return 0;
  }
  double rC2 = rCut2;
  if (r > 0.0)
    rC2 = r * r;
  Vec pos1 = positions[a1];
  int nNeighbors = 0;
  for (int i = i0; i < i1; i++) {
    int a2 = neighborList[i];
    diffs[nNeighbors] = positions[a2] - pos1;
    double d2 = Length2(diffs[nNeighbors]);
    if (d2 < rC2) {
      diffs2[nNeighbors] = d2;
      if (a2 < nSize)
        neighbors[nNeighbors] = a2;
      else
        neighbors[nNeighbors] = images[a2 - nSize].number;
      nNeighbors++;
    }
  }
  size -= nNeighbors;
  assert(size >= 0);
  return nNeighbors;
}

const int *NeighborList::GetList(int a, int &n) {
  n = indices[a + 1] - indices[a];
  return neighborList + indices[a];
}
