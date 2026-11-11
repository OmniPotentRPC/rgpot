// Emacs: this is -*- C++ -*-

#include "Image.h"
#include <vector>
using std::vector;

class SuperCell;
class Atoms;
class Vec;

namespace rgpot::nlist {
class Neighbours;
}

/// Half neighbour list backed by the shared cell walk.
///
/// GetNeighbors returns one direction of each pair inside the cutoff.
/// A stored call keeps a Verlet list. A streaming call keeps the bins
/// and writes one atom's partners into the caller buffers.
class NeighborList {
public:
  /// `mode` is 0 (auto), 1 (stored) or 2 (stream).
  NeighborList(Atoms *a, double rCut, double driftfactor = 0.05, int mode = 1);
  ~NeighborList();

  bool CheckAndUpdateNeighborList();

  int GetNeighbors(int n, int *neighbors, Vec *diffs, double *diffs2, int &size,
                   double r = -1.0) const;

  void UpdateSuperCell(const SuperCell *superCell);

  const int *GetList(int a, int &n);

  const Image &GetImage(int i) { return images[i]; }

  int MaxNeighborListLength() const { return maxlistlen; }

  int GetNumberOfAtoms() const { return nAtoms; }

  /// 0 auto, 1 stored, 2 stream. The next update rebuilds.
  void setMode(int mode);

  /// Stored partner indices. Zero while streaming.
  int slots() const;

private:
  void MakeList();
  void Allocate();
  void bindCells();

private:
  Atoms *atoms;
  int nAtoms;
  int nSize;
  double rCut;
  double rCut2;
  double drift2;
  int mode_;
  bool forceRebuild;
  Vec *positions;
  vector<Vec> referencePositions;
  vector<Image> images;
  int maxlistlen;
  rgpot::nlist::Neighbours *cells;
};
