// Emacs: this is -*- C++ -*-

#include "Image.h"
#include "rgpot/nlist/Neighbours.hpp"
#include <memory>
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
                   double r = -1.0);

  void UpdateSuperCell(const SuperCell *superCell);

  const int *GetList(int a, int &n);

  const Image &GetImage(int i) { return images[i]; }

  int MaxNeighborListLength() const { return maxlistlen; }

  int GetNumberOfAtoms() const { return nAtoms; }

  /// 0 auto, 1 stored, 2 stream. The next update rebuilds.
  void setMode(int mode);

  /// Stored partner indices. Zero while streaming.
  int slots() const;

  /// Null when the stored path should walk atoms in index order.
  const int *cellOrder() const noexcept {
    return order_.empty() ? nullptr : order_.data();
  }

  [[nodiscard]] bool streaming() const noexcept;

  /// One half-pair walk, r_j - r_i, j > i, inside the cutoff.
  /// A streaming walk keeps the pairs so `replayHalf` can feed the force
  /// pass without walking the cells again.
  template <typename Fn>
  void forHalf(Fn &&fn) {
    if (cells == nullptr || nAtoms < 2) {
      return;
    }
    const double cut2 = rCut2;
    const bool record = cells->streaming();
    if (record) {
      stream_pairs_.clear();
      stream_pairs_.reserve(static_cast<std::size_t>(nAtoms) * 8);
    }
    cells->forHalf(0, static_cast<std::int64_t>(nAtoms),
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
                     if (!(r2 > 0.0) || !(r2 < cut2)) {
                       return;
                     }
                     const double x = -dx;
                     const double y = -dy;
                     const double z = -dz;
                     if (record) {
                       stream_pairs_.push_back(StreamPair{static_cast<int>(i), static_cast<int>(j), x,
                                                          y, z, r2});
                     }
                     fn(static_cast<int>(i), static_cast<int>(j), x, y, z, r2);
                   });
  }

  template <typename Fn>
  void replayHalf(Fn &&fn) const {
    for (const StreamPair &p : stream_pairs_) {
      fn(p.i, p.j, p.x, p.y, p.z, p.r2);
    }
  }

private:
  /// Half pair kept from the last stored build. The image shifts let a
  /// later call form r_j - r_i from the current coordinates.
  struct HalfBond {
    int j;
    int sx;
    int sy;
    int sz;
  };
  struct CellRows {
    std::vector<int> atoms;
    std::vector<int> off;
    std::vector<int> nbr;
    std::vector<double> xyz;
    std::vector<double> r2;
  };
  void slide(int cell);
  void fillPlane(int ix);
  void fillCell(int cell);
  void dropCell(int cell);
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
  int center_ = -2;
  int plane_ = -2;
  rgpot::nlist::Neighbours *cells;
  std::vector<int> order_;
  std::vector<int> place_;
  std::vector<int> lookup_;
  std::vector<std::unique_ptr<CellRows>> cached_;
  std::vector<int> half_off_;
  std::vector<HalfBond> half_;
  struct StreamPair {
    int i;
    int j;
    double x;
    double y;
    double z;
    double r2;
  };
  std::vector<StreamPair> stream_pairs_;
};
