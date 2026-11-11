#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Public neighbour query. Cells is the reusable list the classical
// potentials walk. Neighbours is the cell list under it. The stable list
// is a structure of arrays: a pair shell or a cluster of four, a cutoff
// per pair type, triclinic and partial periodic boundaries, a skin
// rebuild, and a callback or a chunked walk that does not store the pairs.
//
// Distinct Table, Builder and Walk objects may be used from different
// threads. One object is not shared. build() may start an OpenMP team
// when OMP_NUM_THREADS is set, so do not call it from inside an OpenMP
// region. There is no mutable global. A Walk borrows positions and the
// box until it is destroyed. A View is invalidated by the next update
// of the table that owns it.

#include "rgpot/nlist/Neighbours.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace rgpot::nlist {

/// Status values shared with the C ABI. Ok is never thrown.
enum class Code : std::int32_t {
  Ok = 0,
  Abi = 1,
  Null = 2,
  Cutoff = 3,
  Box = 4,
  Shape = 5,
  Short = 6,
  Nomem = 7,
  Skin = 8
};

/// A rejected query. code() is the C ABI value. what() is a sentence
/// the caller can show. Builders throw this and leave a previous table
/// in place.
class Error : public std::runtime_error {
public:
  Error(Code code, const std::string &message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] Code code() const noexcept { return code_; }

private:
  Code code_;
};

/// Half keeps one direction of each pair, with the same image rule vesin
/// uses for a self image. Full keeps both directions.
enum class Shell : std::uint8_t { Half, Full };

/// One pair per interaction. `shell` chooses the direction.
struct Pairs {
  Shell shell = Shell::Half;
  friend bool operator==(Pairs a, Pairs b) noexcept { return a.shell == b.shell; }
};

/// Four consecutive central atoms, one partner, and a 4-bit mask.
/// The interactions are the full set. There is no half-list cluster:
/// the type has no shell.
struct Cluster4 {
  friend bool operator==(Cluster4, Cluster4) noexcept { return true; }
};

/// Pairs or a cluster of four. A value cannot be both.
using Shape = std::variant<Pairs, Cluster4>;

/// Strict keeps pairs with distance strictly below the cutoff. Loose keeps
/// pairs out to cutoff + skin, which is what a Verlet rebuild needs.
enum class Cut : std::uint8_t { Strict, Loose };

/// What a stored table or a chunk actually carries. Unset fields are left
/// empty rather than filled with zeros.
struct Fields {
  bool shifts = true;
  bool vectors = true;
  bool distances = true;
};

/// Cutoffs that depend on the pair of atom types. `types` has one id per
/// atom, each in `[0, ntypes)`. `cutoff` is row-major and holds
/// `ntypes * ntypes` entries: atom `i` with atom `j` uses
/// `cutoff[types[i] * ntypes + types[j]]`. A non-positive entry drops
/// that pair type. `ntypes == 0` means every pair uses `Query::cutoff`.
/// The pointers are borrowed for the call. `Builder`, `Walk` and `Cells`
/// copy what they keep.
struct PairCutoffs {
  const std::int32_t *types = nullptr;
  const double *cutoff = nullptr;
  std::int32_t ntypes = 0;
};

/// One neighbour query. `build` and `Builder` store the pairs. `Walk`
/// streams them. There is no mode flag that can store nothing.
struct Query {
  double cutoff = 0;
  double skin = 0;
  Shape shape = Pairs{};
  Cut cut = Cut::Strict;
  std::array<bool, 3> periodic{{true, true, true}};
  Fields fields{};
};

/// Non-owning view of a structure-of-arrays neighbour table.
/// `shift` is empty or 3 entries per pair, `vector` is empty or 3,
/// `mask` is one byte per pair (0xff for a pair list; bits 0..3 for
/// a cluster of four). The spans are invalidated by the next rebuild
/// of the table that owns them.
struct View {
  std::span<const std::int32_t> first;
  std::span<const std::int32_t> second;
  std::span<const std::int32_t> shift;
  std::span<const double> vector;
  std::span<const double> distance;
  std::span<const std::uint8_t> mask;
};

/// Caller-owned buffers for one chunk. `first` and `second` have the
/// same length, which is the maximum number of pairs this pull writes.
/// `shift` and `vector` are empty or three times that length.
struct Buffers {
  std::span<std::int32_t> first;
  std::span<std::int32_t> second;
  std::span<std::int32_t> shift;
  std::span<double> vector;
  std::span<double> distance;
  std::span<std::uint8_t> mask;
};

/// Owning table. The arrays are the storage a caller can wrap without
/// copying. `device` is 0 for host memory. A GPU fill would use the same
/// layout and a non-zero device id; this build fills host memory.
struct Table {
  std::vector<std::int32_t> first;
  std::vector<std::int32_t> second;
  std::vector<std::int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<std::uint8_t> mask;
  int device = 0;
  bool stored = false;

  [[nodiscard]] std::size_t size() const noexcept { return first.size(); }

  /// `vector` as a Fortran `(3, npairs)` array. Pair `p` is
  /// `x, y, z` at elements `3p`, `3p+1`, `3p+2`. Same memory as `vector`.
  [[nodiscard]] std::span<const double> column_major_vectors() const noexcept {
    return vector;
  }
  [[nodiscard]] View view() const noexcept {
    return View{first, second,
                shift.empty() ? std::span<const std::int32_t>{} : std::span<const std::int32_t>{shift},
                vector.empty() ? std::span<const double>{} : std::span<const double>{vector},
                distance.empty() ? std::span<const double>{} : std::span<const double>{distance},
                mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>{mask}};
  }
};

/// Stored list with a skin rebuild. A later update keeps the table while
/// every atom has moved by at most half the skin and the cell and the
/// query are unchanged. Skin 0 rebuilds unless the coordinates match.
/// Throws Error and keeps the previous table when the geometry is
/// rejected. The same builder is used by one thread. Two builders may
/// update at the same time.
class Builder {
public:
  void update(const double *positions, std::size_t n, const double *box9,
              const Query &query, PairCutoffs cuts = {});

  [[nodiscard]] const Table &table() const noexcept { return table_; }
  [[nodiscard]] std::uint64_t builds() const noexcept { return builds_; }
  [[nodiscard]] bool reused() const noexcept { return reused_; }

private:
  Table table_;
  std::vector<double> ref_;
  double box_[9]{};
  Query query_{};
  std::size_t n_ = 0;
  std::uint64_t builds_ = 0;
  bool have_ = false;
  bool reused_ = false;
  std::int32_t ntypes_ = 0;
  std::vector<std::int32_t> types_;
  std::vector<double> matrix_;
};

/// Walk pairs for one central atom, or one cluster of four, at a time
/// into caller storage. The walk does not keep a list of the whole
/// system. It borrows `positions` and `box9` until it is destroyed.
/// `pull` writes at most `dest.first.size()` pairs and returns how
/// many it wrote. `done` is set when the geometry has been exhausted.
/// A requested field whose span is too short throws Error and does not
/// advance. One walk is used by one thread.
class Walk {
public:
  Walk(const double *positions, std::size_t n, const double *box9,
       const Query &query, PairCutoffs cuts = {});

  std::size_t pull(Buffers dest, bool &done);

private:
  const double *pos_ = nullptr;
  std::size_t n_ = 0;
  double box_[9]{};
  Query query_{};
  int nmax_[3]{};
  bool one_image_ = false;
  Neighbours cells_;
  std::size_t atom_ = 0;
  std::vector<std::int32_t> hold_first_;
  std::vector<std::int32_t> hold_second_;
  std::vector<std::int32_t> hold_shift_;
  std::vector<double> hold_vector_;
  std::vector<double> hold_distance_;
  std::vector<std::uint8_t> hold_mask_;
  std::size_t hold_at_ = 0;
  bool finished_ = false;
  bool cut_typed_ = false;
  bool cut_inclusive_ = false;
  std::int32_t cut_ntypes_ = 0;
  double cut_search_ = 0;
  std::vector<std::int32_t> cut_types_;
  std::vector<double> cut_limit_;
};

/// Fill `table` from `query`. The table holds every pair. A caller that
/// must not store the list uses Walk. Throws Error before clearing
/// `table` when the geometry is rejected. Two builds on different tables
/// may run at the same time. They share no mutable global. A build of
/// at least 8192 atoms may start an OpenMP team. A smaller build stays
/// on the calling thread.
void build(Table &table, const double *positions, std::size_t n,
           const double *box9, const Query &query, PairCutoffs cuts = {});

/// The pairs `build` would store, one at a time, with no table. The vector
/// is `r_j - r_i + S H` and `r` is its length. The callback runs on the
/// calling thread. A cluster is merged a few atoms at a time. `fn` is
/// `void(int32_t i, int32_t j, int sa, int sb, int sc, double x, double y,
/// double z, double r)`.
void each_raw(const double *positions, std::size_t n, const double *box9,
              const Query &query,
              void (*fn)(void *user, std::int32_t i, std::int32_t j, int sa, int sb, int sc,
                         double x, double y, double z, double r),
              void *user, PairCutoffs cuts = {});

template <typename Fn>
void each(const double *positions, std::size_t n, const double *box9, const Query &query,
          Fn &&fn, PairCutoffs cuts = {}) {
  using FnObj = std::remove_reference_t<Fn>;
  auto *held = &fn;
  each_raw(
      positions, n, box9, query,
      [](void *user, std::int32_t i, std::int32_t j, int sa, int sb, int sc, double x, double y,
         double z, double r) {
        (*static_cast<FnObj *>(user))(i, j, sa, sb, sc, x, y, z, r);
      },
      held, cuts);
}

/// The same pairs as `build`, gathered one atom at a time. A CUDA, HIP
/// or Kokkos launch gives one thread one atom and runs this walk. The
/// host `build` remains the source of truth.
void buildByAtom(Table &table, const double *positions, std::size_t n,
                 const double *box9, const Query &query, PairCutoffs cuts = {});

/// Several systems that do not interact. `positions` is the concatenation
/// of every system, `count` atoms each. `box9` holds nine lattice values
/// per system, in the same order. `count[s] == 0` is an empty system.
struct Systems {
  const double *positions = nullptr;
  const std::int32_t *count = nullptr;
  std::size_t systems = 0;
  const double *box9 = nullptr;
};

/// Pairs of a `Systems` build. `first` and `second` index the concatenated
/// positions. `system[p]` is the system that owns pair `p`.
struct Batch {
  Table pairs;
  std::vector<std::int32_t> system;
};

/// One query for every system. The systems do not see each other's atoms.
/// A rejected system throws before `batch` is cleared.
void build(Batch &batch, const Systems &systems, const Query &query,
           PairCutoffs cuts = {});

/// Reusable cell list for a force kernel. `dx, dy, dz` are `r_i - r_j`
/// after the image, which is the negation of the vector `build` stores,
/// and `r2` is the squared length. No square root and no shift are
/// computed. With an empty `PairCutoffs` the walk is the cell list the
/// classical potentials used directly: the same pairs, in the same order.
/// A matrix keeps candidates out to that pair's cutoff plus the skin
/// (`r2 <= (cutoff + skin)^2`). One `Cells` object is used by one thread.
/// `forCellTeam` may start an OpenMP team. Do not call it from inside
/// another OpenMP region.
class Cells {
public:
  void update(const double *positions, std::size_t n, const double *box9, double cutoff,
              double skin, Image image, Mode mode, const std::array<bool, 3> &periodic,
              PairCutoffs cuts = {});

  [[nodiscard]] bool streaming() const noexcept { return cells_.streaming(); }
  [[nodiscard]] bool reused() const noexcept { return cells_.reused(); }
  [[nodiscard]] std::uint64_t builds() const noexcept { return cells_.builds(); }
  [[nodiscard]] std::size_t slots() const noexcept { return cells_.slots(); }
  [[nodiscard]] const std::vector<std::int32_t> &offsets() const noexcept {
    return cells_.offsets();
  }
  [[nodiscard]] const std::vector<std::int32_t> &neighbours() const noexcept {
    return cells_.neighbours();
  }

  template <typename Fn>
  void forDirected(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (!typed_) {
      cells_.forDirected(begin, end, std::forward<Fn>(fn));
      return;
    }
    using FnObj = std::remove_reference_t<Fn>;
    filter_directed(
        begin, end,
        [](void *user, std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
          (*static_cast<FnObj *>(user))(i, j, dx, dy, dz, r2);
        },
        &fn);
  }

  template <typename Fn>
  void forHalf(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (!typed_) {
      cells_.forHalf(begin, end, std::forward<Fn>(fn));
      return;
    }
    using FnObj = std::remove_reference_t<Fn>;
    filter_half(
        begin, end,
        [](void *user, std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
          (*static_cast<FnObj *>(user))(i, j, dx, dy, dz, r2);
        },
        &fn);
  }

  template <typename Fn>
  void forAtom(std::int32_t i, Fn &&fn) const {
    if (!typed_) {
      cells_.forAtom(i, std::forward<Fn>(fn));
      return;
    }
    using FnObj = std::remove_reference_t<Fn>;
    filter_atom(
        i,
        [](void *user, std::int32_t a, std::int32_t j, double dx, double dy, double dz, double r2) {
          (*static_cast<FnObj *>(user))(a, j, dx, dy, dz, r2);
        },
        &fn);
  }

  template <typename Fn>
  void forCellTeam(bool half, int nthreads, Fn &&fn) const {
    if (!typed_) {
      cells_.forCellTeam(half, nthreads, std::forward<Fn>(fn));
      return;
    }
    using FnObj = std::remove_reference_t<Fn>;
    filter_team(
        half, nthreads,
        [](void *user, int thread, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
           double r2) {
          (*static_cast<FnObj *>(user))(thread, i, j, dx, dy, dz, r2);
        },
        &fn);
  }

private:
  using PairFn = void (*)(void *user, std::int32_t i, std::int32_t j, double dx, double dy,
                          double dz, double r2);
  using TeamFn = void (*)(void *user, int thread, std::int32_t i, std::int32_t j, double dx,
                          double dy, double dz, double r2);

  void filter_directed(std::int64_t begin, std::int64_t end, PairFn fn, void *user) const;
  void filter_half(std::int64_t begin, std::int64_t end, PairFn fn, void *user) const;
  void filter_atom(std::int32_t i, PairFn fn, void *user) const;
  void filter_team(bool half, int nthreads, TeamFn fn, void *user) const;

  [[nodiscard]] bool allow(std::int32_t i, std::int32_t j, double r2) const {
    const double lim = limit_[static_cast<std::size_t>(types_[static_cast<std::size_t>(i)]) *
                                  static_cast<std::size_t>(ntypes_) +
                              static_cast<std::size_t>(types_[static_cast<std::size_t>(j)])];
    return (lim > 0.0) && r2 <= lim;
  }

  Neighbours cells_;
  bool typed_ = false;
  std::int32_t ntypes_ = 0;
  std::vector<std::int32_t> types_;
  std::vector<double> limit_;
};

/// Geometry callback with a skin. `update` keeps the bins, and on a wide
/// orthorhombic cell the candidate index, while every atom stays within
/// half the skin. `each` and `eachTeam` choose a half or a full shell
/// without rebuilding. The vector is `r_j - r_i + S H`. A cluster is not
/// a sweep. One sweep is used by one thread. `eachTeam` may start an
/// OpenMP team on a wide orthorhombic cell and must not be called from
/// inside another one. A narrow or sheared cell stays on the calling
/// thread. `threads` is at least 1. Callbacks for different threads may
/// run at the same time.
class Sweep {
public:
  void update(const double *positions, std::size_t n, const double *box9, const Query &query,
              PairCutoffs cuts = {});

  [[nodiscard]] bool reused() const noexcept { return reused_; }

  template <typename Fn>
  void each(Shell shell, Fn &&fn) const {
    using FnObj = std::remove_reference_t<Fn>;
    each_raw(
        shell, 1,
        [](void *user, int, std::int32_t i, std::int32_t j, int sa, int sb, int sc, double x,
           double y, double z, double r) {
          (*static_cast<FnObj *>(user))(i, j, sa, sb, sc, x, y, z, r);
        },
        &fn);
  }

  template <typename Fn>
  void eachTeam(Shell shell, int threads, Fn &&fn) const {
    using FnObj = std::remove_reference_t<Fn>;
    each_raw(
        shell, threads,
        [](void *user, int thread, std::int32_t i, std::int32_t j, int sa, int sb, int sc, double x,
           double y, double z, double r) {
          (*static_cast<FnObj *>(user))(thread, i, j, sa, sb, sc, x, y, z, r);
        },
        &fn);
  }

private:
  using GeomFn = void (*)(void *user, int thread, std::int32_t i, std::int32_t j, int sa, int sb,
                          int sc, double x, double y, double z, double r);
  void each_raw(Shell shell, int threads, GeomFn fn, void *user) const;

  Neighbours cells_;
  Query query_{};
  double box_[9]{};
  int nmax_[3]{};
  double search_ = 0;
  const double *pos_ = nullptr;
  std::size_t n_ = 0;
  bool one_image_ = false;
  bool have_ = false;
  bool reused_ = false;
  bool cut_typed_ = false;
  bool cut_inclusive_ = false;
  std::int32_t cut_ntypes_ = 0;
  std::vector<std::int32_t> cut_types_;
  std::vector<double> cut_limit_;
};

/// Bind `cells` to this geometry. `cutoff` is the pair cutoff and `skin`
/// is the Verlet buffer the kernels drop inside the cutoff. `image` is
/// Round for an orthorhombic cell, Half after coordinates have been
/// wrapped, and General for a sheared cell. `mode` stores the index when
/// it fits and streams otherwise, unless the caller asks.
inline void updateCells(Neighbours &cells, const double *positions, std::size_t n,
                        const double *box9, double cutoff, double skin, Image image, Mode mode,
                        const std::array<bool, 3> &periodic) {
  Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.image = image;
  spec.mode = mode;
  spec.periodic = periodic;
  cells.update(positions, n, box9, spec);
}

/// Pairs for atoms in `[begin, end)`. `dx` is `r_i - r_j` after the image,
/// the negation of the vector a stored table holds for the minimum image.
template <typename Fn>
inline void visitDirected(const Neighbours &cells, std::int64_t begin, std::int64_t end,
                          Fn &&fn) {
  cells.forDirected(begin, end, std::forward<Fn>(fn));
}

/// The pairs of `visitDirected` with `j > i` only.
template <typename Fn>
inline void visitHalf(const Neighbours &cells, std::int64_t begin, std::int64_t end, Fn &&fn) {
  cells.forHalf(begin, end, std::forward<Fn>(fn));
}

} // namespace rgpot::nlist
