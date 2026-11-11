#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Public neighbour query. The classical potentials keep using Neighbours
// for a minimum-image walk. This type is the stable list: structure of
// arrays, half / full / cluster-of-four layouts, triclinic and partial
// periodic boundaries, a skin rebuild, and a chunked walk that does not
// store the pairs.

#include "rgpot/nlist/Neighbours.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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
  Nomem = 7
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
/// uses for a self image. Full keeps both directions. Cluster4 groups
/// four consecutive central atoms and stores one partner with a 4-bit mask.
/// Cluster4 is the full interaction set packed by fours. It is not a
/// half list.
enum class Format : std::uint8_t { Half, Full, Cluster4 };

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

/// One neighbour query.
struct Query {
  double cutoff = 0;
  double skin = 0;
  Format format = Format::Half;
  Cut cut = Cut::Strict;
  std::array<bool, 3> periodic{{true, true, true}};
  Fields fields{};
  Mode mode = Mode::Stored;
};

/// Non-owning view of a structure-of-arrays neighbour table.
/// `shift` has 3 entries per pair, `vector` has 3, `mask` is one byte per
/// pair (0xff for Half and Full; bits 0..3 for Cluster4).
struct SoA {
  const std::int32_t *first = nullptr;
  const std::int32_t *second = nullptr;
  const std::int32_t *shift = nullptr;
  const double *vector = nullptr;
  const double *distance = nullptr;
  const std::uint8_t *mask = nullptr;
  std::size_t count = 0;
};

/// Writable structure of arrays for one chunk of a walk.
struct OutSoA {
  std::int32_t *first = nullptr;
  std::int32_t *second = nullptr;
  std::int32_t *shift = nullptr;
  double *vector = nullptr;
  double *distance = nullptr;
  std::uint8_t *mask = nullptr;
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
  [[nodiscard]] SoA view() const noexcept {
    return SoA{first.data(),
               second.data(),
               shift.empty() ? nullptr : shift.data(),
               vector.empty() ? nullptr : vector.data(),
               distance.empty() ? nullptr : distance.data(),
               mask.empty() ? nullptr : mask.data(),
               first.size()};
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
              const Query &query);

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
};

/// Walk pairs for one central atom, or one cluster of four, at a time
/// into caller storage. The walk does not keep a list of the whole
/// system. It borrows `positions` and `box9` until it is destroyed.
/// `pull` writes at most `cap` pairs and returns how many it wrote.
/// `done` is set when the geometry has been exhausted. A null first or
/// second pointer with a positive cap throws Error and does not advance.
/// One walk is used by one thread.
class Walk {
public:
  Walk(const double *positions, std::size_t n, const double *box9,
       const Query &query);

  std::size_t pull(OutSoA dest, std::size_t cap, bool &done);

private:
  const double *pos_ = nullptr;
  std::size_t n_ = 0;
  double box_[9]{};
  Query query_{};
  int nmax_[3]{};
  std::size_t atom_ = 0;
  std::vector<std::int32_t> hold_first_;
  std::vector<std::int32_t> hold_second_;
  std::vector<std::int32_t> hold_shift_;
  std::vector<double> hold_vector_;
  std::vector<double> hold_distance_;
  std::vector<std::uint8_t> hold_mask_;
  std::size_t hold_at_ = 0;
  bool finished_ = false;
};

/// Fill `table` from `query`. Stream mode leaves the table empty and sets
/// `table.stored` false; the caller uses Walk. Auto stores when a bound on
/// the pair count fits in the memory the OS reports as available, and
/// otherwise leaves the table empty. Throws Error before clearing `table`
/// when the geometry is rejected. Two builds on different tables may run
/// at the same time. They share no mutable global.
void build(Table &table, const double *positions, std::size_t n,
           const double *box9, const Query &query);

} // namespace rgpot::nlist
