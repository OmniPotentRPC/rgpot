#pragma once
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace rgpot {
namespace linked_cell {

// One directed pair. The owning atom is the slice index, not a field,
// so the record stays 16 bytes and the write stays inside the atom's block.
struct Pair {
  double dist2 = 0;
  std::uint32_t neighbor = 0;
  std::int8_t s0 = 0;
  std::int8_t s1 = 0;
  std::int8_t s2 = 0;
  std::int8_t pad = 0;
};

inline bool before(const Pair &a, const Pair &b) {
  if (a.dist2 != b.dist2)
    return a.dist2 < b.dist2;
  if (a.neighbor != b.neighbor)
    return a.neighbor < b.neighbor;
  if (a.s0 != b.s0)
    return a.s0 < b.s0;
  if (a.s1 != b.s1)
    return a.s1 < b.s1;
  return a.s2 < b.s2;
}

struct Scratch {
  std::vector<std::uint32_t> cell_of;
  std::vector<std::uint32_t> placed;
  std::vector<std::size_t> hist;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<std::vector<Pair>> parts;
  std::vector<std::uint32_t> slice_at;
  std::vector<std::uint32_t> slice_n;
  std::vector<std::uint32_t> slice_thread;
  std::vector<double> ref_pos;
  double ref_box[9]{};
  int ref_n = -1;
  double ref_search = -1;
  const void *stable = nullptr;
  std::size_t stable_n = 0;
};

inline bool orthorhombic(const double *box) {
  return box[0] > 0.0 && box[4] > 0.0 && box[8] > 0.0 && std::fabs(box[1]) < 1e-12 &&
         std::fabs(box[2]) < 1e-12 && std::fabs(box[3]) < 1e-12 &&
         std::fabs(box[5]) < 1e-12 && std::fabs(box[6]) < 1e-12 &&
         std::fabs(box[7]) < 1e-12;
}

// Exclusive prefix. `count` grows by one and the last entry is the total.
inline void exclusive_prefix(std::vector<std::size_t> &count, int threads) {
  const std::size_t m = count.size();
  if (m == 0)
    return;
  // A few thousand bins are cheaper to add on this thread than to
  // hand to a freshly spawned worker.
  // One thread on a long prefix still uses the blocked loops below.
  if (m < 8192) {
    std::size_t run = 0;
    count.push_back(0);
    for (std::size_t i = 0; i < m; ++i) {
      const std::size_t value = count[i];
      count[i] = run;
      run += value;
    }
    count[m] = run;
    return;
  }
  threads = std::max(1, std::min(threads, static_cast<int>(m)));
  std::vector<std::size_t> block(static_cast<std::size_t>(threads) + 1, 0);
  auto sum_block = [&](int t) {
    const std::size_t b = (static_cast<std::size_t>(t) * m) / static_cast<std::size_t>(threads);
    const std::size_t e =
        (static_cast<std::size_t>(t + 1) * m) / static_cast<std::size_t>(threads);
    std::size_t s = 0;
    for (std::size_t i = b; i < e; ++i)
      s += count[i];
    block[static_cast<std::size_t>(t) + 1] = s;
  };
  if (threads == 1) {
    sum_block(0);
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back(sum_block, t);
    for (auto &worker : pool)
      worker.join();
  }
  for (int t = 1; t <= threads; ++t)
    block[static_cast<std::size_t>(t)] += block[static_cast<std::size_t>(t - 1)];
  std::vector<std::size_t> out(m + 1, 0);
  auto fill = [&](int t) {
    const std::size_t b = (static_cast<std::size_t>(t) * m) / static_cast<std::size_t>(threads);
    const std::size_t e =
        (static_cast<std::size_t>(t + 1) * m) / static_cast<std::size_t>(threads);
    std::size_t run = block[static_cast<std::size_t>(t)];
    for (std::size_t i = b; i < e; ++i) {
      out[i] = run;
      run += count[i];
    }
    if (t + 1 == threads)
      out[m] = run;
  };
  if (threads == 1) {
    fill(0);
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back(fill, t);
    for (auto &worker : pool)
      worker.join();
  }
  count.swap(out);
}

// Bin size is at least `search`. Atoms are counting-sorted into bins.
// Each thread writes one contiguous pair block, and each atom's pairs
// are sorted before the next atom is visited. False when the box is
// not orthorhombic or a side is shorter than three search lengths.
inline bool build(const double *pos, std::size_t n, const double *box, double search,
                  int threads, Scratch &scratch) {
  if (n == 0 || !(search > 0.0) || !orthorhombic(box))
    return false;
  const double side[3] = {box[0], box[4], box[8]};
  if (side[0] < 3.0 * search || side[1] < 3.0 * search || side[2] < 3.0 * search)
    return false;
  threads = std::max(1, threads);
  int ncell[3];
  int cells = 1;
  for (int a = 0; a < 3; ++a) {
    ncell[a] = std::max(1, static_cast<int>(std::floor(side[a] / search)));
    cells *= ncell[a];
  }
  scratch.cell_of.resize(n);
  scratch.hist.assign(static_cast<std::size_t>(cells), 0);
  for (std::size_t i = 0; i < n; ++i) {
    int ix[3];
    for (int a = 0; a < 3; ++a) {
      double f = pos[3 * i + a] / side[a];
      f -= std::floor(f);
      int bin = static_cast<int>(std::floor(f * static_cast<double>(ncell[a])));
      if (bin >= ncell[a])
        bin = ncell[a] - 1;
      // The wrap puts the fraction in [0, 1], so the bin is not negative.
      ix[a] = bin;
    }
    const int cell = ix[0] + ncell[0] * (ix[1] + ncell[1] * ix[2]);
    scratch.cell_of[i] = static_cast<std::uint32_t>(cell);
    ++scratch.hist[static_cast<std::size_t>(cell)];
  }
  exclusive_prefix(scratch.hist, threads);
  scratch.placed.resize(n);
  auto slot = scratch.hist;
  for (std::size_t i = 0; i < n; ++i) {
    const auto cell = static_cast<std::size_t>(scratch.cell_of[i]);
    scratch.placed[slot[cell]++] = static_cast<std::uint32_t>(i);
  }
  scratch.x.resize(n);
  scratch.y.resize(n);
  scratch.z.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    const std::uint32_t id = scratch.placed[k];
    scratch.x[k] = pos[3 * id];
    scratch.y[k] = pos[3 * id + 1];
    scratch.z[k] = pos[3 * id + 2];
  }

  if (static_cast<int>(scratch.parts.size()) < threads)
    scratch.parts.resize(static_cast<std::size_t>(threads));
  scratch.slice_at.resize(n);
  scratch.slice_n.resize(n);
  scratch.slice_thread.resize(n);
  const double cut2 = search * search;
  auto work = [&](int t) {
    const std::size_t begin =
        (static_cast<std::size_t>(t) * n) / static_cast<std::size_t>(threads);
    const std::size_t end =
        (static_cast<std::size_t>(t + 1) * n) / static_cast<std::size_t>(threads);
    auto &out = scratch.parts[static_cast<std::size_t>(t)];
    out.clear();
    out.reserve((end - begin) * 128);
    for (std::size_t k = begin; k < end; ++k) {
      const std::uint32_t id = scratch.placed[k];
      const std::uint32_t slice = static_cast<std::uint32_t>(out.size());
      const int cell = static_cast<int>(scratch.cell_of[id]);
      const int ix = cell % ncell[0];
      const int iy = (cell / ncell[0]) % ncell[1];
      const int iz = cell / (ncell[0] * ncell[1]);
      const double cx = scratch.x[k];
      const double cy = scratch.y[k];
      const double cz = scratch.z[k];
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            int jx = ix + dx;
            int jy = iy + dy;
            int jz = iz + dz;
            int sx = 0, sy = 0, sz = 0;
            if (jx < 0) {
              jx += ncell[0];
              sx = -1;
            } else if (jx >= ncell[0]) {
              jx -= ncell[0];
              sx = 1;
            }
            if (jy < 0) {
              jy += ncell[1];
              sy = -1;
            } else if (jy >= ncell[1]) {
              jy -= ncell[1];
              sy = 1;
            }
            if (jz < 0) {
              jz += ncell[2];
              sz = -1;
            } else if (jz >= ncell[2]) {
              jz -= ncell[2];
              sz = 1;
            }
            const int jcell = jx + ncell[0] * (jy + ncell[1] * jz);
            const std::size_t lo = scratch.hist[static_cast<std::size_t>(jcell)];
            const std::size_t hi = scratch.hist[static_cast<std::size_t>(jcell) + 1];
            for (std::size_t p = lo; p < hi; ++p) {
              const std::uint32_t jd = scratch.placed[p];
              if (jd == id && sx == 0 && sy == 0 && sz == 0)
                continue;
              const double ddx = scratch.x[p] - cx + static_cast<double>(sx) * side[0];
              const double ddy = scratch.y[p] - cy + static_cast<double>(sy) * side[1];
              const double ddz = scratch.z[p] - cz + static_cast<double>(sz) * side[2];
              const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (!(d2 < cut2))
                continue;
              Pair row;
              row.dist2 = d2;
              row.neighbor = jd;
              row.s0 = static_cast<std::int8_t>(sx);
              row.s1 = static_cast<std::int8_t>(sy);
              row.s2 = static_cast<std::int8_t>(sz);
              out.push_back(row);
            }
          }
        }
      }
      const auto count = static_cast<std::uint32_t>(out.size() - slice);
      scratch.slice_at[id] = slice;
      scratch.slice_n[id] = count;
      scratch.slice_thread[id] = static_cast<std::uint32_t>(t);
      if (count > 1)
        std::sort(out.begin() + static_cast<std::ptrdiff_t>(slice), out.end(), before);
    }
  };
  if (threads == 1) {
    work(0);
  } else {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back(work, t);
    for (auto &worker : pool)
      worker.join();
  }
  scratch.stable = scratch.parts[0].data();
  scratch.stable_n = 0;
  for (const auto &part : scratch.parts)
    scratch.stable_n += part.size();
  return true;
}

inline double max_move(const double *pos, const std::vector<double> &ref) {
  double peak = 0.0;
  for (std::size_t i = 0; i < ref.size(); i += 3) {
    const double dx = pos[i] - ref[i];
    const double dy = pos[i + 1] - ref[i + 1];
    const double dz = pos[i + 2] - ref[i + 2];
    peak = std::max(peak, dx * dx + dy * dy + dz * dz);
  }
  return peak;
}

} // namespace linked_cell
} // namespace rgpot
