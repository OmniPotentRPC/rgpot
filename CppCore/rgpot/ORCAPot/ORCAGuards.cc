// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAGuards.hpp"

#include "rgpot/ParamHash.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

namespace rgpot::orca {

namespace fs = std::filesystem;

namespace {

// Most lattice vectors the box check tries. A padded cluster box needs a
// few dozen; only a cell skewed far past any padding reaches this.
constexpr double kMaxImageVectors = 2.0e5;

struct timespec modificationTime(const struct stat &st) noexcept {
#if defined(__APPLE__)
  return st.st_mtimespec;
#else
  return st.st_mtim;
#endif
}

struct timespec changeTime(const struct stat &st) noexcept {
#if defined(__APPLE__)
  return st.st_ctimespec;
#else
  return st.st_ctim;
#endif
}

void cross(const double *u, const double *v, double *w) {
  w[0] = u[1] * v[2] - u[2] * v[1];
  w[1] = u[2] * v[0] - u[0] * v[2];
  w[2] = u[0] * v[1] - u[1] * v[0];
}

double dot(const double *u, const double *v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

} // namespace

bool stampOpenFile(int fd, uint64_t *stamp) noexcept {
  struct stat st {};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    return false;
  }
  Fnv1a content;
  std::array<unsigned char, 16384> chunk;
  uint64_t total = 0;
  for (;;) {
    const ssize_t got = ::pread(fd, chunk.data(), chunk.size(),
                                static_cast<off_t>(total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (got == 0) {
      break;
    }
    content.bytes(chunk.data(), static_cast<std::size_t>(got));
    total += static_cast<uint64_t>(got);
  }
  const struct timespec modified = modificationTime(st);
  const struct timespec changed = changeTime(st);
  Fnv1a fp;
  fp.u64(static_cast<uint64_t>(st.st_size));
  fp.i64(static_cast<int64_t>(modified.tv_sec));
  fp.i64(static_cast<int64_t>(modified.tv_nsec));
  fp.i64(static_cast<int64_t>(changed.tv_sec));
  fp.i64(static_cast<int64_t>(changed.tv_nsec));
  fp.u64(static_cast<uint64_t>(st.st_dev));
  fp.u64(static_cast<uint64_t>(st.st_ino));
  fp.u64(total);
  fp.u64(content.h);
  *stamp = fp.h;
  return true;
}

bool identifyExecutable(const std::string &executable, std::string *real,
                        uint64_t *stamp) noexcept {
  try {
    std::error_code ec;
    const fs::path resolved = fs::canonical(executable, ec);
    if (ec) {
      return false;
    }
    int fd;
    do {
      fd = ::open(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
      return false;
    }
    const bool stamped = stampOpenFile(fd, stamp);
    ::close(fd);
    if (!stamped) {
      return false;
    }
    *real = resolved.string();
    return true;
  } catch (...) {
    return false;
  }
}

bool executableUnchanged(const std::string &executable, const std::string &real,
                         uint64_t stamp) noexcept {
  std::string now_real;
  uint64_t now_stamp = 0;
  return identifyExecutable(executable, &now_real, &now_stamp) &&
         now_real == real && now_stamp == stamp;
}

// A box is taken as padding around an isolated cluster. It passes when it
// is absent or all zero, or when every pair of atoms is its own nearest
// periodic image. An image of atom j shifted by the lattice vector L is
// closer to atom i than j itself exactly when the pair displacement d has
// 2 d.L + |L|^2 < 0, so the rule is that along every nonzero L the atoms
// span at most |L| / 2. In a skewed cell the shortest such L need not be a
// box vector, so every L short enough to matter is tried. The positions
// are not moved, so a cluster may sit across the origin.
void checkClusterBox(const char *who, const double *box, std::size_t n,
                     const double *pos) {
  if (box == nullptr) {
    return;
  }
  bool open = true;
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(box[k])) {
      throw std::invalid_argument(std::string(who) +
                                  ": the box holds a non-finite entry");
    }
    open = open && box[k] == 0.0;
  }
  if (open) {
    return;
  }
  for (std::size_t k = 0; k < 3 * n; ++k) {
    if (!std::isfinite(pos[k])) {
      throw std::invalid_argument(
          std::string(who) +
          ": a position is not finite, so the box cannot be checked");
    }
  }
  const double *rows[3] = {box, box + 3, box + 6};
  double normals[3][3];
  for (int k = 0; k < 3; ++k) {
    cross(rows[(k + 1) % 3], rows[(k + 2) % 3], normals[k]);
  }
  const double volume = std::abs(dot(rows[0], normals[0]));
  const double scale = std::sqrt(dot(rows[0], rows[0]) * dot(rows[1], rows[1]) *
                                 dot(rows[2], rows[2]));
  if (!(volume > 1.0e-12 * scale)) {
    throw std::invalid_argument(
        std::string(who) +
        ": the box vectors are not independent; pass an all-zero box "
        "for an isolated system, or a padded cell of three independent "
        "vectors");
  }
  const auto spanAlong = [&](const double *v) {
    double lo = 0.0;
    double hi = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double along = dot(pos + 3 * i, v);
      lo = i == 0 ? along : std::min(lo, along);
      hi = i == 0 ? along : std::max(hi, along);
    }
    return hi - lo;
  };

  // No pair is farther apart than the diagonal of the atoms' bounding box,
  // and an image shifted by more than twice that cannot be the closer one.
  // Coefficient k of a lattice vector is its projection on reciprocal
  // vector k, so it is at most the vector's length over cell width k.
  double reach2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double unit[3] = {axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0,
                            axis == 2 ? 1.0 : 0.0};
    const double span = spanAlong(unit);
    reach2 += span * span;
  }
  const double reach = std::sqrt(reach2);
  const auto tooSkewed = [who] {
    return std::invalid_argument(
        std::string(who) +
        ": the box is too skewed, or too small for the cluster, to "
        "check every periodic image; pass an all-zero box for an isolated "
        "system, or pad the cluster into a less skewed cell");
  };
  long bound[3];
  double vectors = 1.0;
  for (int k = 0; k < 3; ++k) {
    const double width = volume / std::sqrt(dot(normals[k], normals[k]));
    const double coefficient =
        std::floor(2.0 * reach / width * (1.0 + 1.0e-9));
    // Converting a NaN, an infinity or a value past the range of long to
    // long is undefined, so the range is checked on the double first.
    if (!std::isfinite(coefficient) || !(coefficient <= kMaxImageVectors)) {
      throw tooSkewed();
    }
    bound[k] = static_cast<long>(coefficient);
    vectors *= 2.0 * coefficient + 1.0;
  }
  if (vectors > kMaxImageVectors) {
    throw tooSkewed();
  }

  const auto check = [&](long a, long b, long c) {
    double v[3];
    for (int j = 0; j < 3; ++j) {
      v[j] = static_cast<double>(a) * rows[0][j] +
             static_cast<double>(b) * rows[1][j] +
             static_cast<double>(c) * rows[2][j];
    }
    const double length2 = dot(v, v);
    const double span = spanAlong(v);
    // The slack keeps a box made exactly twice the extent, as rgpot.ase
    // makes for a cluster wider than 50 Angstrom, from failing on rounding.
    if (!(2.0 * span > length2 * (1.0 + 1.0e-12))) {
      return;
    }
    const double length = std::sqrt(length2);
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << who << ": the atoms span " << span / length << " Angstrom along ";
    const long coef[3] = {a, b, c};
    const int nonzero = (a != 0) + (b != 0) + (c != 0);
    if (nonzero == 1 && std::abs(a + b + c) == 1) {
      text << "box vector " << (a != 0 ? 1 : b != 0 ? 2 : 3);
    } else {
      text << "the lattice vector ";
      bool first = true;
      for (int j = 0; j < 3; ++j) {
        if (coef[j] == 0) {
          continue;
        }
        if (!first) {
          text << (coef[j] < 0 ? " - " : " + ");
        } else if (coef[j] < 0) {
          text << '-';
        }
        if (std::abs(coef[j]) != 1) {
          text << std::abs(coef[j]) << ' ';
        }
        text << 'a' << (j + 1);
        first = false;
      }
      text << " (a1, a2 and a3 being the box vectors)";
    }
    text << ", more than half its length of " << length
         << " Angstrom, so a periodic image is closer than the atom itself "
            "and the atoms would wrap across a periodic boundary. ORCA has "
            "no periodic boundary conditions: pass an all-zero box for an "
            "isolated system, or pad the cluster into a box in which every "
            "lattice vector is at least twice the extent of the atoms along "
            "it, as rgpot.ase does with a cube of at least 100 Angstrom";
    throw std::invalid_argument(text.str());
  };

  // The box vectors first, so an orthogonal box is reported by name.
  check(1, 0, 0);
  check(0, 1, 0);
  check(0, 0, 1);
  // L and -L give the same span, so half of the vectors are enough.
  for (long a = 0; a <= bound[0]; ++a) {
    for (long b = a == 0 ? 0 : -bound[1]; b <= bound[1]; ++b) {
      for (long c = (a == 0 && b == 0) ? 1 : -bound[2]; c <= bound[2]; ++c) {
        check(a, b, c);
      }
    }
  }
}


} // namespace rgpot::orca
