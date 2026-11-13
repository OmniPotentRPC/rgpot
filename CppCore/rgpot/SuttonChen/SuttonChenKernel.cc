// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/SuttonChen/SuttonChenKernel.hpp"

#include "rgpot/stress.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace rgpot {
namespace {

void *alloc_aligned(std::size_t alignment, std::size_t bytes) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  void *mem = nullptr;
  return posix_memalign(&mem, alignment, bytes) == 0 ? mem : nullptr;
#endif
}

void free_aligned(void *mem) {
#if defined(_WIN32)
  _aligned_free(mem);
#else
  std::free(mem);
#endif
}

// One allocation: coordinates and densities, then the accepted pairs.
// A later call of the same size reuses both.
struct Scratch {
  double *x{nullptr};
  double *y{nullptr};
  double *z{nullptr};
  double *rho{nullptr};
  double *w{nullptr};
  double *dx{nullptr};
  double *dy{nullptr};
  double *dz{nullptr};
  double *phi{nullptr};
  double *rep{nullptr};
  double *inv{nullptr};
  std::int32_t *ia{nullptr};
  std::int32_t *ib{nullptr};
  std::size_t atom_cap{0};
  std::size_t pair_cap{0};
  void *atom_mem{nullptr};
  void *pair_mem{nullptr};

  Scratch() = default;
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  ~Scratch() {
    free_aligned(atom_mem);
    free_aligned(pair_mem);
  }

  void ensure_atoms(std::size_t n) {
    if (n <= atom_cap) {
      return;
    }
    std::size_t nc = atom_cap == 0 ? 32 : atom_cap * 2;
    if (nc < n) {
      nc = n;
    }
    nc = (nc + 7u) & ~std::size_t{7};
    void *mem = alloc_aligned(64, nc * 5 * sizeof(double));
    if (mem == nullptr) {
      throw std::bad_alloc();
    }
    free_aligned(atom_mem);
    atom_mem = mem;
    x = static_cast<double *>(mem);
    y = x + nc;
    z = y + nc;
    rho = z + nc;
    w = rho + nc;
    atom_cap = nc;
  }

  void ensure_pairs(std::size_t n) {
    if (n <= pair_cap) {
      return;
    }
    std::size_t nc = pair_cap == 0 ? 64 : pair_cap * 2;
    if (nc < n) {
      nc = n;
    }
    // Six doubles and two int32s. The int32 pair follows the doubles,
    // so it stays 8-byte aligned.
    const std::size_t bytes =
        nc * (6 * sizeof(double) + 2 * sizeof(std::int32_t));
    void *mem = alloc_aligned(64, bytes);
    if (mem == nullptr) {
      throw std::bad_alloc();
    }
    free_aligned(pair_mem);
    pair_mem = mem;
    dx = static_cast<double *>(mem);
    dy = dx + nc;
    dz = dy + nc;
    phi = dz + nc;
    rep = phi + nc;
    inv = rep + nc;
    ia = reinterpret_cast<std::int32_t *>(inv + nc);
    ib = ia + nc;
    pair_cap = nc;
  }
};

Scratch &scratch() {
  thread_local Scratch soa;
  return soa;
}

// Binary power of (a/r)^2, the same products as the previous kernel.
double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

double pow_ar(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp & 1) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

// ipow(u, 3) and ipow(u, 6). u2 * u4 is the exponent-6 product, not
// (u^3)^2, which rounds differently.
inline void powers_12_6(double u, double &phi, double &rep) {
  const double u2 = u * u;
  phi = u * u2;
  const double u4 = u2 * u2;
  rep = u2 * u4;
}

void load_soa(const double *pos, std::size_t n, Scratch &s) {
  for (std::size_t i = 0; i < n; ++i) {
    s.x[i] = pos[3 * i];
    s.y[i] = pos[3 * i + 1];
    s.z[i] = pos[3 * i + 2];
    s.rho[i] = 0.0;
  }
}

// One walk writes densities, pair energy, and the bond derivatives.
std::size_t gather(Scratch &s, std::size_t n, double a2, double epsilon,
                   int n_exp, int m_exp, double cutoff2, bool fast_12_6,
                   double &epair) {
  std::size_t p = 0;
  epair = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = s.x[i];
    const double yi = s.y[i];
    const double zi = s.z[i];
    double ri = 0.0;
    for (std::size_t j = i + 1; j < n; ++j) {
      const double dx = xi - s.x[j];
      const double dy = yi - s.y[j];
      const double dz = zi - s.z[j];
      const double r2 = (dx * dx + dy * dy) + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
      }
      const double inv = 1.0 / r2;
      const double u = a2 * inv;
      double phi = 0.0;
      double rep = 0.0;
      if (fast_12_6) {
        powers_12_6(u, phi, rep);
      } else {
        phi = pow_ar(u, m_exp);
        rep = pow_ar(u, n_exp);
      }
      ri += phi;
      s.rho[j] += phi;
      epair += epsilon * rep;
      s.dx[p] = dx;
      s.dy[p] = dy;
      s.dz[p] = dz;
      s.phi[p] = phi;
      s.rep[p] = rep;
      s.inv[p] = inv;
      s.ia[p] = static_cast<std::int32_t>(i);
      s.ib[p] = static_cast<std::int32_t>(j);
      ++p;
    }
    s.rho[i] += ri;
  }
  return p;
}

#if defined(__x86_64__)
struct Quad {
  alignas(32) double r2[4];
  alignas(32) double phi[4];
  alignas(32) double rep[4];
  alignas(32) double inv[4];
  alignas(32) double dx[4];
  alignas(32) double dy[4];
  alignas(32) double dz[4];
};

__attribute__((target("avx2"))) void
sc_quad_126(double xi, double yi, double zi, const double *x, const double *y,
            const double *z, std::size_t j, __m256d va2, Quad &q) {
  const __m256d dx = _mm256_sub_pd(_mm256_set1_pd(xi), _mm256_loadu_pd(x + j));
  const __m256d dy = _mm256_sub_pd(_mm256_set1_pd(yi), _mm256_loadu_pd(y + j));
  const __m256d dz = _mm256_sub_pd(_mm256_set1_pd(zi), _mm256_loadu_pd(z + j));
  const __m256d r2 =
      _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                    _mm256_mul_pd(dz, dz));
  const __m256d inv = _mm256_div_pd(_mm256_set1_pd(1.0), r2);
  const __m256d u = _mm256_mul_pd(va2, inv);
  const __m256d u2 = _mm256_mul_pd(u, u);
  const __m256d phi = _mm256_mul_pd(u, u2);
  const __m256d u4 = _mm256_mul_pd(u2, u2);
  const __m256d rep = _mm256_mul_pd(u2, u4);
  _mm256_store_pd(q.r2, r2);
  _mm256_store_pd(q.phi, phi);
  _mm256_store_pd(q.rep, rep);
  _mm256_store_pd(q.inv, inv);
  _mm256_store_pd(q.dx, dx);
  _mm256_store_pd(q.dy, dy);
  _mm256_store_pd(q.dz, dz);
}

__attribute__((target("avx2"))) std::size_t
sc_avx2_126(Scratch &s, std::size_t n, double a2, double epsilon,
            double cutoff2, double &epair) {
  const __m256d va2 = _mm256_set1_pd(a2);
  std::size_t p = 0;
  epair = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = s.x[i];
    const double yi = s.y[i];
    const double zi = s.z[i];
    double ri = 0.0;
    std::size_t j = i + 1;
    Quad q;
    for (; j + 3 < n; j += 4) {
      sc_quad_126(xi, yi, zi, s.x, s.y, s.z, j, va2, q);
      for (int k = 0; k < 4; ++k) {
        const double r2 = q.r2[k];
        if (!(r2 > 0.0) || r2 > cutoff2) {
          continue;
        }
        const double phi = q.phi[k];
        const double rep = q.rep[k];
        const auto jj =
            static_cast<std::size_t>(j + static_cast<std::size_t>(k));
        ri += phi;
        s.rho[jj] += phi;
        epair += epsilon * rep;
        s.dx[p] = q.dx[k];
        s.dy[p] = q.dy[k];
        s.dz[p] = q.dz[k];
        s.phi[p] = phi;
        s.rep[p] = rep;
        s.inv[p] = q.inv[k];
        s.ia[p] = static_cast<std::int32_t>(i);
        s.ib[p] = static_cast<std::int32_t>(jj);
        ++p;
      }
    }
    for (; j < n; ++j) {
      const double dx = xi - s.x[j];
      const double dy = yi - s.y[j];
      const double dz = zi - s.z[j];
      const double r2 = (dx * dx + dy * dy) + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
      }
      const double inv = 1.0 / r2;
      double phi = 0.0;
      double rep = 0.0;
      powers_12_6(a2 * inv, phi, rep);
      ri += phi;
      s.rho[j] += phi;
      epair += epsilon * rep;
      s.dx[p] = dx;
      s.dy[p] = dy;
      s.dz[p] = dz;
      s.phi[p] = phi;
      s.rep[p] = rep;
      s.inv[p] = inv;
      s.ia[p] = static_cast<std::int32_t>(i);
      s.ib[p] = static_cast<std::int32_t>(j);
      ++p;
    }
    s.rho[i] += ri;
  }
  return p;
}
#endif

void embed_and_forces(Scratch &s, std::size_t n, std::size_t npair,
                      double epsilon, double c_coef, int n_exp, int m_exp,
                      double epair, double *forces, double &energy,
                      double *strain) {
  double eemb = 0.0;
  const double pref = epsilon * c_coef;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(s.rho[i] > 0.0)) {
      s.w[i] = 0.0;
      continue;
    }
    const double root = std::sqrt(s.rho[i]);
    eemb -= pref * root;
    s.w[i] = pref / (2.0 * root);
  }
  const double n_real = static_cast<double>(n_exp);
  const double m_real = static_cast<double>(m_exp);
  for (std::size_t p = 0; p < npair; ++p) {
    const auto i = static_cast<std::size_t>(s.ia[p]);
    const auto j = static_cast<std::size_t>(s.ib[p]);
    const double dx = s.dx[p];
    const double dy = s.dy[p];
    const double dz = s.dz[p];
    const double fscale =
        (epsilon * n_real * s.rep[p] - m_real * s.phi[p] * (s.w[i] + s.w[j])) *
        s.inv[p];
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    forces[3 * i] += fx;
    forces[3 * i + 1] += fy;
    forces[3 * i + 2] += fz;
    forces[3 * j] -= fx;
    forces[3 * j + 1] -= fy;
    forces[3 * j + 2] -= fz;
    if (strain != nullptr) {
      accumulatePairStrain(strain, fscale, dx, dy, dz);
    }
  }
  energy = epair + eemb;
}

} // namespace

void sutton_chen_direct(const double *pos, std::size_t n, double epsilon,
                        double c, double a, int n_exp, int m_exp, double cutoff,
                        double *forces, double *energy, double *strain) {
  if (energy == nullptr) {
    return;
  }
  if (n < 2 || forces == nullptr || pos == nullptr) {
    *energy = 0.0;
    return;
  }
  Scratch &soa = scratch();
  soa.ensure_atoms(n);
  soa.ensure_pairs(n * (n - 1) / 2);
  load_soa(pos, n, soa);
  const double a2 = a * a;
  const double cutoff2 = cutoff * cutoff;
  const bool fast_12_6 = n_exp == 12 && m_exp == 6;
  double epair = 0.0;
  std::size_t npair = 0;
#if defined(__x86_64__)
  if (fast_12_6 && __builtin_cpu_supports("avx2")) {
    npair = sc_avx2_126(soa, n, a2, epsilon, cutoff2, epair);
  } else
#endif
  {
    npair =
        gather(soa, n, a2, epsilon, n_exp, m_exp, cutoff2, fast_12_6, epair);
  }
  double acc = 0.0;
  embed_and_forces(soa, n, npair, epsilon, c, n_exp, m_exp, epair, forces, acc,
                   strain);
  *energy = acc;
}

} // namespace rgpot
