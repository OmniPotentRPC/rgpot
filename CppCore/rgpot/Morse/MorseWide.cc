// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <vector>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/OrderedWalk.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/stress.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {

struct MorseHit {
  std::int32_t i = 0;
  std::int32_t j = 0;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double r = 0.0;
};

void MorsePot::forceParallel(const ForceInput &in, ForceOut *out) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
  *U = 0.0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }

  const auto n_atoms = static_cast<std::size_t>(N);
  nlist::Mode mode = mode_;
  if (mode == nlist::Mode::Auto && n_atoms >= 20000) {
    mode = nlist::Mode::Stored;
  }
  const double skin = mode == nlist::Mode::Stream ? 0.0 : neighbourSkin(cuttOffR);
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(neighbours_, R, n_atoms, box, cuttOffR, skin, nlist::Image::Round, mode,
                     periodic);
  const double cutoff2 = cuttOffR * cuttOffR;
  const double twoDeA = 2.0 * De * a;
  const bool switched = m_config.switch_width != 0.0;
  const int nt = nlist::threadsRequested();

  constexpr int kChunk = 128;
  struct Bag {
    std::vector<nlist::HalfPart> parts;
    int nt = -1;
  };
  thread_local Bag bag;
  if (bag.nt != nt) {
    bag.parts.assign(static_cast<std::size_t>(nt), {});
    bag.nt = nt;
  }
  for (auto &part : bag.parts) {
    part.terms.clear();
  }
  nlist::HalfPart *parts = bag.parts.data();
  const double well = De;
  const double alpha = a;
  const double eq = re;
  const double shift = energyCutoff;
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    const auto range = nlist::partition(N, thread, nt);
    auto &mine = parts[thread].terms;
    MorseHit hits[kChunk];
    double args[kChunk];
    double exps[kChunk];
    int nbuf = 0;
    auto flush = [&]() {
      if (nbuf == 0) {
        return;
      }
      simd::exp_fixed_n(args, exps, static_cast<std::size_t>(nbuf));
      for (int k = 0; k < nbuf; ++k) {
        const MorseHit &h = hits[k];
        const double d = 1.0 - exps[k];
        const double bare = well * d * d - well;
        nlist::HalfTerm term;
        term.i = h.i;
        term.j = h.j;
        term.dx = h.dx;
        term.dy = h.dy;
        term.dz = h.dz;
        if (switched) {
          const auto s = m_switch(h.r);
          term.e = bare * s.s;
          term.fscale = (twoDeA * d * (d - 1.0) * s.s - bare * s.dsdr) / h.r;
        } else {
          term.e = bare - shift;
          term.fscale = twoDeA * d * (d - 1.0) / h.r;
        }
        mine.push_back(term);
      }
      nbuf = 0;
    };
    neighbours_.forHalf(range.begin, range.end,
                        [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                            double r2) {
                          if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
                            return;
                          }
                          if (nbuf == kChunk) {
                            flush();
                          }
                          const double r = std::sqrt(r2);
                          hits[nbuf] = MorseHit{i, j, dx, dy, dz, r};
                          args[nbuf] = -alpha * (r - eq);
                          ++nbuf;
                        });
    flush();
  }
  double energy = 0.0;
  nlist::replay_half(bag.parts, F, energy, [&](const nlist::HalfTerm &term) {
    accumulatePairStrain(strain, term.fscale, term.dx, term.dy, term.dz);
  });
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
