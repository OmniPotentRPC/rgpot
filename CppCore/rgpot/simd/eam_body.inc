// Included twice, once inside each EAM entry. RGPOT_EAM_HW is true or false.
  const double b0 = box[0];
  const double b1 = box[1];
  const double b2 = box[2];
  const double h0 = b0 * 0.5;
  const double h1 = b1 * 0.5;
  const double h2 = b2 * 0.5;
  struct Scratch {
    std::vector<double> drho;
    std::vector<double> dxs;
    std::vector<double> dys;
    std::vector<double> dzs;
    std::vector<double> rs;
    std::vector<double> r2s;
    std::vector<long> js;
  };
  thread_local Scratch scratch;
  const auto nn = static_cast<std::size_t>(n);
  if (scratch.drho.size() != nn * 3) {
    scratch.drho.assign(nn * 3, 0.0);
  }
  if (scratch.dxs.size() < nn) {
    scratch.dxs.resize(nn);
    scratch.dys.resize(nn);
    scratch.dzs.resize(nn);
    scratch.rs.resize(nn);
    scratch.r2s.resize(nn);
    scratch.js.resize(nn);
  }
  for (long i = 0; i < n; ++i) {
    const EamAtom &a = atoms[i];
    Part8 dens_part{};
    Part8 pair_part{};
    int nd = 0;
    int npair = 0;
    long ns = 0;
    const double xi = R[3 * i];
    const double yi = R[3 * i + 1];
    const double zi = R[3 * i + 2];
    auto consider = [&](long j, double dx, double dy, double dz, double r2) {
      if (j == i || !(r2 <= a.cutoff2)) {
        return;
      }
      const auto us = static_cast<std::size_t>(ns);
      scratch.dxs[us] = dx;
      scratch.dys[us] = dy;
      scratch.dzs[us] = dz;
      scratch.rs[us] = std::sqrt(r2);
      scratch.r2s[us] = r2;
      scratch.js[us] = j;
      ++ns;
    };
    long j = 0;
    for (; j + 1 < n; j += 2) {
      double dx0 = 0, dy0 = 0, dz0 = 0, dx1 = 0, dy1 = 0, dz1 = 0;
      if (j != i) {
        dx0 = xi - R[3 * j];
        dy0 = yi - R[3 * j + 1];
        dz0 = zi - R[3 * j + 2];
        half_box(dx0, dy0, dz0, b0, b1, b2, h0, h1, h2);
      }
      if (j + 1 != i) {
        dx1 = xi - R[3 * (j + 1)];
        dy1 = yi - R[3 * (j + 1) + 1];
        dz1 = zi - R[3 * (j + 1) + 2];
        half_box(dx1, dy1, dz1, b0, b1, b2, h0, h1, h2);
      }
      const __m128d dxv = _mm_set_pd(dx1, dx0);
      const __m128d dyv = _mm_set_pd(dy1, dy0);
      const __m128d dzv = _mm_set_pd(dz1, dz0);
      __m128d r2v;
      if constexpr (RGPOT_EAM_HW) {
        r2v = _mm_fmadd_pd(dzv, dzv, _mm_fmadd_pd(dyv, dyv, _mm_mul_pd(dxv, dxv)));
      } else {
        r2v = fma2_trunc(dzv, dzv, fma2_trunc(dyv, dyv, _mm_mul_pd(dxv, dxv)));
      }
      alignas(16) double r2a[2];
      _mm_store_pd(r2a, r2v);
      consider(j, dx0, dy0, dz0, r2a[0]);
      consider(j + 1, dx1, dy1, dz1, r2a[1]);
    }
    if (j < n && j != i) {
      double dx = xi - R[3 * j];
      double dy = yi - R[3 * j + 1];
      double dz = zi - R[3 * j + 2];
      half_box(dx, dy, dz, b0, b1, b2, h0, h1, h2);
      double r2 = 0;
      if constexpr (RGPOT_EAM_HW) {
        r2 = r2_fast<true>(dx, dy, dz);
      } else {
        r2 = fma_trunc(dz, dz, fma_trunc(dy, dy, dx * dx));
      }
      consider(j, dx, dy, dz, r2);
    }
    const double two_b2 = -2.0 * a.beta2;
    auto consume = [&](long nb, double dx, double dy, double dz, double r, double r2,
                       double eb1, double eb2, double ex) {
      const double r2sq = r2 * r2;
      const double r6 = r2sq * r2;
      const double r4 = r2sq;
      const double r5 = r4 * r;
      // 512 is a power of two, so the scale is exact and the add is the fma.
      const double rho_pair = 512.0 * eb2 + eb1;
      const double dens_term = r6 * rho_pair;
      const double t_a = (6.0 * r5) * rho_pair;
      const double neg_b1 = -a.beta1 * eb1;
      const double b2_term = (1024.0 * a.beta2) * eb2;
      const double inner = neg_b1 - b2_term;
      double mag = 0;
      if constexpr (RGPOT_EAM_HW) {
        mag = __builtin_fma(r6, inner, t_a);
      } else {
        mag = fma_trunc(r6, inner, t_a);
      }
      const double inv_r = 1.0 / r;
      const double fscale = -mag * inv_r;
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      const int kd = nd & 7;
      dens_part.e[kd] += dens_term;
      dens_part.fx[kd] += fx;
      dens_part.fy[kd] += fy;
      dens_part.fz[kd] += fz;
      ++nd;
      scratch.drho[static_cast<std::size_t>(3 * nb)] -= fx;
      scratch.drho[static_cast<std::size_t>(3 * nb + 1)] -= fy;
      scratch.drho[static_cast<std::size_t>(3 * nb + 2)] -= fz;
      if (nb > i) {
        const double md = 1.0 - ex;
        const double dd = a.d_m * md;
        double phi = 0;
        if constexpr (RGPOT_EAM_HW) {
          phi = __builtin_fma(dd, md, -a.d_m);
        } else {
          phi = fma_trunc(dd, md, -a.d_m);
        }
        const double mag_f = ((2.0 * a.alpha_m) * a.d_m) * md * (md - 1.0);
        const double fcomp = mag_f * inv_r;
        const double pfx = fcomp * dx;
        const double pfy = fcomp * dy;
        const double pfz = fcomp * dz;
        const int kp = npair & 7;
        pair_part.e[kp] += phi;
        pair_part.fx[kp] += pfx;
        pair_part.fy[kp] += pfy;
        pair_part.fz[kp] += pfz;
        ++npair;
        F[3 * nb] += pfx;
        F[3 * nb + 1] += pfy;
        F[3 * nb + 2] += pfz;
      }
    };
    long t = 0;
    for (; t + 8 <= ns; t += 8) {
      alignas(16) double a1[8], a2[8], a3[8], e1[8], e2[8], e3[8];
      for (int k = 0; k < 8; ++k) {
        const double r = scratch.rs[static_cast<std::size_t>(t + k)];
        a1[k] = -a.beta1 * r;
        a2[k] = two_b2 * r;
        a3[k] = -a.alpha_m * (r - a.r_m);
      }
      if constexpr (RGPOT_EAM_HW) {
        exp8_hw(a1, e1);
        exp8_hw(a2, e2);
        exp8_hw(a3, e3);
      } else {
        exp8(a1, e1);
        exp8(a2, e2);
        exp8(a3, e3);
      }
      for (int k = 0; k < 8; ++k) {
        const auto us = static_cast<std::size_t>(t + k);
        consume(scratch.js[us], scratch.dxs[us], scratch.dys[us], scratch.dzs[us],
                scratch.rs[us], scratch.r2s[us], e1[k], e2[k], e3[k]);
      }
    }
    for (; t < ns; ++t) {
      const auto us = static_cast<std::size_t>(t);
      const double r = scratch.rs[us];
      const double eb1 = exp(-a.beta1 * r);
      const double eb2 = exp(two_b2 * r);
      const double ex = exp(-a.alpha_m * (r - a.r_m));
      consume(scratch.js[us], scratch.dxs[us], scratch.dys[us], scratch.dzs[us], r,
              scratch.r2s[us], eb1, eb2, ex);
    }
    const double dens = combine8(dens_part.e);
    scratch.drho[static_cast<std::size_t>(3 * i)] += combine8(dens_part.fx);
    scratch.drho[static_cast<std::size_t>(3 * i + 1)] += combine8(dens_part.fy);
    scratch.drho[static_cast<std::size_t>(3 * i + 2)] += combine8(dens_part.fz);
    *U += combine8(pair_part.e);
    *U += embed_horner<false>(a.coeff, dens);
    F[3 * i] -= combine8(pair_part.fx);
    F[3 * i + 1] -= combine8(pair_part.fy);
    F[3 * i + 2] -= combine8(pair_part.fz);
    const double dF = embed_force<false>(a.coeff, dens);
    auto apply_rho = [&](long atom) {
      const std::size_t base = static_cast<std::size_t>(3 * atom);
      F[base] -= dF * scratch.drho[base];
      F[base + 1] -= dF * scratch.drho[base + 1];
      F[base + 2] -= dF * scratch.drho[base + 2];
      scratch.drho[base] = 0.0;
      scratch.drho[base + 1] = 0.0;
      scratch.drho[base + 2] = 0.0;
    };
    apply_rho(i);
    for (long u = 0; u < ns; ++u) {
      apply_rho(scratch.js[static_cast<std::size_t>(u)]);
    }
  }
