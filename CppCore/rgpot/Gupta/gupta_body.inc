/* Cleri–Rosato sum, every pair, no cutoff. Included in a function whose
 * arguments are x, n, slot, n_species, a, xi, p, q, r0, energy, gradient.
 * The first pass stores the pair derivative the second pass consumes, in
 * the same i < j order, so the density and the force match that walk.
 */
{
  double rho_stack[RGPOT_GUPTA_STACK_ATOMS];
  double pack_stack[RGPOT_GUPTA_PACK_STRIDE * RGPOT_GUPTA_STACK_PAIRS];
  double *rho_heap = NULL;
  double *pack_heap = NULL;
  double *rho = rho_stack;
  double *pack = pack_stack;
  size_t pack_limit = 0;
  if (n > RGPOT_GUPTA_STACK_ATOMS) {
    rho_heap = (double *)malloc(n * sizeof(double));
    if (rho_heap == NULL) {
      *energy = INFINITY;
      for (size_t k = 0; k < 3 * n; k++) {
        gradient[k] = 0.0;
      }
      return;
    }
    rho = rho_heap;
  }
  if (n >= 2) {
    const size_t a_n = (n & 1) ? n : n / 2;
    const size_t b_n = (n & 1) ? (n - 1) / 2 : n - 1;
    pack_limit = a_n > RGPOT_GUPTA_MAX_PACK_PAIRS / b_n
                     ? RGPOT_GUPTA_MAX_PACK_PAIRS
                     : a_n * b_n;
    if (pack_limit > RGPOT_GUPTA_STACK_PAIRS) {
      pack_heap = (double *)malloc(pack_limit * RGPOT_GUPTA_PACK_STRIDE * sizeof(double));
      if (pack_heap == NULL) {
        pack_limit = RGPOT_GUPTA_STACK_PAIRS;
      } else {
        pack = pack_heap;
      }
    }
  }
  for (size_t i = 0; i < n; i++) {
    rho[i] = 0.0;
  }
  for (size_t k = 0; k < 3 * n; k++) {
    gradient[k] = 0.0;
  }
  double total = 0.0;
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    const size_t base_i = 3 * i;
    const size_t si = (size_t)slot[i];
    for (size_t j = i + 1; j < n; j++) {
      const size_t base_j = 3 * j;
      const double d0 = x[base_i] - x[base_j];
      const double d1 = x[base_i + 1] - x[base_j + 1];
      const double d2 = x[base_i + 2] - x[base_j + 2];
      const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
      /* A coincidence is not differentiable. NaN is not <= 0, so it
       * falls through and the energy and the forces become NaN. */
      if (r2 <= 0.0) {
        free(pack_heap);
        free(rho_heap);
        *energy = INFINITY;
        return;
      }
      const size_t key = si * n_species + (size_t)slot[j];
      const double r = sqrt(r2);
      const double u = r / r0[key] - 1.0;
      const double phi = 2.0 * a[key] * exp(-p[key] * u);
      const double psi = xi[key] * xi[key] * exp(-2.0 * q[key] * u);
      total += phi;
      rho[i] += psi;
      rho[j] += psi;
      if (m < pack_limit) {
        double *rec = pack + RGPOT_GUPTA_PACK_STRIDE * m;
        rec[RGPOT_GUPTA_PACK_DX] = d0;
        rec[RGPOT_GUPTA_PACK_DY] = d1;
        rec[RGPOT_GUPTA_PACK_DZ] = d2;
        rec[RGPOT_GUPTA_PACK_R] = r;
        rec[RGPOT_GUPTA_PACK_DPHI] = -(p[key] / r0[key]) * phi;
        rec[RGPOT_GUPTA_PACK_DPSI] = -(2.0 * q[key] / r0[key]) * psi;
      }
      m++;
    }
  }
  for (size_t i = 0; i < n; i++) {
    total -= sqrt(rho[i]);
  }
  m = 0;
  for (size_t i = 0; i < n; i++) {
    const size_t base_i = 3 * i;
    const size_t si = (size_t)slot[i];
    const double inv_i = rho[i] == 0.0 ? 0.0 : 1.0 / sqrt(rho[i]);
    for (size_t j = i + 1; j < n; j++) {
      double d0, d1, d2, r, dphi, dpsi;
      if (m < pack_limit) {
        const double *rec = pack + RGPOT_GUPTA_PACK_STRIDE * m;
        d0 = rec[RGPOT_GUPTA_PACK_DX];
        d1 = rec[RGPOT_GUPTA_PACK_DY];
        d2 = rec[RGPOT_GUPTA_PACK_DZ];
        r = rec[RGPOT_GUPTA_PACK_R];
        dphi = rec[RGPOT_GUPTA_PACK_DPHI];
        dpsi = rec[RGPOT_GUPTA_PACK_DPSI];
      } else {
        const size_t base_j = 3 * j;
        const size_t key = si * n_species + (size_t)slot[j];
        d0 = x[base_i] - x[base_j];
        d1 = x[base_i + 1] - x[base_j + 1];
        d2 = x[base_i + 2] - x[base_j + 2];
        const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
        r = sqrt(r2);
        const double u = r / r0[key] - 1.0;
        const double phi = 2.0 * a[key] * exp(-p[key] * u);
        const double psi = xi[key] * xi[key] * exp(-2.0 * q[key] * u);
        dphi = -(p[key] / r0[key]) * phi;
        dpsi = -(2.0 * q[key] / r0[key]) * psi;
      }
      const double inv_j = rho[j] == 0.0 ? 0.0 : 1.0 / sqrt(rho[j]);
      const double embed = -0.5 * (inv_i + inv_j);
      const double coef = (dphi + embed * dpsi) / r;
      gradient[3 * i] += coef * d0;
      gradient[3 * i + 1] += coef * d1;
      gradient[3 * i + 2] += coef * d2;
      gradient[3 * j] -= coef * d0;
      gradient[3 * j + 1] -= coef * d1;
      gradient[3 * j + 2] -= coef * d2;
      m++;
    }
  }
  free(pack_heap);
  free(rho_heap);
  *energy = total;
}
