/*
 * rgpot plugin: Morse pairwise potential (C).
 *
 * E = D_e * sum_{i<j} (1 - exp(-alpha*(r_ij - r_e)))^2
 * F_i = -dE/dr_i (analytic)
 *
 * Config JSON keys: "D_e" (default 1.0), "alpha" (1.0), "r_e" (1.0),
 *                   "cutoff" (10.0).
 *
 * Build: cc -shared -fPIC -o rgpot_morse.so morse_plugin.c -I../../../include -lm
 */

#include <rgpot/plugin.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double D_e;
  double alpha;
  double r_e;
  double cutoff;
} MorseParams;

static rgpot_plugin_status_t
morse_create(const char *config_json, rgpot_plugin_instance **out) {
  MorseParams *p = (MorseParams *)calloc(1, sizeof(MorseParams));
  if (!p)
    return RGPOT_PLUGIN_ERROR;
  p->D_e = 1.0;
  p->alpha = 1.0;
  p->r_e = 1.0;
  p->cutoff = 10.0;
  /* Minimal JSON parsing for numeric fields */
  if (config_json) {
    const char *s;
    if ((s = strstr(config_json, "\"D_e\"")))
      sscanf(s, "\"D_e\" : %lf", &p->D_e);
    if ((s = strstr(config_json, "\"alpha\"")))
      sscanf(s, "\"alpha\" : %lf", &p->alpha);
    if ((s = strstr(config_json, "\"r_e\"")))
      sscanf(s, "\"r_e\" : %lf", &p->r_e);
    if ((s = strstr(config_json, "\"cutoff\"")))
      sscanf(s, "\"cutoff\" : %lf", &p->cutoff);
  }
  *out = (rgpot_plugin_instance *)p;
  return RGPOT_PLUGIN_OK;
}

static void morse_destroy(rgpot_plugin_instance *inst) { free(inst); }

static rgpot_plugin_status_t
morse_calculate(rgpot_plugin_instance *inst, const rgpot_plugin_input_t *input,
                rgpot_plugin_output_t *output) {
  MorseParams *p = (MorseParams *)inst;
  size_t n = input->n_atoms;
  const double *R = input->positions;
  double *F = output->forces;

  memset(F, 0, n * 3 * sizeof(double));
  double energy = 0.0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double dx = R[3 * i] - R[3 * j];
      double dy = R[3 * i + 1] - R[3 * j + 1];
      double dz = R[3 * i + 2] - R[3 * j + 2];
      double r = sqrt(dx * dx + dy * dy + dz * dz);
      if (r > p->cutoff || r < 1e-10)
        continue;

      double exp_term = exp(-p->alpha * (r - p->r_e));
      double one_minus = 1.0 - exp_term;
      energy += p->D_e * one_minus * one_minus;

      /* dE/dr = 2 * D_e * alpha * (1-exp) * exp */
      double dEdr = 2.0 * p->D_e * p->alpha * one_minus * exp_term;
      double f_over_r = dEdr / r;

      F[3 * i] -= f_over_r * dx;
      F[3 * i + 1] -= f_over_r * dy;
      F[3 * i + 2] -= f_over_r * dz;
      F[3 * j] += f_over_r * dx;
      F[3 * j + 1] += f_over_r * dy;
      F[3 * j + 2] += f_over_r * dz;
    }
  }

  output->energy = energy;
  output->variance = 0.0;
  return RGPOT_PLUGIN_OK;
}

static uint32_t morse_caps(const rgpot_plugin_instance *inst) {
  (void)inst;
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES;
}

static const rgpot_plugin_descriptor_t morse_desc = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "morse",
    "1.0.0",
    NULL, /* angstrom */
    NULL, /* eV */
    morse_create,
    morse_destroy,
    morse_calculate,
    morse_caps,
    NULL,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
  return &morse_desc;
}
