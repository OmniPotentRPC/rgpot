/*
 * rgpot plugin wrapper for the Fortran Coulomb potential.
 *
 * This C file provides the rgpot_plugin_init() entry point and delegates
 * the actual calculation to the Fortran subroutine coulomb_forces() defined
 * in coulomb_plugin.f90.
 *
 * Build both together:
 *   gfortran -c -fPIC coulomb_plugin.f90 -o coulomb_f.o
 *   cc -shared -fPIC -o rgpot_coulomb.so coulomb_plugin_wrapper.c coulomb_f.o \
 *      -I../../../include -lgfortran
 */

#include <rgpot/plugin.h>

#include <stdlib.h>
#include <string.h>

/* Fortran subroutine (bind(C) name) */
extern void coulomb_forces(int n, const double *pos, const int *atmnrs,
                           const double *cell, double *energy, double *forces);

static rgpot_plugin_status_t
coulomb_create(const char *config_json, rgpot_plugin_instance **out) {
  (void)config_json;
  *out = (rgpot_plugin_instance *)(void *)1; /* stateless */
  return RGPOT_PLUGIN_OK;
}

static void coulomb_destroy(rgpot_plugin_instance *inst) { (void)inst; }

static rgpot_plugin_status_t
coulomb_calculate(rgpot_plugin_instance *inst,
                  const rgpot_plugin_input_t *input,
                  rgpot_plugin_output_t *output) {
  (void)inst;
  int n = (int)input->n_atoms;
  output->variance = 0.0;
  memset(output->forces, 0, (size_t)n * 3 * sizeof(double));

  coulomb_forces(n, input->positions, input->atomic_numbers, input->cell,
                 &output->energy, output->forces);

  return RGPOT_PLUGIN_OK;
}

static uint32_t coulomb_caps(const rgpot_plugin_instance *inst) {
  (void)inst;
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES;
}

static const rgpot_plugin_descriptor_t coulomb_desc = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "coulomb",
    "1.0.0",
    "angstrom",
    "eV",
    coulomb_create,
    coulomb_destroy,
    coulomb_calculate,
    coulomb_caps,
    NULL,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
  return &coulomb_desc;
}
