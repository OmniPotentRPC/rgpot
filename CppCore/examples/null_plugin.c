/*
 * Minimal rgpot plugin example.
 *
 * This "null potential" returns zero energy and zero forces for any input.
 * It demonstrates the minimum code needed to implement a plugin.
 *
 * Build:
 *   cc -shared -fPIC -o rgpot_null.so null_plugin.c -I../../include
 *
 * Use:
 *   export RGPOT_PLUGIN_PATH=/path/to/dir/containing/rgpot_null.so
 */

#include <rgpot/plugin.h>
#include <string.h>

static rgpot_plugin_status_t
null_create(const char *config_json, rgpot_plugin_instance **out) {
  (void)config_json;
  /* No state needed -- use a sentinel non-NULL pointer */
  *out = (rgpot_plugin_instance *)(void *)1;
  return RGPOT_PLUGIN_OK;
}

static void null_destroy(rgpot_plugin_instance *inst) { (void)inst; }

static rgpot_plugin_status_t
null_calculate(rgpot_plugin_instance *inst,
               const rgpot_plugin_input_t *input,
               rgpot_plugin_output_t *output) {
  (void)inst;
  output->energy = 0.0;
  output->variance = 0.0;
  memset(output->forces, 0, input->n_atoms * 3 * sizeof(double));
  return RGPOT_PLUGIN_OK;
}

static const rgpot_plugin_descriptor_t null_desc = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "null",
    "0.0.1",
    NULL, /* angstrom */
    NULL, /* eV */
    null_create,
    null_destroy,
    null_calculate,
    NULL,
    NULL,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *
rgpot_plugin_init(void) {
  return &null_desc;
}
