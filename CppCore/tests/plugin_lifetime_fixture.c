/* MIT License */
#include <rgpot/plugin.h>
#include <stdio.h>
#include <stdlib.h>

static rgpot_plugin_status_t create(const char *config, rgpot_plugin_instance **output) {
  (void)config;
  *output = (rgpot_plugin_instance *)malloc(1);
  return *output ? RGPOT_PLUGIN_OK : RGPOT_PLUGIN_ERROR;
}
static void destroy(rgpot_plugin_instance *instance) {
  free(instance);
  puts("plugin instance destroyed");
  fflush(stdout);
}
static rgpot_plugin_status_t calculate(rgpot_plugin_instance *instance,
                                      const rgpot_plugin_input_t *input,
                                      rgpot_plugin_output_t *output) {
  (void)instance;
  output->energy = 37.0;
  output->variance = 0.0;
  for (size_t i = 0; i < 3 * input->n_atoms; ++i)
    output->forces[i] = 0.0;
  return RGPOT_PLUGIN_OK;
}
static const rgpot_plugin_descriptor_t descriptor = {
    sizeof(rgpot_plugin_descriptor_t), RGPOT_PLUGIN_API_VERSION,
    "lifetime-contract", "1", NULL, NULL, create, destroy, calculate, NULL, NULL};
RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
  return &descriptor;
}
