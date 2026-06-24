#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rgpot_metatomic_config_t {
  const char *model_path;
  const char *device;
  const char *length_unit;
  const char *extensions_directory;
  bool check_consistency;
  double uncertainty_threshold;
  const char *dtype_override;
} rgpot_metatomic_config_t;

rgpot_potential_t *
rgpot_metatomic_potential_new(const rgpot_metatomic_config_t *config);

#ifdef __cplusplus
} // extern "C"
#endif
