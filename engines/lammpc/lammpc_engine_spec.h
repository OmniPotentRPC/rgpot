#pragma once

#include <stdint.h>

/*
 * rgpot lammpc engine spec: this fixed header is followed by one flat
 * PotentialConfig and one flat ForceInput. Integers use the host C ABI because
 * the buffer is consumed synchronously by an in-process engine.
 */
#define RGPOT_LAMMPC_SESSION_SPEC_MAGIC UINT64_C(0x5247504f544c4d50)

typedef struct rgpot_lammpc_session_spec_header {
  uint64_t magic;
  uint64_t config_size;
  uint64_t force_input_size;
} rgpot_lammpc_session_spec_header;
