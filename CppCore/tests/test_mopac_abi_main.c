/**
 * @file test_mopac_abi_main.c
 * @brief Link against mopac_abi_stub; verify stub reports unavailable.
 */
#include "mopac_c_abi.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  if (mopacc_available() != 0) {
    fprintf(stderr, "FAIL: stub should report available=0\n");
    return 1;
  }
  const char *ver = mopacc_version();
  if (!ver || strstr(ver, "stub") == NULL) {
    fprintf(stderr, "FAIL: expected stub version string, got '%s'\n",
            ver ? ver : "(null)");
    return 1;
  }
  MopacCResult r = mopacc_energy_gradient(0, NULL, NULL, NULL, 0, NULL);
  if (r.ok != 0) {
    fprintf(stderr, "FAIL: stub energy_gradient should set ok=0\n");
    return 1;
  }
  if (mopacc_set_params(NULL, 0) == 0) {
    fprintf(stderr, "FAIL: stub set_params should fail\n");
    return 1;
  }
  if (mopacc_c_abi_version() != RGPOT_MOPACC_C_ABI_VERSION) {
    fprintf(stderr, "FAIL: stub ABI version\n");
    return 1;
  }
  if (mopacc_abi_version() != MOPACC_ABI_VERSION) {
    fprintf(stderr, "FAIL: stub library ABI version\n");
    return 1;
  }
  if (mopacc_result_size() != sizeof(MopacCResult) ||
      mopacc_params_size() != sizeof(MopacCParams)) {
    fprintf(stderr, "FAIL: stub struct size queries\n");
    return 1;
  }
  printf("ok: stub ABI version=%s\n", ver);
  return 0;
}
