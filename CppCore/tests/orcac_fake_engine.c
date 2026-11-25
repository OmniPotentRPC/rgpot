/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A stand-in for liborcac (OmniPotentRPC/orcac) that runs no ORCA. A session
 * keeps the message it was created from; its energy in Hartree is the length
 * of that message and its forces are 0.1 * (i + 1) Hartree/Bohr.
 * orcac_fake_last copies out the last message and says which create call
 * received it, so a test can check what ORCACPot handed the shim.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

typedef struct OrcaCResult {
  int ok;
  double energy_h;
  int reused_orbitals;
  char message[1024];
} OrcaCResult;

typedef struct OrcaCSession {
  size_t len;
} OrcaCSession;

static unsigned char g_last[65536];
static size_t g_last_len;
static int g_from_config = -1;
static char g_error[256] = "";

EXPORT int orcac_c_abi_version(void) { return 1; }
EXPORT int orcac_available(void) { return 0; }
EXPORT const char *orcac_last_error(void) { return g_error; }

static OrcaCSession *make(const void *msg, size_t len, int from_config) {
  OrcaCSession *s;
  if (len == 0 || len > sizeof(g_last)) {
    snprintf(g_error, sizeof(g_error), "fake orcac: message of %zu bytes", len);
    return NULL;
  }
  memcpy(g_last, msg, len);
  g_last_len = len;
  g_from_config = from_config;
  s = calloc(1, sizeof(*s));
  s->len = len;
  return s;
}

EXPORT OrcaCSession *orcac_session_create(const void *p, size_t n) {
  return make(p, n, 0);
}

EXPORT OrcaCSession *orcac_session_create_from_config(const void *p, size_t n) {
  return make(p, n, 1);
}

EXPORT void orcac_session_destroy(OrcaCSession *s) { free(s); }

EXPORT const char *orcac_session_last_error(const OrcaCSession *s) {
  (void)s;
  return "";
}

EXPORT const char *orcac_session_run_directory(const OrcaCSession *s) {
  (void)s;
  return "/fake/orcac";
}

EXPORT OrcaCResult orcac_session_energy_forces(OrcaCSession *s, int n_atoms,
                                               const double *pos,
                                               const int *z, double *forces) {
  OrcaCResult r;
  int i;
  (void)pos;
  (void)z;
  memset(&r, 0, sizeof(r));
  for (i = 0; i < 3 * n_atoms; ++i) {
    forces[i] = 0.1 * (double)(i + 1);
  }
  r.ok = 1;
  r.energy_h = (double)s->len;
  return r;
}

EXPORT size_t orcac_fake_last(void *out, size_t cap, int *from_config) {
  memcpy(out, g_last, g_last_len < cap ? g_last_len : cap);
  *from_config = g_from_config;
  return g_last_len;
}
