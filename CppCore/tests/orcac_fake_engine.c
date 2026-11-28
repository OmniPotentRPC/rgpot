/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A stand-in for liborcac (OmniPotentRPC/orcac) C ABI 1 that runs no ORCA.
 * It exports exactly the thirteen functions of orcac.h and nothing else
 * (orcac_abi1_exports.txt). A session's energy is -0.25 Hartree per atom
 * and its gradient is 0.1 * (i + 1) Hartree/Bohr. When ORCAC_FAKE_ENGINE_LOG
 * names a file, orcac_create writes the config it received there as
 * key=value lines, so a test can check what ORCACPot handed the shim.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORCAC_BUILD
#include "rgpot/ORCACPot/orcac.h"

struct OrcaCSession {
  int charge;
  int multiplicity;
};

static char g_error[ORCAC_MESSAGE_SIZE] = "";

static int fail(int st, const char *msg) {
  snprintf(g_error, sizeof(g_error), "%s", msg);
  return st;
}

static const char *show(const char *s) { return s != NULL ? s : "(null)"; }

ORCAC_API int orcac_config_init(OrcaCConfig *config, size_t struct_size) {
  if (config == NULL || struct_size < ORCAC_CONFIG_MIN_SIZE) {
    return fail(ORCAC_E_INVALID, "fake orcac: config too small");
  }
  memset(config, 0, struct_size);
  config->magic = ORCAC_CONFIG_MAGIC;
  config->version = ORCAC_CONFIG_VERSION;
  config->struct_size = struct_size;
  config->multiplicity = 1;
  config->nprocs = 1;
  config->guess_policy = ORCAC_GUESS_FRESH;
  return ORCAC_OK;
}

ORCAC_API int orcac_create(const OrcaCConfig *c, OrcaCSession **out,
                           char *errbuf, size_t errbuf_size) {
  const char *log = getenv("ORCAC_FAKE_ENGINE_LOG");
  OrcaCSession *s;
  int st = ORCAC_OK;
  if (out != NULL) {
    *out = NULL;
  }
  if (c == NULL || out == NULL) {
    st = fail(ORCAC_E_INVALID, "fake orcac: NULL config or out");
  } else if (c->magic != ORCAC_CONFIG_MAGIC || c->version != ORCAC_CONFIG_VERSION ||
             c->struct_size != sizeof(OrcaCConfig)) {
    st = fail(ORCAC_E_INVALID, "fake orcac: bad config header");
  } else if (c->multiplicity < 1) {
    st = fail(ORCAC_E_INVALID, "fake orcac: multiplicity must be >= 1");
  }
  if (st != ORCAC_OK) {
    if (errbuf != NULL && errbuf_size > 0) {
      snprintf(errbuf, errbuf_size, "%s", g_error);
    }
    return st;
  }
  if (log != NULL && *log != '\0') {
    FILE *fp = fopen(log, "w");
    if (fp != NULL) {
      fprintf(fp,
              "method=%s\nbasis=%s\nkeywords=%s\nblocks=%s\nexecutable=%s\n"
              "scratch_dir=%s\ncharge=%d\nmultiplicity=%d\nnprocs=%d\n"
              "maxcore_mb=%d\ntimeout_s=%.17g\nguess_policy=%d\nflags=%u\n",
              show(c->method), show(c->basis), show(c->keywords),
              show(c->blocks), show(c->executable), show(c->scratch_dir),
              c->charge, c->multiplicity, c->nprocs, c->maxcore_mb,
              c->timeout_s, c->guess_policy, (unsigned)c->flags);
      fclose(fp);
    }
  }
  s = (OrcaCSession *)calloc(1, sizeof(*s));
  if (s == NULL) {
    return fail(ORCAC_E_ERROR, "fake orcac: out of memory");
  }
  s->charge = c->charge;
  s->multiplicity = c->multiplicity;
  *out = s;
  g_error[0] = '\0';
  return ORCAC_OK;
}

ORCAC_API int orcac_configure(OrcaCSession *session, const OrcaCConfig *c) {
  if (session == NULL || c == NULL) {
    return fail(ORCAC_E_INVALID, "fake orcac: NULL session or config");
  }
  session->charge = c->charge;
  session->multiplicity = c->multiplicity;
  return ORCAC_OK;
}

ORCAC_API int orcac_energy_gradient(OrcaCSession *session, int n_atoms,
                                    const double *positions_ang,
                                    const int *atomic_numbers, uint32_t want,
                                    double *energy_h, double *gradient_h_bohr,
                                    OrcaCResult *meta) {
  int i;
  (void)positions_ang;
  (void)atomic_numbers;
  if (meta != NULL && meta->struct_size < ORCAC_RESULT_MIN_SIZE) {
    return fail(ORCAC_E_INVALID, "fake orcac: result too small");
  }
  if (session == NULL || n_atoms <= 0) {
    return fail(ORCAC_E_INVALID, "fake orcac: bad session or atom count");
  }
  if ((want & ORCAC_WANT_STRESS) != 0) {
    return fail(ORCAC_E_UNSUPPORTED, "fake orcac: no stress");
  }
  if ((want & ORCAC_WANT_ENERGY) != 0) {
    *energy_h = -0.25 * (double)n_atoms;
  }
  if ((want & ORCAC_WANT_GRADIENT) != 0) {
    for (i = 0; i < 3 * n_atoms; ++i) {
      gradient_h_bohr[i] = 0.1 * (double)(i + 1);
    }
  }
  if (meta != NULL) {
    size_t keep = meta->struct_size;
    memset(meta, 0, keep < sizeof(*meta) ? keep : sizeof(*meta));
    meta->struct_size = keep;
    meta->status = ORCAC_OK;
    meta->orca_runs = 0;
    meta->exit_code = -1;
  }
  g_error[0] = '\0';
  return ORCAC_OK;
}

ORCAC_API int orcac_cancel(OrcaCSession *session) {
  return session != NULL ? ORCAC_OK : ORCAC_E_INVALID;
}

ORCAC_API const char *orcac_last_error(const OrcaCSession *session) {
  (void)session;
  return g_error;
}

ORCAC_API void orcac_destroy(OrcaCSession *session) { free(session); }

ORCAC_API const char *orcac_status_name(int status) {
  switch (status) {
  case ORCAC_OK:
    return "ORCAC_OK";
  case ORCAC_E_INVALID:
    return "ORCAC_E_INVALID";
  case ORCAC_E_ERROR:
    return "ORCAC_E_ERROR";
  case ORCAC_E_UNSUPPORTED:
    return "ORCAC_E_UNSUPPORTED";
  default:
    return "ORCAC_E_UNKNOWN";
  }
}

ORCAC_API const char *orcac_version(void) { return "0.1.0-fake"; }

ORCAC_API int orcac_c_abi_version(void) { return RGPOT_ORCAC_C_ABI_VERSION; }

ORCAC_API int orcac_abi_version(void) { return ORCAC_ABI_VERSION; }

ORCAC_API int orcac_available(void) { return 0; }

ORCAC_API int orcac_probe_executable(const OrcaCConfig *config, char *resolved,
                                     size_t resolved_size) {
  (void)config;
  if (resolved != NULL && resolved_size > 0) {
    resolved[0] = '\0';
  }
  return fail(ORCAC_E_NOT_FOUND, "fake orcac: runs no ORCA");
}
