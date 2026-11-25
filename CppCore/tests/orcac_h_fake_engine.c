/*
 * MIT License
 * Copyright 2023--present rgpot developers
 *
 * A fake liborcac for the ORCAPot tests, after mopacc_fake_engine.c. It
 * speaks the vendored orcac.h (ABI 1) and runs no ORCA: the energy is
 * -1 + 0.01 charge + 0.001 (multiplicity - 1) Eh and gradient component k
 * is 0.001 (k + 1) Eh/bohr. Method "FAIL" ends every evaluation with
 * ORCAC_E_ERROR, "NOCONV" with ORCAC_E_NOT_CONVERGED, and "SLOW" waits up
 * to five seconds for orcac_cancel. Multiplicities above 9 are refused.
 * An explicit executable must exist; the default lookup finds $ORCAC_ORCA
 * or nothing, and orcac_available() is always 0.
 *
 * Build variants for the load-time handshake:
 *   ORCAC_FAKE_C_ABI=N        orcac_c_abi_version() returns N
 *   ORCAC_FAKE_ABI=N          orcac_abi_version() returns N
 *   ORCAC_FAKE_SMALL_CONFIG   orcac_config_init() reports a struct_size
 *                             below ORCAC_CONFIG_MIN_SIZE
 *   ORCAC_FAKE_SMALL_RESULT   results report a struct_size below
 *                             ORCAC_RESULT_MIN_SIZE
 *   ORCAC_FAKE_NO_COMPUTE     orcac_energy_gradient is not exported
 */
#define _XOPEN_SOURCE 700
#define ORCAC_BUILD
#include "orcac.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef ORCAC_FAKE_C_ABI
#define ORCAC_FAKE_C_ABI RGPOT_ORCAC_C_ABI_VERSION
#endif
#ifndef ORCAC_FAKE_ABI
#define ORCAC_FAKE_ABI ORCAC_ABI_VERSION
#endif

#if defined(_WIN32)
#define ORCAC_FAKE_EXPORT __declspec(dllexport)
#else
#define ORCAC_FAKE_EXPORT __attribute__((visibility("default")))
#endif

#define FAKE_TEXT 256

/* What the library saw; read by the tests through orcac_fake_state. */
typedef struct orcac_fake_state_t {
  int sessions_created;
  int sessions_live;
  int evaluations; /* orcac_energy_gradient calls with a session */
  int null_probes; /* orcac_energy_gradient calls without one */
  int in_call;     /* 1 while a "SLOW" evaluation waits */
  int guess_policy;
  unsigned flags;
  int charge;
  int multiplicity;
  int nprocs;
  int maxcore_mb;
  double timeout_s;
  int used_guess; /* the last evaluation read a chained guess */
  char method[FAKE_TEXT];
  char basis[FAKE_TEXT];
  char blocks[FAKE_TEXT];
  char executable[FAKE_TEXT];
} orcac_fake_state_t;

static orcac_fake_state_t g_state;
static _Thread_local char g_thread_error[ORCAC_MESSAGE_SIZE];

struct OrcaCSession {
  char method[FAKE_TEXT];
  int charge;
  int multiplicity;
  int guess_policy;
  int has_guess;
  atomic_uint cancel_gen;
  char error[ORCAC_MESSAGE_SIZE];
};

ORCAC_FAKE_EXPORT orcac_fake_state_t *orcac_fake_state(void) {
  return &g_state;
}

static void copy_text(char *dst, const char *src) {
  snprintf(dst, FAKE_TEXT, "%s", src ? src : "");
}

ORCAC_API int orcac_c_abi_version(void) { return ORCAC_FAKE_C_ABI; }

ORCAC_API int orcac_abi_version(void) { return ORCAC_FAKE_ABI; }

ORCAC_API const char *orcac_version(void) { return "orcac-fake/1.0"; }

ORCAC_API const char *orcac_status_name(int status) {
  static const char *names[] = {
      "ORCAC_OK",          "ORCAC_E_INVALID",       "ORCAC_E_ERROR",
      "ORCAC_E_UNSUPPORTED", "ORCAC_E_TIMEOUT",     "ORCAC_E_NOT_CONVERGED",
      "ORCAC_E_BUSY",      "ORCAC_E_NOT_FOUND",     "ORCAC_E_CANCELLED"};
  return status >= 0 && status <= 8 ? names[status] : "unknown orcac status";
}

ORCAC_API int orcac_config_init(OrcaCConfig *config, size_t struct_size) {
  if (config == NULL || struct_size < ORCAC_CONFIG_MIN_SIZE) {
    return ORCAC_E_INVALID;
  }
  memset(config, 0, struct_size);
  config->magic = ORCAC_CONFIG_MAGIC;
  config->version = ORCAC_CONFIG_VERSION;
  config->struct_size = struct_size;
#ifdef ORCAC_FAKE_SMALL_CONFIG
  config->struct_size = ORCAC_CONFIG_MIN_SIZE - 8u;
#endif
  config->multiplicity = 1;
  config->nprocs = 1;
  config->guess_policy = ORCAC_GUESS_FRESH;
  return ORCAC_OK;
}

static int check_config(const OrcaCConfig *c, char *why, size_t why_size) {
  if (c == NULL || c->magic != ORCAC_CONFIG_MAGIC ||
      c->struct_size < ORCAC_CONFIG_MIN_SIZE || c->version == 0u) {
    snprintf(why, why_size, "bad OrcaCConfig header");
    return ORCAC_E_INVALID;
  }
  if (c->version > ORCAC_CONFIG_VERSION) {
    snprintf(why, why_size, "OrcaCConfig version %u", c->version);
    return ORCAC_E_UNSUPPORTED;
  }
  if (c->flags & ~(ORCAC_F_NO_FRESH_RETRY | ORCAC_F_KEEP_MPI_ENV)) {
    snprintf(why, why_size, "unknown flags 0x%x", c->flags);
    return ORCAC_E_INVALID;
  }
  if (c->multiplicity < 1 || c->multiplicity > 9) {
    snprintf(why, why_size, "fake supports multiplicity 1 to 9, got %d",
             c->multiplicity);
    return ORCAC_E_INVALID;
  }
  return ORCAC_OK;
}

static int resolve(const OrcaCConfig *c, char *resolved, size_t size,
                   char *why, size_t why_size) {
  const char *path = c ? c->executable : NULL;
  if (path == NULL || path[0] == '\0') {
    path = getenv("ORCAC_ORCA");
  }
  char real[PATH_MAX];
  if (path == NULL || path[0] == '\0' || realpath(path, real) == NULL) {
    snprintf(why, why_size, "no orca executable at %s",
             path ? path : "(default lookup)");
    return ORCAC_E_NOT_FOUND;
  }
  if (resolved != NULL &&
      (size_t)snprintf(resolved, size, "%s", real) >= size) {
    snprintf(why, why_size, "resolved buffer too small");
    return ORCAC_E_INVALID;
  }
  return ORCAC_OK;
}

ORCAC_API int orcac_probe_executable(const OrcaCConfig *config,
                                     char *resolved, size_t resolved_size) {
  if (config != NULL) {
    const int st =
        check_config(config, g_thread_error, sizeof(g_thread_error));
    if (st != ORCAC_OK) {
      return st;
    }
  }
  return resolve(config, resolved, resolved_size, g_thread_error,
                 sizeof(g_thread_error));
}

ORCAC_API int orcac_available(void) { return 0; }

static void record(const OrcaCConfig *c) {
  g_state.guess_policy = c->guess_policy;
  g_state.flags = c->flags;
  g_state.charge = c->charge;
  g_state.multiplicity = c->multiplicity;
  g_state.nprocs = c->nprocs;
  g_state.maxcore_mb = c->maxcore_mb;
  g_state.timeout_s = c->timeout_s;
  copy_text(g_state.method, c->method);
  copy_text(g_state.basis, c->basis);
  copy_text(g_state.blocks, c->blocks);
  copy_text(g_state.executable, c->executable);
}

ORCAC_API int orcac_create(const OrcaCConfig *config, OrcaCSession **out,
                           char *errbuf, size_t errbuf_size) {
  if (out == NULL || config == NULL) {
    return ORCAC_E_INVALID;
  }
  *out = NULL;
  int st = check_config(config, g_thread_error, sizeof(g_thread_error));
  if (st == ORCAC_OK) {
    st = resolve(config, NULL, 0, g_thread_error, sizeof(g_thread_error));
  }
  if (st != ORCAC_OK) {
    if (errbuf != NULL && errbuf_size > 0) {
      snprintf(errbuf, errbuf_size, "%s", g_thread_error);
    }
    return st;
  }
  OrcaCSession *s = calloc(1, sizeof(*s));
  if (s == NULL) {
    return ORCAC_E_ERROR;
  }
  copy_text(s->method, config->method);
  s->charge = config->charge;
  s->multiplicity = config->multiplicity;
  s->guess_policy = config->guess_policy;
  atomic_init(&s->cancel_gen, 0u);
  g_state.sessions_created += 1;
  g_state.sessions_live += 1;
  record(config);
  *out = s;
  return ORCAC_OK;
}

ORCAC_API int orcac_configure(OrcaCSession *session,
                              const OrcaCConfig *config) {
  if (session == NULL) {
    return ORCAC_E_INVALID;
  }
  const int st = check_config(config, session->error, sizeof(session->error));
  if (st != ORCAC_OK) {
    return st;
  }
  if (config->charge != session->charge ||
      config->multiplicity != session->multiplicity ||
      config->guess_policy == ORCAC_GUESS_FRESH) {
    session->has_guess = 0;
  }
  copy_text(session->method, config->method);
  session->charge = config->charge;
  session->multiplicity = config->multiplicity;
  session->guess_policy = config->guess_policy;
  session->error[0] = '\0';
  record(config);
  return ORCAC_OK;
}

#ifndef ORCAC_FAKE_NO_COMPUTE
static int finish(OrcaCResult *meta, int st, const char *message) {
  if (meta == NULL) {
    return st;
  }
  const size_t size = meta->struct_size;
  OrcaCResult rep;
  memset(&rep, 0, sizeof(rep));
  rep.struct_size = size;
#ifdef ORCAC_FAKE_SMALL_RESULT
  rep.struct_size = ORCAC_RESULT_MIN_SIZE - 8u;
#endif
  rep.status = st;
  rep.exit_code = -1;
  snprintf(rep.message, sizeof(rep.message), "%s", message ? message : "");
  memcpy(meta, &rep, size < sizeof(rep) ? size : sizeof(rep));
  return st;
}

static void nap_ms(long ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

ORCAC_API int orcac_energy_gradient(OrcaCSession *session, int n_atoms,
                                    const double *positions_ang,
                                    const int *atomic_numbers, uint32_t want,
                                    double *energy_h, double *gradient_h_bohr,
                                    OrcaCResult *meta) {
  if (meta != NULL && meta->struct_size < ORCAC_RESULT_MIN_SIZE) {
    return ORCAC_E_INVALID;
  }
  if (session == NULL) {
    g_state.null_probes += 1;
    snprintf(g_thread_error, sizeof(g_thread_error),
             "orcac: session is NULL");
    return finish(meta, ORCAC_E_INVALID, g_thread_error);
  }
  const unsigned seen = atomic_load(&session->cancel_gen);
  g_state.evaluations += 1;
  if (want & ORCAC_WANT_STRESS) {
    snprintf(session->error, sizeof(session->error), "no stress from ORCA");
    return finish(meta, ORCAC_E_UNSUPPORTED, session->error);
  }
  if (n_atoms <= 0 || positions_ang == NULL || atomic_numbers == NULL ||
      ((want & ORCAC_WANT_ENERGY) && energy_h == NULL) ||
      ((want & ORCAC_WANT_GRADIENT) && gradient_h_bohr == NULL)) {
    snprintf(session->error, sizeof(session->error), "bad arguments");
    return finish(meta, ORCAC_E_INVALID, session->error);
  }
  if (strcmp(session->method, "FAIL") == 0) {
    snprintf(session->error, sizeof(session->error),
             "fake ORCA run failed for method FAIL");
    return finish(meta, ORCAC_E_ERROR, session->error);
  }
  if (strcmp(session->method, "NOCONV") == 0) {
    snprintf(session->error, sizeof(session->error),
             "fake SCF did not converge");
    return finish(meta, ORCAC_E_NOT_CONVERGED, session->error);
  }
  if (strcmp(session->method, "SLOW") == 0) {
    g_state.in_call = 1;
    for (int i = 0; i < 5000; ++i) {
      if (atomic_load(&session->cancel_gen) != seen) {
        g_state.in_call = 0;
        snprintf(session->error, sizeof(session->error),
                 "fake ORCA run cancelled");
        return finish(meta, ORCAC_E_CANCELLED, session->error);
      }
      nap_ms(1);
    }
    g_state.in_call = 0;
  }
  g_state.used_guess =
      session->guess_policy == ORCAC_GUESS_CHAIN && session->has_guess;
  session->has_guess = 1;
  if (want & ORCAC_WANT_GRADIENT) {
    for (int i = 0; i < 3 * n_atoms; ++i) {
      gradient_h_bohr[i] = 0.001 * (double)(i + 1);
    }
  }
  if (want & ORCAC_WANT_ENERGY) {
    *energy_h = -1.0 + 0.01 * (double)session->charge +
                0.001 * (double)(session->multiplicity - 1);
  }
  session->error[0] = '\0';
  return finish(meta, ORCAC_OK, "");
}
#endif

ORCAC_API int orcac_cancel(OrcaCSession *session) {
  if (session == NULL) {
    return ORCAC_E_INVALID;
  }
  atomic_fetch_add(&session->cancel_gen, 1u);
  return ORCAC_OK;
}

ORCAC_API const char *orcac_last_error(const OrcaCSession *session) {
  return session ? session->error : g_thread_error;
}

ORCAC_API void orcac_destroy(OrcaCSession *session) {
  if (session != NULL) {
    g_state.sessions_live -= 1;
    free(session);
  }
}
