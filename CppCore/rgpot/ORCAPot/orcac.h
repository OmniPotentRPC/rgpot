/*
 * MIT License
 * Copyright 2023--present rgpot developers
 *
 * Vendored copy of include/orcac.h from orcac, pinned by orcac.h.pin.
 * Everything below the marker line is orcac's header byte for byte;
 * CppCore/tests/test_orcac_header_pin.py checks it against the pin and,
 * when ORCAC_SOURCE_DIR names an orcac checkout, against that checkout.
 */
/* ---- orcac include/orcac.h ---- */
/**
 * @file orcac.h
 * @brief C ABI between an rgpot ORCA frontend and liborcac.so.
 *
 * Split ORCA engine, same role as mopacc and nwchemc. Each evaluation
 * writes an ORCA `EnGrad` input, runs the external `orca` executable in
 * the session's scratch directory, and parses the `.engrad` file. The
 * interface is plain C structs; there is no Cap'n Proto inside orcac,
 * and a host lowers its own configuration into OrcaCConfig.
 *
 * Units: positions are Angstrom in. Energy is Hartree (Eh) and the
 * gradient is Hartree/Bohr (Eh/bohr) out, as ORCA writes them. The
 * gradient is dE/dx, not the force. There is no stress output.
 *
 * Status: every call that can fail returns an int status from
 * OrcaCStatus and writes results through caller-owned out-parameters.
 * No call returns a struct by value. Codes 0 to 3 have the meaning of
 * rgpot's plugin status codes and can be passed through unchanged.
 *
 * Ownership: the library never keeps a caller pointer past the call
 * that received it. Strings in OrcaCConfig are copied by orcac_create
 * and orcac_configure. Out-parameters are written only on ORCAC_OK.
 *
 * Threads: a session runs one call at a time. A second call on the same
 * session while one is running returns ORCAC_E_BUSY at once instead of
 * waiting. Distinct sessions are independent. orcac_cancel may be
 * called from any thread of the creating process at any time.
 *
 * Processes: a session belongs to the process that created it. Calls
 * from a forked child return ORCAC_E_INVALID.
 *
 * Children: ORCA runs in its own process group with stdin on /dev/null
 * and no descriptor other than 0, 1 and 2, whether or not the caller
 * marked its descriptors close-on-exec. orcac waits for ORCA itself, so
 * it needs SIGCHLD left at its default and no waitpid(-1) elsewhere in
 * the process. If ORCA is reaped by someone else anyway, orcac stops
 * signalling its process group, which may by then name an unrelated
 * process, and accepts the run only when ORCA's output shows normal
 * termination and a well-formed engrad; otherwise the call is
 * ORCAC_E_ERROR. A timeout or cancel in that state cannot reach ORCA's
 * remaining helpers.
 *
 * POSIX only. The Windows branch of ORCAC_API only keeps the header
 * parseable; the library is not built there.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#define RGPOT_ORCAC_C_ABI_VERSION 1
#define ORCAC_ABI_VERSION 1 /* matches shared-library soversion 1 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
#define ORCAC_API
#elif defined(ORCAC_BUILD) && defined(__GNUC__)
#define ORCAC_API __attribute__((visibility("default")))
#else
#define ORCAC_API
#endif

/** Status codes. 0 to 3 match rgpot's plugin status codes. */
typedef enum OrcaCStatus {
  ORCAC_OK = 0,              /**< success; out-parameters are written */
  ORCAC_E_INVALID = 1,       /**< bad argument, config, or calling process */
  ORCAC_E_ERROR = 2,         /**< ORCA failed or its output was unusable */
  ORCAC_E_UNSUPPORTED = 3,   /**< valid request orcac does not provide */
  ORCAC_E_TIMEOUT = 4,       /**< ORCA ran past timeout_s and was stopped */
  ORCAC_E_NOT_CONVERGED = 5, /**< the SCF did not converge */
  ORCAC_E_BUSY = 6,          /**< another call is running on the session */
  ORCAC_E_NOT_FOUND = 7,     /**< no ORCA executable resolved */
  ORCAC_E_CANCELLED = 8      /**< stopped by orcac_cancel */
} OrcaCStatus;

/** OrcaCConfig.magic. orcac_config_init sets it. */
#define ORCAC_CONFIG_MAGIC 0x4F524343u
/** Highest OrcaCConfig.version this header describes. */
#define ORCAC_CONFIG_VERSION 1u

/** OrcaCConfig.guess_policy. */
#define ORCAC_GUESS_FRESH 0 /**< every run starts from ORCA's own guess */
#define ORCAC_GUESS_CHAIN 1 /**< start from the last orbitals via MORead */

/** OrcaCConfig.flags. Unknown bits are rejected with ORCAC_E_INVALID. */
#define ORCAC_F_NO_FRESH_RETRY (1u << 0) /**< no fresh rerun after MORead fails */
#define ORCAC_F_KEEP_MPI_ENV (1u << 1)   /**< pass OMPI_*, PMIX_*, PMI_* on */
#define ORCAC_F_ALLOW_NEAR_CONVERGED (1u << 2) /**< reserved; UNSUPPORTED */

/** want mask of orcac_energy_gradient. */
#define ORCAC_WANT_ENERGY (1u << 0)
#define ORCAC_WANT_GRADIENT (1u << 1)
#define ORCAC_WANT_STRESS (1u << 2) /**< always ORCAC_E_UNSUPPORTED */

/** Size of OrcaCResult.message, terminator included. */
#define ORCAC_MESSAGE_SIZE 1024

/** Opaque session. Create with orcac_create, free with orcac_destroy. */
typedef struct OrcaCSession OrcaCSession;

/**
 * Session options. Fill with orcac_config_init, then set fields. Strings
 * are borrowed for the duration of orcac_create or orcac_configure only.
 *
 * struct_size is the number of bytes the caller's layout holds. The
 * library reads only those bytes: any size from ORCAC_CONFIG_MIN_SIZE up
 * is accepted, and fields past struct_size take their orcac_config_init
 * defaults, so a caller built against an older, shorter header keeps
 * working. A larger struct_size from a newer header is accepted when
 * every byte past this layout is zero, and is ORCAC_E_UNSUPPORTED
 * otherwise.
 *
 * orcac_create rejects a struct whose magic is wrong, whose struct_size
 * is below ORCAC_CONFIG_MIN_SIZE, whose version is 0, or whose reserved
 * words within struct_size are not zero (ORCAC_E_INVALID). A version
 * above ORCAC_CONFIG_VERSION is ORCAC_E_UNSUPPORTED. Later versions only
 * use reserved words or append fields.
 */
typedef struct OrcaCConfig {
  uint32_t magic;          /**< ORCAC_CONFIG_MAGIC */
  uint32_t version;        /**< ORCAC_CONFIG_VERSION */
  size_t struct_size;      /**< bytes the caller allocated for this struct */
  const char *method;      /**< `!` line method; NULL is "HF" */
  const char *basis;       /**< `!` line basis; NULL is "STO-3G", "" omits */
  const char *keywords;    /**< extra `!` words; one line, NULL omits */
  const char *blocks;      /**< user % blocks, validated; NULL omits */
  const char *executable;  /**< path or PATH name; NULL or "" uses the
                                default lookup ($ORCAC_ORCA, then PATH) */
  const char *scratch_dir; /**< parent of the run directory; NULL or ""
                                uses $TMPDIR, then /tmp */
  int charge;              /**< net charge */
  int multiplicity;        /**< 2S+1, >= 1 */
  int nprocs;              /**< `%pal nprocs`, >= 1; 1 omits the block */
  int maxcore_mb;          /**< `%maxcore` MB per process; 0 omits it */
  double timeout_s;        /**< wall-clock limit per ORCA run; 0 is none */
  int guess_policy;        /**< ORCAC_GUESS_FRESH or ORCAC_GUESS_CHAIN */
  uint32_t flags;          /**< ORCAC_F_* bits */
  uint64_t reserved[8];    /**< must be zero */
} OrcaCConfig;

/** Smallest OrcaCConfig.struct_size accepted: everything before reserved. */
#define ORCAC_CONFIG_MIN_SIZE (offsetof(OrcaCConfig, flags) + sizeof(uint32_t))

/**
 * Per-call report. Set struct_size to sizeof(OrcaCResult) before the
 * call. The library never writes past struct_size: a shorter layout gets
 * the fields it holds and as much of message, still terminated, as fits;
 * a longer one keeps its extra bytes untouched. Sizes below
 * ORCAC_RESULT_MIN_SIZE are rejected with ORCAC_E_INVALID and nothing is
 * written. Written on every other status.
 */
typedef struct OrcaCResult {
  size_t struct_size; /**< set by the caller */
  int status;         /**< same value the call returned */
  int used_guess;     /**< 1 when the accepted run read MORead orbitals */
  int retried_fresh;  /**< 1 when a MORead run failed and a fresh run ran */
  int orca_runs;      /**< ORCA processes started by this call */
  int exit_code;      /**< exit status of the last run; -1 when none */
  int term_signal;    /**< signal that ended the last run; 0 when none */
  double wall_s;      /**< wall time of the call in seconds */
  char message[ORCAC_MESSAGE_SIZE]; /**< "" on success, else error start */
} OrcaCResult;

/** Smallest OrcaCResult.struct_size accepted: everything before message. */
#define ORCAC_RESULT_MIN_SIZE offsetof(OrcaCResult, message)

/**
 * Zero struct_size bytes at config, then set magic, version, struct_size
 * and the defaults that fit: multiplicity 1, nprocs 1, maxcore_mb 0,
 * timeout_s 0, ORCAC_GUESS_FRESH, no flags. Pass sizeof(OrcaCConfig);
 * sizes below ORCAC_CONFIG_MIN_SIZE are ORCAC_E_INVALID.
 */
ORCAC_API int orcac_config_init(OrcaCConfig *config, size_t struct_size);

/**
 * Validate config (NULL selects the orcac_config_init defaults), resolve
 * the ORCA executable to an absolute path with symlinks resolved, create
 * the run directory, and record the calling process. On success *out is
 * a session the caller frees with orcac_destroy. On failure *out is NULL
 * and the error text goes to errbuf (when non-NULL, truncated to
 * errbuf_size) and to orcac_last_error(NULL). An executable that does
 * not resolve is ORCAC_E_NOT_FOUND.
 */
ORCAC_API int orcac_create(const OrcaCConfig *config, OrcaCSession **out,
                           char *errbuf, size_t errbuf_size);

/**
 * Replace the options of session, validated and resolved as in
 * orcac_create. On failure the session keeps its previous options. A
 * change of charge or multiplicity, or a switch to ORCAC_GUESS_FRESH,
 * drops every stored guess; a new scratch_dir moves the session to a new
 * run directory and removes the old one. Stored guesses are keyed on
 * method, basis, keywords, blocks, the resolved executable, charge,
 * multiplicity and the atomic numbers, so a guess is only read back
 * under the settings that produced it.
 */
ORCAC_API int orcac_configure(OrcaCSession *session,
                              const OrcaCConfig *config);

/**
 * Evaluate n_atoms atoms.
 *
 * positions_ang:   3*n_atoms doubles, x y z per atom, Angstrom.
 * atomic_numbers:  n_atoms atomic numbers, 1 to 118.
 * want:            ORCAC_WANT_* bits; ORCAC_WANT_STRESS is
 *                  ORCAC_E_UNSUPPORTED.
 * energy_h:        written with the energy in Eh; required with
 *                  ORCAC_WANT_ENERGY.
 * gradient_h_bohr: 3*n_atoms doubles written with dE/dx in Eh/bohr;
 *                  required with ORCAC_WANT_GRADIENT.
 * meta:            optional report; NULL skips it.
 *
 * An SCF that does not converge is ORCAC_E_NOT_CONVERGED; orcac always
 * runs ORCA with ConvForced. With ORCAC_GUESS_CHAIN a run that read
 * MORead orbitals and failed is repeated once from a fresh guess unless
 * ORCAC_F_NO_FRESH_RETRY is set.
 */
ORCAC_API int orcac_energy_gradient(OrcaCSession *session, int n_atoms,
                                    const double *positions_ang,
                                    const int *atomic_numbers, uint32_t want,
                                    double *energy_h, double *gradient_h_bohr,
                                    OrcaCResult *meta);

/**
 * Ask the call running on session to stop. The ORCA process group gets
 * SIGTERM, then SIGKILL after 2 s, and the call returns
 * ORCAC_E_CANCELLED. A cancel counts for every call that entered
 * orcac_energy_gradient before it, including one still taking the
 * session lock; it has no effect on calls that enter later. It only
 * increments a lock-free atomic counter (the build requires
 * ATOMIC_INT_LOCK_FREE == 2), so it is safe from another thread or a
 * signal handler.
 */
ORCAC_API int orcac_cancel(OrcaCSession *session);

/**
 * Error text of the last call on session that held it, "" after a
 * success, owned by the session and valid until its next call. NULL
 * returns the text of the last call made on this thread, held in
 * thread-local storage; that also covers failed creates and calls
 * rejected as busy or from another process.
 */
ORCAC_API const char *orcac_last_error(const OrcaCSession *session);

/**
 * Remove the run directory and free session. NULL is a no-op. In a
 * forked child only the memory is freed; the run directory belongs to
 * the creating process. Must not race with another call on session.
 */
ORCAC_API void orcac_destroy(OrcaCSession *session);

/** Static name of a status code, e.g. "ORCAC_E_BUSY". */
ORCAC_API const char *orcac_status_name(int status);

/** Library version as a static string. */
ORCAC_API const char *orcac_version(void);

/** Returns RGPOT_ORCAC_C_ABI_VERSION the library was built with. */
ORCAC_API int orcac_c_abi_version(void);

/** Returns ORCAC_ABI_VERSION the library was built with. */
ORCAC_API int orcac_abi_version(void);

/**
 * 1 when the default lookup resolves an executable, else 0: $ORCAC_ORCA
 * when set and non-empty, otherwise `orca` on PATH. Use
 * orcac_probe_executable for an explicit executable.
 */
ORCAC_API int orcac_available(void);

/**
 * Resolve the executable orcac_create would use for config (NULL is the
 * default lookup) without creating a session. On ORCAC_OK the absolute
 * path is copied to resolved when it is non-NULL. ORCAC_E_NOT_FOUND
 * when nothing resolves; ORCAC_E_INVALID for a bad config header or a
 * resolved buffer that is too small.
 */
ORCAC_API int orcac_probe_executable(const OrcaCConfig *config,
                                     char *resolved, size_t resolved_size);

#ifdef __cplusplus
}
#endif
