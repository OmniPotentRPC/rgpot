#ifndef RGPOT_GUPTA_TEST_H
#define RGPOT_GUPTA_TEST_H
/*
 * Private test seam for the thread local scratch in gupta.c. Kept out of
 * the installed headers. rgpot_gupta_test_set_alloc replaces malloc for a
 * new scratch buffer. A null function pointer restores malloc. The
 * pointer it returns must be one free can release.
 */
#include <stddef.h>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_GUPTA_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_GUPTA_TEST_HIDDEN
#endif

#ifdef __cplusplus
extern "C" {
#endif

RGPOT_GUPTA_TEST_HIDDEN void rgpot_gupta_test_set_alloc(void *(*fn)(size_t));

RGPOT_GUPTA_TEST_HIDDEN size_t rgpot_gupta_test_rho_cap(void);

RGPOT_GUPTA_TEST_HIDDEN size_t rgpot_gupta_test_pack_cap(void);

#ifdef __cplusplus
}
#endif

#undef RGPOT_GUPTA_TEST_HIDDEN
#endif
