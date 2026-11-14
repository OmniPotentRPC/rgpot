/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * RGPOT_API marks a function of the librgpot C ABI.
 *
 * Windows: the sources built into librgpot define RGPOT_BUILDING and
 * export the function from the DLL. A consumer of the DLL imports it.
 * A consumer of a static librgpot defines RGPOT_STATIC; the rgpot meson
 * dependency and rgpot.pc do that for a static build.
 *
 * Elsewhere: default visibility, so the function stays exported when a
 * target builds with -fvisibility=hidden.
 */
#ifndef RGPOT_API_H
#define RGPOT_API_H

#if defined(_WIN32)
#if defined(RGPOT_BUILDING)
#define RGPOT_API __declspec(dllexport)
#elif defined(RGPOT_STATIC)
#define RGPOT_API
#else
#define RGPOT_API __declspec(dllimport)
#endif
#else
#define RGPOT_API __attribute__((visibility("default")))
#endif

#endif
