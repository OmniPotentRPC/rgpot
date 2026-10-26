#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file eindir_abi.hpp
 * @brief Complete @c eindir_objective_t layout so @c rgpot.h compiles.
 *
 * cbindgen emits @c eindir_objective_t as an unresolved name because
 * @c parse_deps is false. The member list matches eindir-core 0.5
 * @c #[repr(C)] @c eindir_objective_t (dim, bounds, eval/grad, user
 * data, free), which eindir-core 0.6 keeps unchanged, plus the
 * @c eindir_abi_stamp_t that eindir-core 0.6 added for
 * @c rgpot_eindir_abi_stamp. Skip this stub when the real eindir header is
 * present.
 */

#ifndef RGPOT_EINDIR_ABI_HPP
#define RGPOT_EINDIR_ABI_HPP

#if !defined(EINDIR_CORE_H) && !defined(EINDIR_OBJECTIVE_T_DEFINED)

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eindir_objective_t {
  size_t dim;
  double *low;
  double *high;
  void *eval_fn;
  void *grad_fn;
  void *user_data;
  void *free_fn;
} eindir_objective_t;

#define EINDIR_OBJECTIVE_T_DEFINED 1

#ifdef __cplusplus
}
#endif

#endif

#if !defined(EINDIR_CORE_H) && !defined(EINDIR_ABI_STAMP_T_DEFINED)

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eindir_abi_stamp_t {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t objective_layout;
  size_t objective_size;
  size_t objective_align;
  uint32_t dlpack_major;
  uint32_t dlpack_minor;
  uint64_t features;
} eindir_abi_stamp_t;

#define EINDIR_ABI_STAMP_T_DEFINED 1

#ifdef __cplusplus
}
#endif

#endif

#endif
