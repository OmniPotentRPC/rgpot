/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * Const accessors of the neighbour C API, called together from several
 * threads, and two builds of different geometries at the same time.
 */
#include "rgpot/nlist/rgpot_nlist.h"

#include <stdio.h>
#include <string.h>

typedef void *(*thread_fn)(void *);

#ifdef _WIN32
#include <windows.h>

typedef struct {
  HANDLE handle;
  thread_fn fn;
  void *arg;
} thread_t;

static DWORD WINAPI thread_entry(LPVOID self) {
  thread_t *t = self;
  t->fn(t->arg);
  return 0;
}

static int thread_start(thread_t *t, thread_fn fn, void *arg) {
  t->fn = fn;
  t->arg = arg;
  t->handle = CreateThread(NULL, 0, thread_entry, t, 0, NULL);
  return t->handle == NULL;
}

static void thread_join(thread_t *t) {
  WaitForSingleObject(t->handle, INFINITE);
  CloseHandle(t->handle);
}
#else
#include <pthread.h>

typedef struct {
  pthread_t handle;
} thread_t;

static int thread_start(thread_t *t, thread_fn fn, void *arg) {
  return pthread_create(&t->handle, NULL, fn, arg);
}

static void thread_join(thread_t *t) { pthread_join(t->handle, NULL); }
#endif

static void *read_accessors(void *arg) {
  int *bad = arg;
  for (int i = 0; i < 200; ++i) {
    if (rgpot_nlist_abi_version() != RGPOT_NLIST_ABI) {
      *bad = 1;
    }
    const char *text = rgpot_nlist_strerror(RGPOT_NLIST_OK);
    if (text == NULL || text[0] == '\0') {
      *bad = 1;
    }
    rgpot_nlist_request request;
    rgpot_nlist_request_init(&request);
    if (request.abi != (uint32_t)RGPOT_NLIST_ABI || request.cutoff != 0.0) {
      *bad = 1;
    }
  }
  return NULL;
}

typedef struct {
  const double *xyz;
  size_t n;
  const double *box;
  double cutoff;
  size_t expect;
  int bad;
} Build;

static void *build_one(void *arg) {
  Build *job = arg;
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = job->cutoff;
  size_t count = 0;
  size_t needed = 0;
  const int code = rgpot_nlist_build(job->xyz, job->n, job->box, &request, NULL, &count, &needed);
  if (code != RGPOT_NLIST_OK || needed != job->expect) {
    job->bad = 1;
  }
  return NULL;
}

int main(void) {
  int bad[4] = {0, 0, 0, 0};
  thread_t threads[4];
  for (int i = 0; i < 4; ++i) {
    if (thread_start(&threads[i], read_accessors, &bad[i]) != 0) {
      fprintf(stderr, "thread start failed\n");
      return 1;
    }
  }
  for (int i = 0; i < 4; ++i) {
    thread_join(&threads[i]);
    if (bad[i]) {
      fprintf(stderr, "const accessor changed under threads\n");
      return 1;
    }
  }
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pair_a[6] = {0, 0, 0, 1, 0, 0};
  const double pair_b[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  Build jobs[2] = {
      {pair_a, 2, box, 2.0, 1, 0},
      {pair_b, 3, box, 2.0, 3, 0},
  };
  thread_t builders[2];
  for (int i = 0; i < 2; ++i) {
    if (thread_start(&builders[i], build_one, &jobs[i]) != 0) {
      return 1;
    }
  }
  for (int i = 0; i < 2; ++i) {
    thread_join(&builders[i]);
    if (jobs[i].bad) {
      fprintf(stderr, "concurrent build disagreed with the serial count\n");
      return 1;
    }
  }
  return 0;
}
