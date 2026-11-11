#pragma once
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace rgpot {

// Successful growth does the same work as the vector call.
// A failure aborts the process instead of throwing, which keeps a
// bad_alloc edge off every call. The timed builder measures this.
template <class T>
inline void grow_reserve(std::vector<T> &v, std::size_t n) noexcept {
  if (v.capacity() < n)
    v.reserve(n);
}

template <class T, class U>
inline void grow_push(std::vector<T> &v, U &&x) noexcept {
  v.push_back(std::forward<U>(x));
}

template <class T>
inline void grow_resize(std::vector<T> &v, std::size_t n) noexcept {
  v.resize(n);
}

template <class T>
inline void grow_assign_n(std::vector<T> &v, std::size_t n, const T &x) noexcept {
  v.assign(n, x);
}

template <class T>
inline void grow_assign_range(std::vector<T> &v, const T *first, const T *last) noexcept {
  v.assign(first, last);
}

template <class It, class Cmp>
inline void grow_sort(It first, It last, Cmp cmp) noexcept {
  std::sort(first, last, cmp);
}

struct QuietLock {
  std::mutex &mu;
  explicit QuietLock(std::mutex &mu) noexcept : mu(mu) { mu.lock(); }
  ~QuietLock() { mu.unlock(); }
};

struct SpawnError {};

// Tests set this to the number of successful starts before a refusal.
// A negative value never refuses.
inline int &spawn_fail_budget() {
  static int budget = -1;
  return budget;
}

inline int &spawn_started() {
  static int count = 0;
  return count;
}

template <class Make>
bool spawn_workers(int threads, Make make) noexcept {
  std::vector<std::thread> pool;
  grow_reserve(pool, static_cast<std::size_t>(threads));
  try {
    for (int t = 0; t < threads; ++t) {
      if (spawn_fail_budget() == 0) {
        spawn_fail_budget() = -1;
        throw SpawnError{};
      }
      if (spawn_fail_budget() > 0)
        --spawn_fail_budget();
      pool.emplace_back(make(t));
      ++spawn_started();
    }
  } catch (...) {
    for (auto &worker : pool)
      worker.join();
    return false;
  }
  for (auto &worker : pool)
    worker.join();
  return true;
}

} // namespace rgpot
