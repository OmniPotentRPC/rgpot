#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Every OpenMP region in rgpot opens here. A caller cuts its work into
// slices before the region and keeps one buffer per slice. The runtime may
// start fewer threads than slices: under OMP_THREAD_LIMIT, inside another
// region, with dynamic teams, or below a TeamCap. A thread then runs several
// slices in turn. No other file reads a thread number or a team size, so a
// buffer is indexed by its slice, and a sum taken in slice order has the same
// bits on any team. CppCore/tests/test_openmp_confined.py fails the test
// suite when OpenMP appears outside Slices.hpp and Slices.cc.
//
// A call over atoms takes its slice count from teamSize, so a small system
// opens no region at all. A batch of such systems is the other way around:
// forEachBatchSystem spreads the systems, and each system stays whole so a
// larger one can still open its own team.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__SANITIZE_THREAD__)
#define RGPOT_TSAN_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RGPOT_TSAN_ACTIVE 1
#endif
#endif

#if defined(RGPOT_TSAN_ACTIVE)
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#endif

#if defined(RGPOT_TSAN_ACTIVE) && defined(_OPENMP) && defined(__GNUC__) &&     \
    !defined(__clang__)
#define RGPOT_GOMP_TSAN 1
extern "C" void GOMP_parallel(void (*)(void *), void *, unsigned, unsigned);
#endif

namespace rgpot {

/// Items `[begin, end)`.
struct Range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

/// Contiguous run of `n` items for `slice` in `[0, nslices)`. The runs
/// cover `n` without overlap, in slice order. Any other slice is empty.
[[nodiscard]] inline Range partition(std::int64_t n, int slice,
                                     int nslices) noexcept {
  if (nslices < 1) {
    nslices = 1;
  }
  if (slice < 0 || slice >= nslices || n <= 0) {
    return {};
  }
  const auto ns = static_cast<std::int64_t>(nslices);
  const auto s = static_cast<std::int64_t>(slice);
  return {(n * s) / ns, (n * (s + 1)) / ns};
}

/// Slice count from OMP_NUM_THREADS, read on every call. Unset, empty, or
/// a non-positive value is one slice, so a call that does not ask for
/// threads matches the serial order. One without OpenMP.
[[nodiscard]] int threadsRequested();

/// Atoms below which a call without a threshold of its own, such as a
/// neighbour-list build, runs on the calling thread alone. A region costs
/// more than such a call: on 8 threads a neighbour search of 64 atoms took
/// several times its one-thread time. Each potential keeps its own
/// thresholds as `kTeamFrom` in its class.
#if !defined(RGPOT_TEAM_MIN_ATOMS)
#define RGPOT_TEAM_MIN_ATOMS 8192
#endif
static_assert(RGPOT_TEAM_MIN_ATOMS > 0);
inline constexpr std::size_t kTeamMinAtoms = RGPOT_TEAM_MIN_ATOMS;

/// An atom count no call reaches, for a team that never opens.
inline constexpr std::size_t kNoTeam = std::numeric_limits<std::size_t>::max();

/// Atoms from which a team of two threads opens, of three, and of four or
/// more. The default opens each at the threshold of the call. A potential
/// keeps its own as `kTeamFrom` and, on any thread count, sums atom by atom
/// in index order from the least of them, its `kTeamMinAtoms`.
struct TeamFrom {
  std::size_t two = 0;
  std::size_t three = 0;
  std::size_t more = 0;

  /// Atoms from which a team of `threads` opens.
  [[nodiscard]] constexpr std::size_t of(int threads) const noexcept {
    return threads <= 2 ? two : (threads == 3 ? three : more);
  }

  /// Atoms from which some team opens.
  [[nodiscard]] constexpr std::size_t least() const noexcept {
    const std::size_t fewer = two < three ? two : three;
    return fewer < more ? fewer : more;
  }
};

/// The threshold of a call whose own is `own`: that of the innermost
/// TeamThreshold alive on this thread, else `own`.
[[nodiscard]] std::size_t teamThreshold(std::size_t own) noexcept;

/// Whether a TeamThreshold is alive on this thread.
[[nodiscard]] bool teamThresholdActive() noexcept;

/// Slices for a call over `n` atoms with threshold `from` and teams
/// `team`: threadsRequested(), at most `n`, once `n` reaches `from` and the
/// atoms `team` asks of that many threads, else one. While a TeamThreshold
/// is alive any count above one opens from `from`.
[[nodiscard]] int teamSize(std::size_t n, std::size_t from, TeamFrom team = {});

/// Same as `teamSize(n, kTeamMinAtoms, {})`. Kept exported and deprecated
/// so binaries built against the one-argument signature keep loading
/// under `kCxxSoversion` 9.
[[deprecated("Use teamSize(n, from, team) with explicit threshold and TeamFrom")]]
[[nodiscard]] int teamSize(std::size_t n);

/// Regions the calling thread has opened through the helpers below, a
/// region the runtime gave one thread included. A call that leaves the
/// count unchanged ran on the calling thread alone.
[[nodiscard]] std::uint64_t teamsOpened() noexcept;

/// The team a region without a thread count would ask for, from
/// omp_set_num_threads or the first OMP_NUM_THREADS entry. One without
/// OpenMP. The runtime may still start fewer.
[[nodiscard]] int poolThreads() noexcept;

/// Sets the team `poolThreads` reports and fixes it against dynamic
/// adjustment. Shrinking it first lets the idle workers go.
void resizePool(int threads) noexcept;

/// While alive, regions this thread opens here start at most `threads`
/// threads. Zero or less lifts the cap. Slices and results do not change.
class TeamCap {
public:
  explicit TeamCap(int threads) noexcept;
  ~TeamCap();
  TeamCap(const TeamCap &) = delete;
  TeamCap &operator=(const TeamCap &) = delete;

private:
  int previous_;
};

/// While alive, a potential called on this thread takes `atoms` as its
/// threshold in place of its own and opens a team on any thread count
/// above one, so a benchmark can time every team on either side of its
/// own threshold. The bits still do not depend on the thread count, but a
/// call between the two thresholds sums in the other order and its energy
/// can differ in the last bits.
class TeamThreshold {
public:
  explicit TeamThreshold(std::size_t atoms) noexcept;
  ~TeamThreshold();
  TeamThreshold(const TeamThreshold &) = delete;
  TeamThreshold &operator=(const TeamThreshold &) = delete;

private:
  std::size_t previous_;
  bool had_previous_;
};

namespace detail {
/// Threads to request for `wanted` slices or ranks under this thread's cap.
[[nodiscard]] int teamFor(int wanted) noexcept;
/// Counts a region for teamsOpened.
void countTeam() noexcept;

#if defined(RGPOT_GOMP_TSAN)
/// libgomp's uninstrumented fork does not give TSan a happens-before edge.
/// Passing our own final state to GOMP_parallel lets the caller release after
/// initialization and every worker acquire that exact address before reading.
template <typename Function>
void gompParallel(int threads, Function &&function) {
  using Fn = std::remove_reference_t<Function>;
  struct State {
    Fn *function;
    std::unique_ptr<std::atomic<int>[]> joined;
    std::atomic<int> fork{0};
    std::atomic<int> actual{0};
  };
  State state{
      std::addressof(function),
      std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(threads)),
  };
  state.fork.store(1, std::memory_order_release);
  GOMP_parallel(
      [](void *opaque) {
        auto &shared = *static_cast<State *>(opaque);
        (void)shared.fork.load(std::memory_order_acquire);
        const int thread = omp_get_thread_num();
        if (thread == 0) {
          shared.actual.store(omp_get_num_threads(), std::memory_order_release);
        }
        (*shared.function)();
        shared.joined[static_cast<std::size_t>(thread)].store(
            1, std::memory_order_release);
      },
      &state, static_cast<unsigned>(threads), 0);
  const int actual = state.actual.load(std::memory_order_acquire);
  for (int thread = 0; thread < actual; ++thread) {
    (void)state.joined[static_cast<std::size_t>(thread)].load(
        std::memory_order_acquire);
  }
}
#endif
} // namespace detail

/// Runs `body(slice)` once for each slice in `[0, nslices)`, possibly
/// several on one thread, so slices must not wait on each other.
///
/// MSVC rejects `#pragma omp` inside a lambda (C2760, token PRAGMA_OMP).
/// The regions live here; the lambdas only supply the bodies.
template <typename Body> void forEachSlice(int nslices, Body &&body) {
#if defined(_OPENMP)
  const int request = nslices > 1 ? detail::teamFor(nslices) : 1;
  if (request > 1) {
    detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
    detail::gompParallel(request, [&] {
      const int team = omp_get_num_threads();
      for (int slice = omp_get_thread_num(); slice < nslices; slice += team) {
        body(slice);
      }
    });
#else
#if defined(RGPOT_TSAN_ACTIVE)
    // Distro OpenMP runtimes (libomp, libgomp) are built without TSan, so
    // their fork/join barriers are invisible to TSan without an explicit
    // release/acquire pair. Concurrent slices still race if they share writes.
    static std::atomic<std::uint64_t> tsan_fork_epoch{0};
    auto tsan_join =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(request));
#endif
#pragma omp parallel num_threads(request)
    {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
      tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
      (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
      const int team = omp_get_num_threads();
      for (int slice = omp_get_thread_num(); slice < nslices; slice += team) {
        body(slice);
      }
#if defined(RGPOT_TSAN_ACTIVE)
      tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
          1, std::memory_order_release);
#endif
    }
#if defined(RGPOT_TSAN_ACTIVE)
    for (int thread = 0; thread < request; ++thread) {
      (void)tsan_join[static_cast<std::size_t>(thread)].load(
          std::memory_order_acquire);
    }
#endif
#endif
    return;
  }
#endif
  for (int slice = 0; slice < nslices; ++slice) {
    body(slice);
  }
}

/// Runs `body(phase, slice)` for each phase in `[0, nphases)` and each
/// slice in `[0, nslices)`, in one region. Every slice of a phase ends
/// before any slice of the next phase starts. Within a phase the slices
/// follow the rule of `forEachSlice`.
template <typename Body>
void forEachSlicePhase(int nphases, int nslices, Body &&body) {
#if defined(_OPENMP)
  const int request = nslices > 1 && nphases > 0 ? detail::teamFor(nslices) : 1;
  if (request > 1) {
    detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
    auto tsan_phase =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nphases));
    detail::gompParallel(request, [&] {
      const int team = omp_get_num_threads();
      const int first = omp_get_thread_num();
      for (int phase = 0; phase < nphases; ++phase) {
        if (phase > 0) {
          tsan_phase[static_cast<std::size_t>(phase)].fetch_add(
              1, std::memory_order_release);
#pragma omp barrier
          (void)tsan_phase[static_cast<std::size_t>(phase)].load(
              std::memory_order_acquire);
        }
        for (int slice = first; slice < nslices; slice += team) {
          body(phase, slice);
        }
      }
    });
#else
#if defined(RGPOT_TSAN_ACTIVE)
    static std::atomic<std::uint64_t> tsan_fork_epoch{0};
    auto tsan_join =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(request));
    auto tsan_phase =
        std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nphases));
#endif
#pragma omp parallel num_threads(request)
    {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
      tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
      (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
      const int team = omp_get_num_threads();
      const int first = omp_get_thread_num();
      for (int phase = 0; phase < nphases; ++phase) {
        if (phase > 0) {
#if defined(RGPOT_TSAN_ACTIVE)
          tsan_phase[static_cast<std::size_t>(phase)].fetch_add(
              1, std::memory_order_release);
#endif
#pragma omp barrier
#if defined(RGPOT_TSAN_ACTIVE)
          (void)tsan_phase[static_cast<std::size_t>(phase)].load(
              std::memory_order_acquire);
#endif
        }
        for (int slice = first; slice < nslices; slice += team) {
          body(phase, slice);
        }
      }
#if defined(RGPOT_TSAN_ACTIVE)
      tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
          1, std::memory_order_release);
#endif
    }
#if defined(RGPOT_TSAN_ACTIVE)
    for (int thread = 0; thread < request; ++thread) {
      (void)tsan_join[static_cast<std::size_t>(thread)].load(
          std::memory_order_acquire);
    }
#endif
#endif
    return;
  }
#endif
  for (int phase = 0; phase < nphases; ++phase) {
    for (int slice = 0; slice < nslices; ++slice) {
      body(phase, slice);
    }
  }
}

/// Runs `body(rank)` for each rank in `[0, nranks)`, all at once on their
/// own threads, so ranks may wait on each other. Returns false and runs
/// nothing when the runtime starts a shorter team, or without OpenMP; the
/// caller then does the same work through `forEachSlice`. One rank runs
/// on the calling thread.
template <typename Body>
[[nodiscard]] bool onFullTeam(int nranks, Body &&body) {
  if (nranks <= 1) {
    if (nranks == 1) {
      body(0);
    }
    return true;
  }
#if defined(_OPENMP)
  if (detail::teamFor(nranks) < nranks) {
    return false;
  }
  bool full = false;
  detail::countTeam();
#if defined(RGPOT_GOMP_TSAN)
  detail::gompParallel(nranks, [&] {
    if (omp_get_num_threads() == nranks) {
      const int rank = omp_get_thread_num();
      if (rank == 0) {
        full = true;
      }
      body(rank);
    }
  });
#else
#if defined(RGPOT_TSAN_ACTIVE)
  static std::atomic<std::uint64_t> tsan_fork_epoch{0};
  auto tsan_join =
      std::make_unique<std::atomic<int>[]>(static_cast<std::size_t>(nranks));
#endif
#pragma omp parallel num_threads(nranks)
  {
#if defined(RGPOT_TSAN_ACTIVE)
#pragma omp master
    tsan_fork_epoch.fetch_add(1, std::memory_order_acq_rel);
#pragma omp barrier
    (void)tsan_fork_epoch.load(std::memory_order_acquire);
#endif
    if (omp_get_num_threads() == nranks) {
      const int rank = omp_get_thread_num();
      if (rank == 0) {
        full = true;
      }
      body(rank);
    }
#if defined(RGPOT_TSAN_ACTIVE)
    tsan_join[static_cast<std::size_t>(omp_get_thread_num())].store(
        1, std::memory_order_release);
#endif
  }
#if defined(RGPOT_TSAN_ACTIVE)
  for (int thread = 0; thread < nranks; ++thread) {
    (void)tsan_join[static_cast<std::size_t>(thread)].load(
        std::memory_order_acquire);
  }
#endif
#endif
  return full;
#else
  (void)body;
  return false;
#endif
}

/// Evaluates `n` independent systems on the thread team.
///
/// `atomsOf(i)` is the atom count of system `i`. `body(i)` evaluates
/// that system. `from` and `team` are the threshold and the teams a
/// single evaluation passes to `teamSize`, and they default to the
/// neighbour-list threshold. When `teamSize` would leave every system
/// on one thread, more than one thread is requested, and this call is
/// not already inside a team, the systems are spread across
/// `threadsRequested()` slices. A system that would open its own team,
/// or a batch that contains one, keeps the serial order, so that team
/// is not nested and a spread worker does not open one. One system, or
/// one requested thread, does the same.
///
/// The split is the one `BatchEvaluator` uses for configurations, run
/// here on the existing team rather than a second pool.
/// `threadsRequested()` honours `OMP_NUM_THREADS`. Each slice may run on
/// its own thread. `body` keeps scratch off any state another slice
/// writes. A system is evaluated whole, so its bits match the serial
/// loop. A slice that throws is caught, and the exception from the
/// lowest numbered slice is rethrown after the team joins.
template <typename AtomsOf, typename Body>
void forEachBatchSystem(std::size_t n, AtomsOf &&atomsOf, Body &&body,
                        std::size_t from = kTeamMinAtoms, TeamFrom team = {}) {
  if (n == 0) {
    return;
  }
  bool spread = n > 1 && threadsRequested() > 1;
#if defined(_OPENMP)
  if (omp_in_parallel()) {
    spread = false;
  }
#endif
  if (spread) {
    // The same rule as one evaluation of this potential. A system that
    // would open a team stays whole, and a system that is spread then
    // sees teamSize of one, so the worker does not open an inner team.
    const std::size_t floor = teamThreshold(from);
    for (std::size_t i = 0; i < n; ++i) {
      if (teamSize(atomsOf(i), floor, team) > 1) {
        spread = false;
        break;
      }
    }
  }
  if (!spread) {
    for (std::size_t i = 0; i < n; ++i) {
      body(i);
    }
    return;
  }
  int slices = threadsRequested();
  if (n < static_cast<std::size_t>(slices)) {
    slices = static_cast<int>(n);
  }
  const bool threshold_active = teamThresholdActive();
  const std::size_t threshold = teamThreshold(from);
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(slices));
  forEachSlice(slices, [&](int slice) {
    auto run = [&] {
      try {
        const Range range =
            partition(static_cast<std::int64_t>(n), slice, slices);
        for (std::int64_t i = range.begin; i < range.end; ++i) {
          body(static_cast<std::size_t>(i));
        }
      } catch (...) {
        errors[static_cast<std::size_t>(slice)] = std::current_exception();
      }
    };
    if (threshold_active) {
      const TeamThreshold worker_threshold(threshold);
      run();
    } else {
      run();
    }
  });
  for (const std::exception_ptr &error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

} // namespace rgpot
