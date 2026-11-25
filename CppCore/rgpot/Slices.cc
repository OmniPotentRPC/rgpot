// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Slices.hpp"

#include <cstdlib>
#include <stdexcept>

namespace rgpot {
namespace {

thread_local int t_team_cap = 0;
thread_local std::uint64_t t_teams_opened = 0;
thread_local bool t_threshold_set = false;
thread_local std::size_t t_threshold = 0;

// kind 1 is a call-scope ThreadCount, kind 2 a potential ThreadBinding.
struct ExplicitFrame {
  int count = 0;
  unsigned char kind = 0;
};

constexpr int kExplicitDepth = 32;
thread_local ExplicitFrame t_explicit[kExplicitDepth];
thread_local int t_explicit_depth = 0;
thread_local int t_sticky = 0;

int resolvedExplicit() noexcept {
  if (t_explicit_depth > 0) {
    return t_explicit[t_explicit_depth - 1].count;
  }
  return t_sticky;
}

bool callScopeActive() noexcept {
  for (int i = 0; i < t_explicit_depth; ++i) {
    if (t_explicit[i].kind == 1) {
      return true;
    }
  }
  return false;
}

void pushExplicit(int count, unsigned char kind) {
  if (t_explicit_depth >= kExplicitDepth) {
    throw std::logic_error("rgpot thread-count nest is deeper than 32");
  }
  t_explicit[t_explicit_depth].count = count;
  t_explicit[t_explicit_depth].kind = kind;
  ++t_explicit_depth;
}

} // namespace

int threadsRequested() {
#if defined(_OPENMP)
  if (const int explicit_count = resolvedExplicit(); explicit_count > 0) {
    return explicit_count;
  }
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }
  char *end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 1) {
    return 1;
  }
  if (parsed > 0x7fffffffL) {
    return 0x7fffffff;
  }
  return static_cast<int>(parsed);
#else
  return 1;
#endif
}

std::size_t teamThreshold(std::size_t own) noexcept {
  return t_threshold_set ? t_threshold : own;
}

bool teamThresholdActive() noexcept { return t_threshold_set; }

int teamSize(std::size_t n, std::size_t from, TeamFrom team) {
  const int nt = threadsRequested();
  // A threshold of zero admits every n, and a call with no atoms or one
  // atom still runs on one slice, never on none.
  if (n < 2 || n < from || nt <= 1 || (!t_threshold_set && n < team.of(nt))) {
    return 1;
  }
  if (static_cast<std::size_t>(nt) > n) {
    return static_cast<int>(n);
  }
  return nt;
}

int teamSize(std::size_t n) { return teamSize(n, kTeamMinAtoms, {}); }

std::uint64_t teamsOpened() noexcept { return t_teams_opened; }

int activeThreads(int fallback) noexcept {
#if defined(_OPENMP)
  if (const int explicit_count = resolvedExplicit(); explicit_count > 0) {
    return explicit_count;
  }
  return fallback > 0 ? fallback : 1;
#else
  (void)fallback;
  return 1;
#endif
}

void setThreadCount(int threads) {
  if (threads < 1) {
    throw std::invalid_argument("rgpot thread count must be at least 1");
  }
  t_sticky = threads;
}

void clearThreadCount() noexcept { t_sticky = 0; }

int explicitThreadCount() noexcept { return resolvedExplicit(); }

ThreadCount::ThreadCount(int threads) : pushed_(false) {
  if (threads < 1) {
    throw std::invalid_argument("rgpot thread count must be at least 1");
  }
  pushExplicit(threads, 1);
  pushed_ = true;
}

ThreadCount::~ThreadCount() {
  if (pushed_ && t_explicit_depth > 0) {
    --t_explicit_depth;
  }
}

ThreadBinding::ThreadBinding(int threads) : pushed_(false) {
  if (threads < 1 || callScopeActive()) {
    return;
  }
  pushExplicit(threads, 2);
  pushed_ = true;
}

ThreadBinding::~ThreadBinding() {
  if (pushed_ && t_explicit_depth > 0) {
    --t_explicit_depth;
  }
}

int poolThreads() noexcept {
#if defined(_OPENMP)
  const int n = omp_get_max_threads();
  return n > 0 ? n : 1;
#else
  return 1;
#endif
}

void resizePool(int threads) noexcept {
#if defined(_OPENMP)
  if (threads < 1) {
    threads = 1;
  }
  omp_set_dynamic(0);
  const int current = poolThreads();
  if (current > threads) {
    // MSVC's OpenMP 2.0 runtime has no pause. GCC reports 4.5 and has one.
#if _OPENMP >= 201511
    omp_pause_resource_all(omp_pause_soft);
#endif
    omp_set_num_threads(threads);
  } else if (current < threads) {
    omp_set_num_threads(threads);
  }
#else
  (void)threads;
#endif
}

TeamCap::TeamCap(int threads) noexcept : previous_(t_team_cap) {
  t_team_cap = threads > 0 ? threads : 0;
}

TeamCap::~TeamCap() { t_team_cap = previous_; }

TeamThreshold::TeamThreshold(std::size_t atoms) noexcept
    : previous_(t_threshold), had_previous_(t_threshold_set) {
  t_threshold = atoms;
  t_threshold_set = true;
}

TeamThreshold::~TeamThreshold() {
  t_threshold = previous_;
  t_threshold_set = had_previous_;
}

namespace detail {

int teamFor(int wanted) noexcept {
  const int cap = t_team_cap;
  return cap > 0 && cap < wanted ? cap : wanted;
}

void countTeam() noexcept { ++t_teams_opened; }

} // namespace detail

} // namespace rgpot
