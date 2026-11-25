// MIT License
// Copyright 2023--present rgpot developers

// Fails the k-th allocation after each bridge call starts. Recording the
// error must not throw out of the extern "C" entry points, and the last
// error must stay readable.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" {
void *eam_bridge_cpp_new_from_file(const char *path, int32_t format,
                                   bool periodic, double skin);
void *eam_bridge_cpp_new_from_table(std::size_t n_elements, const int32_t *z,
                                    int32_t nrho, double drho, int32_t nr,
                                    double dr, double cutoff, double rhomax,
                                    const double *frho, const double *rhor,
                                    const double *z2r, bool periodic,
                                    double skin);
int32_t eam_bridge_cpp_force(void *handle, std::size_t n_atoms,
                             const double *positions,
                             const int32_t *atomic_numbers, const double *cell,
                             double *forces, double *energy);
const char *eam_bridge_cpp_last_error();
}

namespace {

int g_fail_at = 0;

void *allocate(std::size_t n) {
  if (g_fail_at > 0 && --g_fail_at == 0) {
    throw std::bad_alloc();
  }
  if (void *p = std::malloc(n != 0 ? n : 1)) {
    return p;
  }
  throw std::bad_alloc();
}

} // namespace

void *operator new(std::size_t n) { return allocate(n); }
void *operator new[](std::size_t n) { return allocate(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {

int check(const char *what, bool (*call)()) {
  int failures = 0;
  for (int k = 1; k <= 8; ++k) {
    g_fail_at = k;
    bool failed = false;
    try {
      failed = call();
    } catch (...) {
      g_fail_at = 0;
      std::fprintf(stderr, "%s: exception escaped with allocation %d failing\n",
                   what, k);
      ++failures;
      continue;
    }
    g_fail_at = 0;
    const char *message = eam_bridge_cpp_last_error();
    if (!failed || message == nullptr || std::strlen(message) == 0) {
      std::fprintf(stderr, "%s: no error recorded with allocation %d failing\n",
                   what, k);
      ++failures;
    }
  }
  return failures;
}

} // namespace

int main() {
  int failures = 0;
  failures += check("new_from_file", [] {
    return eam_bridge_cpp_new_from_file(nullptr, 1, false, 1.0) == nullptr;
  });
  failures += check("new_from_table", [] {
    return eam_bridge_cpp_new_from_table(1, nullptr, 5, 0.5, 5, 0.5, 2.0, 2.0,
                                         nullptr, nullptr, nullptr, false,
                                         1.0) == nullptr;
  });
  failures += check("force", [] {
    double energy = 0.0;
    return eam_bridge_cpp_force(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                &energy) != 0;
  });
  if (failures == 0) {
    std::puts("eam_table_bridge_oom: ok");
  }
  return failures == 0 ? 0 : 1;
}
