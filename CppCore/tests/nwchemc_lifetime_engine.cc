// MIT License

#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_NWCHEMC_BUILD
#endif
#include "rgpot/NWChemPot/nwchem_c_abi.h"
#include <cstdio>
#include <cstdlib>

namespace {
int calls = 0;
bool finalized = false;
bool registered = false;
void record(const char *event) {
  const char *path = std::getenv("RGPOT_NWCHEM_LIFETIME_LOG");
  if (!path)
    std::abort();
  FILE *file = std::fopen(path, "a");
  if (!file)
    std::abort();
  std::fprintf(file, "%s\n", event);
  std::fclose(file);
}
struct Lifetime {
  Lifetime() { record("load"); }
  ~Lifetime() { record("unload"); }
} lifetime;
}

extern "C" RGPOT_NWCHEMC_API void nwchemc_finalize() {
  if (!finalized) {
    finalized = true;
    record("finalize");
  }
}
extern "C" int nwchemc_set_params(const void *, size_t) {
  if (!registered) {
    std::atexit(nwchemc_finalize);
    registered = true;
  }
  return finalized ? -1 : 0;
}
extern "C" NWChemCResult nwchemc_energy_gradient(
    int n, const double *, const int *, const void *, size_t, double *gradient) {
  NWChemCResult out{};
  if (finalized) {
    std::snprintf(out.message, sizeof(out.message), "engine finalized while in use");
    return out;
  }
  ++calls;
  char event[64];
  std::snprintf(event, sizeof(event), "call %d", calls);
  record(event);
  out.ok = 1;
  out.energy_h = 0.01 * calls;
  for (int i = 0; i < 3 * n; ++i)
    gradient[i] = 0.001 * (i + 1);
  return out;
}
extern "C" const char *nwchemc_version() { return "nwchem-lifetime-fixture"; }
extern "C" int nwchemc_available() { return 1; }
