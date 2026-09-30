#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_CPMDC_BUILD
#endif
#include "rgpot/CPMDPot/cpmd_c_abi.h"

#include <cstring>

#include <mpi.h>

namespace {

constexpr CPMDCFeatureEntry kFeatures[] = {
    {"abi.cpmdc_set_params", CPMDC_FEATURE_ABI, 1, 1},
    {"abi.cpmdc_energy_gradient", CPMDC_FEATURE_ABI, 1, 1},
    {"abi.cpmdc_adopt_calculator_comm", CPMDC_FEATURE_ABI, 1, 1},
};

MPI_Comm g_comm = MPI_COMM_NULL;
bool g_set = false;

} // namespace

extern "C" {

int cpmdc_set_params(const void *params_capnp, size_t params_capnp_size_bytes) {
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  return 0;
}

CPMDCResult cpmdc_energy_gradient(int n_atoms, const double *positions_ang,
                                  const int *atomic_numbers,
                                  const void *params_capnp,
                                  size_t params_capnp_size_bytes,
                                  double *grad_h_bohr) {
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  (void)grad_h_bohr;
  CPMDCResult result{};
  result.ok = 0;
  std::strncpy(result.message, "adopt engine has no energy",
               sizeof(result.message) - 1);
  return result;
}

// Stores the communicator from rgpot. Does not call MPI_Comm_split.
int cpmdc_adopt_calculator_comm(const void *comm, size_t comm_bytes,
                                int ranks_per_calc) {
  if (comm == nullptr || comm_bytes != sizeof(MPI_Comm))
    return -1;
  MPI_Comm incoming = MPI_COMM_NULL;
  std::memcpy(&incoming, comm, sizeof(incoming));
  if (incoming == MPI_COMM_NULL)
    return -1;
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    return -1;
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rpc = ranks_per_calc;
  if (rpc <= 0)
    rpc = size;
  if (rpc > size || size % rpc != 0)
    return -1;
  if (!g_set) {
    g_comm = incoming;
    g_set = true;
  }
  return rank / rpc;
}

int cpmdc_adopted_comm(void *out, size_t nbytes) {
  if (!g_set || out == nullptr || nbytes != sizeof(MPI_Comm))
    return -1;
  std::memcpy(out, &g_comm, sizeof(g_comm));
  return 0;
}

size_t cpmdc_feature_count(void) {
  return sizeof(kFeatures) / sizeof(kFeatures[0]);
}

const CPMDCFeatureEntry *cpmdc_feature_table(void) { return kFeatures; }

const CPMDCFeatureEntry *cpmdc_feature_find(const char *feature_id) {
  if (feature_id == nullptr)
    return nullptr;
  for (const auto &feature : kFeatures) {
    if (std::strcmp(feature.feature_id, feature_id) == 0)
      return &feature;
  }
  return nullptr;
}

} // extern "C"
