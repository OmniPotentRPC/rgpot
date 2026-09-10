// MIT License — UMA engine behind the generic rgpot engine C ABI
// (torch linked into this .so only).
#define RGPOT_ENGINE_BUILD
#include "rgpot/engine_c_abi.h"

#include "rgpot/ForceStructs.hpp"
#include "rgpot/UmaPot/UmaPot.hpp"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/rpc/Potentials.capnp.h"

// When this engine is called from a Python host (pyeonclient / nanobind) the
// GIL is often still held. Torch autograd refuses that. Soft-resolve CPython
// thread APIs so pure C++ hosts (eonclient binary) need no libpython link.
//
// Only SaveThread when PyGILState_Check() is true — nested release (Job.run
// already dropped the GIL) must not call SaveThread again.
namespace {
struct SoftGilRelease {
  using CheckFn = int (*)();
  using SaveFn = void *(*)();
  using RestoreFn = void (*)(void *);
  void *tstate = nullptr;
  RestoreFn restore = nullptr;
  SoftGilRelease() {
    auto *check =
        reinterpret_cast<CheckFn>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    auto *save =
        reinterpret_cast<SaveFn>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    restore = reinterpret_cast<RestoreFn>(
        dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));
    if (check && check() && save && restore) {
      tstate = save();
    }
  }
  ~SoftGilRelease() {
    if (tstate && restore) {
      restore(tstate);
    }
  }
  SoftGilRelease(const SoftGilRelease &) = delete;
  SoftGilRelease &operator=(const SoftGilRelease &) = delete;
};

} // namespace

struct RgpotEnginePot {
  std::unique_ptr<rgpot::UmaPot> impl;
};

static void set_err(char *errbuf, size_t errlen, const char *msg) {
  if (!errbuf || errlen == 0)
    return;
  std::snprintf(errbuf, errlen, "%s", msg ? msg : "unknown");
}

extern "C" {

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  if (!config || config_len == 0 || config_len % sizeof(capnp::word) != 0) {
    set_err(errbuf, errlen,
            "rgpot_engine_create(uma): capnp UmaParams message required");
    return nullptr;
  }
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    const auto params = reader.getRoot<::UmaParams>();

    rgpot::UmaConfig c;
    c.model_path = params.getModelPath().cStr();
    if (c.model_path.empty()) {
      set_err(errbuf, errlen, "rgpot_engine_create(uma): modelPath required");
      return nullptr;
    }
    if (params.getTaskName().size() > 0)
      c.task_name = params.getTaskName().cStr();
    if (params.getDevice().size() > 0)
      c.device = params.getDevice().cStr();
    c.charge = params.getCharge();
    c.spin = params.getSpin() > 0 ? params.getSpin() : 1;
    if (params.getCutoff() > 0.0)
      c.cutoff = params.getCutoff();
    if (params.getMaxNeighbors() > 0)
      c.max_neighbors = params.getMaxNeighbors();
    auto pot = std::make_unique<RgpotEnginePot>();
    pot->impl = std::make_unique<rgpot::UmaPot>(c);
    return pot.release();
  } catch (const std::exception &e) {
    set_err(errbuf, errlen, e.what());
    return nullptr;
  } catch (...) {
    set_err(errbuf, errlen, "rgpot_engine_create(uma): unknown exception");
    return nullptr;
  }
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { delete pot; }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                       const double *positions, const int *atomicNrs,
                       double *forces, double *energy, double *variance,
                       const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transform_user) {
  if (!pot || !pot->impl || !positions || !atomicNrs || !forces || !energy ||
      !box || nAtoms <= 0)
    return 1;
  try {
    SoftGilRelease no_gil;
    // Scratch copies: a host transform must never mutate caller buffers.
    std::vector<double> R(positions, positions + 3 * nAtoms);
    std::vector<double> cell(box, box + 9);
    if (transform) {
      const int rc = transform(transform_user, nAtoms, R.data(), cell.data());
      if (rc != 0)
        return rc;
    }
    std::vector<double> Fbuf(static_cast<size_t>(nAtoms) * 3);
    rgpot::ForceOut out{Fbuf.data(), 0.0, 0.0};
    rgpot::ForceInput in{static_cast<size_t>(nAtoms), R.data(), atomicNrs,
                         cell.data()};
    pot->impl->forceImpl(in, &out);
    *energy = out.energy;
    if (variance)
      *variance = out.variance;
    std::memcpy(forces, Fbuf.data(), Fbuf.size() * sizeof(double));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot_engine_force(uma): %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "rgpot_engine_force(uma): unknown exception\n");
    return 2;
  }
}

int rgpot_engine_force_batch(RgpotEnginePot *pot, long nSystems, long nAtoms,
                             const double *positions, const int *atomicNrs,
                             double *forces, double *energies,
                             double *variances, const double *boxes,
                             rgpot_engine_coord_transform transform,
                             void *transform_user) {
  if (!pot || !pot->impl || !positions || !atomicNrs || !forces || !energies ||
      !boxes || nSystems <= 0 || nAtoms <= 0)
    return 1;
  try {
    SoftGilRelease no_gil;
    const size_t B = static_cast<size_t>(nSystems);
    const size_t stride = static_cast<size_t>(nAtoms) * 3;
    // Scratch copies: a host transform must never mutate caller buffers.
    std::vector<double> R(positions, positions + stride * B);
    std::vector<double> cells(boxes, boxes + 9 * B);
    if (transform) {
      for (size_t i = 0; i < B; ++i) {
        const int rc = transform(transform_user, nAtoms, R.data() + i * stride,
                                 cells.data() + i * 9);
        if (rc != 0)
          return rc;
      }
    }
    std::vector<rgpot::ForceInput> in;
    std::vector<rgpot::ForceOut> out(B);
    in.reserve(B);
    for (size_t i = 0; i < B; ++i) {
      in.push_back(rgpot::ForceInput{static_cast<size_t>(nAtoms),
                                     R.data() + i * stride, atomicNrs,
                                     cells.data() + i * 9});
      out[i] = rgpot::ForceOut{forces + i * stride, 0.0, 0.0};
    }
    const rgpot::ForceBatch batch{B, in.data(), out.data()};
    pot->impl->forceBatchImpl(batch);
    for (size_t i = 0; i < B; ++i) {
      energies[i] = out[i].energy;
      if (variances)
        variances[i] = out[i].variance;
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot_engine_force_batch(uma): %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "rgpot_engine_force_batch(uma): unknown exception\n");
    return 2;
  }
}

} // extern "C"
