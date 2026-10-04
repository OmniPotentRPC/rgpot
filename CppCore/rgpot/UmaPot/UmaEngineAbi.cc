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
#include <mutex>
#include <string>
#include <vector>

#include <ATen/Parallel.h>

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
  /// Message of the last failed call. Evaluations may run concurrently on a
  /// shared instance, so the message has its own lock.
  mutable std::mutex error_mutex;
  std::string error;

  void set_error(const std::string &message) {
    std::lock_guard<std::mutex> lock(error_mutex);
    error = message;
  }
  void clear_error() {
    std::lock_guard<std::mutex> lock(error_mutex);
    error.clear();
  }
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
      !box || nAtoms <= 0) {
    if (pot)
      pot->set_error("rgpot_engine_force(uma): invalid argument");
    return 1;
  }
  try {
    SoftGilRelease no_gil;
    // Scratch copies: a host transform must never mutate caller buffers.
    std::vector<double> R(positions, positions + 3 * nAtoms);
    std::vector<double> cell(box, box + 9);
    if (transform) {
      const int rc = transform(transform_user, nAtoms, R.data(), cell.data());
      if (rc != 0) {
        pot->set_error("coordinate transform returned " + std::to_string(rc));
        return rc;
      }
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
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot_engine_force(uma): %s\n", e.what());
    pot->set_error(e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "rgpot_engine_force(uma): unknown exception\n");
    pot->set_error("unknown exception");
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
      !boxes || nSystems <= 0 || nAtoms <= 0) {
    if (pot)
      pot->set_error("rgpot_engine_force_batch(uma): invalid argument");
    return 1;
  }
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
        if (rc != 0) {
          pot->set_error("coordinate transform returned " +
                         std::to_string(rc));
          return rc;
        }
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
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot_engine_force_batch(uma): %s\n", e.what());
    pot->set_error(e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "rgpot_engine_force_batch(uma): unknown exception\n");
    pot->set_error("unknown exception");
    return 2;
  }
}

RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  RgpotEngineAbiStamp stamp;
  stamp.abi_major = RGPOT_ENGINE_ABI_VERSION;
  stamp.abi_minor = RGPOT_ENGINE_ABI_MINOR;
  stamp.layout_revision = RGPOT_ENGINE_ABI_LAYOUT_REVISION;
  return stamp;
}

int rgpot_engine_abi_compatible(const RgpotEngineAbiStamp *stamp) {
  if (!stamp)
    return 0;
  return stamp->abi_major == RGPOT_ENGINE_ABI_VERSION &&
                 stamp->layout_revision == RGPOT_ENGINE_ABI_LAYOUT_REVISION &&
                 stamp->abi_minor <= RGPOT_ENGINE_ABI_MINOR
             ? 1
             : 0;
}

int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  if (!pot || !pot->impl || !out)
    return 1;
  const rgpot::PotCaps caps = pot->impl->caps();
  RgpotEngineCaps filled;
  std::memset(&filled, 0, sizeof(filled));
  filled.size = sizeof(RgpotEngineCaps);
  filled.reentrancy = static_cast<int>(caps.reentrancy);
  filled.per_image_instances = caps.perImageInstances ? 1 : 0;
  filled.batched = caps.batched ? 1 : 0;
  filled.periodic = caps.periodic ? 1 : 0;
  filled.stress = caps.stress ? 1 : 0;
  filled.group_collective = caps.groupCollective ? 1 : 0;
  // The host's struct may be older (smaller) than this engine's: write only
  // the bytes it announced room for.
  const unsigned room = out->size < sizeof(filled) ? out->size : sizeof(filled);
  if (room < sizeof(unsigned))
    return 1;
  std::memcpy(out, &filled, room);
  out->size = room;
  return 0;
}

size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf,
                               size_t len) {
  if (!pot)
    return 0;
  std::lock_guard<std::mutex> lock(pot->error_mutex);
  if (buf && len > 0) {
    const size_t n = pot->error.size() < len - 1 ? pot->error.size() : len - 1;
    std::memcpy(buf, pot->error.data(), n);
    buf[n] = '\0';
  }
  return pot->error.size();
}

int rgpot_engine_set_num_threads(RgpotEnginePot *pot, int intra_op,
                                 int inter_op) {
  if (!pot || !pot->impl)
    return 1;
  try {
    if (intra_op >= 1)
      at::set_num_threads(intra_op);
    if (inter_op >= 1)
      at::set_num_interop_threads(inter_op);
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    pot->set_error(e.what());
    return 2;
  } catch (...) {
    pot->set_error("unknown exception");
    return 2;
  }
}

int rgpot_engine_set_charge_spin(RgpotEnginePot *pot, int charge, int spin) {
  if (!pot || !pot->impl || spin < 1)
    return 1;
  try {
    pot->impl->setChargeSpin(charge, spin);
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    pot->set_error(e.what());
    return 2;
  } catch (...) {
    pot->set_error("unknown exception");
    return 2;
  }
}

} // extern "C"
