// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/coupled_well_engine.h"
#include "rgpot/engine_c_abi.h"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// The model is a pure function of the create message, so one handle serves
// concurrent calls (reentrancy 0). The only mutable state is the last-error
// pointer, which always refers to a string literal and is published atomically:
// rgpot_engine_last_error may run during an evaluation on the same handle.
struct RgpotEnginePot {
  RgpotCoupledWellConfigV1 params;
  std::atomic<const char *> error;
};

namespace {

constexpr long kAtoms = 1;
constexpr int kCoordinates = 3;
constexpr int kBoxEntries = 9;
// Scratch doubles per system in a batch: coordinates, forces and the energy.
constexpr size_t kBatchScratchPerSystem = 2 * kCoordinates + 1;
constexpr const char kNoError[] = "";

struct Derivatives {
  double energy;
  double gradient[2];
  double hessian[4];
};

void copyMessage(char *destination, size_t capacity, const char *message) {
  if (destination != nullptr && capacity != 0) {
    snprintf(destination, capacity, "%s", message);
  }
}

// `message` must have static storage duration.
int fail(RgpotEnginePot *pot, int code, const char *message) {
  if (pot != nullptr) {
    pot->error.store(message, std::memory_order_release);
  }
  return code;
}

void succeed(RgpotEnginePot *pot) {
  pot->error.store(kNoError, std::memory_order_release);
}

bool allFinite(const double *values, int count) {
  for (int index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

int checkComposition(RgpotEnginePot *pot, long nAtoms,
                     const int *atomicNumbers) {
  if (pot == nullptr || nAtoms != kAtoms || atomicNumbers == nullptr) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well engine requires exactly one atom");
  }
  if (atomicNumbers[0] != pot->params.atomic_number) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well engine atomic number mismatch");
  }
  return RGPOT_ENGINE_OK;
}

// Scratch copy of one system's coordinates after the host transform. The
// caller's buffers are never modified.
int transformed(RgpotEnginePot *pot, const double *positions, const double *box,
                rgpot_engine_coord_transform transform, void *transformUser,
                double *coordinates) {
  double localBox[kBoxEntries];
  memcpy(coordinates, positions, kCoordinates * sizeof *coordinates);
  memcpy(localBox, box, sizeof localBox);
  if (transform != nullptr) {
    const int code = transform(transformUser, kAtoms, coordinates, localBox);
    if (code != RGPOT_ENGINE_OK) {
      return fail(pot, code, "coupled-well coordinate transform failed");
    }
  }
  if (!allFinite(coordinates, kCoordinates)) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well positions must be finite");
  }
  return RGPOT_ENGINE_OK;
}

Derivatives derivatives(const RgpotCoupledWellConfigV1 &p, double x,
                        double y) {
  const double u = x / p.well_position;
  const double u2 = u * u;
  const double displaced = y - p.coupling * x * x;
  const double displacedDx = -2.0 * p.coupling * x;
  const double displacedDxx = -2.0 * p.coupling;
  const double wellShape = u2 - 1.0;

  Derivatives d;
  d.energy = p.barrier_energy * wellShape * wellShape +
             0.5 * p.transverse_spring * displaced * displaced;
  d.gradient[0] = 4.0 * p.barrier_energy * u * wellShape / p.well_position +
                  p.transverse_spring * displaced * displacedDx;
  d.gradient[1] = p.transverse_spring * displaced;
  d.hessian[0] = 4.0 * p.barrier_energy * (3.0 * u2 - 1.0) /
                     (p.well_position * p.well_position) +
                 p.transverse_spring *
                     (displacedDx * displacedDx + displaced * displacedDxx);
  d.hessian[1] = p.transverse_spring * displacedDx;
  d.hessian[2] = d.hessian[1];
  d.hessian[3] = p.transverse_spring;
  return d;
}

bool finiteAnswer(const Derivatives &d) {
  return std::isfinite(d.energy) && allFinite(d.gradient, 2) &&
         allFinite(d.hessian, 4);
}

struct Scratch {
  double *data;
  explicit Scratch(size_t systems)
      : data(static_cast<double *>(
            calloc(systems, kBatchScratchPerSystem * sizeof(double)))) {}
  ~Scratch() { free(data); }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
};

} // namespace

extern "C" {

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  return {RGPOT_ENGINE_ABI_VERSION, RGPOT_ENGINE_ABI_MINOR,
          RGPOT_ENGINE_ABI_LAYOUT_REVISION};
}

int rgpot_engine_abi_compatible(const RgpotEngineAbiStamp *host_stamp) {
  return host_stamp != nullptr &&
         host_stamp->abi_major == RGPOT_ENGINE_ABI_VERSION &&
         host_stamp->layout_revision == RGPOT_ENGINE_ABI_LAYOUT_REVISION &&
         RGPOT_ENGINE_ABI_MINOR <= host_stamp->abi_minor;
}

RgpotEnginePot *rgpot_engine_create(const void *config, size_t configLen,
                                    char *errbuf, size_t errlen) {
  RgpotCoupledWellConfigV1 params;
  if (config == nullptr || configLen != sizeof params) {
    copyMessage(errbuf, errlen,
                "coupled-well engine requires RgpotCoupledWellConfigV1");
    return nullptr;
  }
  memcpy(&params, config, sizeof params);
  if (params.magic != RGPOT_COUPLED_WELL_CONFIG_MAGIC ||
      params.version != RGPOT_COUPLED_WELL_CONFIG_VERSION ||
      params.reserved != 0) {
    copyMessage(errbuf, errlen,
                "coupled-well engine config magic, version, or reserved field");
    return nullptr;
  }
  if (params.atomic_number <= 0) {
    copyMessage(errbuf, errlen,
                "coupled-well atomic_number must be positive");
    return nullptr;
  }
  if (!std::isfinite(params.barrier_energy) || params.barrier_energy <= 0.0) {
    copyMessage(errbuf, errlen,
                "coupled-well barrier_energy must be finite and positive");
    return nullptr;
  }
  if (!std::isfinite(params.well_position) || params.well_position <= 0.0) {
    copyMessage(errbuf, errlen,
                "coupled-well well_position must be finite and positive");
    return nullptr;
  }
  if (!std::isfinite(params.transverse_spring) ||
      params.transverse_spring <= 0.0) {
    copyMessage(errbuf, errlen,
                "coupled-well transverse_spring must be finite and positive");
    return nullptr;
  }
  if (!std::isfinite(params.coupling)) {
    copyMessage(errbuf, errlen, "coupled-well coupling must be finite");
    return nullptr;
  }
  auto *pot = new (std::nothrow) RgpotEnginePot;
  if (pot == nullptr) {
    copyMessage(errbuf, errlen, "coupled-well engine allocation failed");
    return nullptr;
  }
  pot->params = params;
  pot->error.store(kNoError, std::memory_order_relaxed);
  return pot;
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { delete pot; }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                       const double *positions, const int *atomicNumbers,
                       double *forces, double *energy, double *variance,
                       const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transformUser) {
  if (positions == nullptr || forces == nullptr || energy == nullptr ||
      box == nullptr) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well force input or output is null");
  }
  int code = checkComposition(pot, nAtoms, atomicNumbers);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  double coordinates[kCoordinates];
  code = transformed(pot, positions, box, transform, transformUser,
                     coordinates);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  const Derivatives d = derivatives(pot->params, coordinates[0], coordinates[1]);
  if (!finiteAnswer(d)) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL,
                "coupled-well evaluation is not finite at this position");
  }
  *energy = d.energy;
  forces[0] = -d.gradient[0];
  forces[1] = -d.gradient[1];
  forces[2] = 0.0;
  if (variance != nullptr) {
    *variance = 0.0;
  }
  succeed(pot);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_force_batch(RgpotEnginePot *pot, long nSystems, long nAtoms,
                             const double *positions, const int *atomicNumbers,
                             double *forces, double *energies,
                             double *variances, const double *boxes,
                             rgpot_engine_coord_transform transform,
                             void *transformUser) {
  if (nSystems < 1 || nSystems > LONG_MAX / kBoxEntries || nAtoms < 1 ||
      positions == nullptr || forces == nullptr || energies == nullptr ||
      boxes == nullptr) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID, "invalid argument (batch)");
  }
  int code = checkComposition(pot, nAtoms, atomicNumbers);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  const size_t systems = static_cast<size_t>(nSystems);
  Scratch scratch(systems);
  if (scratch.data == nullptr) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL,
                "coupled-well batch allocation failed");
  }
  double *coordinates = scratch.data;
  double *scratchForces = coordinates + kCoordinates * systems;
  double *scratchEnergies = scratchForces + kCoordinates * systems;

  // Transform and evaluate every system before any output is written, so a
  // failure leaves the caller's buffers untouched.
  for (size_t system = 0; system < systems; ++system) {
    double *q = coordinates + kCoordinates * system;
    code = transformed(pot, positions + kCoordinates * system,
                       boxes + kBoxEntries * system, transform, transformUser,
                       q);
    if (code != RGPOT_ENGINE_OK) {
      return code;
    }
  }
  for (size_t system = 0; system < systems; ++system) {
    const double *q = coordinates + kCoordinates * system;
    const Derivatives d = derivatives(pot->params, q[0], q[1]);
    if (!finiteAnswer(d)) {
      return fail(pot, RGPOT_ENGINE_ERR_EVAL,
                  "coupled-well evaluation is not finite at this position");
    }
    double *f = scratchForces + kCoordinates * system;
    f[0] = -d.gradient[0];
    f[1] = -d.gradient[1];
    f[2] = 0.0;
    scratchEnergies[system] = d.energy;
  }
  memcpy(forces, scratchForces, kCoordinates * systems * sizeof *forces);
  memcpy(energies, scratchEnergies, systems * sizeof *energies);
  if (variances != nullptr) {
    for (size_t system = 0; system < systems; ++system) {
      variances[system] = 0.0;
    }
  }
  succeed(pot);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_hvp(RgpotEnginePot *pot, long nAtoms, const double *positions,
                     const int *atomicNumbers, const double *direction,
                     double *product, const double *box) {
  if (positions == nullptr || direction == nullptr || product == nullptr ||
      box == nullptr) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well HVP input or output is null");
  }
  int code = checkComposition(pot, nAtoms, atomicNumbers);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  if (!allFinite(direction, kCoordinates)) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "coupled-well HVP direction must be finite");
  }
  double coordinates[kCoordinates];
  code = transformed(pot, positions, box, nullptr, nullptr, coordinates);
  if (code != RGPOT_ENGINE_OK) {
    return code;
  }
  const Derivatives d = derivatives(pot->params, coordinates[0], coordinates[1]);
  const double result[kCoordinates] = {
      d.hessian[0] * direction[0] + d.hessian[1] * direction[1],
      d.hessian[2] * direction[0] + d.hessian[3] * direction[1], 0.0};
  if (!finiteAnswer(d) || !allFinite(result, kCoordinates)) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL,
                "coupled-well evaluation is not finite at this position");
  }
  memcpy(product, result, sizeof result);
  succeed(pot);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_caps(const RgpotEnginePot *, RgpotEngineCaps *out) {
  if (out == nullptr || out->size < sizeof(unsigned)) {
    return RGPOT_ENGINE_ERR_INVALID;
  }
  RgpotEngineCaps caps{};
  caps.size = sizeof caps;
  caps.reentrancy = 0;
  caps.batched = 1;
  caps.analytic_hvp = 1;
  const size_t room = static_cast<size_t>(out->size) < sizeof caps
                          ? static_cast<size_t>(out->size)
                          : sizeof caps;
  memcpy(out, &caps, room);
  out->size = static_cast<unsigned>(room);
  return RGPOT_ENGINE_OK;
}

size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buffer,
                               size_t length) {
  if (pot == nullptr) {
    return 0;
  }
  const char *message = pot->error.load(std::memory_order_acquire);
  const size_t full = strlen(message);
  if (buffer != nullptr && length != 0) {
    const size_t copied = full < length - 1 ? full : length - 1;
    memcpy(buffer, message, copied);
    buffer[copied] = '\0';
  }
  return full;
}

const char *rgpot_engine_name(void) { return "coupled-well"; }

const char *rgpot_engine_version(void) { return "coupled-well/1"; }

} // extern "C"
