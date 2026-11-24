// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/coupled_well_engine.h"
#include "rgpot/engine_c_abi.h"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#ifndef RGPOT_ENGINE_PROJECT_VERSION
#define RGPOT_ENGINE_PROJECT_VERSION "unknown"
#endif

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

bool rangesOverlap(const void *first, size_t firstBytes, const void *second,
                   size_t secondBytes) {
  const auto firstBegin = reinterpret_cast<uintptr_t>(first);
  const auto secondBegin = reinterpret_cast<uintptr_t>(second);
  if (firstBegin < secondBegin) {
    return secondBegin - firstBegin < firstBytes;
  }
  return firstBegin - secondBegin < secondBytes;
}

bool trailingBytesAreZero(const unsigned char *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) {
      return false;
    }
  }
  return true;
}

// Version 1 is the 48-byte layout: magic, version, atomic_number, reserved,
// then the four doubles. Version 2 carries size at offset 4 and version at
// offset 8. A u32 of 1 at offset 4 is version 1, never a version-2 size
// (those are at least RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE). A 48-byte message
// whose word at offset 4 is below that minimum, and is not 1, has an unknown
// version and is refused before offset 8 (atomic_number in version 1, which
// may be 2) is read as a version-2 version. A word at offset 4 that is at
// least the version-2 size is a truncated version-2 or later message, so
// decoding follows the version-2 path and reports that the buffer is shorter
// than config.size. The stored parameters are the version-1 fields.
bool readConfig(RgpotCoupledWellConfigV1 *out, const void *config, size_t configLen,
                char *errbuf, size_t errlen) {
  if (config == nullptr || configLen < sizeof(uint32_t) * 2) {
    copyMessage(errbuf, errlen,
                "coupled-well engine requires RgpotCoupledWellConfigV1");
    return false;
  }
  const auto *bytes = static_cast<const unsigned char *>(config);
  uint32_t magic = 0;
  uint32_t word4 = 0;
  memcpy(&magic, bytes, sizeof magic);
  memcpy(&word4, bytes + sizeof magic, sizeof word4);
  if (magic != RGPOT_COUPLED_WELL_CONFIG_MAGIC) {
    copyMessage(errbuf, errlen,
                "coupled-well engine config magic, version, or reserved fields");
    return false;
  }
  if (word4 == RGPOT_COUPLED_WELL_CONFIG_VERSION_V1) {
    if (configLen != sizeof(RgpotCoupledWellConfigV1)) {
      copyMessage(errbuf, errlen,
                  "coupled-well engine requires RgpotCoupledWellConfigV1");
      return false;
    }
    RgpotCoupledWellConfigV1 params{};
    memcpy(&params, config, sizeof params);
    if (params.reserved != 0) {
      copyMessage(errbuf, errlen,
                  "coupled-well engine config magic, version, or reserved fields");
      return false;
    }
    *out = params;
    return true;
  }
  if (configLen == sizeof(RgpotCoupledWellConfigV1) &&
      word4 < RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE) {
    copyMessage(errbuf, errlen, "coupled-well config version is unknown");
    return false;
  }
  if (configLen < offsetof(RgpotCoupledWellConfigV2, version) + sizeof(uint32_t)) {
    copyMessage(errbuf, errlen,
                "coupled-well engine requires RgpotCoupledWellConfigV2");
    return false;
  }
  uint32_t version = 0;
  memcpy(&version, bytes + offsetof(RgpotCoupledWellConfigV2, version),
         sizeof version);
  if (version != RGPOT_COUPLED_WELL_CONFIG_VERSION) {
    copyMessage(errbuf, errlen, "coupled-well config version is unknown");
    return false;
  }
  RgpotCoupledWellConfigV2 wire{};
  memcpy(&wire, config, sizeof(uint32_t) * 3);
  if (wire.size < RGPOT_COUPLED_WELL_CONFIG_MIN_SIZE) {
    copyMessage(errbuf, errlen,
                "coupled-well config size is smaller than RgpotCoupledWellConfigV2");
    return false;
  }
  if (configLen < wire.size) {
    copyMessage(errbuf, errlen,
                "coupled-well config buffer is shorter than config.size");
    return false;
  }
  memcpy(&wire, config, sizeof wire);
  if (wire.size > sizeof wire &&
      !trailingBytesAreZero(bytes + sizeof wire, wire.size - sizeof wire)) {
    copyMessage(errbuf, errlen,
                "coupled-well config trailing bytes must be zero");
    return false;
  }
  if (wire.reserved0 != 0 || wire.reserved1 != 0) {
    copyMessage(errbuf, errlen,
                "coupled-well engine config magic, version, or reserved fields");
    return false;
  }
  RgpotCoupledWellConfigV1 params{};
  params.magic = wire.magic;
  params.version = RGPOT_COUPLED_WELL_CONFIG_VERSION_V1;
  params.atomic_number = wire.atomic_number;
  params.reserved = 0;
  params.barrier_energy = wire.barrier_energy;
  params.well_position = wire.well_position;
  params.transverse_spring = wire.transverse_spring;
  params.coupling = wire.coupling;
  *out = params;
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
  const double x0 = p.well_position;
  const double v0 = p.barrier_energy;
  const double k = p.transverse_spring;
  const double alpha = p.coupling;

  const double u = x / x0;
  const double u2 = u * u;
  double wellEnergy;
  double wellGradX;
  if (std::isfinite(u2)) {
    const double wellShape = u2 - 1.0;
    wellEnergy = v0 * wellShape * wellShape;
    wellGradX = 4.0 * v0 * u * wellShape / x0;
  } else {
    const double v0WellShape = (v0 * u) * u - v0;
    wellEnergy = (v0WellShape * u) * u - v0WellShape;
    wellGradX = 4.0 * (v0WellShape / x0) * u;
  }

  const double displaced = y - alpha * x * x;
  const double dDx = -2.0 * alpha * x;
  const double springGradX =
      displaced == 0.0 ? 0.0 : k * displaced * dDx;

  Derivatives d;
  d.energy = wellEnergy + 0.5 * k * displaced * displaced;
  d.gradient[0] = wellGradX + springGradX;
  d.gradient[1] = k * displaced;
  return d;
}

bool finiteEnergyAndGradient(const Derivatives &d) {
  return std::isfinite(d.energy) && allFinite(d.gradient, 2);
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
  if (!readConfig(&params, config, configLen, errbuf, errlen)) {
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
  if (!finiteEnergyAndGradient(d)) {
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
    if (!finiteEnergyAndGradient(d)) {
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
  if (rangesOverlap(product, kCoordinates * sizeof *product, direction,
                    kCoordinates * sizeof *direction) ||
      rangesOverlap(product, kCoordinates * sizeof *product, positions,
                    kCoordinates * sizeof *positions) ||
      rangesOverlap(product, kCoordinates * sizeof *product, atomicNumbers,
                    static_cast<size_t>(kAtoms) * sizeof *atomicNumbers) ||
      rangesOverlap(product, kCoordinates * sizeof *product, box,
                    kBoxEntries * sizeof *box)) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "HVP output overlaps an input buffer");
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
  const double x = coordinates[0];
  const double y = coordinates[1];
  const double dx = direction[0];
  const double dy = direction[1];
  const double x0 = pot->params.well_position;
  const double v0 = pot->params.barrier_energy;
  const double k = pot->params.transverse_spring;
  const double alpha = pot->params.coupling;
  const double hxy = -2.0 * k * alpha * x;
  const double hyy = k;
  double hvx = 0.0;
  double hvy = 0.0;
  if (dx != 0.0) {
    const double u = x / x0;
    const double u2 = u * u;
    const double v0Term =
        std::isfinite(u2) ? v0 * (3.0 * u2 - 1.0) : 3.0 * ((v0 * u) * u) - v0;
    const double wellHxx = (4.0 * (v0Term / x0)) / x0;
    const double displaced = y - alpha * x * x;
    const double dDx = -2.0 * alpha * x;
    const double dDxx = -2.0 * alpha;
    const double hxx =
        wellHxx +
        k * (dDx * dDx + (displaced == 0.0 ? 0.0 : displaced * dDxx));
    hvx = hxx * dx;
    hvy = hxy * dx;
    if (dy != 0.0) {
      hvx += hxy * dy;
      hvy += hyy * dy;
    }
  } else if (dy != 0.0) {
    hvx = hxy * dy;
    hvy = hyy * dy;
  }
  const double result[kCoordinates] = {hvx, hvy, 0.0};
  if (!allFinite(result, kCoordinates)) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL,
                "coupled-well evaluation is not finite at this position");
  }
  memcpy(product, result, sizeof result);
  succeed(pot);
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  if (pot == nullptr || out == nullptr || out->size < sizeof(unsigned)) {
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

const char *rgpot_engine_version(void) {
  return "coupled-well/" RGPOT_ENGINE_PROJECT_VERSION;
}

} // extern "C"
