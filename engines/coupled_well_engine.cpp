// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/coupled_well_engine.h"
#include "rgpot/engine_c_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct RgpotEnginePot {
  RgpotCoupledWellConfigV1 params{};
  char error[256]{};
};

namespace {

void copyMessage(char *destination, size_t capacity, const char *message) {
  if (destination != nullptr && capacity != 0) {
    snprintf(destination, capacity, "%s", message);
  }
}

int invalid(RgpotEnginePot *pot, const char *message) {
  if (pot != nullptr) {
    snprintf(pot->error, sizeof pot->error, "%s", message);
  }
  return RGPOT_ENGINE_ERR_INVALID;
}

int checkRequest(RgpotEnginePot *pot, long nAtoms, const double *positions,
                 const int *atomicNumbers, const double *box) {
  if (pot == nullptr || nAtoms != 1 || positions == nullptr ||
      atomicNumbers == nullptr || box == nullptr) {
    return invalid(pot, "coupled-well engine requires one atom");
  }
  if (atomicNumbers[0] != pot->params.atomic_number) {
    return invalid(pot, "coupled-well engine atomic number mismatch");
  }
  return RGPOT_ENGINE_OK;
}

void gradientHessian(const RgpotCoupledWellConfigV1 &p, double x, double y,
                     double *energy, double gradient[2], double hessian[4]) {
  const double u = x / p.well_position;
  const double u2 = u * u;
  const double displaced = y - p.coupling * x * x;
  const double displacedDx = -2.0 * p.coupling * x;
  const double wellShape = u2 - 1.0;

  *energy = p.barrier_energy * wellShape * wellShape +
            0.5 * p.transverse_spring * displaced * displaced;
  gradient[0] =
      4.0 * p.barrier_energy * u * wellShape / p.well_position +
      p.transverse_spring * displaced * displacedDx;
  gradient[1] = p.transverse_spring * displaced;

  hessian[0] =
      4.0 * p.barrier_energy * (3.0 * u2 - 1.0) /
          (p.well_position * p.well_position) +
      p.transverse_spring *
          (displacedDx * displacedDx - 2.0 * p.coupling * displaced);
  hessian[1] = p.transverse_spring * displacedDx;
  hessian[2] = hessian[1];
  hessian[3] = p.transverse_spring;
}

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
  RgpotCoupledWellConfigV1 params{};
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
  if (params.atomic_number <= 0 || !isfinite(params.barrier_energy) ||
      !isfinite(params.well_position) ||
      !isfinite(params.transverse_spring) ||
      !isfinite(params.coupling) || params.barrier_energy <= 0.0 ||
      params.well_position <= 0.0 || params.transverse_spring <= 0.0) {
    copyMessage(errbuf, errlen,
                "coupled-well engine parameters must be finite and positive");
    return nullptr;
  }
  auto *pot =
      static_cast<RgpotEnginePot *>(calloc(1, sizeof(RgpotEnginePot)));
  if (pot == nullptr) {
    copyMessage(errbuf, errlen, "coupled-well engine allocation failed");
    return nullptr;
  }
  pot->params = params;
  return pot;
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { free(pot); }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                       const double *positions, const int *atomicNumbers,
                       double *forces, double *energy, double *variance,
                       const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transformUser) {
  int code = checkRequest(pot, nAtoms, positions, atomicNumbers, box);
  if (code != RGPOT_ENGINE_OK || forces == nullptr || energy == nullptr) {
    return code == RGPOT_ENGINE_OK
               ? invalid(pot, "coupled-well engine output is null")
               : code;
  }
  double coordinates[3] = {positions[0], positions[1], positions[2]};
  double localBox[9];
  memcpy(localBox, box, sizeof localBox);
  if (transform != nullptr) {
    code = transform(transformUser, nAtoms, coordinates, localBox);
    if (code != RGPOT_ENGINE_OK) {
      snprintf(pot->error, sizeof pot->error,
               "%s", "coupled-well coordinate transform failed");
      return code;
    }
  }
  double gradient[2];
  double hessian[4];
  gradientHessian(pot->params, coordinates[0], coordinates[1], energy,
                  gradient, hessian);
  forces[0] = -gradient[0];
  forces[1] = -gradient[1];
  forces[2] = 0.0;
  if (variance != nullptr) {
    *variance = 0.0;
  }
  pot->error[0] = '\0';
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_hvp(RgpotEnginePot *pot, long nAtoms, const double *positions,
                     const int *atomicNumbers, const double *direction,
                     double *product, const double *box) {
  int code = checkRequest(pot, nAtoms, positions, atomicNumbers, box);
  if (code != RGPOT_ENGINE_OK || direction == nullptr || product == nullptr) {
    return code == RGPOT_ENGINE_OK
               ? invalid(pot, "coupled-well HVP input is null")
               : code;
  }
  double coordinates[3] = {positions[0], positions[1], positions[2]};
  double energy;
  double gradient[2];
  double hessian[4];
  gradientHessian(pot->params, coordinates[0], coordinates[1], &energy,
                  gradient, hessian);
  product[0] = hessian[0] * direction[0] + hessian[1] * direction[1];
  product[1] = hessian[2] * direction[0] + hessian[3] * direction[1];
  product[2] = 0.0;
  pot->error[0] = '\0';
  return RGPOT_ENGINE_OK;
}

int rgpot_engine_caps(const RgpotEnginePot *, RgpotEngineCaps *out) {
  if (out == nullptr || out->size < sizeof(unsigned)) {
    return RGPOT_ENGINE_ERR_INVALID;
  }
  RgpotEngineCaps caps{};
  caps.size = sizeof caps;
  caps.reentrancy = 1;
  caps.per_image_instances = 1;
  caps.analytic_hvp = 1;
  const size_t room =
      static_cast<size_t>(out->size) < sizeof caps
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
  const size_t full = strlen(pot->error);
  if (buffer != nullptr && length != 0) {
    const size_t copied = full < length - 1 ? full : length - 1;
    memcpy(buffer, pot->error, copied);
    buffer[copied] = '\0';
  }
  return full;
}

const char *rgpot_engine_name(void) { return "coupled-well"; }

const char *rgpot_engine_version(void) { return "coupled-well/1"; }

} // extern "C"
