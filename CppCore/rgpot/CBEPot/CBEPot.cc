// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CBEPot/CBEPot.hpp"
#include "rgpot/units.hpp"

#include <potlibc.h>

#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifndef POTLIBC_JET_MAX_ORDER
#error "rgpot's CBE surface needs potlibc with potlibc_gradient_jet"
#endif

namespace rgpot {
namespace {
constexpr std::size_t kAtoms = 6;
constexpr std::size_t kCoordinates = 3 * kAtoms;

void require(potlibc_status status) {
  if (status != POTLIBC_SUCCESS)
    throw std::runtime_error(std::string("potlibc ch52008: ") +
                             potlibc_status_string(status));
}

double toForce(double hartreePerBohr) {
  return hartreePerBohr * units::HARTREE_TO_EV / units::BOHR_TO_ANGSTROM;
}
} // namespace

void CBEPot::HandleDeleter::operator()(potlibc_handle *handle) const noexcept {
  potlibc_destroy(handle);
}

CBEPot::CBEPot() : Potential(PotType::CBE) {
  potlibc_handle *raw = nullptr;
  require(potlibc_create("ch52008", &raw));
  handle_.reset(raw);
  potlibc_metadata metadata{};
  require(potlibc_get_metadata(raw, &metadata));
  if ((metadata.capabilities & POTLIBC_CAP_HVP) == 0 ||
      (metadata.capabilities & POTLIBC_CAP_GRADIENT_JET) == 0)
    throw std::runtime_error(
        "potlibc ch52008 lacks the Hessian product or the gradient jet");
}

CBEPot::~CBEPot() = default;

void CBEPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const rgpot::ThreadBinding bind_threads{threadCount()};
  if (in.nAtoms != kAtoms)
    throw std::invalid_argument("CBE requires six atoms");
  std::array<double, kCoordinates> x{}, force{};
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = in.pos[i] / units::BOHR_TO_ANGSTROM;
  double energy = 0.0;
  std::lock_guard<std::mutex> lock(mutex_);
  require(potlibc_evaluate(handle_.get(), 1, x.data(), &energy, force.data()));
  out->energy = energy * units::HARTREE_TO_EV;
  for (std::size_t i = 0; i < force.size(); ++i)
    out->F[i] = toForce(force[i]);
}

void CBEPot::forceBatchImpl(const ForceBatch &batch) const {
  for (std::size_t s = 0; s < batch.nSystems; ++s)
    if (batch.in[s].nAtoms != kAtoms)
      throw std::invalid_argument("CBE requires six atoms");
  std::vector<double> x(batch.nSystems * kCoordinates);
  std::vector<double> force(x.size());
  std::vector<double> energy(batch.nSystems);
  for (std::size_t s = 0; s < batch.nSystems; ++s)
    for (std::size_t i = 0; i < kCoordinates; ++i)
      x[s * kCoordinates + i] = batch.in[s].pos[i] / units::BOHR_TO_ANGSTROM;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    require(potlibc_evaluate(handle_.get(), batch.nSystems, x.data(),
                             energy.data(), force.data()));
  }
  for (std::size_t s = 0; s < batch.nSystems; ++s) {
    batch.out[s].energy = energy[s] * units::HARTREE_TO_EV;
    for (std::size_t i = 0; i < kCoordinates; ++i)
      batch.out[s].F[i] = toForce(force[s * kCoordinates + i]);
  }
}

void CBEPot::hessianVectorImpl(const ForceInput &in, const double *direction,
                               double *product) const {
  if (in.nAtoms != kAtoms)
    throw std::invalid_argument("CBE requires six atoms");
  std::array<double, kCoordinates> x{}, v{}, hv{};
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = in.pos[i] / units::BOHR_TO_ANGSTROM;
    v[i] = direction[i] / units::BOHR_TO_ANGSTROM;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  require(potlibc_hessian_vector_product(handle_.get(), 1, x.data(), v.data(),
                                         hv.data()));
  for (std::size_t i = 0; i < hv.size(); ++i)
    product[i] = toForce(hv[i]);
}

int CBEPot::maxJetOrder() const noexcept { return POTLIBC_JET_MAX_ORDER; }

DerivativeReport CBEPot::gradientJet(std::size_t systems, std::size_t nAtoms,
                                     const double *positions, const int *,
                                     const double *, const double *directions,
                                     int order, double *jet) const {
  if (nAtoms != kAtoms)
    throw std::invalid_argument("CBE requires six atoms");
  if (order < 1 || order > POTLIBC_JET_MAX_ORDER)
    throw std::invalid_argument("CBE jet order out of range");
  const std::size_t values = systems * kCoordinates;
  std::vector<double> x(values), v(values);
  std::vector<double> derivatives(static_cast<std::size_t>(order) * values);
  for (std::size_t i = 0; i < values; ++i) {
    x[i] = positions[i] / units::BOHR_TO_ANGSTROM;
    v[i] = directions[i] / units::BOHR_TO_ANGSTROM;
  }
  potlibc_jet_report report{};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    require(potlibc_gradient_jet(handle_.get(), systems,
                                 static_cast<std::size_t>(order), x.data(),
                                 v.data(), derivatives.data(), &report));
  }
  for (std::size_t i = 0; i < derivatives.size(); ++i)
    jet[i] = toForce(derivatives[i]);
  return {.tracked = true,
          .operations = report.operations,
          .scale = toForce(report.scale)};
}

} // namespace rgpot
