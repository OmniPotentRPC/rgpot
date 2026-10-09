#include "rgpot/Water/WaterPots.hpp"

#include "rgpot/Water/spce_ccl.hpp"
#include "rgpot/Water/tip4p_ccl.hpp"
#include "rgpot/Water/zhu_philpott.hpp"

#include <stdexcept>

namespace rgpot {
namespace {

constexpr uint64_t kKernelVersion = 1;

uint64_t hashWater(uint64_t kind, const WaterConfig &config) {
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.u64(kind);
  fp.f64(config.cutoff);
  fp.f64(config.switching_width);
  return fp.h;
}

void prepare(const ForceInput &in, ForceOut *out) {
  const auto n = static_cast<long>(in.nAtoms);
  for (long k = 0; k < 3 * n; ++k) {
    out->F[k] = 0.0;
  }
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
}

void diagonalBox(const double *box, double diag[3]) {
  diag[0] = box[0];
  diag[1] = box[4];
  diag[2] = box[8];
}

} // namespace

struct TIP4PPot::Impl {
  explicit Impl(const WaterConfig &config)
      : kernel(config.cutoff, config.switching_width) {}
  forcefields::Tip4p kernel;
};

struct SPCEPot::Impl {
  explicit Impl(const WaterConfig &config)
      : kernel(config.cutoff, config.switching_width) {}
  forcefields::SpceCcl kernel;
};

struct TIP4PPtPot::Impl {
  explicit Impl(const WaterConfig &config)
      : kernel(config.cutoff, config.switching_width) {}
  forcefields::ZhuPhilpott<> kernel;
};

TIP4PPot::TIP4PPot() : TIP4PPot(WaterConfig{}) {}

TIP4PPot::TIP4PPot(const WaterConfig &config)
    : Potential(PotType::TIP4P), m_config(config),
      m_paramsKey(hashWater(1, config)), m_impl(new Impl(config)) {}

TIP4PPot::~TIP4PPot() = default;

void TIP4PPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const int n = static_cast<int>(in.nAtoms);
  prepare(in, out);
  if (n == 0) {
    return;
  }
  if (n % 3 != 0) {
    throw std::invalid_argument(
        "TIP4PPot expects atoms ordered as hydrogen pairs then oxygens");
  }
  double diag[3];
  diagonalBox(in.box, diag);
  double energy = 0.0;
  m_impl->kernel.computeHH_O_(n, in.pos, out->F, energy, diag);
  out->energy = energy;
}

SPCEPot::SPCEPot() : SPCEPot(WaterConfig{}) {}

SPCEPot::SPCEPot(const WaterConfig &config)
    : Potential(PotType::SPCE), m_config(config),
      m_paramsKey(hashWater(2, config)), m_impl(new Impl(config)) {}

SPCEPot::~SPCEPot() = default;

void SPCEPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const int n = static_cast<int>(in.nAtoms);
  prepare(in, out);
  if (n == 0) {
    return;
  }
  if (n % 3 != 0) {
    throw std::invalid_argument(
        "SPCEPot expects atoms ordered as hydrogen pairs then oxygens");
  }
  double diag[3];
  diagonalBox(in.box, diag);
  double energy = 0.0;
  m_impl->kernel.computeHH_O_(n, in.pos, out->F, energy, diag);
  out->energy = energy;
}

TIP4PPtPot::TIP4PPtPot() : TIP4PPtPot(WaterConfig{}) {}

TIP4PPtPot::TIP4PPtPot(const WaterConfig &config)
    : Potential(PotType::TIP4PPt), m_config(config),
      m_paramsKey(hashWater(3, config)), m_impl(new Impl(config)) {}

TIP4PPtPot::~TIP4PPtPot() = default;

void TIP4PPtPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const int n = static_cast<int>(in.nAtoms);
  prepare(in, out);
  if (n == 0) {
    return;
  }
  if (in.atmnrs == nullptr) {
    throw std::invalid_argument("TIP4PPtPot requires atomic numbers");
  }
  int hydrogens = 0;
  while (hydrogens < n && in.atmnrs[hydrogens] == 1) {
    hydrogens += 2;
  }
  if (hydrogens > n) {
    throw std::invalid_argument(
        "TIP4PPtPot expects hydrogens in pairs at the front");
  }
  const int nWater = hydrogens / 2;
  const int nPt = n - (hydrogens * 3) / 2;
  if (nPt < 0) {
    throw std::invalid_argument(
        "TIP4PPtPot atom count does not match the water and platinum split");
  }
  double diag[3];
  diagonalBox(in.box, diag);
  double energy = 0.0;
  m_impl->kernel.computeHH_O_Pt_(nWater, nPt, in.pos, out->F, energy, diag,
                                 nullptr);
  out->energy = energy;
}

} // namespace rgpot
