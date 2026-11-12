#include "rgpot/EAM/EAMPot.hpp"

#include "EAM.h"

namespace rgpot {

struct EAMPot::Impl {
  EAM kernel;
};

EAMPot::EAMPot() : EAMPot(EAMCellConfig{}) {}

EAMPot::EAMPot(const EAMCellConfig &)
    : Potential(PotType::EAMCell), m_impl(new Impl) {
  Fnv1a fp;
  fp.u64(1);
  m_paramsKey = fp.h;
}

EAMPot::~EAMPot() { delete m_impl; }

std::uint64_t EAMPot::listBuilds() const noexcept {
  return m_impl->kernel.listBuilds();
}

std::size_t EAMPot::neighborSlots() const noexcept {
  return m_impl->kernel.neighborSlots();
}

bool EAMPot::listReused() const noexcept { return m_impl->kernel.listReused(); }

std::size_t EAMPot::streamPartners() const noexcept {
  return m_impl->kernel.streamPartners();
}

int EAMPot::threadsUsed() const noexcept {
  return m_impl->kernel.threadsUsed();
}

void EAMPot::setNeighbourMode(nlist::Mode mode) noexcept {
  m_impl->kernel.setNeighbourMode(mode);
}

void EAMPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  double energy = 0.0;
  double variance = 0.0;
  m_impl->kernel.force(static_cast<long>(in.nAtoms), in.pos, in.atmnrs, out->F,
                       &energy, &variance, in.box);
  out->energy = energy;
  out->variance = variance;
  out->has_stress = 0;
}

} // namespace rgpot
