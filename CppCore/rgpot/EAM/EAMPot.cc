#include "rgpot/EAM/EAMPot.hpp"

#include "EAM.h"
#include "rgpot/stress.hpp"

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
  // The kernel reads only the diagonal of the cell, so a force call on any
  // cell runs the same arithmetic as before stress existed. Stress is the
  // strain derivative of that model, which is the physical one only when
  // the cell is orthorhombic; any other cell leaves has_stress at zero.
  StressAccumulator stress(in, out);
  const bool orthorhombic =
      in.box != nullptr && in.box[1] == 0.0 && in.box[2] == 0.0 &&
      in.box[3] == 0.0 && in.box[5] == 0.0 && in.box[6] == 0.0 &&
      in.box[7] == 0.0;
  double energy = 0.0;
  double variance = 0.0;
  m_impl->kernel.force(static_cast<long>(in.nAtoms), in.pos, in.atmnrs, out->F,
                       &energy, &variance, in.box,
                       orthorhombic ? stress.data() : nullptr);
  out->energy = energy;
  out->variance = variance;
  if (orthorhombic) {
    stress.publish(in.box);
  }
}

} // namespace rgpot
