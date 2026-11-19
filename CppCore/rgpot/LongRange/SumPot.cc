// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/SumPot.hpp"

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Nlh.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <algorithm>
#include <stdexcept>

namespace rgpot::lr {

SumPot::SumPot() : Potential(PotType::LrSum) {}

void SumPot::add(std::unique_ptr<PotentialBase> term) {
  if (!term) {
    throw std::invalid_argument("SumPot cannot take a null term");
  }
  terms_.push_back(std::move(term));
}

void SumPot::shareNeighbours(std::shared_ptr<DualList> lists) {
  for (auto &term : terms_) {
    if (auto *c = dynamic_cast<CoulombPot *>(term.get())) {
      c->setNeighbours(lists);
    } else if (auto *d = dynamic_cast<DispersionPot *>(term.get())) {
      d->setNeighbours(lists);
    } else if (auto *n = dynamic_cast<NlhPot *>(term.get())) {
      n->setNeighbours(lists);
    }
  }
}

void SumPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::fill(out->F, out->F + 3 * in.nAtoms, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  std::fill(out->stress, out->stress + 9, 0.0);
  double total_stress[9] = {};
  bool complete_stress = in.compute_stress && !terms_.empty();
  for (const auto &term : terms_) {
    ForceOut part{};
    std::vector<double> f(3 * in.nAtoms, 0.0);
    part.F = f.data();
    term->forceBatch(ForceBatch{.nSystems = 1, .in = &in, .out = &part});
    out->energy += part.energy;
    out->variance += part.variance;
    for (std::size_t i = 0; i < f.size(); ++i) {
      out->F[i] += f[i];
    }
    if (in.compute_stress && part.has_stress) {
      for (int k = 0; k < 9; ++k) {
        total_stress[k] += part.stress[k];
      }
    } else if (in.compute_stress) {
      complete_stress = false;
    }
  }
  if (complete_stress) {
    std::copy(std::begin(total_stress), std::end(total_stress), out->stress);
    out->has_stress = 1;
  }
}

} // namespace rgpot::lr
