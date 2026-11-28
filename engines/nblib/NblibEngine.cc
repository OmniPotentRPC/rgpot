// MIT License
// Copyright 2023--present rgpot developers

#include "NblibEngine.hpp"

#include "rgpot/nlist/MinimumImage.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace rgpot::nblibpot {

namespace {

constexpr std::size_t kListedTypes =
    std::tuple_size<nblib::ListedInteractionData>::value;

// Slots of the energy array of GmxNBForceCalculatorCpu::compute, the order of
// gmx::NonBondedEnergyTerms for one energy group.
constexpr std::size_t kEnergyEpoch = 256;
constexpr std::size_t kNonbondedTerms = 5;
constexpr std::size_t kCoulombSR = 0;
constexpr std::size_t kLJSR = 1;

nblib::ParticleName pname(const MoleculeDesc &molecule, int index) {
  return nblib::ParticleName(molecule.atoms[static_cast<std::size_t>(index)].name);
}

void addListed(nblib::Molecule &molecule, const MoleculeDesc &description,
               const ListedDesc &term) {
  const auto &p = term.params;
  const auto at = [&](std::size_t k) { return pname(description, term.atoms[k]); };
  const auto cast = [](double value) { return static_cast<Real>(value); };
  switch (term.kind) {
  case ListedKind::HarmonicBond:
    molecule.addInteraction(
        at(0), at(1), nblib::HarmonicBondType(cast(p[0]), cast(p[1])));
    break;
  case ListedKind::G96Bond:
    molecule.addInteraction(at(0), at(1),
                            nblib::G96BondType(cast(p[0]), cast(p[1])));
    break;
  case ListedKind::MorseBond:
    molecule.addInteraction(
        at(0), at(1), nblib::MorseBondType(cast(p[0]), cast(p[1]), cast(p[2])));
    break;
  case ListedKind::Pair14:
    molecule.addInteraction(
        at(0), at(1),
        nblib::PairLJType(nblib::C6(cast(p[0])), nblib::C12(cast(p[1]))));
    break;
  case ListedKind::HarmonicAngle:
    molecule.addInteraction(
        at(0), at(1), at(2),
        nblib::HarmonicAngle(cast(p[0]), nblib::Degrees(cast(p[1]))));
    break;
  case ListedKind::G96Angle:
    molecule.addInteraction(
        at(0), at(1), at(2),
        nblib::G96Angle(cast(p[0]), nblib::Degrees(cast(p[1]))));
    break;
  case ListedKind::ProperDihedral:
    molecule.addInteraction(
        at(0), at(1), at(2), at(3),
        nblib::ProperDihedral(nblib::Degrees(cast(p[0])), cast(p[1]),
                              static_cast<int>(p[2])));
    break;
  case ListedKind::ImproperDihedral:
    molecule.addInteraction(
        at(0), at(1), at(2), at(3),
        nblib::ImproperDihedral(nblib::Degrees(cast(p[0])), cast(p[1])));
    break;
  case ListedKind::RyckaertBelleman:
    molecule.addInteraction(
        at(0), at(1), at(2), at(3),
        nblib::RyckaertBellemanDihedral(cast(p[0]), cast(p[1]), cast(p[2]),
                                        cast(p[3]), cast(p[4]), cast(p[5])));
    break;
  }
}

nblib::Topology buildTopology(const TopologyDesc &description) {
  std::map<std::string, nblib::ParticleType> particleTypes;
  nblib::ParticleTypesInteractions interactions(nblib::CombinationRule::Geometric);
  for (const AtomTypeDesc &type : description.types) {
    particleTypes.emplace(
        type.name, nblib::ParticleType(nblib::ParticleTypeName(type.name),
                                       nblib::Mass(static_cast<Real>(type.mass))));
    interactions.add(nblib::ParticleTypeName(type.name),
                     nblib::C6(static_cast<Real>(type.c6)),
                     nblib::C12(static_cast<Real>(type.c12)));
  }
  for (const PairDesc &pair : description.pairs) {
    interactions.add(nblib::ParticleTypeName(pair.a),
                     nblib::ParticleTypeName(pair.b),
                     nblib::C6(static_cast<Real>(pair.c6)),
                     nblib::C12(static_cast<Real>(pair.c12)));
  }
  nblib::TopologyBuilder builder;
  const bool charged = description.coulomb == CoulombMode::Cutoff;
  for (const MoleculeDesc &entry : description.molecules) {
    nblib::Molecule molecule{nblib::MoleculeName(entry.name)};
    for (const AtomDesc &atom : entry.atoms) {
      const Real charge = charged ? static_cast<Real>(atom.charge) : Real(0);
      molecule.addParticle(nblib::ParticleName(atom.name),
                           nblib::Charge(charge), particleTypes.at(atom.type));
    }
    // nblib stores an exclusion symmetrically; a pair given twice is a
    // duplicate to it.
    std::set<std::array<int, 2>> excluded;
    for (const auto &pair : entry.exclusions) {
      const std::array<int, 2> key{std::min(pair[0], pair[1]),
                                   std::max(pair[0], pair[1])};
      if (excluded.insert(key).second) {
        molecule.addExclusion(pname(entry, key[0]), pname(entry, key[1]));
      }
    }
    for (const ListedDesc &term : entry.listed) {
      addListed(molecule, entry, term);
    }
    builder.addMolecule(molecule, entry.count);
  }
  builder.addParticleTypesInteractions(interactions);
  return builder.buildTopology();
}

nblib::NBKernelOptions kernelOptions(const TopologyDesc &description) {
  nblib::NBKernelOptions options;
  options.useGpu = false;
  options.numOpenMPThreads = 1;
  options.nbnxmSimd = nblib::SimdKernels::SimdNo;
  options.pairlistCutoff = static_cast<Real>(description.cutoff);
  options.coulombType = nblib::CoulombType::Cutoff;
  return options;
}

} // namespace

nblib::Box Engine::makeBox(const std::array<double, 3> &edges) {
  return nblib::Box(static_cast<Real>(edges[0]), static_cast<Real>(edges[1]),
                    static_cast<Real>(edges[2]));
}

Engine::Engine(TopologyDesc description)
    : description_(std::move(description)), atoms_(description_.atomCount()),
      instances_(description_.instances()) {
  topology_ = std::make_unique<nblib::Topology>(buildTopology(description_));
  if (static_cast<std::size_t>(topology_->numParticles()) != atoms_) {
    throw std::logic_error("nblib built a topology of another size");
  }
  masses_.reserve(atoms_);
  for (const AtomTypeDesc *type : description_.atomTypes()) {
    masses_.push_back(type->mass);
  }
  nonbonded_ = nblib::setupGmxForceCalculatorCpu(*topology_,
                                                 kernelOptions(description_));
  wrapped_.resize(atoms_);
  nonbondedForces_.resize(atoms_);
  listedForces_.resize(atoms_);
}

Engine::~Engine() = default;

void Engine::rebuildListed(const std::array<double, 3> &box) {
  if (listed_ != nullptr && box == listedBox_) {
    return;
  }
  using ImproperData = nblib::ListedTypeData<nblib::ImproperDihedral>;
  nblib::ListedInteractionData rest = topology_->getInteractionData();
  nblib::ListedInteractionData only;
  std::get<ImproperData>(only) = std::get<ImproperData>(rest);
  std::get<ImproperData>(rest) = ImproperData{};
  listed_ = std::make_unique<nblib::ListedForceCalculator>(rest, atoms_, 1,
                                                           makeBox(box));
  impropers_.reset();
  if (!std::get<ImproperData>(only).indices.empty()) {
    impropers_ = std::make_unique<nblib::ListedForceCalculator>(
        only, atoms_, 1, makeBox(box));
  }
  listedBox_ = box;
}

void Engine::evaluate(const double *positions, const std::array<double, 3> &box,
                      bool wantVirial, Evaluation &out) {
  const double shortest = std::min({box[0], box[1], box[2]});
  if (!(shortest > 2.0 * description_.cutoff)) {
    throw std::invalid_argument(
        "nblib needs every box edge longer than twice the cutoff");
  }
  const nblib::Box nbox = makeBox(box);
  rebuildListed(box);

  for (std::size_t i = 0; i < atoms_; ++i) {
    wrapped_[i] = nblib::Vec3(static_cast<Real>(positions[3 * i]),
                              static_cast<Real>(positions[3 * i + 1]),
                              static_cast<Real>(positions[3 * i + 2]));
  }
  if (epochCalls_ >= kEnergyEpoch) {
    nonbonded_ = nblib::setupGmxForceCalculatorCpu(*topology_,
                                                   kernelOptions(description_));
    accumulated_.fill(0.0);
    epochCalls_ = 0;
  }
  nonbonded_->updatePairlist(wrapped_, nbox);

  std::fill(nonbondedForces_.begin(), nonbondedForces_.end(),
            nblib::Vec3(0, 0, 0));
  std::vector<Real> nonbondedVirial;
  if (wantVirial) {
    nonbondedVirial.assign(9, Real(0));
  }
  std::vector<Real> nonbondedEnergy(kNonbondedTerms, Real(0));
  nonbonded_->compute(wrapped_, nbox, nonbondedForces_, nonbondedVirial,
                      nonbondedEnergy);
  ++epochCalls_;
  for (std::size_t term = 0; term < kNonbondedTerms; ++term) {
    const double total = nonbondedEnergy[term];
    nonbondedEnergy[term] = static_cast<Real>(total - accumulated_[term]);
    accumulated_[term] = total;
  }

  std::fill(listedForces_.begin(), listedForces_.end(), nblib::Vec3(0, 0, 0));
  std::vector<Real> listedEnergy(kListedTypes, Real(0));
  listed_->compute(wrapped_, listedForces_, listedEnergy, true);
  if (impropers_ != nullptr) {
    improperForces_.assign(atoms_, nblib::Vec3(0, 0, 0));
    std::vector<Real> improperEnergy(kListedTypes, Real(0));
    impropers_->compute(wrapped_, improperForces_, improperEnergy, true);
    for (std::size_t i = 0; i < atoms_; ++i) {
      listedForces_[i] -= improperForces_[i];
    }
    for (std::size_t term = 0; term < kListedTypes; ++term) {
      listedEnergy[term] += improperEnergy[term];
    }
  }

  out.energy = nonbondedEnergy[kCoulombSR] + nonbondedEnergy[kLJSR];
  for (const Real term : listedEnergy) {
    out.energy += term;
  }
  out.forces.resize(3 * atoms_);
  for (std::size_t i = 0; i < atoms_; ++i) {
    for (std::size_t d = 0; d < 3; ++d) {
      out.forces[3 * i + d] = nonbondedForces_[i][d] + listedForces_[i][d];
    }
  }

  out.virial.fill(0.0);
  if (!wantVirial) {
    return;
  }
  for (std::size_t k = 0; k < 9; ++k) {
    out.virial[k] = nonbondedVirial[k];
  }
  // The listed forces sum to zero on each molecule, so their virial does not
  // depend on the image of the molecule as long as it is whole. Each atom is
  // placed relative to the one before it, which keeps a chain whole when it
  // is longer than half the cell.
  const std::array<double, 9> cell{box[0], 0.0, 0.0, 0.0, box[1],
                                   0.0,    0.0, 0.0, box[2]};
  const rgpot::nlist::MinimumImage image(cell.data(), {true, true, true});
  for (const auto &instance : instances_) {
    const std::size_t first = instance[0];
    double r[3] = {positions[3 * first], positions[3 * first + 1],
                   positions[3 * first + 2]};
    for (std::size_t k = 0; k < instance[1]; ++k) {
      if (k > 0) {
        double d[3];
        for (std::size_t a = 0; a < 3; ++a) {
          d[a] = positions[3 * (first + k) + a] - positions[3 * (first + k - 1) + a];
        }
        image.fold(d[0], d[1], d[2]);
        for (std::size_t a = 0; a < 3; ++a) {
          r[a] += d[a];
        }
      }
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          out.virial[3 * a + b] -= 0.5 * r[a] * listedForces_[first + k][b];
        }
      }
    }
  }
}

} // namespace rgpot::nblibpot
