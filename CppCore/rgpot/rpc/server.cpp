// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the standalone Cap'n Proto potential server.
 *
 * This file implements a basic RPC server which exposes toy potentials over a
 * network interface. It utilizes the @c EzRpcServer for handling requests.
 */

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/debug.h>

#ifdef RGPOT_POTSERV_MPI
#include "tools/MpiPefSession.hpp"
#endif

namespace rgpot::tools {
class MpiPefSession;
}

#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif // RGPOT_HAS_FORTRAN_POTS

#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif // RGPOT_HAS_XTB

#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif // RGPOT_HAS_TBLITE

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif // RGPOT_HAS_DFTD3

#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#include "rgpot/rpc/D4Configure.hpp"
#endif // RGPOT_HAS_DFTD4

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#endif // RGPOT_HAS_METATOMIC

#ifdef RGPOT_HAS_UMA
#include "rgpot/UmaPot/UmaPot.hpp"
#endif
#ifdef RGPOT_HAS_SKALA
#include "rgpot/SkalaPot/SkalaPot.hpp"
#endif // RGPOT_HAS_UMA

#ifdef RGPOT_HAS_CP2K
#include "rgpot/CP2KPot/CP2KPot.hpp"
#endif
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#ifdef RGPOT_HAS_CBE
#include "rgpot/CBEPot/CBEPot.hpp"
#endif
#include "rgpot/abi/Handshake.hpp"
#ifdef RGPOT_HAS_IPI
#include "rgpot/IPIPot/IPIPot.hpp"
#endif
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"
#include "rgpot/rpc/CompositeConfigure.hpp"
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/MullerBrownConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"
#include "rgpot/units.hpp"

#ifdef RGPOT_HAS_DFTD4
#define RGPOT_OWN_ARM_TAIL "D2, D2Cpmd and D4"
#else
#define RGPOT_OWN_ARM_TAIL "D2 and D2Cpmd"
#endif
#ifdef RGPOT_HAS_CP2K
constexpr const char *kOwnArmBackends =
    "NWChem, CPMD, CP2K, Morse, Composite, MullerBrown, " RGPOT_OWN_ARM_TAIL;
#else
constexpr const char *kOwnArmBackends =
    "NWChem, CPMD, Morse, Composite, MullerBrown, " RGPOT_OWN_ARM_TAIL;
#endif
#undef RGPOT_OWN_ARM_TAIL
#ifdef RGPOT_POTSERV_MPI
#ifdef RGPOT_HAS_CP2K
constexpr const char *kCollectiveBackends = "NWChem, CPMD or CP2K";
#else
constexpr const char *kCollectiveBackends = "NWChem or CPMD";
#endif
#endif

namespace {

constexpr std::uint32_t kDefaultStreamLimit = 4096;

[[nodiscard]] double potentialCutoffAngstrom(const rgpot::PotentialBase &pot) noexcept {
  if (const auto *lj = dynamic_cast<const rgpot::LJPot *>(&pot))
    return lj->locality().cutoff;
  if (const auto *cluster = dynamic_cast<const rgpot::LJClusterPot *>(&pot))
    return cluster->config().cutoff > 0.0 ? cluster->config().cutoff : 0.0;
  if (const auto *morse = dynamic_cast<const rgpot::MorsePot *>(&pot))
    return morse->locality().cutoff;
  if (const auto *zbl = dynamic_cast<const rgpot::ZBLPot *>(&pot))
    return zbl->config().cut_global > 0.0 ? zbl->config().cut_global : 0.0;
#ifdef RGPOT_HAS_FORTRAN_POTS
  if (dynamic_cast<const rgpot::fortranpots::CuH2Pot *>(&pot) != nullptr)
    return 6.1;
  if (dynamic_cast<const rgpot::fortranpots::EAMAlPot *>(&pot) != nullptr)
    return 7.1;
  if (dynamic_cast<const rgpot::fortranpots::SWPot *>(&pot) != nullptr)
    return 1.8 * 2.0951;
  if (dynamic_cast<const rgpot::fortranpots::EDIPPot *>(&pot) != nullptr)
    return 3.1213820;
  if (dynamic_cast<const rgpot::fortranpots::LenoskyPot *>(&pot) != nullptr)
    return 4.5;
  if (dynamic_cast<const rgpot::fortranpots::TersoffPot *>(&pot) != nullptr)
    return 3.0;
  if (dynamic_cast<const rgpot::fortranpots::FeHePot *>(&pot) != nullptr)
    return 5.4;
#endif
  return 0.0;
}

void setNeighborError(::NeighborList::Builder out, rgpot::nlist::Code code,
                      const std::string &message) {
  out.setStatus(static_cast<std::int32_t>(code));
  out.setMessage(message);
  out.setMore(false);
  out.setNext(0);
}

void answerNeighborRequest(const rgpot::PotentialBase &pot,
                           const ::ForceInput::Reader &fip,
                           const ::NeighborRequest::Reader &request,
                           ::NeighborList::Builder out) {
  const auto mode = request.getMode();
  if (mode != 1 && mode != 2) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "mode must be 1 (stored) or 2 (stream)");
    return;
  }
  const auto format = request.getFormat();
  if (format > 2) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "format must be 0 (half pairs), 1 (full pairs) or 2 (cluster of four)");
    return;
  }
  const auto cut = request.getCut();
  if (cut > 1) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "cut must be 0 (strict) or 1 (loose)");
    return;
  }
  const auto periodic_list = request.getPeriodic();
  if (periodic_list.size() != 0 && periodic_list.size() != 3) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "periodic must be empty or hold 3 flags");
    return;
  }

  double len_to_angstrom = 1.0;
  double angstrom_to_caller = 1.0;
  try {
    const std::string length_unit = fip.getLengthUnit().cStr();
    len_to_angstrom = rgpot::units::unit_conversion_factor(length_unit, "angstrom");
    angstrom_to_caller = rgpot::units::unit_conversion_factor("angstrom", length_unit);
  } catch (const std::exception &err) {
    setNeighborError(out, rgpot::nlist::Code::Cutoff, err.what());
    return;
  }

  double cutoff = request.getCutoff();
  const double skin = request.getSkin();
  if (!std::isfinite(cutoff) || cutoff < 0.0 || !std::isfinite(skin) || skin < 0.0) {
    setNeighborError(out, rgpot::nlist::Code::Cutoff,
                     rgpot_nlist_strerror(RGPOT_NLIST_CUTOFF));
    return;
  }
  if (cutoff == 0.0) {
    const double fallback_a = potentialCutoffAngstrom(pot);
    if (fallback_a > 0.0)
      cutoff = fallback_a * angstrom_to_caller;
  }

  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = skin;
  if (format == 2) {
    query.shape = rgpot::nlist::Cluster4{};
  } else {
    query.shape = rgpot::nlist::Pairs{format == 1 ? rgpot::nlist::Shell::Full
                                                  : rgpot::nlist::Shell::Half};
  }
  query.cut = cut == 0 ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
  if (periodic_list.size() == 3) {
    query.periodic = {periodic_list[0], periodic_list[1], periodic_list[2]};
  } else if (!pot.caps().periodic) {
    query.periodic = {false, false, false};
  } else {
    query.periodic = {true, true, true};
  }
  query.fields.shifts = request.getShifts();
  query.fields.vectors = request.getVectors();
  query.fields.distances = request.getDistances();

  const auto pos_reader = fip.getPos();
  if (pos_reader.size() % 3 != 0) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "positions length must be a multiple of 3");
    return;
  }
  const std::size_t n_atoms = pos_reader.size() / 3;
  const auto atm_reader = fip.getAtmnrs();
  if (atm_reader.size() != 0 && atm_reader.size() != n_atoms) {
    setNeighborError(out, rgpot::nlist::Code::Shape,
                     "atomic numbers size mismatch");
    return;
  }
  std::vector<double> positions(pos_reader.size());
  for (std::size_t i = 0; i < pos_reader.size(); ++i) {
    positions[i] = pos_reader[static_cast<unsigned>(i)];
  }

  rgpot::nlist::Table table;
  std::vector<std::int32_t> system_col;
  try {
    const auto systems_reader = request.getSystems();
    if (systems_reader.size() > 0) {
      std::vector<std::int32_t> counts(systems_reader.size());
      std::vector<double> boxes(systems_reader.size() * 9, 0.0);
      std::size_t total_atoms = 0;
      for (unsigned s = 0; s < systems_reader.size(); ++s) {
        const auto sys_entry = systems_reader[s];
        counts[s] = static_cast<std::int32_t>(sys_entry.getCount());
        total_atoms += static_cast<std::size_t>(sys_entry.getCount());
        const auto sys_box = sys_entry.getBox();
        if (sys_box.size() != 9) {
          setNeighborError(out, rgpot::nlist::Code::Box,
                           "each system box must have 9 elements");
          return;
        }
        for (unsigned k = 0; k < 9; ++k) {
          boxes[static_cast<std::size_t>(s) * 9 + k] = sys_box[k];
        }
      }
      if (total_atoms != n_atoms) {
        setNeighborError(out, rgpot::nlist::Code::Shape,
                         "systems counts do not sum to the number of positions");
        return;
      }
      rgpot::nlist::Systems systems{positions.data(), counts.data(),
                                    systems_reader.size(), boxes.data()};
      rgpot::nlist::Batch batch;
      rgpot::nlist::build(batch, systems, query);
      table = std::move(batch.pairs);
      system_col = std::move(batch.system);
    } else {
      const auto box_reader = fip.getBox();
      if (box_reader.size() != 9) {
        setNeighborError(out, rgpot::nlist::Code::Box,
                         "box must have 9 elements");
        return;
      }
      double box[9]{};
      for (unsigned k = 0; k < 9; ++k) {
        box[k] = box_reader[k];
      }
      rgpot::nlist::build(table, positions.data(), n_atoms, box, query);
    }
  } catch (const rgpot::nlist::Error &err) {
    setNeighborError(out, err.code(), err.what());
    return;
  } catch (const std::bad_alloc &) {
    setNeighborError(out, rgpot::nlist::Code::Nomem,
                     rgpot_nlist_strerror(RGPOT_NLIST_NOMEM));
    return;
  } catch (const std::exception &err) {
    setNeighborError(out, rgpot::nlist::Code::Shape, err.what());
    return;
  }

  (void)len_to_angstrom;
  const std::size_t total = table.size();
  std::size_t start = 0;
  std::size_t count = total;
  bool more = false;
  std::uint64_t next = total;
  if (mode == 2) {
    const std::uint64_t cursor = request.getCursor();
    if (cursor > total) {
      setNeighborError(out, rgpot::nlist::Code::Short,
                       "cursor is past the end of the neighbour list");
      return;
    }
    start = static_cast<std::size_t>(cursor);
    const std::uint32_t raw_limit = request.getLimit();
    const std::size_t limit = raw_limit == 0 ? kDefaultStreamLimit : raw_limit;
    count = std::min(limit, total - start);
    next = cursor + count;
    more = next < total;
  }

  auto first_out = out.initFirst(static_cast<unsigned>(count));
  auto second_out = out.initSecond(static_cast<unsigned>(count));
  auto mask_out = out.initMask(static_cast<unsigned>(count));
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t src = start + i;
    first_out.set(static_cast<unsigned>(i), table.first[src]);
    second_out.set(static_cast<unsigned>(i), table.second[src]);
    mask_out.set(static_cast<unsigned>(i), table.mask[src]);
  }
  if (query.fields.shifts) {
    auto shift_out = out.initShift(static_cast<unsigned>(count * 3));
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t src = start + i;
      shift_out.set(static_cast<unsigned>(3 * i), table.shift[3 * src]);
      shift_out.set(static_cast<unsigned>(3 * i + 1), table.shift[3 * src + 1]);
      shift_out.set(static_cast<unsigned>(3 * i + 2), table.shift[3 * src + 2]);
    }
  }
  if (query.fields.vectors) {
    auto vec_out = out.initVector(static_cast<unsigned>(count * 3));
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t src = start + i;
      vec_out.set(static_cast<unsigned>(3 * i), table.vector[3 * src]);
      vec_out.set(static_cast<unsigned>(3 * i + 1), table.vector[3 * src + 1]);
      vec_out.set(static_cast<unsigned>(3 * i + 2), table.vector[3 * src + 2]);
    }
  }
  if (query.fields.distances) {
    auto dist_out = out.initDistance(static_cast<unsigned>(count));
    for (std::size_t i = 0; i < count; ++i) {
      dist_out.set(static_cast<unsigned>(i), table.distance[start + i]);
    }
  }
  if (!system_col.empty()) {
    auto sys_out = out.initSystem(static_cast<unsigned>(count));
    for (std::size_t i = 0; i < count; ++i) {
      sys_out.set(static_cast<unsigned>(i), system_col[start + i]);
    }
  }
  out.setMore(more);
  out.setNext(next);
  out.setStatus(0);
  out.setMessage("ok");
}

} // namespace

/**
 * @class GenericPotImpl
 * @brief Server implementation for the Potential RPC interface.
 *
 * This class wraps a polymorphic @c PotentialBase instance and dispatches
 * RPC calculate requests to the underlying physics engine.
 */
class GenericPotImpl final : public Potential::Server {
private:
  std::unique_ptr<rgpot::PotentialBase>
      m_potential; //!< The polymorphic potential engine.
  /// Optional typed handle when backend is NWChem (for configure()).
  rgpot::NWChemPot *m_nwchem = nullptr;
  /// Optional typed handle when backend is CPMD (for configure()).
  rgpot::CPMDPot *m_cpmd = nullptr;
  /// Set when the backend is Morse; configure() replaces the pot it names.
  rgpot::MorsePot *m_morse = nullptr;
  /// Set when the backend is Composite; configure() replaces the sum.
  rgpot::SumPot *m_sum = nullptr;
  /// Set when the backend is D2; configure() replaces the pot.
  rgpot::D2Pot *m_d2 = nullptr;
  /// Set when the backend is D2Cpmd; configure() replaces the pot.
  rgpot::D2CpmdPot *m_d2cpmd = nullptr;
  /// Set when the backend is Muller-Brown; configure() replaces the surface.
  rgpot::MullerBrownPot *m_muller_brown = nullptr;
#ifdef RGPOT_HAS_DFTD4
  /// Set when the backend is D4; configure() replaces the pot.
  rgpot::D4Pot *m_d4 = nullptr;
#endif
#ifdef RGPOT_HAS_IPI
  /// Set when the backend is the i-PI socket server. calculate copies the
  /// Cauchy stress stashed by the last force call.
  rgpot::IPIPot *m_ipi = nullptr;
#endif
#ifdef RGPOT_HAS_CP2K
  /// Optional typed handle when backend is CP2K (for configure()).
  rgpot::CP2KPot *m_cp2k = nullptr;
#endif
  rgpot::tools::MpiPefSession *m_pef = nullptr;

public:
  /**
   * @brief Constructor for GenericPotImpl.
   * @param pot Ownership of a PotentialBase derived object.
   */
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)),
        m_morse(dynamic_cast<rgpot::MorsePot *>(m_potential.get())),
        m_sum(dynamic_cast<rgpot::SumPot *>(m_potential.get())),
        m_d2(dynamic_cast<rgpot::D2Pot *>(m_potential.get())),
        m_d2cpmd(dynamic_cast<rgpot::D2CpmdPot *>(m_potential.get())),
        m_muller_brown(dynamic_cast<rgpot::MullerBrownPot *>(m_potential.get()))
#ifdef RGPOT_HAS_DFTD4
        ,
        m_d4(dynamic_cast<rgpot::D4Pot *>(m_potential.get()))
#endif
#ifdef RGPOT_HAS_IPI
        ,
        m_ipi(dynamic_cast<rgpot::IPIPot *>(m_potential.get()))
#endif
  {
  }

  /**
   * @brief Constructor retaining an NWChemPot pointer for configure().
   * Members initialize in declaration order (m_potential then m_nwchem).
   */
  GenericPotImpl(std::unique_ptr<rgpot::NWChemPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_nwchem(static_cast<rgpot::NWChemPot *>(m_potential.get())), m_pef(pef) {}

  GenericPotImpl(std::unique_ptr<rgpot::CPMDPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_cpmd(static_cast<rgpot::CPMDPot *>(m_potential.get())), m_pef(pef) {}

#ifdef RGPOT_HAS_CP2K
  GenericPotImpl(std::unique_ptr<rgpot::CP2KPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_cp2k(static_cast<rgpot::CP2KPot *>(m_potential.get())), m_pef(pef) {}
#endif

private:
  template <typename Context>
  kj::Promise<void> calculateImpl(Context context, bool requireStress) {
    const bool stressCapable = m_potential->caps().stress;
    KJ_REQUIRE(!requireStress || stressCapable,
               "the selected potential does not provide stress");

    auto fip = context.getParams().getFip();
    const size_t numAtoms = fip.getPos().size() / 3;

    KJ_REQUIRE(fip.getAtmnrs().size() == numAtoms, "AtomNumbers size mismatch");

    // Unit conversion factors (caller units -> internal angstrom/eV)
    std::string lengthUnit = fip.getLengthUnit();
    std::string energyUnit = fip.getEnergyUnit();
    double len_to_angstrom =
        rgpot::units::unit_conversion_factor(lengthUnit, "angstrom");
    double ev_to_caller =
        rgpot::units::unit_conversion_factor("eV", energyUnit);
    // F_caller = dE_caller/dx_caller
    //          = (d(E*E_eV)/d(x_A/len_to_angstrom))
    //          = ev_to_caller * len_to_angstrom * F_A.
    double force_to_caller = ev_to_caller * len_to_angstrom;

    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    if (len_to_angstrom != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        nativePositions.data()[i] *= len_to_angstrom;
      }
    }

    std::vector<int> nativeAtomTypes =
        rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(
            fip.getAtmnrs());
    std::array<std::array<double, 3>, 3> nativeBoxMatrix =
        rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(fip.getBox());
    if (len_to_angstrom != 1.0) {
      for (auto &row : nativeBoxMatrix)
        for (auto &val : row)
          val *= len_to_angstrom;
    }

    const std::optional<std::int64_t> orbitalKey =
        fip.getHasOrbitalKey()
            ? std::optional<std::int64_t>(fip.getOrbitalKey())
            : std::nullopt;

    // Potential always computes in eV/angstrom.
    std::vector<double> nativeForces(numAtoms * 3, 0.0);
    rgpot::ForceInput in_native{numAtoms, nativePositions.data(),
                                nativeAtomTypes.data(), nativeBoxMatrix[0].data()};
    rgpot::ForceOut out_native{};
    out_native.F = nativeForces.data();
    auto evaluate = [&]() -> rgpot::ForceOut {
#ifdef RGPOT_POTSERV_MPI
      if (m_pef) {
        try {
          auto [energy, forces, variance] =
              m_pef->calculate(nativePositions, nativeAtomTypes, nativeBoxMatrix,
                               orbitalKey);
          rgpot::ForceOut out{};
          out.F = nativeForces.data();
          out.energy = energy;
          out.variance = variance;
          std::copy(forces.data(), forces.data() + (3 * numAtoms), out.F);
          return out;
        } catch (const std::exception &error) {
          throw KJ_EXCEPTION(FAILED, error.what());
        }
      }
#endif
      if (orbitalKey) {
        if (m_cpmd) {
          m_cpmd->selectOrbitals(*orbitalKey);
#ifdef RGPOT_HAS_CP2K
        } else if (m_cp2k) {
          m_cp2k->selectOrbitals(*orbitalKey);
#endif
        } else {
          throw KJ_EXCEPTION(
              FAILED,
              "orbitalKey is supported only by CPMD and CP2K backends");
        }
      }
      m_potential->forceBatch(
          rgpot::ForceBatch{.nSystems = 1, .in = &in_native, .out = &out_native});
      return out_native;
    };
    out_native = evaluate();

    auto pres = context.getResults().initResult();
    pres.setEnergy(out_native.energy * ev_to_caller);

    auto forcesList = pres.initForces(numAtoms * 3);
    if (force_to_caller != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        nativeForces[i] *= force_to_caller;
      }
    }
    for (size_t i = 0; i < numAtoms * 3; ++i) {
      forcesList.set(i, nativeForces[i]);
    }

    bool hasStress = out_native.has_stress != 0;
    double stress[9] = {};
    if (hasStress) {
      std::copy(out_native.stress, out_native.stress + 9, stress);
    }
#ifdef RGPOT_HAS_IPI
    if (!hasStress && m_ipi != nullptr) {
      hasStress = m_ipi->copyStress(stress);
    }
#endif
    if (hasStress) {
      // sigma_caller = sigma_eV/A^3 * (eV -> caller) *
      //                (A -> caller length)^3
      const double stress_to_caller =
          ev_to_caller * len_to_angstrom * len_to_angstrom * len_to_angstrom;
      auto stressList = pres.initStress(9);
      for (unsigned int i = 0; i < 9; ++i) {
        stressList.set(i, stress[i] * stress_to_caller);
      }
    }

    KJ_REQUIRE(!requireStress || hasStress,
               "the selected potential returned no stress");

    if (fip.hasNeighbors()) {
      answerNeighborRequest(*m_potential, fip, fip.getNeighbors(),
                            pres.initNeighborList());
    }

    return kj::READY_NOW;
  }

public:
  /**
   * @brief Reports what this server speaks, so a client can refuse an
   * incompatible peer before the first calculate call.
   */
  kj::Promise<void> getCapabilities(GetCapabilitiesContext context) override {
    auto caps = context.getResults().initCapabilities();
    caps.setBackendName("rgpot");
    caps.setBackendVersion(RGPOT_BUILD_VERSION);
    caps.setAvailable(true);
    const bool hasHvp = m_potential->caps().analyticHvp;
    const bool hasStress = m_potential->caps().stress;
    auto ops = caps.initOperations(3 + (hasHvp ? 1u : 0u) + (hasStress ? 1u : 0u));
    unsigned int opIdx = 0;
    ops.set(opIdx++, ::Capabilities::Operation::ENERGY);
    ops.set(opIdx++, ::Capabilities::Operation::FORCES);
    if (hasHvp) {
      ops.set(opIdx++, ::Capabilities::Operation::HESSIAN_VECTOR);
    }
    if (hasStress) {
      ops.set(opIdx++, ::Capabilities::Operation::STRESS);
    }
    ops.set(opIdx++, ::Capabilities::Operation::NEIGHBORS);
    rgpot::abi::fill_compatibility(caps);
    return kj::READY_NOW;
  }

  /**
   * @details
   * This method performs the following translation steps:
   * 1. Extracts the @c ForceInput (fip) from the RPC context.
   * 2. Validates the size of the atomic number list.
   * 3. Converts Cap'n Proto lists into native @c AtomMatrix and @c std::vector
   * types.
   * 4. Executes the calculation via the @c PotentialBase virtual operator.
   * 5. Populates the @c PotentialResult with energy and force data.
   *
   * @param context The Cap'n Proto RPC call context.
   * @return An asynchronous promise for completion.
   */
  kj::Promise<void> calculate(CalculateContext context) override {
    return calculateImpl(kj::mv(context), false);
  }

  kj::Promise<void> calculateStress(CalculateStressContext context) override {
    return calculateImpl(kj::mv(context), true);
  }

  kj::Promise<void> neighbours(NeighboursContext context) override {
    auto params = context.getParams();
    answerNeighborRequest(*m_potential, params.getFip(), params.getRequest(),
                          context.getResults().initNeighbors());
    return kj::READY_NOW;
  }

  kj::Promise<void> calculateHvp(CalculateHvpContext context) override {
    auto params = context.getParams();
    auto fip = params.getFip();
    auto direction = params.getDirection();
    const std::size_t n = fip.getPos().size() / 3;
    KJ_REQUIRE(fip.getPos().size() == 3 * n, "position size mismatch");
    KJ_REQUIRE(fip.getAtmnrs().size() == n, "atomic number size mismatch");
    KJ_REQUIRE(fip.getBox().size() == 9, "box size mismatch");
    KJ_REQUIRE(direction.size() == 3 * n, "direction size mismatch");
    KJ_REQUIRE(m_potential->caps().analyticHvp,
               "potential has no analytic Hessian-vector product");
    const double lengthToAngstrom = rgpot::units::unit_conversion_factor(
        std::string(fip.getLengthUnit()), "angstrom");
    const double evToCaller = rgpot::units::unit_conversion_factor(
        "eV", std::string(fip.getEnergyUnit()));
    std::vector<double> positions(3 * n), vector(3 * n), product(3 * n);
    for (std::size_t i = 0; i < 3 * n; ++i) {
      positions[i] = fip.getPos()[i] * lengthToAngstrom;
      vector[i] = direction[i] * lengthToAngstrom;
    }
    std::vector<int> numbers(n);
    for (std::size_t i = 0; i < n; ++i)
      numbers[i] = fip.getAtmnrs()[i];
    std::array<double, 9> box{};
    for (std::size_t i = 0; i < 9; ++i)
      box[i] = fip.getBox()[i] * lengthToAngstrom;
    rgpot::ForceInput input{n, positions.data(), numbers.data(), box.data()};
    m_potential->hessianVectorImpl(input, vector.data(), product.data());
    auto output = context.getResults().initProduct(3 * n);
    // HVP follows the same chain rule as force once direction has been
    // converted: (H v)_caller = evToCaller * lengthToAngstrom * (H v)_A.
    const double productToCaller = evToCaller * lengthToAngstrom;
    for (std::size_t i = 0; i < 3 * n; ++i)
      output.set(i, product[i] * productToCaller);
    return kj::READY_NOW;
  }

  /// NWChem, CPMD and CP2K take every arm through setPotentialConfig.
  [[nodiscard]] bool appliesOwnArm() const {
#ifdef RGPOT_HAS_CP2K
    if (m_cp2k)
      return true;
#endif
    return m_nwchem || m_cpmd;
  }

  bool applyOwnArm(const ::PotentialConfig::Reader &cfg, std::string *msg) {
#ifdef RGPOT_HAS_CP2K
    if (m_cp2k)
      return m_cp2k->setPotentialConfig(cfg, msg);
#endif
    return m_nwchem ? m_nwchem->setPotentialConfig(cfg, msg)
                    : m_cpmd->setPotentialConfig(cfg, msg);
  }

  kj::Promise<void> configure(ConfigureContext context) override {
    auto cfg = context.getParams().getConfig();
    auto results = context.getResults();
    if (m_morse) {
      auto applied = rgpot::rpc::configureMorse(cfg, *m_morse);
      if (applied.replacement) {
        m_morse = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
    if (m_d2) {
      auto applied = rgpot::rpc::configureD2(cfg);
      if (applied.replacement) {
        m_d2 = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
    if (m_d2cpmd) {
      auto applied = rgpot::rpc::configureD2Cpmd(cfg);
      if (applied.replacement) {
        m_d2cpmd = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
#ifdef RGPOT_HAS_DFTD4
    if (m_d4) {
      auto applied = rgpot::rpc::configureD4(cfg);
      if (applied.replacement) {
        m_d4 = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
#endif
    if (m_sum) {
      auto applied = rgpot::rpc::configureComposite(cfg);
      if (applied.replacement) {
        m_sum = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
    if (m_muller_brown) {
      auto applied = rgpot::rpc::configureMullerBrown(cfg);
      if (applied.replacement) {
        m_muller_brown = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
    if (!appliesOwnArm()) {
      if (cfg.which() == ::PotentialConfig::NONE) {
        results.setOk(true);
        results.setMessage("no-op (rgpot params: none)");
        return kj::READY_NOW;
      }
      results.setOk(false);
      results.setMessage("configure() with the " +
                         rgpot::rpc::potentialConfigArm(cfg) +
                         " arm is not supported by this backend; only " +
                         kOwnArmBackends + " apply their own arm");
      return kj::READY_NOW;
    }
    std::string msg;
    bool ok = false;
#ifdef RGPOT_POTSERV_MPI
    if (m_pef) {
      try {
        msg = m_pef->configure(cfg);
        ok = true;
      } catch (const std::exception &error) {
        msg = error.what();
      }
    } else
#endif
    {
      ok = applyOwnArm(cfg, &msg);
    }
    results.setOk(ok);
    results.setMessage(msg);
    return kj::READY_NOW;
  }
};

/**
 * @details
 * The main entry point handles command-line arguments to specify the
 * network port and the potential type. It instantiates the requested
 * physics engine and blocks until the server is terminated.
 *
 * # Usage
 * @code
 * ./potserv <port> <PotentialType>
 * @endcode
 *
 * PotentialType is one of LJ, LJCluster, Morse, ZBL, Eckart, MullerBrown,
 * ModifiedMullerBrown, LEPS-HH2, LEPS-HCl2, LEPS-FH2, DoubleWell,
 * CoupledProtonWell, Composite, D2 and D2Cpmd, plus the backends this build
 * compiled in. The printed usage line is the same list.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on initialization failure.
 */
int runServer(int argc, char *argv[]
#ifdef RGPOT_POTSERV_MPI
              , const rgpot::tools::MpiHost &host
#endif
) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <PotentialType>" << std::endl;
    std::cerr
        << "  Available PotentialTypes: LJ, LJCluster, Morse, SWSi, ZBL, Eckart, "
           "MullerBrown, ModifiedMullerBrown, LEPS-HH2, LEPS-HCl2, "
           "LEPS-FH2, Composite, D2, D2Cpmd"
#ifdef RGPOT_HAS_FORTRAN_POTS
              << ", CuH2, EAMAl"
#endif
#ifdef RGPOT_HAS_IPI
              << ", IPI:unix:<path>, IPI:tcp:<host>:<port>"
#endif
#ifdef RGPOT_HAS_XTB
              << ", XTB, GFNFF, GFN0xTB, GFN1xTB"
#endif
#ifdef RGPOT_HAS_TBLITE
              << ", TBLite, TBLiteGFN2, TBLiteGFN1, TBLiteIPEA1"
#endif
#ifdef RGPOT_HAS_DFTD3
              << ", D3"
#endif
#ifdef RGPOT_HAS_DFTD4
              << ", D4"
#endif
#ifdef RGPOT_HAS_METATOMIC
              << ", Metatomic:<model_path>[:<device>]"
#endif
#ifdef RGPOT_HAS_UMA
              << ", Uma:<model_path>, Uma:<model_path>:<task>"
#endif
#ifdef RGPOT_HAS_SKALA
              << ", Skala, Skala:<basis>"
#endif
              << ", NWChem[:basis[:theory[:xc]]]"
              << ", CPMD"
#ifdef RGPOT_HAS_CP2K
              << ", CP2K"
#endif
              << ", DoubleWell, CoupledProtonWell"
              << std::endl;
    return 1;
  }

  int port = 12345;
  try {
    port = std::stoi(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid port argument '" << argv[1]
              << "'. Using default 12345." << std::endl;
  }

  std::string pot_type = argv[2];
#ifdef RGPOT_POTSERV_MPI
  host.collective([&] {
    bool collective_backend = pot_type == "NWChem" ||
        pot_type.rfind("NWChem:", 0) == 0 || pot_type == "CPMD";
#ifdef RGPOT_HAS_CP2K
    collective_backend = collective_backend || pot_type == "CP2K";
#endif
    if (host.size() > 1 && !collective_backend)
      throw std::runtime_error(std::string("multi-rank RPC requires ") +
                               kCollectiveBackends);
  });
#endif
  std::unique_ptr<rgpot::PotentialBase> potential_to_use;

#ifdef RGPOT_HAS_FORTRAN_POTS
  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::fortranpots::CuH2Pot>();
  } else if (pot_type == "EAMAl") {
    std::cout << "Loading EAMAl potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::fortranpots::EAMAlPot>();
  } else
#endif // RGPOT_HAS_FORTRAN_POTS
      if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
  } else if (pot_type == "LJCluster") {
    std::cout << "Loading LJ cluster potential (free boundaries)..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::LJClusterPot>();
  } else if (pot_type == "Morse") {
    std::cout << "Loading Morse potential (Pt parameters)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::MorsePot>();
  } else if (pot_type == "SWSi") {
    std::cout << "Loading Stillinger-Weber silicon potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::StillingerWeberSi>();
  } else if (pot_type == "ZBL") {
    std::cout << "Loading ZBL potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::ZBLPot>();
  } else if (pot_type == "Eckart") {
    potential_to_use = std::make_unique<rgpot::EckartPot>();
  } else if (pot_type == "MullerBrown") {
    potential_to_use = std::make_unique<rgpot::MullerBrownPot>();
  } else if (pot_type == "ModifiedMullerBrown") {
    potential_to_use = std::make_unique<rgpot::ModifiedMullerBrownPot>();
  } else if (pot_type == "LEPS-HH2") {
    potential_to_use =
        std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::HH2);
  } else if (pot_type == "LEPS-HCl2") {
    potential_to_use =
        std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::HCl2);
  } else if (pot_type == "LEPS-FH2") {
    potential_to_use =
        std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::FH2);
  } else if (pot_type == "DoubleWell") {
    potential_to_use =
        std::make_unique<rgpot::DoubleWellPot>(2, 1.0, 2.0, 1.0, 0.25);
  } else if (pot_type == "CoupledProtonWell") {
    potential_to_use =
        std::make_unique<rgpot::CoupledProtonWellPot>(1.0, 5.0, 8.0 / 25.0);
#ifdef RGPOT_HAS_CBE
  } else if (pot_type == "CBE") {
    potential_to_use = std::make_unique<rgpot::CBEPot>();
#endif
  } else if (pot_type == "Composite") {
    std::cout << "Loading an empty composite (configure() with the composite arm sets its terms)..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::SumPot>();
  } else if (pot_type == "D2") {
    std::cout << "Loading DFT-D2 (BLYP parameters)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D2Pot>();
  } else if (pot_type == "D2Cpmd") {
    std::cout << "Loading CPMD's empirical D2 (BLYP s6)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D2CpmdPot>();
#ifdef RGPOT_HAS_IPI
  } else if (pot_type == "IPI" || pot_type.rfind("IPI:", 0) == 0) {
    if (pot_type.size() < 5) {
      std::cerr << "Error: IPI requires an address, for example "
                   "IPI:unix:/tmp/ipi_rgpot or IPI:tcp:127.0.0.1:31415"
                << std::endl;
      return 1;
    }
    rgpot::IPIConfig cfg;
    cfg.address = pot_type.substr(4);
    std::cout << "Loading IPI socket server at " << cfg.address << "..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::IPIPot>(cfg);
#endif
#ifdef RGPOT_HAS_XTB
  } else if (pot_type == "XTB") {
    std::cout << "Loading XTB potential (GFN2-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>();
  } else if (pot_type == "GFNFF") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFNFF;
    std::cout << "Loading XTB potential (GFNFF)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN1xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN1xTB;
    std::cout << "Loading XTB potential (GFN1-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN0xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN0xTB;
    std::cout << "Loading XTB potential (GFN0-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
#endif // RGPOT_HAS_XTB
#ifdef RGPOT_HAS_TBLITE
  } else if (pot_type == "TBLite" || pot_type == "TBLiteGFN2") {
    std::cout << "Loading TBLite potential (GFN2)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>();
  } else if (pot_type == "TBLiteGFN1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::GFN1;
    std::cout << "Loading TBLite potential (GFN1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
  } else if (pot_type == "TBLiteIPEA1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::IPEA1;
    std::cout << "Loading TBLite potential (IPEA1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
#endif // RGPOT_HAS_TBLITE
#ifdef RGPOT_HAS_DFTD3
  } else if (pot_type == "D3") {
    std::cout << "Loading D3 potential (BJ PBE ATM)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D3Pot>();
#endif // RGPOT_HAS_DFTD3
#ifdef RGPOT_HAS_DFTD4
  } else if (pot_type == "D4") {
    std::cout << "Loading D4 potential (PBE ATM charge 0, molecular; configure() takes "
                 "the d4 arm)..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::D4Pot>();
#endif // RGPOT_HAS_DFTD4
#ifdef RGPOT_HAS_METATOMIC
  } else if (pot_type.rfind("Metatomic:", 0) == 0) {
    auto spec = pot_type.substr(10);
    rgpot::MetatomicConfig cfg;
    // Metatomic:<model.pt>[:<device>] -- the device is part of the request,
    // cuda on an accelerated node, cpu elsewhere.
    auto colon = spec.rfind(':');
    if (colon != std::string::npos &&
        spec.substr(colon + 1).find('/') == std::string::npos) {
      cfg.device = spec.substr(colon + 1);
      spec = spec.substr(0, colon);
    }
    cfg.model_path = spec;
    auto model_path = spec;
    std::cout << "Loading Metatomic potential from '" << model_path << "'..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::MetatomicPot>(cfg);
#endif // RGPOT_HAS_METATOMIC
#ifdef RGPOT_HAS_UMA
  } else if (pot_type.rfind("Uma:", 0) == 0) {
    rgpot::UmaConfig cfg;
    auto rest = pot_type.substr(4);
    const auto colon = rest.rfind(':');
    if (colon != std::string::npos) {
      const auto tail = rest.substr(colon + 1);
      if (tail == "omol" || tail == "omat" || tail == "oc20" ||
          tail == "oc22" || tail == "oc25" || tail == "odac" ||
          tail == "omc") {
        cfg.task_name = tail;
        rest = rest.substr(0, colon);
      }
    }
    cfg.model_path = rest;
    std::cout << "Loading UMA AOTI package '" << cfg.model_path
              << "' task='" << cfg.task_name << "'..." << std::endl;
    potential_to_use = std::make_unique<rgpot::UmaPot>(cfg);
#endif // RGPOT_HAS_UMA
#ifdef RGPOT_HAS_SKALA
  } else if (pot_type == "Skala" || pot_type.rfind("Skala:", 0) == 0) {
    rgpot::SkalaConfig cfg;
    if (pot_type.rfind("Skala:", 0) == 0)
      cfg.basis = pot_type.substr(6);
    std::cout << "Loading Skala XC via NWChem DFT (xc=" << cfg.xc
              << " basis=" << cfg.basis << ")..." << std::endl;
    potential_to_use = std::make_unique<rgpot::SkalaPot>(cfg);
#endif // RGPOT_HAS_SKALA
  } else if (pot_type.rfind("NWChem", 0) == 0) {
    std::cout << "Loading NWChem potential (dlopen libnwchemc)..."
              << std::endl;
    // NWChem[:<basis>[:<theory>[:<xc>]]] -- unset fields keep the schema
    // defaults; commas in the xc field stand in for spaces so a request
    // like NWChem:6-31g*:dft:xpbe96,cpbe96 stays one shell token.
    std::string nw_basis, nw_theory, nw_xc;
    {
      std::vector<std::string> parts;
      std::stringstream ss(pot_type);
      std::string tok;
      while (std::getline(ss, tok, ':'))
        parts.push_back(tok);
      if (parts.size() > 1)
        nw_basis = parts[1];
      if (parts.size() > 2)
        nw_theory = parts[2];
      if (parts.size() > 3)
        nw_xc = parts[3];
      std::replace(nw_xc.begin(), nw_xc.end(), ',', ' ');
    }
    std::unique_ptr<rgpot::NWChemPot> nw;
    auto construct = [&] {
    if (nw_basis.empty() && nw_theory.empty() && nw_xc.empty()) {
      nw = std::make_unique<rgpot::NWChemPot>();
    } else {
      capnp::MallocMessageBuilder nw_msg;
      auto nw_params = nw_msg.initRoot<NWChemParams>();
      if (!nw_basis.empty())
        nw_params.setBasis(nw_basis);
      if (!nw_theory.empty())
        nw_params.setTheory(nw_theory);
      if (!nw_xc.empty()) {
        // The xc directive rides the raw input-block escape hatch; the
        // scfType field is the HF reference choice, not the functional.
        // The iteration cap bounds a pathological geometry's SCF: the
        // engine returns a convergence failure the driver refuses and
        // moves past, instead of one stuck call eating the RPC budget.
        auto nw_blocks = nw_params.initInputBlocks(1);
        nw_blocks.set(0, std::string("dft\n  xc " + nw_xc +
                                     "\n  iterations 60\nend"));
      }
      std::cout << "  basis='" << nw_basis << "' theory='" << nw_theory
                << "' xc='" << nw_xc << "'" << std::endl;
      nw = std::make_unique<rgpot::NWChemPot>(nw_params.asReader());
    }
    };
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::NWChemPot::probeAvailable())
        throw std::runtime_error("NWChem engine is not available on every rank");
    });
    host.collective(construct);
#else
    construct();
#endif
    if (!nw->available()) {
      std::cerr << "Warning: libnwchemc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr; // use dedicated path below
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *nw);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(nw), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else if (pot_type == "CP2K") {
#ifdef RGPOT_HAS_CP2K
    std::cout << "Loading CP2K potential (dlopen libcp2kc_rgpot)..."
              << std::endl;
    std::unique_ptr<rgpot::CP2KPot> cp;
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::CP2KPot::probeAvailable())
        throw std::runtime_error("CP2K shim is not available on every rank");
    });
    host.collective([&] { cp = std::make_unique<rgpot::CP2KPot>(); });
#else
    cp = std::make_unique<rgpot::CP2KPot>();
#endif
    if (!cp->available()) {
      std::cerr << "Warning: libcp2kc_rgpot not loaded; calculate() will fail "
                   "until the shim is available (configure() still accepted)."
                << std::endl;
    }
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *cp);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(cp), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
#else
    std::cerr << "Error: this potserv was built without CP2K; rebuild with "
                 "-Dwith_cp2k=enabled (CMake: -DRGPOT_WITH_CP2K=ON) against "
                 "libcp2k."
              << std::endl;
    return 1;
#endif
  } else if (pot_type == "CPMD") {
    std::cout << "Loading CPMD potential (dlopen libcpmdc)..." << std::endl;
    std::unique_ptr<rgpot::CPMDPot> cp;
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::CPMDPot::probeAvailable())
        throw std::runtime_error("CPMD engine is not available on every rank");
    });
    host.collective([&] { cp = std::make_unique<rgpot::CPMDPot>(); });
#else
    cp = std::make_unique<rgpot::CPMDPot>();
#endif
    if (!cp->available()) {
      std::cerr << "Warning: libcpmdc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr;
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *cp);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(cp), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else {
    std::cerr << "Error: Unknown potential type '" << pot_type << "'"
              << std::endl;
    return 1;
  }

  if (!potential_to_use) {
    std::cerr << "Error: potential not initialized" << std::endl;
    return 1;
  }

  capnp::EzRpcServer server(
      kj::heap<GenericPotImpl>(std::move(potential_to_use)), "localhost", port);

  auto &waitScope = server.getWaitScope();
  std::cout << "Server running on port " << port << " with " << pot_type
            << " potential." << std::endl;
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}

int main(int argc, char **argv) {
  try {
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiHost host(argc, argv);
    return runServer(argc, argv, host);
#else
    return runServer(argc, argv);
#endif
  } catch (const kj::Exception &error) {
    std::cerr << error.getDescription().cStr() << std::endl;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
  }
  return 1;
}
