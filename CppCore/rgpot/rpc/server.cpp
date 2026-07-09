// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the standalone Cap'n Proto potential server.
 *
 * This file implements a basic RPC server which exposes toy potentials over a
 * network interface. It utilizes the @c EzRpcServer for handling requests.
 *
 * When built with RGPOT_POTSERV_MPI and launched under mpirun for NWChem or
 * CPMD, rank 0 owns the Cap'n Proto TCP socket; ranks 1..P-1 run a PEF worker
 * loop. On every calculate/configure, rank 0 broadcasts work so all ranks
 * enter the same pot call (host-owned MPI; Cap'n Proto stays free of ranks).
 */

#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/debug.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef RGPOT_POTSERV_MPI
#include <mpi.h>
#endif

#ifdef RGPOT_HAS_FORTRAN
#include "rgpot/CuH2/CuH2Pot.hpp"
#endif // RGPOT_HAS_FORTRAN

#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif // RGPOT_HAS_XTB

#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif // RGPOT_HAS_TBLITE

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#endif // RGPOT_HAS_METATOMIC

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"
#include "rgpot/units.hpp"

namespace {

#ifdef RGPOT_POTSERV_MPI
// Host-owned PEF command channel (rank 0 root). Cap'n Proto never sees ranks.
enum class PefCmd : int { CALC = 1, CONFIG = 2, EXIT = 3 };

struct MpiWorld {
  int rank = 0;
  int size = 1;
  bool active = false; // true when MPI_Init succeeded
};

static MpiWorld g_mpi;

static bool pef_multi_rank() { return g_mpi.active && g_mpi.size > 1; }

static void pef_bcast_cmd(PefCmd cmd) {
  int c = static_cast<int>(cmd);
  MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static PefCmd pef_recv_cmd() {
  int c = 0;
  MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return static_cast<PefCmd>(c);
}

/** Broadcast geometry already in angstrom (identical on every rank after). */
static void pef_bcast_calc_inputs(int *n, std::vector<double> *pos,
                                  std::vector<int> *Z,
                                  std::array<double, 9> *box_flat) {
  MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (*n < 0)
    *n = 0;
  pos->resize(static_cast<size_t>(*n) * 3u);
  Z->resize(static_cast<size_t>(*n));
  if (*n > 0) {
    MPI_Bcast(pos->data(), *n * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Z->data(), *n, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(box_flat->data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/** Broadcast flat Cap'n Proto PotentialConfig words from rank 0. */
static void pef_bcast_config_words(std::vector<::capnp::word> *words) {
  int nwords = static_cast<int>(words->size());
  MPI_Bcast(&nwords, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (nwords < 0)
    nwords = 0;
  words->resize(static_cast<size_t>(nwords));
  if (nwords > 0) {
    MPI_Bcast(reinterpret_cast<char *>(words->data()),
              nwords * static_cast<int>(sizeof(::capnp::word)), MPI_BYTE, 0,
              MPI_COMM_WORLD);
  }
}

static void pef_apply_config_words(rgpot::NWChemPot *nwchem,
                                   rgpot::CPMDPot *cpmd,
                                   const std::vector<::capnp::word> &words,
                                   std::string *msg_out) {
  if (words.empty()) {
    if (msg_out)
      *msg_out = "empty PotentialConfig broadcast";
    return;
  }
  ::capnp::FlatArrayMessageReader reader(
      kj::ArrayPtr<const ::capnp::word>(words.data(), words.size()));
  auto cfg = reader.getRoot<::PotentialConfig>();
  if (nwchem)
    nwchem->setPotentialConfig(cfg, msg_out);
  else if (cpmd)
    cpmd->setPotentialConfig(cfg, msg_out);
}

static std::pair<double, rgpot::types::AtomMatrix>
pef_call_force(rgpot::PotentialBase *pot, int n,
               const std::vector<double> &pos, const std::vector<int> &Z,
               const std::array<double, 9> &box_flat) {
  rgpot::types::AtomMatrix positions(static_cast<size_t>(n), 3);
  for (int i = 0; i < n * 3; ++i)
    positions.data()[i] = pos[static_cast<size_t>(i)];
  std::vector<int> atmtypes = Z;
  std::array<std::array<double, 3>, 3> box = {
      {{box_flat[0], box_flat[1], box_flat[2]},
       {box_flat[3], box_flat[4], box_flat[5]},
       {box_flat[6], box_flat[7], box_flat[8]}}};
  return (*pot)(positions, atmtypes, box);
}

/**
 * Worker ranks (1..P-1): mirror rank-0 calculate/configure via MPI_Bcast, then
 * enter the same pot call. Results are discarded (engine collectives matter).
 */
static void pef_worker_loop(rgpot::PotentialBase *pot, rgpot::NWChemPot *nwchem,
                            rgpot::CPMDPot *cpmd) {
  for (;;) {
    const PefCmd cmd = pef_recv_cmd();
    if (cmd == PefCmd::EXIT)
      break;
    if (cmd == PefCmd::CALC) {
      int n = 0;
      std::vector<double> pos;
      std::vector<int> Z;
      std::array<double, 9> box_flat{};
      pef_bcast_calc_inputs(&n, &pos, &Z, &box_flat);
      try {
        (void)pef_call_force(pot, n, pos, Z, box_flat);
      } catch (const std::exception &ex) {
        std::cerr << "potserv PEF worker rank " << g_mpi.rank
                  << " calculate error: " << ex.what() << std::endl;
      }
    } else if (cmd == PefCmd::CONFIG) {
      std::vector<::capnp::word> words;
      pef_bcast_config_words(&words);
      std::string msg;
      try {
        pef_apply_config_words(nwchem, cpmd, words, &msg);
      } catch (const std::exception &ex) {
        std::cerr << "potserv PEF worker rank " << g_mpi.rank
                  << " configure error: " << ex.what() << std::endl;
      }
    } else {
      std::cerr << "potserv PEF worker rank " << g_mpi.rank
                << " unknown cmd=" << static_cast<int>(cmd) << std::endl;
    }
  }
}

static void pef_mpi_finalize_if_needed() {
  if (!g_mpi.active)
    return;
  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
  g_mpi.active = false;
}

static bool is_pef_backend(const std::string &pot_type) {
  return pot_type == "NWChem" || pot_type == "CPMD";
}
#endif // RGPOT_POTSERV_MPI

} // namespace

/**
 * @class GenericPotImpl
 * @brief Server implementation for the Potential RPC interface.
 *
 * This class wraps a polymorphic @c PotentialBase instance and dispatches
 * RPC calculate requests to the underlying physics engine. When multi-rank
 * PEF is active, calculate/configure broadcast work to worker ranks first.
 */
class GenericPotImpl final : public Potential::Server {
private:
  std::unique_ptr<rgpot::PotentialBase>
      m_potential; //!< The polymorphic potential engine.
  /// Optional typed handle when backend is NWChem (for configure()).
  rgpot::NWChemPot *m_nwchem = nullptr;
  /// Optional typed handle when backend is CPMD (for configure()).
  rgpot::CPMDPot *m_cpmd = nullptr;
#ifdef RGPOT_POTSERV_MPI
  bool m_pef = false; //!< Broadcast PEF work on calculate/configure.
#endif

public:
  /**
   * @brief Constructor for GenericPotImpl.
   * @param pot Ownership of a PotentialBase derived object.
   */
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)) {}

  /**
   * @brief Constructor retaining an NWChemPot pointer for configure().
   * Members initialize in declaration order (m_potential then m_nwchem).
   */
  GenericPotImpl(std::unique_ptr<rgpot::NWChemPot> pot, bool pef = false)
      : m_potential(std::move(pot)),
        m_nwchem(static_cast<rgpot::NWChemPot *>(m_potential.get()))
#ifdef RGPOT_POTSERV_MPI
        ,
        m_pef(pef)
#endif
  {
#ifndef RGPOT_POTSERV_MPI
    (void)pef;
#endif
  }

  GenericPotImpl(std::unique_ptr<rgpot::CPMDPot> pot, bool pef = false)
      : m_potential(std::move(pot)),
        m_cpmd(static_cast<rgpot::CPMDPot *>(m_potential.get()))
#ifdef RGPOT_POTSERV_MPI
        ,
        m_pef(pef)
#endif
  {
#ifndef RGPOT_POTSERV_MPI
    (void)pef;
#endif
  }

  ~GenericPotImpl() {
#ifdef RGPOT_POTSERV_MPI
    // Best-effort: if the server object is destroyed with workers still
    // blocked on Bcast, release them. mpirun kill of all ranks remains the
    // usual shutdown path (EzRpcServer blocks forever on NEVER_DONE).
    if (m_pef && pef_multi_rank() && g_mpi.rank == 0) {
      pef_bcast_cmd(PefCmd::EXIT);
    }
#endif
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
    // Force unit: energy/length in caller's system
    double force_to_caller = ev_to_caller / len_to_angstrom;

    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    // Convert positions from caller's length unit to angstrom
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
    // Convert box from caller's length unit to angstrom
    if (len_to_angstrom != 1.0) {
      for (auto &row : nativeBoxMatrix)
        for (auto &val : row)
          val *= len_to_angstrom;
    }

#ifdef RGPOT_POTSERV_MPI
    if (m_pef && pef_multi_rank()) {
      pef_bcast_cmd(PefCmd::CALC);
      int n = static_cast<int>(numAtoms);
      std::vector<double> pos(static_cast<size_t>(n) * 3u);
      for (int i = 0; i < n * 3; ++i)
        pos[static_cast<size_t>(i)] = nativePositions.data()[i];
      std::array<double, 9> box_flat = {
          nativeBoxMatrix[0][0], nativeBoxMatrix[0][1], nativeBoxMatrix[0][2],
          nativeBoxMatrix[1][0], nativeBoxMatrix[1][1], nativeBoxMatrix[1][2],
          nativeBoxMatrix[2][0], nativeBoxMatrix[2][1], nativeBoxMatrix[2][2]};
      pef_bcast_calc_inputs(&n, &pos, &nativeAtomTypes, &box_flat);
    }
#endif

    // Potential always computes in eV/angstrom
    auto [energy, forces] =
        (*m_potential)(nativePositions, nativeAtomTypes, nativeBoxMatrix);

    // Convert results to caller's units
    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy * ev_to_caller);

    auto forcesList = pres.initForces(numAtoms * 3);
    if (force_to_caller != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        forces.data()[i] *= force_to_caller;
      }
    }
    rgpot::types::adapt::capnp::populateForcesToCapnp(forcesList, forces);

    return kj::READY_NOW;
  }

  kj::Promise<void> configure(ConfigureContext context) override {
    auto cfg = context.getParams().getConfig();
    auto results = context.getResults();
    if (!m_nwchem && !m_cpmd) {
      results.setOk(false);
      results.setMessage(
          "configure() only supported for NWChem or CPMD backend");
      return kj::READY_NOW;
    }

#ifdef RGPOT_POTSERV_MPI
    if (m_pef && pef_multi_rank()) {
      pef_bcast_cmd(PefCmd::CONFIG);
      ::capnp::MallocMessageBuilder msg;
      msg.setRoot(cfg);
      auto flat = ::capnp::messageToFlatArray(msg);
      std::vector<::capnp::word> words(flat.begin(), flat.end());
      pef_bcast_config_words(&words);
    }
#endif

    std::string msg;
    bool ok = m_nwchem ? m_nwchem->setPotentialConfig(cfg, &msg)
                       : m_cpmd->setPotentialConfig(cfg, &msg);
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
 * @c ./potserv <port> <PotentialType>
 * @c mpirun -np P ./potserv <port> NWChem   # rank0 socket; ranks PEF workers
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on initialization failure.
 */
int main(int argc, char *argv[]) {
#ifdef RGPOT_POTSERV_MPI
  if (MPI_Init(&argc, &argv) == MPI_SUCCESS) {
    g_mpi.active = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi.size);
  }
#endif

  if (argc < 3) {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0) {
#endif
      std::cerr << "Usage: " << argv[0] << " <port> <PotentialType>"
                << std::endl;
      std::cerr << "  Available PotentialTypes: CuH2, LJ"
#ifdef RGPOT_HAS_XTB
                << ", XTB, GFNFF, GFN0xTB, GFN1xTB"
#endif
#ifdef RGPOT_HAS_TBLITE
                << ", TBLite, TBLiteGFN1, TBLiteIPEA1"
#endif
#ifdef RGPOT_HAS_METATOMIC
                << ", Metatomic:<model_path>"
#endif
                << ", NWChem"
                << ", CPMD"
                << std::endl;
#ifdef RGPOT_POTSERV_MPI
      if (g_mpi.active && g_mpi.size > 1) {
        std::cerr << "  Multi-rank PEF (mpirun -np P): NWChem, CPMD only; "
                     "rank 0 binds the RPC port, ranks 1..P-1 are workers."
                  << std::endl;
      }
    }
    pef_mpi_finalize_if_needed();
#endif
    return 1;
  }

  int port = 12345;
  try {
    port = std::stoi(argv[1]);
  } catch (const std::exception &e) {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0)
#endif
      std::cerr << "Invalid port argument '" << argv[1]
                << "'. Using default 12345." << std::endl;
  }

  std::string pot_type = argv[2];

#ifdef RGPOT_POTSERV_MPI
  // Non-PEF backends stay single-process; refuse multi-rank so Finalize stays
  // collective and we do not leave workers stuck without a protocol.
  if (g_mpi.active && g_mpi.size > 1 && !is_pef_backend(pot_type)) {
    if (g_mpi.rank == 0) {
      std::cerr << "potserv: multi-rank PEF is only supported for NWChem and "
                   "CPMD (got '"
                << pot_type << "' with np=" << g_mpi.size << ")" << std::endl;
    }
    pef_mpi_finalize_if_needed();
    return g_mpi.rank == 0 ? 1 : 0;
  }
  const bool pef = g_mpi.active && g_mpi.size > 1 && is_pef_backend(pot_type);
#else
  const bool pef = false;
  (void)pef;
#endif

  std::unique_ptr<rgpot::PotentialBase> potential_to_use;

#ifdef RGPOT_HAS_FORTRAN
  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::CuH2Pot>();
  } else
#endif // RGPOT_HAS_FORTRAN
      if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
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
#ifdef RGPOT_HAS_METATOMIC
  } else if (pot_type.rfind("Metatomic:", 0) == 0) {
    auto model_path = pot_type.substr(10);
    rgpot::MetatomicConfig cfg;
    cfg.model_path = model_path;
    std::cout << "Loading Metatomic potential from '" << model_path << "'..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::MetatomicPot>(cfg);
#endif // RGPOT_HAS_METATOMIC
  } else if (pot_type == "NWChem") {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0)
#endif
      std::cout << "Loading NWChem potential (dlopen libnwchemc)..."
                << std::endl;
    // Scope pot so ~NWChemPot / engine GA+MPI teardown runs before host
    // MPI_Finalize (same contract as nwchem_mpi_force_host).
    {
      auto nw = std::make_unique<rgpot::NWChemPot>();
#ifdef RGPOT_POTSERV_MPI
      if ((!g_mpi.active || g_mpi.rank == 0) && !nw->available()) {
#else
      if (!nw->available()) {
#endif
        std::cerr << "Warning: libnwchemc not loaded; calculate() will fail "
                     "until engine is available (configure() still accepted)."
                  << std::endl;
      }
#ifdef RGPOT_POTSERV_MPI
      if (pef && g_mpi.rank > 0) {
        if (g_mpi.rank == 1) {
          std::cout << "potserv PEF: ranks 1.." << (g_mpi.size - 1)
                    << " in NWChem worker loop (rank 0 binds port " << port
                    << ")" << std::endl;
        }
        rgpot::NWChemPot *raw = nw.get();
        pef_worker_loop(raw, raw, nullptr);
        nw.reset();
        pef_mpi_finalize_if_needed();
        return 0;
      }
#endif
      capnp::EzRpcServer server(
          kj::heap<GenericPotImpl>(std::move(nw), pef), "localhost", port);
      auto &waitScope = server.getWaitScope();
#ifdef RGPOT_POTSERV_MPI
      if (g_mpi.active && g_mpi.size > 1)
        std::cout << "Server running on port " << port << " with " << pot_type
                  << " potential (PEF ranks=" << g_mpi.size << ")."
                  << std::endl;
      else
#endif
        std::cout << "Server running on port " << port << " with " << pot_type
                  << " potential." << std::endl;
      kj::NEVER_DONE.wait(waitScope);
    }
#ifdef RGPOT_POTSERV_MPI
    pef_mpi_finalize_if_needed();
#endif
    return 0;
  } else if (pot_type == "CPMD") {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0)
#endif
      std::cout << "Loading CPMD potential (dlopen libcpmdc)..." << std::endl;
    {
      auto cp = std::make_unique<rgpot::CPMDPot>();
#ifdef RGPOT_POTSERV_MPI
      if ((!g_mpi.active || g_mpi.rank == 0) && !cp->available()) {
#else
      if (!cp->available()) {
#endif
        std::cerr << "Warning: libcpmdc not loaded; calculate() will fail "
                     "until engine is available (configure() still accepted)."
                  << std::endl;
      }
#ifdef RGPOT_POTSERV_MPI
      if (pef && g_mpi.rank > 0) {
        if (g_mpi.rank == 1) {
          std::cout << "potserv PEF: ranks 1.." << (g_mpi.size - 1)
                    << " in CPMD worker loop (rank 0 binds port " << port << ")"
                    << std::endl;
        }
        rgpot::CPMDPot *raw = cp.get();
        pef_worker_loop(raw, nullptr, raw);
        cp.reset();
        pef_mpi_finalize_if_needed();
        return 0;
      }
#endif
      capnp::EzRpcServer server(
          kj::heap<GenericPotImpl>(std::move(cp), pef), "localhost", port);
      auto &waitScope = server.getWaitScope();
#ifdef RGPOT_POTSERV_MPI
      if (g_mpi.active && g_mpi.size > 1)
        std::cout << "Server running on port " << port << " with " << pot_type
                  << " potential (PEF ranks=" << g_mpi.size << ")."
                  << std::endl;
      else
#endif
        std::cout << "Server running on port " << port << " with " << pot_type
                  << " potential." << std::endl;
      kj::NEVER_DONE.wait(waitScope);
    }
#ifdef RGPOT_POTSERV_MPI
    pef_mpi_finalize_if_needed();
#endif
    return 0;
  } else {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0)
#endif
      std::cerr << "Error: Unknown potential type '" << pot_type << "'"
                << std::endl;
#ifdef RGPOT_POTSERV_MPI
    pef_mpi_finalize_if_needed();
#endif
    return 1;
  }

  if (!potential_to_use) {
#ifdef RGPOT_POTSERV_MPI
    if (!g_mpi.active || g_mpi.rank == 0)
#endif
      std::cerr << "Error: potential not initialized" << std::endl;
#ifdef RGPOT_POTSERV_MPI
    pef_mpi_finalize_if_needed();
#endif
    return 1;
  }

  capnp::EzRpcServer server(
      kj::heap<GenericPotImpl>(std::move(potential_to_use)), "localhost", port);

  auto &waitScope = server.getWaitScope();
  std::cout << "Server running on port " << port << " with " << pot_type
            << " potential." << std::endl;
  kj::NEVER_DONE.wait(waitScope);

#ifdef RGPOT_POTSERV_MPI
  pef_mpi_finalize_if_needed();
#endif
  return 0;
}
