#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file IPIPot.hpp
 * @author rgpot Developers
 * @date 2026-10-10
 * @brief Potential that drives an i-PI force engine over a socket.
 *
 * i-PI socket protocol: Ceriotti, More and Manolopoulos, Comput. Phys.
 * Commun. 185, 1019 (2014), doi:10.1016/j.cpc.2013.10.027; i-PI 3.0,
 * J. Chem. Phys. 161, 062504 (2024), doi:10.1063/5.0215869.
 *
 * IPIPot takes the server role. It binds a UNIX or TCP socket and accepts
 * one driver (CP2K, Quantum ESPRESSO, FHI-aims, LAMMPS, i-pi-py_driver).
 * Each force call sends STATUS until the driver reports READY (INIT is
 * sent when the driver reports NEEDINIT), then POSDATA and GETFORCE, and
 * reads FORCEREADY. The destructor sends EXIT.
 *
 * Wire contract, matching the reference C wrappers (native endianness,
 * 12-byte space-padded headers, int32, float64):
 * - lengths in bohr, energies in hartree, forces in hartree/bohr
 * - cell and its inverse are 3x3, lattice vectors in the columns
 * - virial is the extensive tensor in hartree, row-major
 *   W_ij = sum_a F_a,i r_a,j, not divided by the cell volume
 *
 * rgpot keeps Angstrom, eV and eV/Angstrom. The returned stress is the
 * Cauchy stress sigma = (1/V) dE/dstrain, so sigma = -W/V after W is
 * converted to eV and V to Angstrom^3. Atomic numbers are not on the
 * wire; the driver keeps its own species list. timeout_s is the budget
 * for one force call, including the wait for the driver to connect.
 */

#include "rgpot/Potential.hpp"

#include <mutex>
#include <string>

namespace rgpot {

/**
 * @brief Where the driver connects, and how long one force call may wait.
 *
 * address is `unix:/absolute/path`, `unix:<name>` (binds
 * `/tmp/ipi_<name>`, the i-PI default prefix), `tcp:<host>:<port>`, or
 * `tcp:<port>` (host 127.0.0.1). Port 0 asks the kernel for a free port;
 * endpoint() reports the port that was chosen.
 */
struct IPIConfig {
  std::string address;
  double timeout_s = 30.0;
};

/**
 * @brief Server-side i-PI socket potential.
 * @ingroup rgpot_potentials
 */
class IPIPot : public Potential<IPIPot> {
public:
  explicit IPIPot(const IPIConfig &config);
  ~IPIPot() override;

  IPIPot(const IPIPot &) = delete;
  IPIPot &operator=(const IPIPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One connection and one driver. Stress is the Cauchy tensor.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  [[nodiscard]] const IPIConfig &config() const noexcept { return m_config; }

  /// Bound address. A requested TCP port of 0 is the kernel's choice.
  [[nodiscard]] const std::string &endpoint() const noexcept {
    return m_endpoint;
  }

  /// True after a force call that published Cauchy stress.
  [[nodiscard]] bool hasStress() const;

  /// Copies the last Cauchy stress, row-major. Returns false when none.
  bool copyStress(double out[9]) const;

private:
  void bindSocket();
  void closeConn() const;
  void closeListen();
  /// Accepts a driver if needed, then runs one STATUS/POSDATA/GETFORCE
  /// exchange. timeout_s covers both steps.
  void evaluateLocked(const double h[9], const double ih[9],
                      const double *pos_bohr, size_t n_atoms, double *energy,
                      double *forces, double virial[9]) const;

  IPIConfig m_config;
  std::string m_endpoint;
  std::string m_unix_path;
  uint64_t m_paramsKey = 0;
  bool m_tcp = false;

  mutable std::mutex m_mu;
  mutable int m_listen = -1;
  mutable int m_conn = -1;
  mutable double m_stress[9]{};
  mutable int m_has_stress = 0;
};

} // namespace rgpot
