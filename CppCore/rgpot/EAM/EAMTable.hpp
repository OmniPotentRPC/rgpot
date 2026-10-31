#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Tabulated embedded-atom potential read from DYNAMO setfl files.
 *
 *   E = sum_i F_a(rho_i) + sum_{i<j} phi_ab(r_ij)
 *   rho_i = sum_{j != i} rho_ba(r_ij)
 *
 * Daw and Baskes, Phys. Rev. B 29, 6443 (1984). A setfl (eam/alloy)
 * file stores F and rho per element and r phi per pair of elements, the
 * alloy form of Foiles, Baskes and Daw, Phys. Rev. B 33, 7983 (1986).
 * The Finnis-Sinclair variant (eam/fs), after Phil. Mag. A 50, 45
 * (1984), gives every ordered pair of elements its own density.
 *
 * Reading and interpolation follow LAMMPS pair_style eam/alloy and
 * eam/fs (src/MANYBODY/pair_eam.cpp): every table becomes a cubic per
 * interval whose knot slopes are five-point differences, and a density
 * past the F table extends F linearly. Units are those of the file, eV
 * and Angstrom for LAMMPS metal files.
 *
 * The pair list keeps one image per pair, so every periodic cell width
 * must exceed twice the table cutoff.
 */

// clang-format off
#include <array>
#include <cstdint>
#include <string>
#include <vector>
// clang-format on
#include "rgpot/Potential.hpp"

namespace rgpot {

enum class EAMTableFormat : uint8_t { Setfl, FinnisSinclair };

/// Tables as LAMMPS keeps them after file2array. Index 0 of every table
/// is unused, so knot m sits at (m - 1) * dr or (m - 1) * drho.
struct EAMTableData {
  int nrho = 0;
  int nr = 0;
  double drho = 0.0;
  double dr = 0.0;
  double cut = 0.0;
  std::vector<int> atomic_numbers;       //!< Per element, from the file.
  std::vector<std::vector<double>> frho; //!< F of element a.
  /// Entry a * n + b: density an element-a atom puts on an element-b atom.
  std::vector<std::vector<double>> rhor;
  /// Entry a * n + b: r phi of the pair, symmetric in a and b.
  std::vector<std::vector<double>> z2r;
};

[[nodiscard]] EAMTableData read_eam_table(const std::string &path,
                                          EAMTableFormat format);

struct EAMTableConfig {
  std::string path;
  EAMTableFormat format = EAMTableFormat::Setfl;
  /// Atomic number of each file element in file order. Empty takes the
  /// numbers on the element lines of the file.
  std::vector<int> atomic_numbers;
  bool periodic = true;
};

/**
 * @class EAMTablePot
 * @brief Energy and forces of a DYNAMO EAM table on the cached pair list.
 * @ingroup rgpot_potentials
 */
class EAMTablePot : public Potential<EAMTablePot> {
public:
  explicit EAMTablePot(const EAMTableConfig &c);

  [[nodiscard]] const EAMTableData &tables() const noexcept { return m_data; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.periodic = m_periodic};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  static constexpr uint64_t kKernelVersion = 1;
  using Spline = std::vector<std::array<double, 7>>;

  EAMTableData m_data;
  std::vector<Spline> m_frho, m_rhor, m_z2r;
  bool m_periodic = true;
  uint64_t m_paramsKey{0};
};

} // namespace rgpot
