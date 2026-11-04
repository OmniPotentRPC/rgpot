// Dump independent LJ / Morse / optional D3 sums for ExprPot goldens.
// Pins are eV and eV/A. Run from the source root via
// scripts/regen_expr_goldens.py on a host listed in RGPOT_REGEN_HOSTS.
// Do not invoke from meson test.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_io.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif

using rgpot::testio::load_npz;
using rgpot::testio::save_npy;
using rgpot::types::AtomMatrix;

namespace {

constexpr const char *kData = "CppCore/tests/data/expr";

struct Geom {
  std::size_t nat = 0;
  AtomMatrix positions;
  std::vector<int> numbers;
  std::array<std::array<double, 3>, 3> box{};
};

Geom load_geom() {
  const std::string path = std::string(kData) + "/geometry.npz";
  auto arrays = load_npz(path);
  const auto &pos = arrays.at("positions");
  const auto &num = arrays.at("numbers");
  const auto &box = arrays.at("box");
  if (pos.shape.size() != 2 || pos.shape[1] != 3) {
    throw std::runtime_error("geometry positions must be (n,3)");
  }
  if (num.shape.size() != 1 || num.shape[0] != pos.shape[0]) {
    throw std::runtime_error("geometry numbers must be (n,)");
  }
  if (box.shape.size() != 2 || box.shape[0] != 3 || box.shape[1] != 3) {
    throw std::runtime_error("geometry box must be (3,3)");
  }
  Geom g;
  g.nat = pos.shape[0];
  g.positions = AtomMatrix(g.nat, 3);
  g.numbers.resize(g.nat);
  for (std::size_t i = 0; i < g.nat; ++i) {
    g.numbers[i] = static_cast<int>(std::lround(num.data[i]));
    for (std::size_t j = 0; j < 3; ++j) {
      g.positions(i, j) = pos.data[3 * i + j];
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      g.box[i][j] = box.data[3 * i + j];
    }
  }
  return g;
}

void write_pin(const std::string &stem, double energy, const AtomMatrix &forces) {
  const std::size_t nat = forces.rows();
  std::vector<double> flat(3 * nat);
  for (std::size_t i = 0; i < nat; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      flat[3 * i + j] = forces(i, j);
    }
  }
  save_npy(std::string(kData) + "/" + stem + "_energy.npy", {energy}, {});
  save_npy(std::string(kData) + "/" + stem + "_forces.npy", flat, {nat, 3});
  std::printf("%s  E=%.16e eV\n", stem.c_str(), energy);
}

AtomMatrix scaleAdd(double wa, const AtomMatrix &a, double wb,
                    const AtomMatrix &b) {
  AtomMatrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      out(i, j) = wa * a(i, j) + wb * b(i, j);
    }
  }
  return out;
}

} // namespace

int main() {
  try {
    const Geom g = load_geom();
    rgpot::LJPot lj;
    rgpot::MorsePot morse;
    auto [e_lj, f_lj, v_lj] = lj(g.positions, g.numbers, g.box);
    auto [e_m, f_m, v_m] = morse(g.positions, g.numbers, g.box);
    (void)v_lj;
    (void)v_m;

    write_pin("identity", e_lj, f_lj);
    write_pin("half_lj_plus_morse", 0.5 * e_lj + e_m, scaleAdd(0.5, f_lj, 1.0, f_m));
    write_pin("two_lj_minus_lj", e_lj, f_lj);
    write_pin("half_paren_lj_plus_morse", 0.5 * (e_lj + e_m),
              scaleAdd(0.5, f_lj, 0.5, f_m));

#ifdef RGPOT_HAS_DFTD3
    rgpot::D3Pot d3;
    auto [e_d3, f_d3, v_d3] = d3(g.positions, g.numbers, g.box);
    (void)v_d3;
    write_pin("half_lj_plus_d3", 0.5 * e_lj + e_d3, scaleAdd(0.5, f_lj, 1.0, f_d3));
#endif
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "dump_expr_goldens: %s\n", ex.what());
    return 1;
  }
}
