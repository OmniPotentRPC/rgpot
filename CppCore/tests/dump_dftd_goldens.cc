// Dump s-dftd3 / dftd4 C-API energy (Hartree) and gradient (Hartree/Bohr)
// for the committed water-octamer fixture. Same structure setup as D3Pot /
// D4Pot (Angstrom -> Bohr with units.hpp). Overwrites the npy pins.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef RGPOT_HAS_DFTD3
#include <dftd3.h>
#endif
#ifdef RGPOT_HAS_DFTD4
#include <dftd4.h>
#endif

#include "npy_io.hpp"
#include "rgpot/units.hpp"

using rgpot::testio::load_npz;
using rgpot::testio::save_npy;
using rgpot::units::ANGSTROM_TO_BOHR;

namespace {

constexpr const char *kData = "CppCore/tests/data/dftd";

struct Geom {
  int nat = 0;
  std::vector<int> numbers;
  std::vector<double> pos_bohr;
  double box_bohr[9]{};
  bool periodic[3] = {false, false, false};
};

Geom load_geom() {
  auto arrays = load_npz(std::string(kData) + "/geometry.npz");
  const auto &pos = arrays.at("positions");
  const auto &num = arrays.at("numbers");
  if (pos.shape.size() != 2 || pos.shape[1] != 3) {
    throw std::runtime_error("geometry positions must be (n,3)");
  }
  Geom g;
  g.nat = static_cast<int>(pos.shape[0]);
  g.numbers.resize(static_cast<std::size_t>(g.nat));
  g.pos_bohr.resize(static_cast<std::size_t>(3 * g.nat));
  for (int i = 0; i < g.nat; ++i) {
    g.numbers[static_cast<std::size_t>(i)] =
        static_cast<int>(std::lround(num.data[static_cast<std::size_t>(i)]));
    for (int j = 0; j < 3; ++j) {
      g.pos_bohr[static_cast<std::size_t>(3 * i + j)] =
          pos.data[static_cast<std::size_t>(3 * i + j)] * ANGSTROM_TO_BOHR;
    }
  }
  // Same dummy cell D3Pot/D4Pot pass for a non-periodic 100 A box.
  const double box_a[9] = {100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};
  for (int i = 0; i < 9; ++i) {
    g.box_bohr[i] = box_a[i] * ANGSTROM_TO_BOHR;
  }
  return g;
}

void write_pin(const std::string &stem, double energy,
               const std::vector<double> &grad, int nat) {
  save_npy(std::string(kData) + "/" + stem + "_energy.npy", {energy}, {});
  save_npy(std::string(kData) + "/" + stem + "_grad.npy", grad,
           {static_cast<std::size_t>(nat), 3});
  std::printf("%s  %.16e Eh\n", stem.c_str(), energy);
}

#ifdef RGPOT_HAS_DFTD3
void dump_d3(const Geom &g, bool atm, const char *stem) {
  dftd3_error err = dftd3_new_error();
  if (!err) {
    throw std::runtime_error("dftd3_new_error failed");
  }
  std::vector<char> method{'p', 'b', 'e', '\0'};
  dftd3_param param = dftd3_load_rational_damping(err, method.data(), atm);
  if (dftd3_check_error(err) != 0 || !param) {
    throw std::runtime_error("dftd3_load_rational_damping failed");
  }
  dftd3_structure mol = dftd3_new_structure(err, g.nat, g.numbers.data(),
                                            g.pos_bohr.data(), g.box_bohr,
                                            g.periodic);
  if (dftd3_check_error(err) != 0 || !mol) {
    throw std::runtime_error("dftd3_new_structure failed");
  }
  dftd3_model model = dftd3_new_d3_model(err, mol);
  if (dftd3_check_error(err) != 0 || !model) {
    throw std::runtime_error("dftd3_new_d3_model failed");
  }
  double energy = 0.0;
  std::vector<double> grad(static_cast<std::size_t>(3 * g.nat), 0.0);
  double sigma[9] = {};
  dftd3_get_dispersion(err, mol, model, param, &energy, grad.data(), sigma);
  if (dftd3_check_error(err) != 0) {
    throw std::runtime_error("dftd3_get_dispersion failed");
  }
  write_pin(stem, energy, grad, g.nat);
  dftd3_delete_param(&param);
  dftd3_delete_model(&model);
  dftd3_delete_structure(&mol);
  dftd3_delete_error(&err);
}
#endif

#ifdef RGPOT_HAS_DFTD4
void dump_d4(const Geom &g, const char *stem) {
  dftd4_error err = dftd4_new_error();
  if (!err) {
    throw std::runtime_error("dftd4_new_error failed");
  }
  std::vector<char> method{'p', 'b', 'e', '\0'};
  dftd4_param param = dftd4_load_rational_damping(err, method.data(), true);
  if (dftd4_check_error(err) != 0 || !param) {
    throw std::runtime_error("dftd4_load_rational_damping failed");
  }
  const double charge = 0.0;
  dftd4_structure mol = dftd4_new_structure(err, g.nat, g.numbers.data(),
                                            g.pos_bohr.data(), &charge,
                                            g.box_bohr, g.periodic);
  if (dftd4_check_error(err) != 0 || !mol) {
    throw std::runtime_error("dftd4_new_structure failed");
  }
  dftd4_model model = dftd4_new_d4_model(err, mol);
  if (dftd4_check_error(err) != 0 || !model) {
    throw std::runtime_error("dftd4_new_d4_model failed");
  }
  double energy = 0.0;
  std::vector<double> grad(static_cast<std::size_t>(3 * g.nat), 0.0);
  dftd4_get_dispersion(err, mol, model, param, &energy, grad.data(), nullptr);
  if (dftd4_check_error(err) != 0) {
    throw std::runtime_error("dftd4_get_dispersion failed");
  }
  write_pin(stem, energy, grad, g.nat);
  dftd4_delete_param(&param);
  dftd4_delete_model(&model);
  dftd4_delete_structure(&mol);
  dftd4_delete_error(&err);
}
#endif

} // namespace

int main() {
  try {
    Geom g = load_geom();
#ifdef RGPOT_HAS_DFTD3
    dump_d3(g, false, "d3_bj_pbe_atm_off");
    dump_d3(g, true, "d3_bj_pbe_atm_on");
#endif
#ifdef RGPOT_HAS_DFTD4
    dump_d4(g, "d4_pbe");
#endif
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "dump_dftd_goldens: %s\n", ex.what());
    return 1;
  }
  return 0;
}
