// Dump s-dftd3 / dftd4 C-API energy and gradient on the Baker water fixture.
// Writes Hartree / Hartree/Bohr pins. Does not go through D3Pot/D4Pot.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

using rgpot::units::ANGSTROM_TO_BOHR;

// Baker water (Angstrom): O + 2H. Same geometry as D3PotTest / D4PotTest.
static const double kWaterPosA[] = {
    0.00000000, 0.00000000, 0.11779000,  // O
    0.00000000, 0.75545000, -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000 // H
};
static const int kWaterZ[] = {8, 1, 1};
static constexpr int kNat = 3;
static constexpr double kBoxA = 100.0;

[[noreturn]] static void die(const char *msg) {
  std::fprintf(stderr, "dump_dftd_goldens: %s\n", msg);
  std::exit(2);
}

static std::string join(const char *dir, const char *name) {
  return std::string(dir) + "/" + name;
}

#ifdef RGPOT_HAS_DFTD3
static void check_d3(dftd3_error err, const char *what) {
  if (dftd3_check_error(err) == 0) {
    return;
  }
  char buf[512] = {};
  dftd3_get_error(err, buf, nullptr);
  std::fprintf(stderr, "s-dftd3 %s: %s\n", what, buf);
  std::exit(3);
}

static void dump_d3(const char *dir, bool atm, const char *tag) {
  std::vector<double> pos_bohr(3 * kNat);
  for (int i = 0; i < 3 * kNat; ++i) {
    pos_bohr[static_cast<std::size_t>(i)] =
        kWaterPosA[i] * ANGSTROM_TO_BOHR;
  }
  double box_bohr[9] = {};
  box_bohr[0] = box_bohr[4] = box_bohr[8] = kBoxA * ANGSTROM_TO_BOHR;
  const bool periodicity[3] = {false, false, false};

  dftd3_error err = dftd3_new_error();
  if (!err) {
    die("dftd3_new_error");
  }
  char method[] = "pbe";
  dftd3_param param = dftd3_load_rational_damping(err, method, atm);
  check_d3(err, "load rational damping");
  if (!param) {
    die("null d3 param");
  }
  dftd3_structure mol = dftd3_new_structure(
      err, kNat, kWaterZ, pos_bohr.data(), box_bohr, periodicity);
  check_d3(err, "new structure");
  dftd3_model model = dftd3_new_d3_model(err, mol);
  check_d3(err, "new D3 model");

  double energy = 0.0;
  std::vector<double> grad(3 * kNat, 0.0);
  dftd3_get_dispersion(err, mol, model, param, &energy, grad.data(), nullptr);
  check_d3(err, "get dispersion");

  const std::string epath = join(dir, (std::string(tag) + "_energy.npy").c_str());
  const std::string gpath = join(dir, (std::string(tag) + "_grad.npy").c_str());
  rgpot::testio::save_npy(epath, {energy}, {1});
  rgpot::testio::save_npy(gpath, grad, {static_cast<std::size_t>(kNat), 3});

  dftd3_delete_param(&param);
  dftd3_delete_model(&model);
  dftd3_delete_structure(&mol);
  dftd3_delete_error(&err);
}
#endif

#ifdef RGPOT_HAS_DFTD4
static void check_d4(dftd4_error err, const char *what) {
  if (dftd4_check_error(err) == 0) {
    return;
  }
  char buf[512] = {};
  dftd4_get_error(err, buf, nullptr);
  std::fprintf(stderr, "dftd4 %s: %s\n", what, buf);
  std::exit(4);
}

static void dump_d4(const char *dir, const char *tag) {
  std::vector<double> pos_bohr(3 * kNat);
  for (int i = 0; i < 3 * kNat; ++i) {
    pos_bohr[static_cast<std::size_t>(i)] =
        kWaterPosA[i] * ANGSTROM_TO_BOHR;
  }
  double box_bohr[9] = {};
  box_bohr[0] = box_bohr[4] = box_bohr[8] = kBoxA * ANGSTROM_TO_BOHR;
  const bool periodicity[3] = {false, false, false};
  const double charge = 0.0;

  dftd4_error err = dftd4_new_error();
  if (!err) {
    die("dftd4_new_error");
  }
  char method[] = "pbe";
  dftd4_param param = dftd4_load_rational_damping(err, method, true);
  check_d4(err, "load rational damping");
  if (!param) {
    die("null d4 param");
  }
  dftd4_structure mol = dftd4_new_structure(
      err, kNat, kWaterZ, pos_bohr.data(), &charge, box_bohr, periodicity);
  check_d4(err, "new structure");
  dftd4_model model = dftd4_new_d4_model(err, mol);
  check_d4(err, "new D4 model");

  double energy = 0.0;
  std::vector<double> grad(3 * kNat, 0.0);
  dftd4_get_dispersion(err, mol, model, param, &energy, grad.data(), nullptr);
  check_d4(err, "get dispersion");

  const std::string epath = join(dir, (std::string(tag) + "_energy.npy").c_str());
  const std::string gpath = join(dir, (std::string(tag) + "_grad.npy").c_str());
  rgpot::testio::save_npy(epath, {energy}, {1});
  rgpot::testio::save_npy(gpath, grad, {static_cast<std::size_t>(kNat), 3});

  dftd4_delete_param(&param);
  dftd4_delete_model(&model);
  dftd4_delete_structure(&mol);
  dftd4_delete_error(&err);
}
#endif

static void dump_fixture(const char *dir) {
  std::vector<double> pos(kWaterPosA, kWaterPosA + 3 * kNat);
  std::vector<double> z(kNat);
  for (int i = 0; i < kNat; ++i) {
    z[static_cast<std::size_t>(i)] = static_cast<double>(kWaterZ[i]);
  }
  rgpot::testio::save_npy(join(dir, "water_pos.npy"), pos,
                          {static_cast<std::size_t>(kNat), 3});
  rgpot::testio::save_npy(join(dir, "water_z.npy"), z,
                          {static_cast<std::size_t>(kNat)});

  const std::string xyz = join(dir, "water.xyz");
  std::FILE *fp = std::fopen(xyz.c_str(), "w");
  if (!fp) {
    die("cannot write water.xyz");
  }
  std::fprintf(fp, "3\nBaker water (Angstrom)\n");
  std::fprintf(fp, "O  %.8f  %.8f  %.8f\n", kWaterPosA[0], kWaterPosA[1],
               kWaterPosA[2]);
  std::fprintf(fp, "H  %.8f  %.8f  %.8f\n", kWaterPosA[3], kWaterPosA[4],
               kWaterPosA[5]);
  std::fprintf(fp, "H  %.8f  %.8f  %.8f\n", kWaterPosA[6], kWaterPosA[7],
               kWaterPosA[8]);
  std::fclose(fp);
}

int main(int argc, char **argv) {
  const char *dir = "CppCore/tests/data/dftd";
  if (argc > 1) {
    dir = argv[1];
  }

#if !defined(RGPOT_HAS_DFTD3) || !defined(RGPOT_HAS_DFTD4)
  die("rebuild dump_dftd_goldens with -Dwith_dftd3=true -Dwith_dftd4=true");
#endif

  dump_fixture(dir);
#ifdef RGPOT_HAS_DFTD3
  dump_d3(dir, false, "d3_bj_pbe_atm_off");
  dump_d3(dir, true, "d3_bj_pbe_atm_on");
#endif
#ifdef RGPOT_HAS_DFTD4
  dump_d4(dir, "d4_pbe");
#endif
  std::fprintf(stdout, "wrote pins under %s\n", dir);
  return 0;
}
