// MIT License
// Copyright 2023--present rgpot developers

// Prints the units.hpp constants as Rust `f64` bit patterns. rgpot-core's
// `units` feature compiles and runs this program; the header stays the
// only table.

#include "rgpot/units.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static void check_parser_matches(const char *name, double header_value,
                                 double parsed_value) {
  double scale = std::fabs(header_value);
  if (std::fabs(parsed_value - header_value) > 1e-11 * scale) {
    throw std::runtime_error(std::string("units.cc disagrees with ") + name);
  }
}

static void emit(const char *name, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::printf("pub const %s: f64 = f64::from_bits(0x%016llx);\n", name,
              static_cast<unsigned long long>(bits));
}

int main() {
  using namespace rgpot::units;
  try {
    check_parser_matches("BOHR_TO_ANGSTROM", BOHR_TO_ANGSTROM,
                         unit_conversion_factor("bohr", "angstrom"));
    check_parser_matches("HARTREE_TO_EV", HARTREE_TO_EV,
                         unit_conversion_factor("hartree", "eV"));
    check_parser_matches("DALTON_IN_ELECTRON_MASSES",
                         DALTON_IN_ELECTRON_MASSES,
                         unit_conversion_factor("dalton", "electron_mass"));
    check_parser_matches("ATOMIC_UNIT_OF_TIME_S", ATOMIC_UNIT_OF_TIME_S,
                         unit_conversion_factor("hbar", "hartree*s"));
  } catch (const std::exception &err) {
    std::fprintf(stderr, "units export: %s\n", err.what());
    return 1;
  }
  emit("BOHR_TO_ANGSTROM", BOHR_TO_ANGSTROM);
  emit("ANGSTROM_TO_BOHR", ANGSTROM_TO_BOHR);
  emit("HARTREE_TO_EV", HARTREE_TO_EV);
  emit("EV_TO_HARTREE", EV_TO_HARTREE);
  emit("KB_HARTREE", KB_HARTREE);
  emit("KB_EV", KB_EV);
  emit("DALTON_IN_ELECTRON_MASSES", DALTON_IN_ELECTRON_MASSES);
  emit("ATOMIC_UNIT_OF_TIME_S", ATOMIC_UNIT_OF_TIME_S);
  return 0;
}
