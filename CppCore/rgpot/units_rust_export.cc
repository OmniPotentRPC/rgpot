// MIT License
// Copyright 2023--present rgpot developers

// Prints the units.hpp constants as Rust `f64` bit patterns. rgpot-core's
// `units` feature compiles and runs this program; the header stays the
// only table.

#include "rgpot/units.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static void emit(const char *name, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::printf("pub const %s: f64 = f64::from_bits(0x%016llx);\n", name,
              static_cast<unsigned long long>(bits));
}

int main() {
  using namespace rgpot::units;
  emit("BOHR_TO_ANGSTROM", BOHR_TO_ANGSTROM);
  emit("ANGSTROM_TO_BOHR", ANGSTROM_TO_BOHR);
  emit("HARTREE_TO_EV", HARTREE_TO_EV);
  emit("EV_TO_HARTREE", EV_TO_HARTREE);
  emit("KB_HARTREE", KB_HARTREE);
  emit("KB_EV", KB_EV);
  try {
    emit("DALTON_IN_ELECTRON_MASSES",
         unit_conversion_factor("dalton", "electron_mass"));
    emit("ATOMIC_UNIT_OF_TIME_S",
         unit_conversion_factor("hbar", "hartree*s"));
  } catch (const std::exception &err) {
    std::fprintf(stderr, "units export: %s\n", err.what());
    return 1;
  }
  return 0;
}
