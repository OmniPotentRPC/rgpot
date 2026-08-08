// MIT License
// Copyright 2023--present rgpot developers
#include "rgpot/SharedPot/SharedPot.hpp"

#include <dlfcn.h>
#include <stdexcept>

namespace rgpot {

SharedPot::SharedPot(const std::string &library_path)
    : Potential(PotType::Shared) {
  m_handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (m_handle == nullptr) {
    throw std::runtime_error(std::string("SharedPot: dlopen failed: ") +
                             dlerror());
  }
  // The one exported symbol is the whole contract.
  m_force = reinterpret_cast<decltype(m_force)>(
      dlsym(m_handle, "rgpot_shared_force"));
  if (m_force == nullptr) {
    dlclose(m_handle);
    m_handle = nullptr;
    throw std::runtime_error(
        "SharedPot: library exports no rgpot_shared_force");
  }
}

SharedPot::~SharedPot() {
  if (m_handle != nullptr) {
    dlclose(m_handle);
  }
}

void SharedPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const int status = m_force(in.nAtoms, in.pos, in.atmnrs, in.box, out->F,
                             &out->energy, &out->variance);
  if (status != 0) {
    throw std::runtime_error("SharedPot: backend returned " +
                             std::to_string(status));
  }
}

} // namespace rgpot
