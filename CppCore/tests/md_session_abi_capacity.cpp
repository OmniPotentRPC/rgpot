// MIT License
// Copyright 2023--present rgpot developers
//
// Verify capacity negotiation for rgpot_md_session_ops across the C++ host
// (rgpot_md_session_create) and simulated older/newer engines.

#include "rgpot/ExternalPot/external_potentials.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct OwnedSession {
  rgpot_md_session *handle = nullptr;
  ~OwnedSession() { rgpot_md_session_destroy(handle); }
};

void check_host_modes(const char *engine_path) {
  char error[512] = {};

  // Current engine: host passes sizeof(rgpot_md_session_ops) as capacity.
  {
    OwnedSession s{rgpot_md_session_create(
        nullptr, engine_path, nullptr, 0, "current", error, sizeof(error))};
    require(s.handle != nullptr, std::string("current create: ") + error);
    double masses[2] = {};
    require(rgpot_md_session_get_masses(s.handle, masses, 2, error,
                                        sizeof(error)) == 0,
            error);
    require(masses[0] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "host did not pass sizeof(rgpot_md_session_ops) as capacity");
    require(masses[1] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "current engine did not fill sizeof(rgpot_md_session_ops)");
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(s.handle, box, error,
                                              sizeof(error)) == 0,
            error);
  }

  // Older engine: fills only up to destroy; optional calls fail cleanly.
  {
    OwnedSession s{rgpot_md_session_create(
        nullptr, engine_path, nullptr, 0, "older", error, sizeof(error))};
    require(s.handle != nullptr, std::string("older create: ") + error);
    double masses[2] = {};
    require(rgpot_md_session_get_masses(s.handle, masses, 2, error,
                                        sizeof(error)) == 0,
            error);
    require(masses[0] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "host did not pass capacity to older engine");
    require(masses[1] == static_cast<double>(RGPOT_MD_OPS_END(destroy)),
            "older engine did not report truncated size");
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(s.handle, box, error,
                                              sizeof(error)) != 0,
            "older engine unexpectedly supported set_periodic_box");
  }

  // Newer engine: engine has a larger vtable, writes only min(capacity,
  // sizeof(newer)) = sizeof(rgpot_md_session_ops), no stack overrun.
  {
    OwnedSession s{rgpot_md_session_create(
        nullptr, engine_path, nullptr, 0, "newer", error, sizeof(error))};
    require(s.handle != nullptr, std::string("newer create: ") + error);
    double masses[2] = {};
    require(rgpot_md_session_get_masses(s.handle, masses, 2, error,
                                        sizeof(error)) == 0,
            error);
    require(masses[0] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "host did not pass capacity to newer engine");
    require(masses[1] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "newer engine wrote more than host capacity");
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(s.handle, box, error,
                                              sizeof(error)) == 0,
            error);
  }

  // Misbehaving engine reporting struct_size > sizeof(rgpot_md_session_ops):
  // host clamps safely and still works.
  {
    OwnedSession s{rgpot_md_session_create(
        nullptr, engine_path, nullptr, 0, "oversized", error, sizeof(error))};
    require(s.handle != nullptr, std::string("oversized create: ") + error);
    const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    require(rgpot_md_session_set_periodic_box(s.handle, box, error,
                                              sizeof(error)) == 0,
            error);
  }
}

void check_direct_capacities(const char *engine_path) {
  void *handle = dlopen(engine_path, RTLD_NOW | RTLD_LOCAL);
  require(handle != nullptr, "dlopen failed");
  struct DlCloser {
    void *h;
    ~DlCloser() {
      if (h != nullptr) {
        dlclose(h);
      }
    }
  } closer{handle};

  auto create_fn = reinterpret_cast<rgpot_md_session_engine_create_fn>(
      dlsym(handle, "rgpot_md_session_engine_create"));
  require(create_fn != nullptr, "missing rgpot_md_session_engine_create");

  char error[512] = {};

  // Smaller host capacity calling "newer" engine: guard bytes stay intact.
  for (const size_t cap :
       {RGPOT_MD_OPS_END(destroy), RGPOT_MD_OPS_END(set_periodic_box) + 1}) {
    std::vector<unsigned char> buf(sizeof(rgpot_md_session_ops) + 64, 0xA5);
    std::memset(buf.data(), 0, cap);
    auto *ops = reinterpret_cast<rgpot_md_session_ops *>(buf.data());
    ops->struct_size = cap;
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(nullptr, 0, "newer", ops, &ctx, &n_atoms, error,
                      sizeof(error)) == 0,
            std::string("newer with small cap: ") + error);
    require(ops->struct_size == cap, "newer engine filled size mismatch");
    for (size_t i = cap; i < buf.size(); ++i) {
      require(buf[i] == 0xA5, "newer engine overran truncated host buffer");
    }
    OwnedSession s{
        rgpot_md_session_from_ops(ctx, ops, n_atoms, error, sizeof(error))};
    require(s.handle != nullptr, std::string("from_ops small cap: ") + error);
    double masses[2] = {};
    require(rgpot_md_session_get_masses(s.handle, masses, 2, error,
                                        sizeof(error)) == 0,
            error);
    require(masses[0] == static_cast<double>(cap) &&
                masses[1] == static_cast<double>(cap),
            "engine recorded wrong capacity/filled");
  }

  // Larger host capacity calling "current" engine: reports sizeof(ops) and
  // leaves trailing host fields intact.
  {
    struct LargerHostOps {
      rgpot_md_session_ops base;
      void (*future_cb)(void *);
      uint64_t future_tag;
      unsigned char guard[32];
    };
    LargerHostOps host{};
    std::memset(&host, 0xA5, sizeof(host));
    std::memset(&host.base, 0, sizeof(host.base));
    host.base.struct_size = sizeof(host);
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(nullptr, 0, "current", &host.base, &ctx, &n_atoms, error,
                      sizeof(error)) == 0,
            std::string("current with larger host: ") + error);
    require(host.base.struct_size == sizeof(rgpot_md_session_ops),
            "current engine did not report sizeof(rgpot_md_session_ops)");
    const auto *tail = reinterpret_cast<const unsigned char *>(&host) +
                       sizeof(rgpot_md_session_ops);
    for (size_t i = 0; i < sizeof(host) - sizeof(rgpot_md_session_ops); ++i) {
      require(tail[i] == 0xA5, "current engine wrote past its own struct");
    }
    OwnedSession s{rgpot_md_session_from_ops(ctx, &host.base, n_atoms, error,
                                             sizeof(error))};
    require(s.handle != nullptr, std::string("from_ops larger host: ") + error);
    double masses[2] = {};
    require(rgpot_md_session_get_masses(s.handle, masses, 2, error,
                                        sizeof(error)) == 0,
            error);
    require(masses[0] == static_cast<double>(sizeof(host)) &&
                masses[1] == static_cast<double>(sizeof(rgpot_md_session_ops)),
            "current engine capacity/filled mismatch for larger host");
  }

  // Larger host capacity calling "newer" engine: fills FakeNewerOps (larger
  // than current rgpot_md_session_ops) without exceeding host capacity.
  {
    struct LargerHostOps {
      rgpot_md_session_ops base;
      void (*future_cb)(void *);
      uint64_t future_tag;
      unsigned char guard[32];
    };
    LargerHostOps host{};
    std::memset(&host, 0xA5, sizeof(host));
    std::memset(&host.base, 0, sizeof(host.base));
    host.base.struct_size = sizeof(host);
    void *ctx = nullptr;
    size_t n_atoms = 0;
    require(create_fn(nullptr, 0, "newer", &host.base, &ctx, &n_atoms, error,
                      sizeof(error)) == 0,
            std::string("newer with larger host: ") + error);
    const size_t expected_filled =
        offsetof(LargerHostOps, future_tag) + sizeof(host.future_tag);
    require(host.base.struct_size == expected_filled,
            "newer engine did not fill up to its own size");
    require(host.future_cb != nullptr &&
                host.future_tag == UINT64_C(0x123456789abcdef0),
            "newer engine did not populate extra fields for larger host");
    for (unsigned char b : host.guard) {
      require(b == 0xA5, "newer engine wrote into trailing guard bytes");
    }
    OwnedSession s{rgpot_md_session_from_ops(ctx, &host.base, n_atoms, error,
                                             sizeof(error))};
    require(s.handle != nullptr, std::string("from_ops newer+larger: ") + error);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: md_session_abi_capacity <fake-engine>");
    check_host_modes(argv[1]);
    check_direct_capacities(argv[1]);
    std::cout << "MD session ABI capacity checks passed\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
