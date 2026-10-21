#pragma once
// MIT License — thin XTB frontend: dlopen libxtb_engine.so
#include "rgpot/ForceStructs.hpp"
#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/XTBPot/xtb_c_abi.h"

#include <string>

namespace rgpot {

/**
 * Plugin path: ``dlopen`` ``libxtb_engine.so``, which implements the same
 * ISO_C_BINDING singlepoint as linked ``XTBPot`` behind ``xtb_c_abi.h``.
 *
 * A non-empty ``engine_path`` is the only candidate, and it must contain a
 * directory separator (``/`` or ``\``). A bare soname such as
 * ``libxtb_engine.so`` is rejected: the loader would resolve it through
 * ``LD_LIBRARY_PATH``, ``DYLD_LIBRARY_PATH``, or ``PATH`` and could open a
 * different library. If the path does not open, or the library lacks the C
 * ABI, construction throws. The error names that path and, on POSIX, the
 * ``dlerror`` text. Environment variables, the bare soname, and directory
 * lists are not consulted.
 *
 * When ``engine_path`` is empty, search ``RGPOT_XTB_ENGINE``, ``XTB_ENGINE``,
 * the platform bare soname (``libxtb_engine.so`` on Linux,
 * ``libxtb_engine.dylib`` then ``.so`` on Apple, ``xtb_engine.dll`` then
 * ``libxtb_engine.dll`` on Windows), then ``EON_POTENTIALS_PATH`` /
 * ``RGPOT_ENGINE_PATH``.
 */
struct XTBDlopenConfig {
  XTBConfig xtb{};
  std::string engine_path;
};

class XTBDlopen {
public:
  explicit XTBDlopen(const XTBDlopenConfig &config);
  explicit XTBDlopen(const XTBConfig &xtb = XTBConfig{});
  ~XTBDlopen();

  XTBDlopen(const XTBDlopen &) = delete;
  XTBDlopen &operator=(const XTBDlopen &) = delete;

  [[nodiscard]] bool available() const noexcept { return m_pot != nullptr; }

  void forceImpl(const ForceInput &in, ForceOut *out) const;

  /// Convenience operator matching Potential interface shape.
  std::tuple<double, types::AtomMatrix, double>
  operator()(const types::AtomMatrix &positions,
             const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) const;

private:
  void *m_lib{nullptr};
  RgpotXtbPot *m_pot{nullptr};
  using create_fn = RgpotXtbPot *(*)(const RgpotXtbConfig *, char *, size_t);
  using destroy_fn = void (*)(RgpotXtbPot *);
  using force_fn = int (*)(RgpotXtbPot *, long, const double *, const int *,
                           double *, double *, double *, const double *);
  create_fn m_create{nullptr};
  destroy_fn m_destroy{nullptr};
  force_fn m_force{nullptr};
};

} // namespace rgpot
