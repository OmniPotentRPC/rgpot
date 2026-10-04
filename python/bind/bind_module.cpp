// MIT License
// Copyright 2023--present rgpot developers
//
// Python surface: nanobind (stable ABI / abi3 when Python >= 3.12).
// LJ + MetatomicDlopen + optional ExprPot / D3Pot / D4Pot / XcKernel.

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"

#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif
#ifdef RGPOT_HAS_EXPR
#include "rgpot/ExprPot/ExprPot.hpp"
#endif

#ifndef RGPOT_PY_VERSION
#define RGPOT_PY_VERSION "0.0.0"
#endif

namespace nb = nanobind;
using rgpot::types::AtomMatrix;

using NpF64 = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
using NpI32 = nb::ndarray<nb::numpy, int32_t, nb::c_contig, nb::device::cpu>;

namespace {

AtomMatrix numpy_to_atom_matrix(const NpF64 &arr) {
  if (arr.ndim() != 2 || arr.shape(1) != 3) {
    throw std::invalid_argument("positions must have shape (n_atoms, 3)");
  }
  const auto n = static_cast<size_t>(arr.shape(0));
  AtomMatrix mat(n, 3);
  std::memcpy(mat.data(), arr.data(), n * 3 * sizeof(double));
  return mat;
}

std::vector<int> numpy_to_types(const NpI32 &arr, size_t n_atoms) {
  if (arr.ndim() != 1 || static_cast<size_t>(arr.shape(0)) != n_atoms) {
    throw std::invalid_argument("atom_types length must match n_atoms");
  }
  std::vector<int> types(n_atoms);
  for (size_t i = 0; i < n_atoms; ++i) {
    types[i] = static_cast<int>(arr.data()[i]);
  }
  return types;
}

std::array<std::array<double, 3>, 3> numpy_to_box(const NpF64 &arr) {
  if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3) {
    throw std::invalid_argument("box must have shape (3, 3)");
  }
  std::array<std::array<double, 3>, 3> box{};
  const double *p = arr.data();
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      box[i][j] = p[i * 3 + j];
  return box;
}

nb::ndarray<nb::numpy, double, nb::c_contig>
atom_matrix_to_numpy(const AtomMatrix &mat) {
  const size_t rows = mat.rows();
  const size_t cols = mat.cols();
  double *buf = new double[rows * cols];
  std::memcpy(buf, mat.data(), rows * cols * sizeof(double));
  nb::capsule owner(buf, [](void *p) noexcept {
    delete[] static_cast<double *>(p);
  });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}

nb::ndarray<nb::numpy, double, nb::c_contig>
buffer_to_numpy(const double *src, size_t rows, size_t cols) {
  double *buf = new double[rows * cols];
  std::memcpy(buf, src, rows * cols * sizeof(double));
  nb::capsule owner(buf, [](void *p) noexcept {
    delete[] static_cast<double *>(p);
  });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}

void flat_box(const std::array<std::array<double, 3>, 3> &box, double out[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      out[i * 3 + j] = box[i][j];
}

template <typename Pot>
nb::tuple call_pot(Pot &pot, const NpF64 &positions, const NpI32 &atom_types,
                   const NpF64 &box) {
  auto pos = numpy_to_atom_matrix(positions);
  auto types = numpy_to_types(atom_types, pos.rows());
  auto cell = numpy_to_box(box);
  auto [energy, forces, variance] = pot(pos, types, cell);
  return nb::make_tuple(energy, atom_matrix_to_numpy(forces), variance);
}

nb::tuple evaluate_lj(const NpF64 &positions, const NpI32 &atom_types,
                      const NpF64 &box) {
  rgpot::LJPot pot;
  return call_pot(pot, positions, atom_types, box);
}

nb::tuple evaluate_metatomic_dlopen(const NpF64 &positions,
                                    const NpI32 &atom_types, const NpF64 &box,
                                    const std::string &model_path,
                                    const std::string &engine_path,
                                    const std::string &device) {
  if (positions.ndim() != 2 || atom_types.ndim() != 1 || box.ndim() != 2) {
    throw std::invalid_argument(
        "evaluate_metatomic_dlopen expects positions (n,3), atom_types (n,), "
        "box (3,3)");
  }
  if (model_path.empty()) {
    throw std::invalid_argument("model_path is required");
  }
  auto pos = numpy_to_atom_matrix(positions);
  auto types = numpy_to_types(atom_types, pos.rows());
  auto cell = numpy_to_box(box);

  rgpot::MetatomicConfig cfg;
  cfg.model_path = model_path;
  cfg.engine_path = engine_path;
  cfg.device = device.empty() ? "cpu" : device;

  rgpot::MetatomicDlopen front(cfg);
  if (!front.available()) {
    throw std::runtime_error("MetatomicDlopen: engine not available");
  }

  const size_t n = pos.rows();
  AtomMatrix forces = AtomMatrix::Zero(n, 3);
  double flat[9];
  flat_box(cell, flat);
  rgpot::ForceInput in{n, pos.data(), types.data(), flat};
  rgpot::ForceOut out{forces.data(), 0.0, 0.0};
  {
    nb::gil_scoped_release release;
    front.forceImpl(in, &out);
  }
  return nb::make_tuple(out.energy, atom_matrix_to_numpy(forces), out.variance);
}

#ifdef RGPOT_HAS_EXPR
std::string child_kind(nb::handle spec) {
  if (nb::isinstance<nb::str>(spec)) {
    return nb::cast<std::string>(spec);
  }
  if (nb::isinstance<nb::dict>(spec)) {
    auto d = nb::cast<nb::dict>(spec);
    if (d.contains("kind")) {
      return nb::cast<std::string>(d["kind"]);
    }
    if (d.contains("type")) {
      return nb::cast<std::string>(d["type"]);
    }
  }
  throw std::invalid_argument(
      "ExprPot child spec must be a kind string or a dict with 'kind'");
}

std::unique_ptr<rgpot::PotentialBase> make_expr_child(nb::handle spec) {
  const std::string kind = child_kind(spec);
  if (kind == "lj") {
    return std::make_unique<rgpot::LJPot>();
  }
  if (kind == "morse") {
    return std::make_unique<rgpot::MorsePot>();
  }
#ifdef RGPOT_HAS_DFTD3
  if (kind == "d3") {
    rgpot::D3Config cfg;
    if (nb::isinstance<nb::dict>(spec)) {
      auto d = nb::cast<nb::dict>(spec);
      if (d.contains("functional")) {
        cfg.functional = nb::cast<std::string>(d["functional"]);
      }
      if (d.contains("atm")) {
        cfg.atm = nb::cast<bool>(d["atm"]);
      }
      if (d.contains("damping")) {
        const auto damp = nb::cast<std::string>(d["damping"]);
        cfg.damping = (damp == "zero" || damp == "Zero")
                          ? rgpot::D3Damping::Zero
                          : rgpot::D3Damping::BJ;
      }
    }
    return std::make_unique<rgpot::D3Pot>(cfg);
  }
#endif
#ifdef RGPOT_HAS_DFTD4
  if (kind == "d4") {
    rgpot::D4Config cfg;
    if (nb::isinstance<nb::dict>(spec)) {
      auto d = nb::cast<nb::dict>(spec);
      if (d.contains("functional")) {
        cfg.functional = nb::cast<std::string>(d["functional"]);
      }
      if (d.contains("atm")) {
        cfg.atm = nb::cast<bool>(d["atm"]);
      }
      if (d.contains("charge")) {
        cfg.charge = nb::cast<double>(d["charge"]);
      }
    }
    return std::make_unique<rgpot::D4Pot>(cfg);
  }
#endif
  throw std::invalid_argument("unknown ExprPot child kind: " + kind);
}

std::unique_ptr<rgpot::ExprPot> make_expr_pot(const std::string &expression,
                                              const nb::dict &terms) {
  std::vector<rgpot::ExprPot::Term> children;
  children.reserve(terms.size());
  for (auto item : terms) {
    auto name = nb::cast<std::string>(item.first);
    children.emplace_back(std::move(name), make_expr_child(item.second));
  }
  return std::make_unique<rgpot::ExprPot>(expression, std::move(children));
}
#endif

#ifdef RGPOT_HAS_XCKERNEL
struct NumpyXcGrid {
  rgpot::XcGrid grid;
  NpF64 dchi, lapl, hess;
};

NumpyXcGrid grid_from_numpy(const NpF64 &chi, nb::handle dchi_obj,
                            nb::handle lapl_obj, nb::handle hess_obj) {
  if (chi.ndim() != 2) {
    throw std::invalid_argument("chi must have shape (nbf, npts)");
  }
  NumpyXcGrid storage;
  auto &grid = storage.grid;
  grid.nbf = static_cast<std::int64_t>(chi.shape(0));
  grid.npts = static_cast<std::int64_t>(chi.shape(1));
  grid.chi = chi.data();
  if (!dchi_obj.is_none()) {
    storage.dchi = nb::cast<NpF64>(dchi_obj);
    const auto &dchi = storage.dchi;
    if (dchi.ndim() != 3 || dchi.shape(0) != 3 ||
        dchi.shape(1) != chi.shape(0) || dchi.shape(2) != chi.shape(1)) {
      throw std::invalid_argument("dchi must have shape (3, nbf, npts)");
    }
    grid.dchi = dchi.data();
  }
  if (!lapl_obj.is_none()) {
    storage.lapl = nb::cast<NpF64>(lapl_obj);
    const auto &lapl = storage.lapl;
    if (lapl.ndim() != 2 || lapl.shape(0) != chi.shape(0) ||
        lapl.shape(1) != chi.shape(1)) {
      throw std::invalid_argument("lapl_chi must have shape (nbf, npts)");
    }
    grid.lapl_chi = lapl.data();
  }
  if (!hess_obj.is_none()) {
    storage.hess = nb::cast<NpF64>(hess_obj);
    const auto &hess = storage.hess;
    if (hess.ndim() != 3 || hess.shape(0) != 6 ||
        hess.shape(1) != chi.shape(0) || hess.shape(2) != chi.shape(1)) {
      throw std::invalid_argument("hess_chi must have shape (6, nbf, npts)");
    }
    grid.hess_chi = hess.data();
  }
  return storage;
}

struct NumpyXcScalars {
  std::map<std::string, const double *> pointers;
  std::vector<NpF64> arrays;
};

NumpyXcScalars scal_from_dict(const nb::dict &scal, std::size_t npts) {
  NumpyXcScalars storage;
  storage.arrays.reserve(scal.size());
  for (auto item : scal) {
    auto name = nb::cast<std::string>(item.first);
    auto arr = nb::cast<NpF64>(item.second);
    if (!((arr.ndim() == 1 && arr.shape(0) == npts) ||
          (arr.ndim() == 2 && arr.shape(0) == npts && arr.shape(1) == 1))) {
      throw std::invalid_argument("scalar " + name +
                                  " must have shape (npts,) or (npts, 1)");
    }
    storage.pointers.emplace(std::move(name), arr.data());
    storage.arrays.push_back(std::move(arr));
  }
  return storage;
}
#endif

} // namespace

NB_MODULE(_core, m) {
  m.doc() =
      "rgpot Python bindings (nanobind): LJ, Metatomic dlopen, ExprPot, "
      "D3Pot, D4Pot, and XcKernel when the wheel/meson flags are on.";
  m.attr("__version__") = RGPOT_PY_VERSION;
  m.attr("has_metatomic_dlopen") = true;
#ifdef RGPOT_HAS_CACHE
  m.attr("has_cache") = true;
#else
  m.attr("has_cache") = false;
#endif
#ifdef RGPOT_HAS_EXPR
  m.attr("has_expr") = true;
#else
  m.attr("has_expr") = false;
#endif
#ifdef RGPOT_HAS_DFTD3
  m.attr("has_dftd3") = true;
#else
  m.attr("has_dftd3") = false;
#endif
#ifdef RGPOT_HAS_DFTD4
  m.attr("has_dftd4") = true;
#else
  m.attr("has_dftd4") = false;
#endif
#ifdef RGPOT_HAS_XCKERNEL
  m.attr("has_xckernel") = true;
#else
  m.attr("has_xckernel") = false;
#endif

  m.def("evaluate_lj", &evaluate_lj, nb::arg("positions"),
        nb::arg("atom_types"), nb::arg("box"));

  m.def("evaluate_metatomic_dlopen", &evaluate_metatomic_dlopen,
        nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"),
        nb::arg("model_path"), nb::arg("engine_path") = "",
        nb::arg("device") = "cpu",
        "Evaluate forces via MetatomicDlopen (dlopen libmetatomic_engine.so). "
        "engine_path may be empty to use RGPOT_METATOMIC_ENGINE / "
        "package-bundled multi-ABI engine under rgpot/lib/torch-X.Y/.");

#ifdef RGPOT_HAS_CACHE
  nb::class_<rgpot::cache::PotentialCache>(m, "PotentialCache")
      .def(nb::init<const std::string &, bool>(), nb::arg("path"),
           nb::arg("create_if_missing") = true)
      .def(
          "counts",
          [](const rgpot::cache::PotentialCache &self) {
            const auto c = self.counts();
            return nb::make_tuple(c.computed, c.served);
          },
          "Return (computed, served): results the kernel produced against "
          "results answered from this cache since construction or the last "
          "reset_counts().")
      .def("reset_counts", &rgpot::cache::PotentialCache::reset_counts);
#endif

  nb::class_<rgpot::LJPot>(m, "LJPot")
      .def(nb::init<>())
#ifdef RGPOT_HAS_CACHE
      .def(
          "set_cache",
          [](rgpot::LJPot &self, rgpot::cache::PotentialCache *cache) {
            self.set_cache(cache);
          },
          nb::arg("cache").none(), nb::keep_alive<1, 2>())
#endif
      .def(
          "__call__",
          [](rgpot::LJPot &self, const NpF64 &positions,
             const NpI32 &atom_types, const NpF64 &box) {
            return call_pot(self, positions, atom_types, box);
          },
          nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"));

#ifdef RGPOT_HAS_DFTD3
  nb::class_<rgpot::D3Pot>(m, "D3Pot")
      .def(nb::init<>())
      .def(
          "__init__",
          [](rgpot::D3Pot *self, const std::string &damping,
             const std::string &functional, bool atm) {
            rgpot::D3Config cfg;
            cfg.functional = functional;
            cfg.atm = atm;
            cfg.damping = (damping == "zero" || damping == "Zero")
                              ? rgpot::D3Damping::Zero
                              : rgpot::D3Damping::BJ;
            new (self) rgpot::D3Pot(cfg);
          },
          nb::arg("damping") = "bj", nb::arg("functional") = "pbe",
          nb::arg("atm") = true)
      .def(
          "__call__",
          [](rgpot::D3Pot &self, const NpF64 &positions,
             const NpI32 &atom_types, const NpF64 &box) {
            return call_pot(self, positions, atom_types, box);
          },
          nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"));
#endif

#ifdef RGPOT_HAS_DFTD4
  nb::class_<rgpot::D4Pot>(m, "D4Pot")
      .def(nb::init<>())
      .def(
          "__init__",
          [](rgpot::D4Pot *self, const std::string &functional, double charge,
             bool atm) {
            rgpot::D4Config cfg;
            cfg.functional = functional;
            cfg.charge = charge;
            cfg.atm = atm;
            new (self) rgpot::D4Pot(cfg);
          },
          nb::arg("functional") = "pbe", nb::arg("charge") = 0.0,
          nb::arg("atm") = true)
      .def(
          "__call__",
          [](rgpot::D4Pot &self, const NpF64 &positions,
             const NpI32 &atom_types, const NpF64 &box) {
            return call_pot(self, positions, atom_types, box);
          },
          nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"));
#endif

#ifdef RGPOT_HAS_EXPR
  nb::class_<rgpot::ExprPot>(m, "ExprPot")
      .def(
          "__init__",
          [](rgpot::ExprPot *self, const std::string &expression,
             const nb::dict &terms) {
            auto pot = make_expr_pot(expression, terms);
            new (self) rgpot::ExprPot(std::move(*pot));
          },
          nb::arg("expression"), nb::arg("terms"),
          "expression is a Lepton string; terms maps names to 'lj'/'morse'/"
          "'d3'/'d4' or a dict with kind plus D3/D4 fields.")
      .def(
          "__call__",
          [](rgpot::ExprPot &self, const NpF64 &positions,
             const NpI32 &atom_types, const NpF64 &box) {
            return call_pot(self, positions, atom_types, box);
          },
          nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"))
      .def_prop_ro("expression", &rgpot::ExprPot::expression)
      .def("d_energy_d_term", &rgpot::ExprPot::dEnergyDTerm, nb::arg("name"));
#endif

#ifdef RGPOT_HAS_XCKERNEL
  nb::class_<rgpot::XcKernel>(m, "XcKernel")
      .def(nb::init<std::string>(), nb::arg("name"))
      .def_static("catalog", &rgpot::XcKernel::catalog)
      .def_prop_ro("name", &rgpot::XcKernel::name)
      .def("scal_names", &rgpot::XcKernel::scalNames)
      .def(
          "contract",
          [](const rgpot::XcKernel &self, const NpF64 &chi, nb::handle dchi,
             const nb::dict &scal, nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            auto scalars = scal_from_dict(scal, static_cast<size_t>(grid.npts));
            const auto nbf = static_cast<size_t>(grid.nbf);
            std::vector<double> out(nbf * nbf, 0.0);
            const int rc = self.contract(grid, scalars.pointers, out.data());
            if (rc != 0) {
              throw std::runtime_error("XcKernel.contract rc=" +
                                       std::to_string(rc));
            }
            return buffer_to_numpy(out.data(), nbf, nbf);
          },
          nb::arg("chi"), nb::arg("dchi") = nb::none(), nb::arg("scal"),
          nb::arg("lapl_chi") = nb::none(), nb::arg("hess_chi") = nb::none())
      .def_static(
          "fields_from_density",
          [](const NpF64 &chi, nb::handle dchi, const NpF64 &density,
             nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            if (density.ndim() != 2 ||
                density.shape(0) != chi.shape(0) ||
                density.shape(1) != chi.shape(0)) {
              throw std::invalid_argument("density must have shape (nbf, nbf)");
            }
            auto fields = rgpot::XcKernel::fieldsFromDensity(grid, density.data());
            nb::dict out;
            auto vec1 = [](const std::vector<double> &v, size_t n) {
              return buffer_to_numpy(v.data(), n, 1);
            };
            const auto npts = static_cast<size_t>(grid.npts);
            out["rho"] = vec1(fields.rho, npts);
            out["sigma"] = vec1(fields.sigma, npts);
            out["tau"] = vec1(fields.tau, npts);
            out["lapl"] = vec1(fields.lapl, npts);
            if (!fields.grad_rho.empty()) {
              out["grad_rho"] = buffer_to_numpy(fields.grad_rho.data(), 3, npts);
            }
            return out;
          },
          nb::arg("chi"), nb::arg("dchi"), nb::arg("density"),
          nb::arg("lapl_chi") = nb::none(), nb::arg("hess_chi") = nb::none());
#endif
}
