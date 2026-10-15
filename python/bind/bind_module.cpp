// MIT License
// Copyright 2023--present rgpot developers
//
// Python surface: nanobind. F1 free-threaded when the interpreter is
// built that way (see python/meson.build). F4 releases the thread
// around every potential evaluation. F9 exports only PyInit__core.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "rgbind/rgbind.hpp"

#include "pot_holder.hpp"

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#endif

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
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

#ifndef RGPOT_PY_VERSION
#define RGPOT_PY_VERSION "0.0.0"
#endif

namespace nb = nanobind;
using rgpot::pybind::F64;
using rgpot::pybind::I32;
using rgpot::pybind::PotHolder;
using rgpot::types::AtomMatrix;

namespace {

void translate_invalid_argument(const std::exception_ptr &ptr, void *) {
  try {
    std::rethrow_exception(ptr);
  } catch (const std::invalid_argument &err) {
    PyErr_SetString(PyExc_ValueError, err.what());
  }
}

#ifdef RGPOT_HAS_EXPR
struct ChildSpec {
  std::string name;
  std::string kind;
  std::string functional = "pbe";
  std::string damping = "bj";
  bool atm = true;
  double charge = 0.0;
};

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
  throw nb::value_error(
      "ExprPot child spec must be a kind string or a dict with 'kind'");
}

ChildSpec child_spec_from(const std::string &name, nb::handle spec) {
  ChildSpec out;
  out.name = name;
  out.kind = child_kind(spec);
  if (nb::isinstance<nb::dict>(spec)) {
    auto d = nb::cast<nb::dict>(spec);
    if (d.contains("functional")) {
      out.functional = nb::cast<std::string>(d["functional"]);
    }
    if (d.contains("atm")) {
      out.atm = nb::cast<bool>(d["atm"]);
    }
    if (d.contains("damping")) {
      out.damping = nb::cast<std::string>(d["damping"]);
    }
    if (d.contains("charge")) {
      out.charge = nb::cast<double>(d["charge"]);
    }
  }
  return out;
}

std::unique_ptr<rgpot::PotentialBase> child_from_spec(const ChildSpec &spec) {
  if (spec.kind == "lj") {
    return std::make_unique<rgpot::LJPot>();
  }
  if (spec.kind == "morse") {
    return std::make_unique<rgpot::MorsePot>();
  }
#ifdef RGPOT_HAS_DFTD3
  if (spec.kind == "d3") {
    rgpot::D3Config cfg;
    cfg.functional = spec.functional;
    cfg.atm = spec.atm;
    cfg.damping = (spec.damping == "zero" || spec.damping == "Zero")
                      ? rgpot::D3Damping::Zero
                      : rgpot::D3Damping::BJ;
    return std::make_unique<rgpot::D3Pot>(cfg);
  }
#endif
#ifdef RGPOT_HAS_DFTD4
  if (spec.kind == "d4") {
    rgpot::D4Config cfg;
    cfg.functional = spec.functional;
    cfg.charge = spec.charge;
    cfg.atm = spec.atm;
    return std::make_unique<rgpot::D4Pot>(cfg);
  }
#endif
  throw std::invalid_argument("unknown ExprPot child kind: " + spec.kind);
}

std::unique_ptr<rgpot::ExprPot>
expr_from_specs(const std::string &expression,
                const std::vector<ChildSpec> &specs) {
  std::vector<rgpot::ExprPot::Term> children;
  children.reserve(specs.size());
  for (const ChildSpec &spec : specs) {
    children.emplace_back(spec.name, child_from_spec(spec));
  }
  return std::make_unique<rgpot::ExprPot>(expression, std::move(children));
}
#endif

#ifdef RGPOT_HAS_XCKERNEL
struct NumpyXcGrid {
  rgpot::XcGrid grid;
  F64 dchi, lapl, hess;
};

NumpyXcGrid grid_from_numpy(const F64 &chi, nb::handle dchi_obj,
                            nb::handle lapl_obj, nb::handle hess_obj) {
  rgbind::check_f64_rank(chi, 2, "chi");
  NumpyXcGrid storage;
  auto &grid = storage.grid;
  grid.nbf = static_cast<std::int64_t>(chi.shape(0));
  grid.npts = static_cast<std::int64_t>(chi.shape(1));
  grid.chi = chi.data();
  if (!dchi_obj.is_none()) {
    storage.dchi = nb::cast<F64>(dchi_obj, false);
    const auto &dchi = storage.dchi;
    if (dchi.ndim() != 3 || dchi.shape(0) != 3 ||
        dchi.shape(1) != chi.shape(0) || dchi.shape(2) != chi.shape(1)) {
      throw nb::value_error("dchi: expected shape (3, nbf, npts)");
    }
    grid.dchi = dchi.data();
  }
  if (!lapl_obj.is_none()) {
    storage.lapl = nb::cast<F64>(lapl_obj, false);
    const auto &lapl = storage.lapl;
    if (lapl.ndim() != 2 || lapl.shape(0) != chi.shape(0) ||
        lapl.shape(1) != chi.shape(1)) {
      throw nb::value_error("lapl_chi: expected shape (nbf, npts)");
    }
    grid.lapl_chi = lapl.data();
  }
  if (!hess_obj.is_none()) {
    storage.hess = nb::cast<F64>(hess_obj, false);
    const auto &hess = storage.hess;
    if (hess.ndim() != 3 || hess.shape(0) != 6 ||
        hess.shape(1) != chi.shape(0) || hess.shape(2) != chi.shape(1)) {
      throw nb::value_error("hess_chi: expected shape (6, nbf, npts)");
    }
    grid.hess_chi = hess.data();
  }
  return storage;
}

struct NumpyXcScalars {
  std::map<std::string, const double *> pointers;
  std::vector<F64> arrays;
};

NumpyXcScalars scal_from_dict(const nb::dict &scal, std::size_t npts) {
  NumpyXcScalars storage;
  storage.arrays.reserve(scal.size());
  for (auto item : scal) {
    auto name = nb::cast<std::string>(item.first);
    auto arr = nb::cast<F64>(item.second, false);
    if (!((arr.ndim() == 1 && arr.shape(0) == npts) ||
          (arr.ndim() == 2 && arr.shape(0) == npts && arr.shape(1) == 1))) {
      throw nb::value_error(
          ("scalar " + name + ": expected shape (npts,) or (npts, 1)").c_str());
    }
    storage.pointers.emplace(std::move(name), arr.data());
    storage.arrays.push_back(std::move(arr));
  }
  return storage;
}
#endif

#ifdef RGPOT_HAS_XCKERNEL
nb::ndarray<nb::numpy, double, nb::c_contig>
buffer_to_numpy(const double *src, size_t rows, size_t cols) {
  const size_t n = rows * cols;
  double *buf = new double[n == 0 ? 1 : n];
  if (n != 0 && src != nullptr) {
    std::memcpy(buf, src, n * sizeof(double));
  }
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}
#endif

#ifdef RGPOT_HAS_FORTRAN_POTS
template <typename Pot>
void bind_default_pot(nb::module_ &m, const char *name) {
  auto cls = nb::class_<PotHolder<Pot>>(m, name);
  cls.def("__init__", [](PotHolder<Pot> *self) {
    new (self) PotHolder<Pot>([] { return std::make_unique<Pot>(); });
  });
  rgpot::pybind::bind_calls(cls);
}
#endif

} // namespace

NB_MODULE(_core, m) {
  // F1, F3, F7. install_policy registers NumericalError and the
  // in-use translator. A second translator maps invalid_argument to
  // ValueError (F7).
  rgbind::install_policy(m);
  nb::register_exception_translator(translate_invalid_argument);

  m.doc() =
      "rgpot Python bindings (nanobind): LJ, Fortran kernels, Metatomic "
      "dlopen, ExprPot, D3Pot, D4Pot, and XcKernel when the build flags "
      "are on. Evaluations release the thread (F4).";
  m.attr("__version__") = RGPOT_PY_VERSION;
#ifdef RGPOT_HAS_METATOMIC
  m.attr("has_metatomic_dlopen") = true;
#else
  m.attr("has_metatomic_dlopen") = false;
#endif
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
#ifdef RGPOT_HAS_FORTRAN_POTS
  m.attr("has_fortran_pots") = true;
#else
  m.attr("has_fortran_pots") = false;
#endif

  // F6 round trip. -0.0, NaN, and subnormals are copied unchanged.
  m.def(
      "echo_f64",
      [](const F64 &values) { return rgbind::echo_f64(values); },
      nb::arg("values").noconvert());

  m.def(
      "evaluate_lj",
      [](const F64 &positions, const I32 &atom_types, const F64 &box) {
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        rgpot::LJPot pot;
        double energy = 0.0;
        double variance = 0.0;
        AtomMatrix forces;
        {
          nb::gil_scoped_release release;
          std::tie(energy, forces, variance) =
              pot(geom.pos, geom.types, geom.box);
        }
        if (!std::isfinite(energy)) {
          throw rgbind::NumericalError("energy is not finite");
        }
        return nb::make_tuple(energy, rgpot::pybind::matrix_to_numpy(forces),
                              variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types").noconvert(),
      nb::arg("box").noconvert());

#ifdef RGPOT_HAS_METATOMIC
  m.def(
      "evaluate_metatomic_dlopen",
      [](const F64 &positions, const I32 &atom_types, const F64 &box,
         const std::string &model_path, const std::string &engine_path,
         const std::string &device) {
        if (model_path.empty()) {
          throw nb::value_error("model_path is required");
        }
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        rgpot::MetatomicConfig cfg;
        cfg.model_path = model_path;
        cfg.engine_path = engine_path;
        cfg.device = device.empty() ? "cpu" : device;
        rgpot::MetatomicDlopen front(cfg);
        if (!front.available()) {
          throw std::runtime_error("MetatomicDlopen: engine not available");
        }
        const size_t n = geom.pos.rows();
        AtomMatrix forces = AtomMatrix::Zero(n, 3);
        double flat[9];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            flat[i * 3 + j] =
                geom.box[static_cast<size_t>(i)][static_cast<size_t>(j)];
          }
        }
        rgpot::ForceInput in{n, geom.pos.data(), geom.types.data(), flat};
        rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
        {
          nb::gil_scoped_release release;
          front.forceImpl(in, &out);
        }
        if (!std::isfinite(out.energy)) {
          throw rgbind::NumericalError("energy is not finite");
        }
        return nb::make_tuple(out.energy, rgpot::pybind::matrix_to_numpy(forces),
                              out.variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types").noconvert(),
      nb::arg("box").noconvert(), nb::arg("model_path"),
      nb::arg("engine_path") = "", nb::arg("device") = "cpu",
      "Evaluate forces via MetatomicDlopen (dlopen libmetatomic_engine.so). "
      "engine_path may be empty to use RGPOT_METATOMIC_ENGINE / "
      "package-bundled multi-ABI engine under rgpot/lib/torch-X.Y/. "
      "Each call builds its own frontend (F4).");
#endif

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

  auto lj = nb::class_<PotHolder<rgpot::LJPot>>(m, "LJPot");
  lj.def(
      "__init__",
      [](PotHolder<rgpot::LJPot> *self, double u0, double cutoff, double psi,
         double switch_width) {
        new (self) PotHolder<rgpot::LJPot>([=] {
          rgpot::LJConfig cfg;
          cfg.u0 = u0;
          cfg.cutoff = cutoff;
          cfg.psi = psi;
          cfg.switch_width = switch_width;
          return std::make_unique<rgpot::LJPot>(cfg);
        });
      },
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 15.0, nb::arg("psi") = 1.0,
      nb::arg("switch_width") = 0.0);
  lj.def("config", [](const PotHolder<rgpot::LJPot> &self) {
    const rgpot::LJConfig &cfg = self.primary().config();
    nb::dict out;
    out["u0"] = cfg.u0;
    out["cutoff"] = cfg.cutoff;
    out["psi"] = cfg.psi;
    out["switch_width"] = cfg.switch_width;
    return out;
  });
#ifdef RGPOT_HAS_CACHE
  lj.def(
      "set_cache",
      [](PotHolder<rgpot::LJPot> &self, rgpot::cache::PotentialCache *cache) {
        self.primary().set_cache(cache);
      },
      nb::arg("cache").none(), nb::keep_alive<1, 2>());
#endif
  rgpot::pybind::bind_calls(lj);

#ifdef RGPOT_HAS_FORTRAN_POTS
  bind_default_pot<rgpot::fortranpots::SWPot>(m, "SWPot");
  bind_default_pot<rgpot::fortranpots::EDIPPot>(m, "EDIPPot");
  bind_default_pot<rgpot::fortranpots::LenoskyPot>(m, "LenoskyPot");
  bind_default_pot<rgpot::fortranpots::TersoffPot>(m, "TersoffPot");
  bind_default_pot<rgpot::fortranpots::EAMAlPot>(m, "EAMAlPot");
  bind_default_pot<rgpot::fortranpots::FeHePot>(m, "FeHePot");
  bind_default_pot<rgpot::fortranpots::CuH2Pot>(m, "CuH2Pot");
  bind_default_pot<rgpot::fortranpots::WaterHPot>(m, "WaterHPot");
#endif

#ifdef RGPOT_HAS_DFTD3
  auto d3 = nb::class_<PotHolder<rgpot::D3Pot>>(m, "D3Pot");
  d3.def(
      "__init__",
      [](PotHolder<rgpot::D3Pot> *self, const std::string &damping,
         const std::string &functional, bool atm) {
        new (self) PotHolder<rgpot::D3Pot>([=] {
          rgpot::D3Config cfg;
          cfg.functional = functional;
          cfg.atm = atm;
          cfg.damping = (damping == "zero" || damping == "Zero")
                            ? rgpot::D3Damping::Zero
                            : rgpot::D3Damping::BJ;
          return std::make_unique<rgpot::D3Pot>(cfg);
        });
      },
      nb::arg("damping") = "bj", nb::arg("functional") = "pbe",
      nb::arg("atm") = true);
  rgpot::pybind::bind_calls(d3);
#endif

#ifdef RGPOT_HAS_DFTD4
  auto d4 = nb::class_<PotHolder<rgpot::D4Pot>>(m, "D4Pot");
  d4.def(
      "__init__",
      [](PotHolder<rgpot::D4Pot> *self, const std::string &functional,
         double charge, bool atm) {
        new (self) PotHolder<rgpot::D4Pot>([=] {
          rgpot::D4Config cfg;
          cfg.functional = functional;
          cfg.charge = charge;
          cfg.atm = atm;
          return std::make_unique<rgpot::D4Pot>(cfg);
        });
      },
      nb::arg("functional") = "pbe", nb::arg("charge") = 0.0,
      nb::arg("atm") = true);
  rgpot::pybind::bind_calls(d4);
#endif

#ifdef RGPOT_HAS_EXPR
  auto expr = nb::class_<PotHolder<rgpot::ExprPot>>(m, "ExprPot");
  expr.def(
      "__init__",
      [](PotHolder<rgpot::ExprPot> *self, const std::string &expression,
         const nb::dict &terms) {
        std::vector<ChildSpec> specs;
        specs.reserve(terms.size());
        for (auto item : terms) {
          auto name = nb::cast<std::string>(item.first);
          specs.push_back(child_spec_from(name, item.second));
        }
        new (self) PotHolder<rgpot::ExprPot>([=] {
          return expr_from_specs(expression, specs);
        });
      },
      nb::arg("expression"), nb::arg("terms"),
      "expression is a Lepton string; terms maps names to 'lj'/'morse'/"
      "'d3'/'d4' or a dict with kind plus D3/D4 fields.");
  expr.def_prop_ro("expression", [](const PotHolder<rgpot::ExprPot> &self) {
    return self.primary().expression();
  });
  expr.def(
      "d_energy_d_term",
      [](PotHolder<rgpot::ExprPot> &self, const std::string &name) {
        nb::gil_scoped_release release;
        return self.slot().dEnergyDTerm(name);
      },
      nb::arg("name"));
  rgpot::pybind::bind_calls(expr);
#endif

#ifdef RGPOT_HAS_XCKERNEL
  nb::class_<rgpot::XcKernel>(m, "XcKernel")
      .def(nb::init<std::string>(), nb::arg("name"))
      .def_static("catalog", &rgpot::XcKernel::catalog)
      .def_prop_ro("name", &rgpot::XcKernel::name)
      .def("scal_names", &rgpot::XcKernel::scalNames)
      .def(
          "contract",
          [](const rgpot::XcKernel &self, const F64 &chi, nb::handle dchi,
             const nb::dict &scal, nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            auto scalars = scal_from_dict(scal, static_cast<size_t>(grid.npts));
            const auto nbf = static_cast<size_t>(grid.nbf);
            std::vector<double> out(nbf * nbf, 0.0);
            int rc = 0;
            {
              nb::gil_scoped_release release;
              rc = self.contract(grid, scalars.pointers, out.data());
            }
            if (rc != 0) {
              throw std::runtime_error("XcKernel.contract rc=" +
                                       std::to_string(rc));
            }
            return buffer_to_numpy(out.data(), nbf, nbf);
          },
          nb::arg("chi").noconvert(), nb::arg("dchi"), nb::arg("scal"),
          nb::arg("lapl_chi") = nb::none(),
          nb::arg("hess_chi") = nb::none())
      .def_static(
          "fields_from_density",
          [](const F64 &chi, nb::handle dchi, const F64 &density,
             nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            rgbind::check_f64_2d(density, static_cast<size_t>(chi.shape(0)),
                                 static_cast<size_t>(chi.shape(0)), "density");
            std::vector<double> density_copy(
                density.data(), density.data() + density.size());
            rgpot::XcFields fields;
            {
              nb::gil_scoped_release release;
              fields = rgpot::XcKernel::fieldsFromDensity(grid,
                                                         density_copy.data());
            }
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
          nb::arg("chi").noconvert(), nb::arg("dchi"),
          nb::arg("density").noconvert(), nb::arg("lapl_chi") = nb::none(),
          nb::arg("hess_chi") = nb::none());
#endif
}
