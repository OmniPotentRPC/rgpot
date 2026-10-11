// MIT License
// Copyright 2023--present rgpot developers
//
// Per-call evaluation helpers for the free-threaded module (F2, F4, F6).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "rgbind/rgbind.hpp"

#include "rgpot/BatchEvaluator.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace nb = nanobind;

namespace rgpot::pybind {

using F64 = rgbind::F64Array;
using I32 = nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu>;
using AtomMatrix = types::AtomMatrix;

/// One geometry copied out of NumPy before the caller detaches (F4, F6).
struct Geom {
  AtomMatrix pos;
  std::vector<int> types;
  std::array<std::array<double, 3>, 3> box{};
};

inline std::string shape_of(const F64 &arr) {
  std::string text = "(";
  for (size_t i = 0; i < arr.ndim(); ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += std::to_string(arr.shape(i));
  }
  if (arr.ndim() == 1) {
    text += ",";
  }
  text += ")";
  return text;
}

inline std::string shape_of_i(const I32 &arr) {
  std::string text = "(";
  for (size_t i = 0; i < arr.ndim(); ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += std::to_string(arr.shape(i));
  }
  if (arr.ndim() == 1) {
    text += ",";
  }
  text += ")";
  return text;
}

/// F6: positions are (N, 3) float64, C contiguous. N is not fixed.
inline void check_positions(const F64 &arr) {
  rgbind::check_f64_rank(arr, 2, "positions");
  if (arr.shape(1) != 3) {
    const std::string msg = "positions: expected shape (N, 3), got " +
                            shape_of(arr);
    throw nb::value_error(msg.c_str());
  }
}

inline void check_types(const I32 &arr, size_t n) {
  if (arr.ndim() == 1 && static_cast<size_t>(arr.shape(0)) == n) {
    return;
  }
  const std::string msg = "atom_types: expected shape (" + std::to_string(n) +
                          ",), got " + shape_of_i(arr);
  throw nb::value_error(msg.c_str());
}

inline Geom read_geom(const F64 &positions, const I32 &atom_types,
                      const F64 &box) {
  check_positions(positions);
  const size_t n = static_cast<size_t>(positions.shape(0));
  check_types(atom_types, n);
  rgbind::check_f64_2d(box, 3, 3, "box");
  Geom g;
  g.pos = AtomMatrix(n, 3);
  std::memcpy(g.pos.data(), positions.data(), n * 3 * sizeof(double));
  g.types.resize(n);
  for (size_t i = 0; i < n; ++i) {
    g.types[i] = static_cast<int>(atom_types.data()[i]);
  }
  const double *p = box.data();
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      g.box[i][j] = p[i * 3 + j];
    }
  }
  return g;
}

inline nb::ndarray<nb::numpy, double, nb::c_contig>
matrix_to_numpy(const AtomMatrix &mat) {
  const size_t rows = mat.rows();
  const size_t cols = mat.cols();
  const size_t n = rows * cols;
  double *buf = new double[n == 0 ? 1 : n];
  if (n != 0) {
    std::memcpy(buf, mat.data(), n * sizeof(double));
  }
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}

inline nb::ndarray<nb::numpy, double, nb::c_contig>
stress_to_numpy(const double stress[9]) {
  double *buf = new double[9];
  std::memcpy(buf, stress, 9 * sizeof(double));
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {3, 3}, owner);
}

inline nb::dict caps_dict(const PotCaps &caps) {
  nb::dict out;
  const char *reentrancy = "shared";
  if (caps.reentrancy == Reentrancy::PerInstance) {
    reentrancy = "per_instance";
  } else if (caps.reentrancy == Reentrancy::ProcessSerial) {
    reentrancy = "process_serial";
  }
  out["reentrancy"] = reentrancy;
  out["per_image"] = caps.perImageInstances;
  out["batched"] = caps.batched;
  out["periodic"] = caps.periodic;
  out["group_collective"] = caps.groupCollective;
  out["stress"] = caps.stress;
  // PotCaps does not report per-atom energies. Do not invent the flag.
  return out;
}

struct FullResult {
  double energy = 0.0;
  AtomMatrix forces;
  double variance = 0.0;
  bool has_stress = false;
  double stress[9]{};
};

/// forceImpl while the caller is already detached (F4).
template <typename Pot>
FullResult run_force(Pot &pot, const Geom &g) {
  AtomMatrix forces = AtomMatrix::Zero(g.pos.rows(), 3);
  double flat[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      flat[i * 3 + j] = g.box[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  ForceOut fo{.F = forces.data(),
              .energy = 0.0,
              .variance = 0.0,
              .stress = {},
              .has_stress = 0};
  ForceInput in{.nAtoms = g.pos.rows(),
                .pos = g.pos.data(),
                .atmnrs = g.types.data(),
                .box = flat};
  pot.forceImpl(in, &fo);
  if (!std::isfinite(fo.energy)) {
    throw rgbind::NumericalError("energy is not finite");
  }
  FullResult result;
  result.energy = fo.energy;
  result.forces = std::move(forces);
  result.variance = fo.variance;
  result.has_stress = fo.has_stress != 0;
  std::memcpy(result.stress, fo.stress, sizeof(result.stress));
  return result;
}

inline nb::dict result_dict(const FullResult &result) {
  nb::dict out;
  out["energy"] = result.energy;
  out["forces"] = matrix_to_numpy(result.forces);
  out["variance"] = result.variance;
  if (result.has_stress) {
    out["stress"] = stress_to_numpy(result.stress);
  } else {
    out["stress"] = nb::none();
  }
  out["energies"] = nb::none();
  return out;
}

/**
 * One potential object, reentrant (F2, F3).
 *
 * SharedInstance kernels (LJ) keep a single instance: concurrent calls
 * are the same calls the GIL used to serialise. Every other kernel gets
 * one instance per calling thread, created on first use under mu_ while
 * the caller is detached, and destroyed with this object.
 *
 * The Calculator path does not use this as a session claim. Only the
 * RPC client claims its connection.
 *
 * extra_ keeps one instance per caller thread id other than the
 * constructing thread. A thread that has exited does not remove its
 * entry, so the map grows with the number of distinct threads that have
 * called this object and is released in clear() when the object is
 * destroyed. BatchEvaluator does not use the map. It builds a fixed
 * set of instances, one per worker.
 */
template <typename Pot> class PotHolder {
public:
  using Factory = std::function<std::unique_ptr<Pot>()>;

  explicit PotHolder(Factory factory)
      : make_(std::move(factory)), primary_(make_()),
        owner_(std::this_thread::get_id()) {
    if (!primary_) {
      throw std::invalid_argument("potential factory returned null");
    }
    shared_ = primary_->caps().reentrancy == Reentrancy::SharedInstance;
  }

  PotHolder(const PotHolder &) = delete;
  PotHolder &operator=(const PotHolder &) = delete;

  ~PotHolder() noexcept {
    try {
      if (Py_IsInitialized() != 0) {
        nb::gil_scoped_release release;
        clear();
      } else {
        clear();
      }
    } catch (...) {
      try {
        clear();
      } catch (...) {
      }
    }
  }

  /// Caller is detached. The mutex covers lookup and creation only.
  Pot &slot() {
    if (shared_ || std::this_thread::get_id() == owner_) {
      return *primary_;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto &extra = extra_[std::this_thread::get_id()];
    if (!extra) {
      extra = make_();
      if (!extra) {
        throw std::invalid_argument("potential factory returned null");
      }
    }
    return *extra;
  }

  Pot &primary() { return *primary_; }
  const Pot &primary() const { return *primary_; }

  std::unique_ptr<Pot> make() const { return make_(); }

private:
  void clear() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    extra_.clear();
    primary_.reset();
  }

  Factory make_;
  std::unique_ptr<Pot> primary_;
  std::thread::id owner_;
  bool shared_ = false;
  std::mutex mu_;
  std::unordered_map<std::thread::id, std::unique_ptr<Pot>> extra_;
};

template <typename Pot>
void bind_calls(nb::class_<PotHolder<Pot>> &cls) {
  cls.def("caps", [](const PotHolder<Pot> &self) {
    return caps_dict(self.primary().caps());
  });
  cls.def(
      "__call__",
      [](PotHolder<Pot> &self, const F64 &positions, const I32 &atom_types,
         const F64 &box) {
        Geom geom = read_geom(positions, atom_types, box);
        double energy = 0.0;
        double variance = 0.0;
        AtomMatrix forces;
        {
          nb::gil_scoped_release release;
          Pot &pot = self.slot();
          std::tie(energy, forces, variance) =
              pot(geom.pos, geom.types, geom.box);
        }
        if (!std::isfinite(energy)) {
          throw rgbind::NumericalError("energy is not finite");
        }
        return nb::make_tuple(energy, matrix_to_numpy(forces), variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types").noconvert(),
      nb::arg("box").noconvert());
  cls.def(
      "evaluate",
      [](PotHolder<Pot> &self, const F64 &positions, const I32 &atom_types,
         const F64 &box) {
        Geom geom = read_geom(positions, atom_types, box);
        FullResult result;
        {
          nb::gil_scoped_release release;
          result = run_force(self.slot(), geom);
        }
        return result_dict(result);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types").noconvert(),
      nb::arg("box").noconvert(),
      "Energy, forces, variance, and stress when the kernel writes it. "
      "F4: the kernel runs detached.");
  cls.def(
      "evaluate_batch",
      [](PotHolder<Pot> &self, nb::list positions, nb::list atom_types,
         nb::list boxes, unsigned n_threads) {
        if (positions.size() != atom_types.size() ||
            positions.size() != boxes.size()) {
          throw nb::value_error(
              "evaluate_batch: positions, atom_types, and box lengths differ");
        }
        if (positions.size() == 0) {
          return nb::list();
        }
        std::vector<BatchInput> inputs;
        inputs.reserve(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
          auto pos = nb::cast<F64>(positions[i], false);
          auto types = nb::cast<I32>(atom_types[i], false);
          auto box = nb::cast<F64>(boxes[i], false);
          Geom geom = read_geom(pos, types, box);
          BatchInput item;
          item.positions = std::move(geom.pos);
          item.atomic_numbers = std::move(geom.types);
          item.box = geom.box;
          inputs.push_back(std::move(item));
        }
        std::vector<BatchResult> results;
        {
          // F4. One BatchEvaluator call. Each worker owns one instance.
          // Do not construct more instances than there are configurations:
          // a single large cell would otherwise pay for unused workspaces.
          unsigned width = n_threads == 0
                               ? std::max(1u, std::thread::hardware_concurrency())
                               : n_threads;
          if (width > inputs.size()) {
            width = static_cast<unsigned>(inputs.size());
          }
          nb::gil_scoped_release release;
          BatchEvaluator eval([&] { return self.make(); }, width);
          results = eval.evaluate(inputs);
        }
        nb::list out;
        for (const BatchResult &item : results) {
          if (!std::isfinite(item.energy)) {
            throw rgbind::NumericalError("energy is not finite");
          }
          nb::dict row;
          row["energy"] = item.energy;
          row["forces"] = matrix_to_numpy(item.forces);
          row["variance"] = item.variance;
          out.append(row);
        }
        return out;
      },
      nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"),
      nb::arg("n_threads") = 0,
      "One detached BatchEvaluator call. results[i] matches a serial "
      "call on inputs[i], bit for bit. n_threads 0 uses the hardware "
      "concurrency.");
}

} // namespace rgpot::pybind
