// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file test_module.cpp
 * @author rgpot Developers
 * @date 2026-10-11
 * @brief Nanobind module that exercises rgbind from pytest.
 *
 * Built only by the bind/ project and the py-* workflows. The root
 * meson build does not compile this file.
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "rgbind/rgbind.hpp"

namespace nb = nanobind;

namespace {

struct Session {
  rgbind::Busy busy;

  int work(int hold_ms) {
    // F4: the hold is native work, so drop the GIL before the claim
    // is taken. F3: the claim itself does not lock a mutex.
    nb::gil_scoped_release rel;
    rgbind::Claim claim(busy.flag(), "Session");
    if (hold_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    }
    return 1;
  }

  void work_throw() {
    nb::gil_scoped_release rel;
    rgbind::Claim claim(busy.flag(), "Session");
    throw std::runtime_error("boom");
  }
};

void call_on_worker(nb::callable fn) {
  rgbind::CallContext ctx;
  rgbind::trampoline<nb::callable> slot(fn, &ctx, false);
  using Callback = int (*)(void *);
  Callback cb = &rgbind::trampoline<nb::callable>::entry<>;
  std::thread worker([cb, &slot]() { (void)cb(&slot); });
  {
    nb::gil_scoped_release rel;
    worker.join();
  }
  ctx.rethrow_if_set();
}

int call_on_main(nb::callable fn, bool check_signals) {
  rgbind::CallContext ctx;
  rgbind::trampoline<nb::callable> slot(fn, &ctx, check_signals);
  using Callback = int (*)(void *);
  Callback cb = &rgbind::trampoline<nb::callable>::entry<>;
  const int rc = cb(&slot);
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  return rc;
}

std::string schedule_canonical(nb::list rows, int n_workers) {
  std::vector<rgbind::OwnerRow> parsed;
  parsed.reserve(nb::len(rows));
  for (nb::handle item : rows) {
    nb::tuple tup = nb::cast<nb::tuple>(item);
    rgbind::OwnerRow row;
    row.row_id = nb::cast<std::int64_t>(tup[0]);
    row.owner = static_cast<std::uint64_t>(nb::cast<std::int64_t>(tup[1]));
    row.reentrant = nb::cast<bool>(tup[2]);
    parsed.push_back(row);
  }
  return rgbind::serialise_by_owner(parsed, n_workers).canonical();
}

} // namespace

NB_MODULE(_rgbind_test, m) {
  m.doc() = "rgbind policy tests (F1-F9).";
  m.attr("RGBIND_VERSION") = RGBIND_VERSION;
  rgbind::install_policy(m);

  nb::class_<Session>(m, "Session")
      .def(nb::init<>())
      .def("work", &Session::work, nb::arg("hold_ms") = 1,
           "Claim the session, hold, and release (F3, F4).")
      .def("work_throw", &Session::work_throw,
           "Claim, throw, and release via RAII (F3).");

  m.def("call_on_worker", &call_on_worker, nb::arg("fn").noconvert(),
        "Run fn on a foreign thread through the F5 trampoline.");
  m.def("call_on_main", &call_on_main, nb::arg("fn").noconvert(),
        nb::arg("check_signals") = false,
        "Run fn on this thread through the F5 trampoline.");
  m.def("schedule_canonical", &schedule_canonical, nb::arg("rows"),
        nb::arg("n_workers"),
        "Canonical F5/F8 schedule. Independent of n_workers.");
  m.def("check_f64_1d", &rgbind::check_f64_1d, nb::arg("arr").noconvert(),
        nb::arg("n"), nb::arg("name"));
  m.def("check_f64_2d", &rgbind::check_f64_2d, nb::arg("arr").noconvert(),
        nb::arg("rows"), nb::arg("cols"), nb::arg("name"));
  m.def("check_f64_3d", &rgbind::check_f64_3d, nb::arg("arr").noconvert(),
        nb::arg("d0"), nb::arg("d1"), nb::arg("d2"), nb::arg("name"));
  m.def("echo_f64", &rgbind::echo_f64, nb::arg("arr").noconvert(),
        "Byte-copy a float64 array into a capsule-owned numpy array (F6).");
  m.def(
      "map_status",
      [](int code, const std::string &last_error, const std::string &type_name) {
        const rgbind::MappedStatus mapped =
            rgbind::map_status(static_cast<rgbind::Status>(code), last_error,
                               type_name);
        return std::make_tuple(std::string(rgbind::error_kind_name(mapped.kind)),
                               mapped.message);
      },
      nb::arg("code"), nb::arg("last_error") = "", nb::arg("type_name") = "Object",
      "Return (kind, message) for an F7 status code.");
  m.def(
      "raise_status",
      [](int code, const std::string &last_error, const std::string &type_name) {
        const rgbind::MappedStatus mapped =
            rgbind::map_status(static_cast<rgbind::Status>(code), last_error,
                               type_name);
        return rgbind::apply_status(mapped, nullptr);
      },
      nb::arg("code"), nb::arg("last_error") = "", nb::arg("type_name") = "Object",
      "Raise the F7 exception, or return False when not converged.");
}
