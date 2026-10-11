// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file rgbind.hpp
 * @author rgpot Developers
 * @date 2026-10-11
 * @brief Header-only nanobind policy for free-threaded bindings.
 *
 * @ingroup rgbind
 *
 * Created on: 2026-10-11
 *
 * One header, C++17, nanobind only. The rules below are the binding
 * contract (cite them by number):
 *
 * - F1. Extension modules are built FREE_THREADED so nanobind sets
 *   Py_mod_gil = Py_MOD_GIL_NOT_USED.
 * - F2. No global mutable state. This header has none. install_policy
 *   registers the exception type with a temporary nb::exception, which
 *   does not keep a Python object past that statement.
 * - F3. Mutable sessions take rgbind::Claim on an atomic flag. A second
 *   caller gets InUse, which the module maps to
 *   RuntimeError("<Type> is in use by another thread"). Do not hold a
 *   std::mutex while attached, and do not use nb::lock_self or
 *   nb::ft_mutex across a compute region.
 * - F4. Native work longer than a few microseconds runs under
 *   nb::gil_scoped_release on GIL and free-threaded builds. Validate
 *   arrays and take pointers before releasing.
 * - F5. Python callbacks go through rgbind::trampoline. The trampoline
 *   re-attaches with nb::gil_scoped_acquire, catches every exception into
 *   CallContext, and returns a non-zero status. Attach failure at
 *   shutdown returns a status and does not throw. rethrow_if_set
 *   re-raises after re-attaching. Rows that share an owner run in
 *   ascending row id. Worker count is not part of that order.
 * - F6. Float64, C contiguous, CPU arrays. Wrong dtype or order is a
 *   TypeError (nb::arg::noconvert). Wrong shape is a ValueError that
 *   names the argument and both shapes. -0.0, NaN, and subnormals pass
 *   through unchanged.
 * - F7. map_status implements the status table. Packages specialise
 *   status_traits for their own enum.
 * - F8. serialise_by_owner is a pure function of the rows. The worker
 *   count is checked and then ignored, so the schedule does not depend
 *   on thread count, GIL mode, or caller interleaving.
 * - F9. Sub-interpreters are unsupported (nanobind sets
 *   Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED). Hidden visibility and
 *   the export list live in the build files, not here.
 */

#ifndef RGBIND_RGBIND_HPP_
#define RGBIND_RGBIND_HPP_

#define RGBIND_VERSION 1

#ifndef RGBIND_NO_PYTHON
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rgbind {

/**
 * @brief Row of a batched evaluation, addressed by owner and row id.
 *
 * The owner key is id(calc) (F5). Rows that share an owner commit in
 * ascending row id (F8).
 */
struct OwnerRow {
  std::int64_t row_id = 0;
  std::uint64_t owner = 0;
  bool reentrant = false;
};

/**
 * @brief One owner and its rows in ascending row id.
 *
 * reentrant is the owner's flag (F5). A true flag means the pure
 * protocol may run the rows concurrently. The commit order is still
 * row_ids, which is sorted ascending.
 */
struct OwnerGroup {
  std::uint64_t owner = 0;
  bool reentrant = false;
  std::vector<std::int64_t> row_ids;
};

/**
 * @brief Deterministic per-owner schedule (F5, F8).
 *
 * canonical() is the byte string that tests hash. It does not contain
 * the worker count.
 */
struct Schedule {
  std::vector<OwnerGroup> groups;

  /**
   * @brief Serialise the commit order.
   * @return One line per group: "<owner> <0|1> <row ids...>".
   */
  std::string canonical() const {
    std::string out;
    for (const OwnerGroup &group : groups) {
      out += std::to_string(group.owner);
      out += group.reentrant ? " 1" : " 0";
      for (std::int64_t row_id : group.row_ids) {
        out += ' ';
        out += std::to_string(row_id);
      }
      out += '\n';
    }
    return out;
  }
};

/**
 * @brief Group rows by owner and sort each group by ascending row id.
 *
 * The worker count is not an input to the groups. n_workers is
 * checked and then ignored (F8). Two different permutations of the
 * same rows produce the same groups. A repeated row_id is rejected
 * because the tie-break would otherwise be the owner address.
 *
 * @param rows Input rows. Order does not affect the result.
 * @param n_workers Caller thread count. Must be >= 1. Not an input to
 *        the groups (F8): 1, 2, 3, and 8 workers return the same groups.
 * @pre n_workers >= 1. row_id values are unique.
 * @return Owner groups ordered by minimum row id, then owner id.
 *         Throws std::invalid_argument on a bad worker count, a
 *         duplicate row_id, or mixed reentrant flags for one owner.
 */
inline Schedule serialise_by_owner(const std::vector<OwnerRow> &rows,
                                   int n_workers) {
  if (n_workers < 1) {
    throw std::invalid_argument(
        "rgbind::serialise_by_owner: n_workers must be >= 1");
  }
  // F8: the worker count is not an input to the groups. The argument
  // is accepted so a test can hash the schedule at 1, 2, 3, and 8
  // workers and so callers can record the decomposition they asked for.

  struct Acc {
    bool reentrant = false;
    bool have_flag = false;
    std::int64_t min_row = 0;
    std::vector<std::int64_t> row_ids;
  };
  std::map<std::uint64_t, Acc> by_owner;
  std::set<std::int64_t> seen_rows;
  for (const OwnerRow &row : rows) {
    if (!seen_rows.insert(row.row_id).second) {
      throw std::invalid_argument(
          "rgbind::serialise_by_owner: duplicate row_id");
    }
    Acc &acc = by_owner[row.owner];
    if (acc.have_flag && acc.reentrant != row.reentrant) {
      throw std::invalid_argument(
          "rgbind::serialise_by_owner: reentrant flag differs for one owner");
    }
    if (!acc.have_flag) {
      acc.reentrant = row.reentrant;
      acc.have_flag = true;
      acc.min_row = row.row_id;
    } else if (row.row_id < acc.min_row) {
      acc.min_row = row.row_id;
    }
    acc.row_ids.push_back(row.row_id);
  }

  struct Keyed {
    std::int64_t min_row;
    std::uint64_t owner;
    OwnerGroup group;
  };
  std::vector<Keyed> keyed;
  keyed.reserve(by_owner.size());
  for (auto &item : by_owner) {
    Acc &acc = item.second;
    std::sort(acc.row_ids.begin(), acc.row_ids.end());
    Keyed key;
    key.min_row = acc.min_row;
    key.owner = item.first;
    key.group.owner = item.first;
    key.group.reentrant = acc.reentrant;
    key.group.row_ids = std::move(acc.row_ids);
    keyed.push_back(std::move(key));
  }
  std::sort(keyed.begin(), keyed.end(), [](const Keyed &a, const Keyed &b) {
    if (a.min_row != b.min_row) {
      return a.min_row < b.min_row;
    }
    return a.owner < b.owner;
  });

  Schedule schedule;
  schedule.groups.reserve(keyed.size());
  for (Keyed &key : keyed) {
    schedule.groups.push_back(std::move(key.group));
  }
  return schedule;
}

/**
 * @brief F7 status class after map_status.
 */
enum class ErrorKind {
  ok,
  value_error,
  type_error,
  not_converged,
  callback,
  numerical,
  runtime_error,
  not_implemented,
};

/**
 * @brief Reference status enum for the F7 table.
 *
 * A package with a different numeric encoding specialises
 * status_traits. The default trait static_casts the enum to int and
 * expects these codes.
 */
enum class Status : int {
  ok = 0,
  build = 1,
  key = 2,
  preset = 3,
  dtype = 4,
  order = 5,
  not_converged = 6,
  callback = 7,
  numerical = 8,
  in_use = 9,
  unavailable = 10,
  other = 11,
};

/**
 * @brief Map a package status enum onto the F7 integer codes.
 */
template <typename Enum> struct status_traits {
  static int code(Enum value) { return static_cast<int>(value); }
};

/**
 * @brief Result of map_status. kind selects the Python exception (F7).
 */
struct MappedStatus {
  ErrorKind kind = ErrorKind::ok;
  std::string message;
};

/**
 * @brief Name of an ErrorKind, stable for tests and stubs.
 * @param kind Error class.
 * @return A short snake_case name.
 */
inline const char *error_kind_name(ErrorKind kind) noexcept {
  switch (kind) {
  case ErrorKind::ok:
    return "ok";
  case ErrorKind::value_error:
    return "value_error";
  case ErrorKind::type_error:
    return "type_error";
  case ErrorKind::not_converged:
    return "not_converged";
  case ErrorKind::callback:
    return "callback";
  case ErrorKind::numerical:
    return "numerical";
  case ErrorKind::runtime_error:
    return "runtime_error";
  case ErrorKind::not_implemented:
    return "not_implemented";
  }
  return "runtime_error";
}

/**
 * @brief Map an integer F7 code to an exception class and message.
 *
 * @param code Status code. See Status for the table.
 * @param last_error Thread-local last-error text. Used for build, key,
 *        preset, numerical, unavailable, and other (F7).
 * @param type_name Session type used in the in-use message (F3, F7).
 * @return The mapped status. not_converged does not raise. run()
 *         returns false instead (ASE semantics).
 */
inline MappedStatus map_status_code(int code, std::string_view last_error,
                                     std::string_view type_name) {
  const std::string last(last_error);
  const std::string fallback_type =
      type_name.empty() ? std::string("Object") : std::string(type_name);
  MappedStatus mapped;
  switch (code) {
  case static_cast<int>(Status::ok):
    mapped.kind = ErrorKind::ok;
    break;
  case static_cast<int>(Status::build):
    mapped.kind = ErrorKind::value_error;
    mapped.message = last.empty() ? "build error" : last;
    break;
  case static_cast<int>(Status::key):
    mapped.kind = ErrorKind::value_error;
    mapped.message = last.empty() ? "unknown key" : last;
    break;
  case static_cast<int>(Status::preset):
    mapped.kind = ErrorKind::value_error;
    mapped.message = last.empty() ? "unknown preset" : last;
    break;
  case static_cast<int>(Status::dtype):
    mapped.kind = ErrorKind::type_error;
    mapped.message = last.empty() ? "dtype must be float64" : last;
    break;
  case static_cast<int>(Status::order):
    mapped.kind = ErrorKind::type_error;
    mapped.message = last.empty() ? "array must be C contiguous" : last;
    break;
  case static_cast<int>(Status::not_converged):
    mapped.kind = ErrorKind::not_converged;
    break;
  case static_cast<int>(Status::callback):
    mapped.kind = ErrorKind::callback;
    mapped.message = last;
    break;
  case static_cast<int>(Status::numerical):
    mapped.kind = ErrorKind::numerical;
    mapped.message = last.empty() ? "NaN energy or singular update" : last;
    break;
  case static_cast<int>(Status::in_use):
    mapped.kind = ErrorKind::runtime_error;
    mapped.message = fallback_type + " is in use by another thread";
    break;
  case static_cast<int>(Status::unavailable):
    mapped.kind = ErrorKind::not_implemented;
    mapped.message = last.empty() ? "compiled-out feature" : last;
    break;
  case static_cast<int>(Status::other):
  default:
    mapped.kind = ErrorKind::runtime_error;
    mapped.message = last;
    break;
  }
  return mapped;
}

/**
 * @brief map_status parameterised by the package status enum (F7).
 * @param code Package status value.
 * @param last_error Thread-local last-error text.
 * @param type_name Session type for the in-use message.
 * @return The mapped status.
 */
template <typename Enum>
inline MappedStatus map_status(Enum code, std::string_view last_error,
                               std::string_view type_name = "Object") {
  return map_status_code(status_traits<Enum>::code(code), last_error,
                         type_name);
}

} // namespace rgbind

#ifndef RGBIND_NO_PYTHON

#include <new>
#include <type_traits>

namespace rgbind {

namespace nb = nanobind;

/**
 * @brief Thrown when compare-exchange fails to claim a session (F3).
 *
 * what() is "<Type> is in use by another thread". install_policy maps
 * this type to RuntimeError with that text.
 */
class InUse : public std::runtime_error {
public:
  /**
   * @brief Build the F3 error for a session type.
   * @param type_name Python type name, without brackets.
   */
  explicit InUse(std::string type_name)
      : std::runtime_error(type_name + " is in use by another thread"),
        type_name_(std::move(type_name)) {}

  /**
   * @brief Session type that was busy.
   * @return The type name passed to the constructor.
   */
  const std::string &type_name() const noexcept { return type_name_; }

private:
  std::string type_name_;
};

/**
 * @brief NaN energy or a singular update (F7).
 *
 * install_policy registers this as <module>.NumericalError, a subclass
 * of ArithmeticError.
 */
class NumericalError : public std::runtime_error {
public:
  /**
   * @brief Build a numerical error.
   * @param message Text carried to Python.
   */
  explicit NumericalError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief Per-session busy flag (F3).
 *
 * The flag starts clear. Claim is the only supported mutator.
 */
class Busy {
public:
  Busy() noexcept = default;
  Busy(const Busy &) = delete;
  Busy &operator=(const Busy &) = delete;

  /**
   * @brief Atomic flag claimed by compare-exchange.
   * @return The flag. true means a Claim is alive.
   */
  std::atomic<bool> &flag() noexcept { return flag_; }

  /**
   * @brief Atomic flag claimed by compare-exchange.
   * @return The flag.
   */
  const std::atomic<bool> &flag() const noexcept { return flag_; }

private:
  std::atomic<bool> flag_{false};
};

/**
 * @brief RAII try-claim of a Busy flag (F3).
 *
 * compare_exchange_strong from false to true. Failure throws InUse and
 * does not own the flag. The destructor stores false.
 */
class Claim {
public:
  /**
   * @brief Try to claim @a busy.
   * @param busy Session flag.
   * @param type_name Used in the InUse message. The message string is
   *        built only when the claim fails.
   * @pre The caller is not holding a std::mutex while attached (F3).
   */
  Claim(std::atomic<bool> &busy, const char *type_name) : busy_(&busy) {
    bool expected = false;
    if (!busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      busy_ = nullptr;
      const char *name = type_name == nullptr ? "Object" : type_name;
      throw InUse(name);
    }
  }

  Claim(const Claim &) = delete;
  Claim &operator=(const Claim &) = delete;

  /**
   * @brief Transfer ownership of the claim.
   * @param other Claim that currently owns the flag.
   */
  Claim(Claim &&other) noexcept : busy_(other.busy_) { other.busy_ = nullptr; }

  Claim &operator=(Claim &&) = delete;

  /**
   * @brief Release the flag if this object still owns it.
   * @return Void.
   */
  ~Claim() {
    if (busy_ != nullptr) {
      busy_->store(false, std::memory_order_release);
    }
  }

  /**
   * @brief True when this object holds the flag.
   * @return Ownership.
   */
  bool owns() const noexcept { return busy_ != nullptr; }

private:
  std::atomic<bool> *busy_;
};

/**
 * @brief Per-call slot for the first callback failure (F5).
 *
 * Workers catch nb::python_error and std::exception, store the first
 * one, and return a status. The run() thread calls rethrow_if_set
 * after joining, which re-attaches and re-raises. A Python exception
 * never unwinds through foreign frames.
 *
 * Capture uses an atomic claim so two workers do not need a mutex
 * while attached (F3). rethrow_if_set runs after those workers have
 * been joined.
 */
class CallContext {
public:
  CallContext() = default;
  CallContext(const CallContext &) = delete;
  CallContext &operator=(const CallContext &) = delete;

  /**
   * @brief Cooperative stop flag. Trampolines return before calling
   *        when this is set (F5).
   */
  std::atomic<bool> stop{false};

  /**
   * @brief Store the first Python exception. Later calls drop theirs.
   * @param err Exception caught under the GIL. Moved from on success.
   * @return Void.
   */
  void capture(nb::python_error &err) {
    int expected = 0;
    if (!state_.compare_exchange_strong(expected, -1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return;
    }
    // F5: other workers see stop and return before calling Python.
    stop.store(true, std::memory_order_release);
    py_ = new (std::nothrow) nb::python_error(std::move(err));
    if (py_ == nullptr) {
      state_.store(0, std::memory_order_release);
      return;
    }
    state_.store(1, std::memory_order_release);
  }

  /**
   * @brief Store the first C++ exception.
   * @param ptr Current exception. Ignored when a slot is already taken.
   * @return Void.
   */
  void capture(std::exception_ptr ptr) {
    int expected = 0;
    if (!state_.compare_exchange_strong(expected, -1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return;
    }
    stop.store(true, std::memory_order_release);
    cxx_ = new (std::nothrow) std::exception_ptr(std::move(ptr));
    if (cxx_ == nullptr) {
      state_.store(0, std::memory_order_release);
      return;
    }
    state_.store(2, std::memory_order_release);
  }

  /**
   * @brief True after a successful capture.
   * @return Whether an error is stored.
   */
  bool has_error() const noexcept {
    const int state = state_.load(std::memory_order_acquire);
    return state == 1 || state == 2;
  }

  /**
   * @brief Re-attach and re-raise the stored error (F5).
   *
   * No-op when the slot is empty. Attach failure throws
   * std::runtime_error rather than touching a dead interpreter.
   *
   * @return Void.
   */
  void rethrow_if_set() {
    const int state = state_.load(std::memory_order_acquire);
    if (state != 1 && state != 2) {
      return;
    }
    nb::gil_scoped_acquire gil;
    if (!gil.is_valid()) {
      throw std::runtime_error(
          "rgbind::CallContext: attach failed while rethrowing");
    }
    if (state == 1 && py_ != nullptr) {
      py_->restore();
      delete py_;
      py_ = nullptr;
      state_.store(0, std::memory_order_release);
      throw nb::python_error();
    }
    if (state == 2 && cxx_ != nullptr) {
      std::exception_ptr ptr = *cxx_;
      delete cxx_;
      cxx_ = nullptr;
      state_.store(0, std::memory_order_release);
      std::rethrow_exception(ptr);
    }
  }

  /**
   * @brief Drop a stored Python error under a cleanup guard.
   * @return Void.
   */
  ~CallContext() {
    delete py_;
    py_ = nullptr;
    delete cxx_;
    cxx_ = nullptr;
  }

private:
  // 0 empty, -1 claim in progress, 1 python, 2 C++.
  std::atomic<int> state_{0};
  nb::python_error *py_ = nullptr;
  std::exception_ptr *cxx_ = nullptr;
};

/**
 * @brief Status codes returned by trampoline::entry. Zero is success.
 */
enum class TrampolineStatus : int {
  ok = 0,
  stopped = 1,
  attach_failed = 2,
  python_error = 3,
  cxx_error = 4,
  bad_user = 5,
};

/**
 * @brief C callback wrapper around a callable (F5).
 *
 * entry has the shape int (*)(void *user, Args...). It acquires the
 * GIL, calls @a F, and on any exception stores it in CallContext and
 * returns a non-zero status. It never throws. Attach failure at
 * shutdown returns TrampolineStatus::attach_failed.
 *
 * The object itself is the user pointer. Destroy it on a thread that
 * may touch the callable, after every entry call has returned.
 */
template <typename F> class trampoline {
public:
  F fn;
  CallContext *ctx = nullptr;
  bool check_signals = false;

  /**
   * @brief Bind a callable to a per-call context.
   * @param function Callable invoked under the GIL.
   * @param context Slot for the first error. May be null.
   * @param signals When true, call PyErr_CheckSignals after a
   *        successful call (main thread, F5).
   */
  trampoline(F function, CallContext *context, bool signals)
      : fn(std::move(function)), ctx(context), check_signals(signals) {}

  /**
   * @brief C callback. @a user is a trampoline<F>*.
   * @param user Trampoline pointer.
   * @param args Extra arguments forwarded to F.
   * @return 0 on success, otherwise a TrampolineStatus.
   */
  template <typename... Args>
  static int entry(void *user, Args... args) noexcept {
    auto *self = static_cast<trampoline *>(user);
    if (self == nullptr) {
      return static_cast<int>(TrampolineStatus::bad_user);
    }
    if (self->ctx != nullptr &&
        self->ctx->stop.load(std::memory_order_acquire)) {
      return static_cast<int>(TrampolineStatus::stopped);
    }
    try {
      nb::gil_scoped_acquire gil;
      if (!gil.is_valid()) {
        if (self->ctx != nullptr) {
          self->ctx->stop.store(true, std::memory_order_release);
        }
        return static_cast<int>(TrampolineStatus::attach_failed);
      }
      try {
        (void)self->fn(args...);
        if (self->check_signals) {
          if (PyErr_CheckSignals() < 0) {
            nb::python_error err;
            if (self->ctx != nullptr) {
              self->ctx->capture(err);
            }
            return static_cast<int>(TrampolineStatus::python_error);
          }
        }
        return static_cast<int>(TrampolineStatus::ok);
      } catch (nb::python_error &err) {
        if (self->ctx != nullptr) {
          self->ctx->capture(err);
        }
        return static_cast<int>(TrampolineStatus::python_error);
      } catch (...) {
        if (self->ctx != nullptr) {
          self->ctx->capture(std::current_exception());
        }
        return static_cast<int>(TrampolineStatus::cxx_error);
      }
    } catch (...) {
      if (self->ctx != nullptr) {
        self->ctx->stop.store(true, std::memory_order_release);
      }
      return static_cast<int>(TrampolineStatus::attach_failed);
    }
  }
};

/**
 * @brief Raise the Python exception selected by map_status (F7).
 *
 * @param mapped Result of map_status.
 * @param ctx Callback context. Used when kind is callback.
 * @return true on ok, false on not_converged (ASE run() semantics).
 * @pre The caller is attached to the interpreter.
 */
inline bool apply_status(const MappedStatus &mapped, CallContext *ctx) {
  switch (mapped.kind) {
  case ErrorKind::ok:
    return true;
  case ErrorKind::not_converged:
    return false;
  case ErrorKind::value_error:
    throw nb::value_error(mapped.message.c_str());
  case ErrorKind::type_error:
    throw nb::type_error(mapped.message.c_str());
  case ErrorKind::callback:
    if (ctx != nullptr) {
      ctx->rethrow_if_set();
    }
    throw std::runtime_error(mapped.message.empty() ? "callback failed"
                                                   : mapped.message);
  case ErrorKind::numerical:
    throw NumericalError(mapped.message);
  case ErrorKind::not_implemented:
    PyErr_SetString(PyExc_NotImplementedError, mapped.message.c_str());
    throw nb::python_error();
  case ErrorKind::runtime_error:
    throw std::runtime_error(mapped.message);
  }
  throw std::runtime_error(mapped.message);
}

/**
 * @brief Register F3 and F7 translators on a module.
 *
 * InUse becomes RuntimeError. NumericalError becomes
 * <module>.NumericalError(ArithmeticError). The nb::exception object
 * is a temporary: the module keeps the type, and this function does
 * not hold a refcounted Python object (F2).
 *
 * @param m Module that owns the Python exception type.
 * @return Void.
 */
inline void install_policy(nb::module_ &m) {
  nb::exception<NumericalError>(m, "NumericalError", PyExc_ArithmeticError);
  nb::register_exception_translator(
      [](const std::exception_ptr &ptr, void *) {
        try {
          std::rethrow_exception(ptr);
        } catch (const InUse &err) {
          PyErr_SetString(PyExc_RuntimeError, err.what());
        }
      });
}

using F64Array = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

namespace detail {

inline std::string shape_list(const std::size_t *dims, std::size_t n) {
  std::string text = "(";
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += std::to_string(dims[i]);
  }
  if (n == 1) {
    text += ",";
  }
  text += ")";
  return text;
}

inline void raise_shape(const char *name, const std::string &expected,
                        const std::string &got) {
  const std::string msg =
      std::string(name == nullptr ? "array" : name) + ": expected shape " +
      expected + ", got " + got;
  throw nb::value_error(msg.c_str());
}

inline void check_f64_nd(const F64Array &arr, const std::size_t *expect,
                         std::size_t n, const char *name) {
  std::vector<std::size_t> got(arr.ndim());
  for (std::size_t i = 0; i < arr.ndim(); ++i) {
    got[i] = arr.shape(i);
  }
  bool match = arr.ndim() == n;
  if (match) {
    for (std::size_t i = 0; i < n; ++i) {
      if (got[i] != expect[i]) {
        match = false;
        break;
      }
    }
  }
  if (!match) {
    raise_shape(name, shape_list(expect, n), shape_list(got.data(), got.size()));
  }
}

} // namespace detail

/**
 * @brief Reject an array whose rank is not @a rank.
 *
 * Call this before reading an extent. A 0-d array has no axis 0, so
 * shape(0) is not a valid length check.
 *
 * @param arr C contiguous float64 array.
 * @param rank Expected rank. Extents are not read.
 * @param name Argument name included in the ValueError.
 * @return Void.
 */
inline void check_f64_rank(const F64Array &arr, std::size_t rank,
                           const char *name) {
  if (arr.ndim() == rank) {
    return;
  }
  std::vector<std::size_t> got(arr.ndim());
  for (std::size_t i = 0; i < arr.ndim(); ++i) {
    got[i] = arr.shape(i);
  }
  const std::string expected = "rank " + std::to_string(rank);
  const std::string got_text = "rank " + std::to_string(arr.ndim()) + " " +
                               detail::shape_list(got.data(), got.size());
  detail::raise_shape(name, expected, got_text);
}

/**
 * @brief F6 shape check for a 1-d float64 vector.
 * @param arr C contiguous float64 array. dtype and order are the
 *        binding's TypeError (noconvert).
 * @param n Expected length.
 * @param name Argument name included in the ValueError.
 * @return Void.
 */
inline void check_f64_1d(const F64Array &arr, std::size_t n, const char *name) {
  const std::size_t expect[1] = {n};
  detail::check_f64_nd(arr, expect, 1, name);
}

/**
 * @brief F6 shape check for a 2-d float64 array.
 * @param arr C contiguous float64 array.
 * @param rows Expected row count.
 * @param cols Expected column count.
 * @param name Argument name included in the ValueError.
 * @return Void.
 */
inline void check_f64_2d(const F64Array &arr, std::size_t rows, std::size_t cols,
                         const char *name) {
  const std::size_t expect[2] = {rows, cols};
  detail::check_f64_nd(arr, expect, 2, name);
}

/**
 * @brief F6 shape check for a 3-d float64 array.
 * @param arr C contiguous float64 array.
 * @param d0 Expected size of axis 0.
 * @param d1 Expected size of axis 1.
 * @param d2 Expected size of axis 2.
 * @param name Argument name included in the ValueError.
 * @return Void.
 */
inline void check_f64_3d(const F64Array &arr, std::size_t d0, std::size_t d1,
                         std::size_t d2, const char *name) {
  const std::size_t expect[3] = {d0, d1, d2};
  detail::check_f64_nd(arr, expect, 3, name);
}

/**
 * @brief Copy a float64 array into a new numpy array with a capsule owner.
 *
 * Byte copy, so -0.0, NaN, and subnormals are unchanged (F6). The
 * returned array owns its buffer through an nb::capsule.
 *
 * @param arr Source array, already validated for dtype and order.
 * @return A new C contiguous float64 array of the same shape.
 */
inline nb::ndarray<nb::numpy, double> echo_f64(const F64Array &arr) {
  const std::size_t nd = arr.ndim();
  std::vector<std::size_t> shape(nd);
  std::size_t count = 1;
  for (std::size_t i = 0; i < nd; ++i) {
    shape[i] = arr.shape(i);
    count *= shape[i];
  }
  double *data = new double[count == 0 ? 1 : count];
  if (count != 0 && arr.data() != nullptr) {
    std::memcpy(data, arr.data(), count * sizeof(double));
  }
  nb::capsule owner(data, [](void *ptr) noexcept {
    delete[] static_cast<double *>(ptr);
  });
  return nb::ndarray<nb::numpy, double>(data, nd, shape.data(), owner);
}

} // namespace rgbind

#endif // RGBIND_NO_PYTHON

#endif // RGBIND_RGBIND_HPP_
