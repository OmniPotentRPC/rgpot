#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file potential.hpp
 * @author rgpot Developers
 * @date 2026-02-15
 * @brief Move-only RAII handle for callback-backed potential calculations.
 *
 * Provides @c PotentialHandle, the primary C++ entry point for performing
 * force/energy calculations through the Rust core.  The handle owns an
 * opaque @c rgpot_potential_t pointer obtained from @c rgpot_potential_new()
 * and releases it via @c rgpot_potential_free() on destruction.
 *
 * Two factory functions are offered:
 *
 * - @c from_impl<Impl>() — wraps an existing C++ potential whose
 *   @c forceImpl() uses the legacy @c ForceInput / @c ForceOut structs
 *   (defined in @c ForceStructs.hpp).  A template trampoline extracts
 *   raw CPU pointers from the DLPack tensors and creates an owning DLPack
 *   forces tensor from the legacy output.
 * - @c from_callback() — registers a bare C function pointer directly.
 *
 * # Example
 * @code
 * rgpot::LJPot lj;
 * auto pot = rgpot::PotentialHandle::from_impl(lj);
 *
 * double pos[] = {0,0,0, 1,0,0};
 * int    atm[] = {1, 1};
 * double box[] = {10,0,0, 0,10,0, 0,0,10};
 * rgpot::InputSpec input(2, pos, atm, box);
 *
 * auto result = pot.calculate(input);
 * double energy = result.energy();
 * @endcode
 *
 * @ingroup rgpot_cpp
 */

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rgpot.h"
#include "rgpot/errors.hpp"
#include "rgpot/types.hpp"

namespace rgpot {

/**
 * @class PotentialHandle
 * @brief Move-only RAII handle around an opaque @c rgpot_potential_t pointer.
 * @ingroup rgpot_cpp
 *
 * The handle is non-copyable.  Move semantics transfer ownership of the
 * underlying Rust allocation.  On destruction the Rust core frees all
 * resources associated with the potential via @c rgpot_potential_free().
 */
class PotentialHandle final {
public:
  /**
   * @brief Adopt an existing raw handle.
   * @pre @a handle was obtained from @c rgpot_potential_new() and has not
   *      been freed.
   * @param handle Raw pointer to the Rust-side potential object.
   */
  explicit PotentialHandle(rgpot_potential_t *handle) : handle_(handle) {}

  /**
   * @brief Destructor — releases the Rust-side allocation.
   *
   * Safe to call on a moved-from handle (internal pointer is @c nullptr).
   */
  ~PotentialHandle() {
    if (handle_) {
      rgpot_potential_free(handle_);
    }
  }

  /**
   * @brief Move constructor — transfers ownership.
   * @param other Handle to move from; left in a null state.
   */
  PotentialHandle(PotentialHandle &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  /**
   * @brief Move assignment — releases current handle, transfers ownership.
   * @param other Handle to move from; left in a null state.
   * @return Reference to @c *this.
   */
  PotentialHandle &operator=(PotentialHandle &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        rgpot_potential_free(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  PotentialHandle(const PotentialHandle &) = delete;
  PotentialHandle &operator=(const PotentialHandle &) = delete;

  /**
   * @brief Perform a force/energy calculation.
   *
   * Creates a @c CalcResult, delegates to @c rgpot_potential_calculate(),
   * and checks the returned status code.  The callback sets the forces
   * tensor in the output struct.
   *
   * @param input The atomic configuration to evaluate.
   * @return A @c CalcResult containing energy, variance, and forces.
   * @throws rgpot::Error when the underlying callback reports failure.
   */
  CalcResult calculate(const InputSpec &input) {
    CalcResult result;
    auto status = rgpot_potential_calculate(handle_, &input.c_struct(),
                                            &result.c_struct());
    details::check_status(status);
    return result;
  }

  /**
   * @brief Create a handle from a legacy C++ potential implementation.
   *
   * Registers a template trampoline that extracts raw CPU pointers from
   * the DLPack tensors in @c rgpot_force_input_t, constructs legacy
   * @c ForceInput / @c ForceOut types for @a Impl::forceImpl(), and
   * wraps the resulting forces in an owning DLPack tensor.
   *
   * The caller retains ownership of @a impl and @b must keep it alive
   * for the lifetime of the returned handle.
   *
   * @tparam Impl A type with
   *         @c void @c forceImpl(const @c ForceInput&, @c ForceOut*).
   * @param impl Reference to the C++ potential object.
   * @return A new @c PotentialHandle wrapping @a impl.
   */
  template <typename Impl> static PotentialHandle from_impl(Impl &impl) {
    auto *handle = rgpot_potential_new(&trampoline_callback<Impl>,
                                       static_cast<void *>(&impl),
                                       nullptr // caller owns impl
    );
    return PotentialHandle(handle);
  }

  /**
   * @brief Create a handle from an explicit C callback and user data.
   *
   * This is the low-level factory; prefer @c from_impl() for C++
   * potentials.
   *
   * @param callback Function pointer matching the @c rgpot_potential_t
   *                 callback signature.
   * @param user_data Opaque pointer forwarded to every @a callback
   *                  invocation.
   * @param free_fn  Optional destructor for @a user_data (may be
   *                 @c nullptr if the caller manages the lifetime).
   * @return A new @c PotentialHandle.
   */
  static PotentialHandle
  from_callback(rgpot_status_t (*callback)(void *, const rgpot_force_input_t *,
                                           rgpot_force_out_t *),
                void *user_data, void (*free_fn)(void *) = nullptr) {
    auto *handle = rgpot_potential_new(callback, user_data, free_fn);
    return PotentialHandle(handle);
  }

  /**
   * @brief Access the raw opaque handle for C interop.
   * @return Pointer to the underlying @c rgpot_potential_t.
   */
  rgpot_potential_t *raw() const { return handle_; }

private:
  rgpot_potential_t *handle_; //!< Owned opaque handle (may be @c nullptr
                              //!< after move).

  /**
   * @brief Template trampoline bridging DLPack-based C ABI types to legacy
   *        C++ types.
   *
   * 1. Extracts raw CPU pointers from DLPack tensors in the input.
   * 2. Constructs legacy @c ForceInput and @c ForceOut.
   * 3. Calls @c Impl::forceImpl().
   * 4. Wraps the output forces in an owning DLPack tensor via
   *    @c rgpot_tensor_owned_cpu_f64_2d() (since the local
   *    @c std::vector goes out of scope).
   *
   * @tparam Impl The concrete potential type.
   * @param user_data Pointer to the @a Impl instance (cast from @c void*).
   * @param input     Pointer to the C input struct (DLPack tensors).
   * @param output    Pointer to the C output struct.
   * @return @c RGPOT_SUCCESS or @c RGPOT_INTERNAL_ERROR.
   */
  template <typename Impl>
  static rgpot_status_t trampoline_callback(void *user_data,
                                            const rgpot_force_input_t *input,
                                            rgpot_force_out_t *output) {
    try {
      auto *self = static_cast<Impl *>(user_data);

      // Extract n_atoms from the positions tensor shape[0]
      if (!input->positions) {
        return RGPOT_INVALID_PARAMETER;
      }
      auto *pos_tensor = &input->positions->dl_tensor;
      size_t n_atoms = static_cast<size_t>(pos_tensor->shape[0]);

      // Extract raw CPU pointers from DLPack tensors
      const double *pos = static_cast<const double *>(pos_tensor->data);
      const int *atmnrs =
          input->atomic_numbers
              ? static_cast<const int *>(input->atomic_numbers->dl_tensor.data)
              : nullptr;
      const double *box =
          input->box_matrix
              ? static_cast<const double *>(input->box_matrix->dl_tensor.data)
              : nullptr;

      // Construct legacy ForceInput
      ::rgpot::ForceInput fi{
          .nAtoms = n_atoms, .pos = pos, .atmnrs = atmnrs, .box = box};

      // Temporary buffer for forces
      std::vector<double> forces(n_atoms * 3, 0.0);
      ::rgpot::ForceOut fo{.F = forces.data(), .energy = 0.0, .variance = 0.0};

      self->forceImpl(fi, &fo);

      output->energy = fo.energy;
      output->variance = fo.variance;

      // Create an owning DLPack tensor that copies the forces data.
      // This is necessary because `forces` is a local vector that goes
      // out of scope when this function returns.
      output->forces = rgpot_tensor_owned_cpu_f64_2d(
          forces.data(), static_cast<int64_t>(n_atoms), 3);

      return RGPOT_SUCCESS;
    } catch (...) {
      return RGPOT_INTERNAL_ERROR;
    }
  }
};

} // namespace rgpot
