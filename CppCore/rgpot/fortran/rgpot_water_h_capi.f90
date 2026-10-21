! MIT License
! Copyright 2023--present rgpot developers

!> C entry point for the H-in-water potential.
!!
!! Only `bind(c)` names leave this library; the physics module exports no
!! external symbols of its own that C could collide with. The parameter
!! set is saved here so a caller can be given a way to retune it without
!! the kernel holding state of its own.
!!
!! Unlike the pair potentials, this one keeps no neighbour table. Its sum
!! has a single centre and no cutoff, so every atom contributes one leg
!! at its nearest periodic image and there is nothing for vesin to build.
module rgpot_water_h_capi
   use rgpot_kinds, only: wp, ip, c_double, c_int
   use, intrinsic :: iso_c_binding, only: c_ptr
   use rgpot_ferror, only: set_error, clear_error
   use rgpot_workspace, only: rgpot_workspace_t, workspace_from_c, &
                              ws_set_error, ws_clear_error
   use rgpot_water_h, only: water_h_params_t, water_h_energy_forces
   implicit none
   private

   public :: rgpot_water_h_force, rgpot_water_h_force_ws, rgpot_water_h_ensure_params

   type(water_h_params_t), save :: params

   !> F2: set by the C++ wrapper under std::call_once.
   !! After that call the block is read-only. Kernels take it
   !! intent(in). The handle-free entry's legacy workspace, when
   !! this module has one, stays process-serial and is not used
   !! by the Python binding.
   logical, save :: params_ready = .false.

contains

   !> Evaluate H-water forces and energy.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `atomic_numbers` is one `int` per atom; `cell` is nine doubles,
   !! row-major, one cell vector per row, and a zero cell means an
   !! isolated cluster. Returns zero on success, non-zero on failure with
   !! the message available through `rgpot_fortran_last_error`.
   !!
   !! The atoms must be hydrogen and oxygen only, in whole water molecules
   !! plus one extra H, and that extra H must be the last atom: nothing in
   !! an atomic-number list distinguishes it from the hydrogens bound in
   !! water, so its position in the list is what identifies it. A list
   !! that does not meet this is rejected with a message naming the fault.
   function rgpot_water_h_force(natoms, positions, atomic_numbers, cell, &
                                forces, energy) result(status) &
      bind(c, name="rgpot_water_h_force")
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      integer(c_int), intent(in) :: atomic_numbers(natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: eval_status
      character(len=:), allocatable :: errmsg

      call clear_error()
      status = 0_c_int
      energy = 0.0_c_double
      forces = 0.0_c_double

      if (natoms < 1_c_int) return

      call water_h_energy_forces(positions, atomic_numbers, cell, params, &
                                 energy, forces, eval_status, errmsg)
      if (eval_status /= 0) then
         call set_error(errmsg)
         status = int(eval_status, c_int)
      end if
   end function rgpot_water_h_force

   !> Evaluate H-water forces and energy in a per-caller
   !! workspace.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `atomic_numbers` is one `int` per atom; `cell` is nine doubles,
   !! row-major, one cell vector per row, and a zero cell means an
   !! isolated cluster. Returns zero on success, non-zero on failure with
   !! the message available
   !! through `rgpot_fortran_workspace_error`. `handle` is a workspace from
   !! `rgpot_fortran_workspace_new`; a NULL handle returns -1.
   !!
   !! The atoms must be hydrogen and oxygen only, in whole water molecules
   !! plus one extra H, and that extra H must be the last atom: nothing in
   !! an atomic-number list distinguishes it from the hydrogens bound in
   !! water, so its position in the list is what identifies it. A list
   !! that does not meet this is rejected with a message naming the fault.
   function rgpot_water_h_force_ws(handle, natoms, positions, atomic_numbers, cell, &
                                forces, energy) result(status) &
      bind(c, name="rgpot_water_h_force_ws")
      type(c_ptr), value :: handle
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      integer(c_int), intent(in) :: atomic_numbers(natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: eval_status
      character(len=:), allocatable :: errmsg
      type(rgpot_workspace_t), pointer :: ws

      ws => workspace_from_c(handle)
      if (.not. associated(ws)) then
         status = -1_c_int
         return
      end if
      call ws_clear_error(ws)
      status = 0_c_int
      energy = 0.0_c_double
      forces = 0.0_c_double

      if (natoms < 1_c_int) return

      call water_h_energy_forces(positions, atomic_numbers, cell, params, &
                                 energy, forces, eval_status, errmsg)
      if (eval_status /= 0) then
         call ws_set_error(ws, errmsg)
         status = int(eval_status, c_int)
      end if
   end function rgpot_water_h_force_ws


   !> F2: default-initialise the saved parameter block once.
   !! The C++ face calls this under std::call_once before any _ws
   !! evaluation. A second call does not write.
   subroutine rgpot_water_h_ensure_params() bind(c, name="rgpot_water_h_ensure_params")
      if (.not. params_ready) then
         params = water_h_params_t()
         params_ready = .true.
      end if
   end subroutine rgpot_water_h_ensure_params

end module rgpot_water_h_capi
