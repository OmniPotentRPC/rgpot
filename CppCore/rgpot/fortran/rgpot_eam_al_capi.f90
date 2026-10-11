! MIT License
! Copyright 2023--present rgpot developers

!> C entry point for the embedded-atom aluminium potential.
!!
!! Only `bind(c)` names leave this library; the physics module exports no
!! external symbols of its own that C could collide with. The `_ws` entry
!! point keeps the neighbour table in the caller's workspace
!! (rgpot_workspace), where vesin reuses its buffers and tracks list
!! staleness for that caller's geometry; the entry point without a handle
!! shares one module-level workspace and is process-serial.
module rgpot_eam_al_capi
   use rgpot_kinds, only: wp, ip, c_double, c_int
   use, intrinsic :: iso_c_binding, only: c_ptr
   use rgpot_ferror, only: set_error, clear_error
   use rgpot_workspace, only: rgpot_workspace_t, workspace_from_c, &
                              ws_set_error, ws_clear_error
   use rgpot_eam_al, only: eam_al_params_t, eam_al_energy_forces
   implicit none
   private

   public :: rgpot_eam_al_force, rgpot_eam_al_force_ws, rgpot_eam_al_ensure_params

   !> Workspace of the entry point without a handle (process-serial).
   type(rgpot_workspace_t), save, target :: legacy
   type(eam_al_params_t), save :: params

   !> F2: set by the C++ wrapper under std::call_once.
   !! After that call the block is read-only. Kernels take it
   !! intent(in). The handle-free entry's legacy workspace, when
   !! this module has one, stays process-serial and is not used
   !! by the Python binding.
   logical, save :: params_ready = .false.

contains

   !> Evaluate embedded-atom aluminium forces and energy.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `cell` is nine doubles, row-major, one cell vector per row. Energies
   !! are eV and forces eV/Angstrom. Returns zero on success, non-zero on
   !! failure with the message available through `rgpot_fortran_last_error`.
   function rgpot_eam_al_force(natoms, positions, cell, forces, energy) &
      result(status) bind(c, name="rgpot_eam_al_force")
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: build_status
      character(len=:), allocatable :: errmsg

      call clear_error()
      status = 0_c_int
      energy = 0.0_c_double
      forces = 0.0_c_double

      if (natoms < 1_c_int) return

      call eam_al_energy_forces(positions, cell, params, legacy%table, energy, &
                                forces, build_status, errmsg)
      if (build_status /= 0) then
         call set_error(errmsg)
         status = int(build_status, c_int)
      end if
   end function rgpot_eam_al_force

   !> Evaluate embedded-atom aluminium forces and energy in a per-caller
   !! workspace.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `cell` is nine doubles, row-major, one cell vector per row. Energies
   !! are eV and forces eV/Angstrom. Returns zero on success, non-zero on
   !! failure with the message available
   !! through `rgpot_fortran_workspace_error`. `handle` is a workspace from
   !! `rgpot_fortran_workspace_new`; a NULL handle returns -1.
   function rgpot_eam_al_force_ws(handle, natoms, positions, cell, forces, energy) &
      result(status) bind(c, name="rgpot_eam_al_force_ws")
      type(c_ptr), value :: handle
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: build_status
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

      call eam_al_energy_forces(positions, cell, params, ws%table, energy, &
                                forces, build_status, errmsg)
      if (build_status /= 0) then
         call ws_set_error(ws, errmsg)
         status = int(build_status, c_int)
      end if
   end function rgpot_eam_al_force_ws


   !> F2: default-initialise the saved parameter block once.
   !! The C++ face calls this under std::call_once before any _ws
   !! evaluation. A second call does not write.
   subroutine rgpot_eam_al_ensure_params() bind(c, name="rgpot_eam_al_ensure_params")
      if (.not. params_ready) then
         params = eam_al_params_t()
         params_ready = .true.
      end if
   end subroutine rgpot_eam_al_ensure_params

end module rgpot_eam_al_capi
