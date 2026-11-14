! MIT License
! Copyright 2023--present rgpot developers

!> Threaded embedded-atom aluminium. Compiled only when the build asks for
!! it. The serial routine stays the one in rgpot_eam_al.f90, and the C++ face
!! calls that routine directly below the atom threshold and on one thread.
module rgpot_eam_al_omp
   use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_eam_al, only: eam_al_params_t
   use rgpot_workspace, only: rgpot_workspace_t, workspace_from_c, &
                              ws_set_error, ws_clear_error
   implicit none
   private

   public :: rgpot_eam_al_force_ws_omp

   integer, parameter :: embed_order = 8
   real(wp), parameter :: truncation_fraction = 0.9999_wp

contains

   function rgpot_eam_al_force_ws_omp(handle, natoms, positions, cell, forces, &
                                      energy) result(status) &
      bind(c, name="rgpot_eam_al_force_ws_omp")
      type(c_ptr), value :: handle
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      type(eam_al_params_t) :: par
      type(rgpot_workspace_t), pointer :: ws
      integer :: build_status
      character(len=:), allocatable :: errmsg

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

      call eam_al_energy_omp(positions, cell, par, ws%table, energy, forces, &
                             build_status, errmsg)
      if (build_status /= 0) then
         call ws_set_error(ws, errmsg)
         status = int(build_status, c_int)
      end if
   end function rgpot_eam_al_force_ws_omp

   subroutine eam_al_energy_omp(positions, cell, par, table, energy, forces, &
                                status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, i, s, j, nent
      integer :: k, e
      real(wp) :: rt, pair_shift, dens_shift
      real(wp) :: r, ea, eb, rpm1, rp, base, y, rhat(3), dedr, pa, pb, phi, dphi
      real(wp) :: p, dp
      real(wp), allocatable :: rho(:), dembed(:), e_pair(:), e_embed(:)
      real(wp), allocatable :: dens(:), ddens(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp
      call table%build(positions, cell, par%cutoff(), status, errmsg)
      if (status /= 0) return

      rt = sqrt(truncation_fraction)*par%rcut
      pair_shift = par%pair_amp_a*exp(-par%pair_decay_a*par%rcut) &
                   + par%pair_amp_b*exp(-par%pair_decay_b*par%rcut)
      dens_shift = par%rcut**par%density_power &
                   *(exp(-par%density_decay_a*par%rcut) &
                     + par%density_weight_b*exp(-par%density_decay_b*par%rcut))
      nent = max(table%row(natoms + 1_ip) - 1_ip, 1_ip)
      allocate (rho(natoms), dembed(natoms), e_pair(natoms), e_embed(natoms))
      allocate (dens(nent), ddens(nent))

      !$omp parallel private(i, s, j, k, e, r, ea, eb, rpm1, rp, base, y, &
      !$omp& rhat, dedr, pa, pb, phi, dphi, p, dp)
      !$omp do schedule(static)
      do i = 1, natoms
         rho(i) = 0.0_wp
         do s = table%row(i), table%row(i + 1_ip) - 1_ip
            r = table%dist(s)
            if (r >= rt) then
               dens(s) = 0.0_wp
               ddens(s) = 0.0_wp
               cycle
            end if
            ea = exp(-par%density_decay_a*r)
            eb = par%density_weight_b*exp(-par%density_decay_b*r)
            y = 1.0_wp
            base = r
            e = par%density_power - 1
            do while (e > 0)
               if (iand(e, 1) == 1) y = y*base
               base = base*base
               e = ishft(e, -1)
            end do
            rpm1 = y
            rp = rpm1*r
            dens(s) = par%density_scale*(rp*(ea + eb) - dens_shift)
            ddens(s) = par%density_scale &
                       *(real(par%density_power, wp)*rpm1*(ea + eb) &
                         - rp*(par%density_decay_a*ea + par%density_decay_b*eb))
            rho(i) = rho(i) + dens(s)
         end do
      end do
      !$omp end do
      !$omp do schedule(static)
      do i = 1, natoms
         p = par%embed_coeff(embed_order)
         dp = 0.0_wp
         do k = embed_order - 1, 1, -1
            dp = dp*rho(i) + p
            p = p*rho(i) + par%embed_coeff(k)
         end do
         e_embed(i) = rho(i)*(par%g_transform + p)
         dembed(i) = par%g_transform + p + rho(i)*dp
      end do
      !$omp end do
      !$omp do schedule(static)
      do i = 1, natoms
         e_pair(i) = 0.0_wp
         forces(:, i) = 0.0_wp
         do s = table%row(i), table%row(i + 1_ip) - 1_ip
            r = table%dist(s)
            if (r >= rt) cycle
            j = table%idx(s)
            rhat = table%vec(:, s)/r
            pa = par%pair_amp_a*exp(-par%pair_decay_a*r)
            pb = par%pair_amp_b*exp(-par%pair_decay_b*r)
            phi = pa + pb - pair_shift - 2.0_wp*par%g_transform*dens(s)
            dphi = -(par%pair_decay_a*pa + par%pair_decay_b*pb) &
                   - 2.0_wp*par%g_transform*ddens(s)
            e_pair(i) = e_pair(i) + 0.5_wp*phi
            dedr = dphi + (dembed(i) + dembed(j))*ddens(s)
            forces(:, i) = forces(:, i) + dedr*rhat
         end do
      end do
      !$omp end do
      !$omp end parallel

      energy = sum(e_pair) + sum(e_embed)
   end subroutine eam_al_energy_omp

end module rgpot_eam_al_omp
