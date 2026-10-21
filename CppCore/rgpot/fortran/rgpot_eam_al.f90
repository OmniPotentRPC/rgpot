! MIT License
! Copyright 2023--present rgpot developers
!
! Double-exponential embedded-atom aluminium. The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/Aluminum),
! BSD-3-Clause, copyright the eOn Development Team.

!> Embedded-atom aluminium with double-exponential pair, density, and an
!! eighth-order polynomial embedding function, in gather form.
!!
!! The energy is a pair sum plus a per-atom embedding term,
!!
!!     E = sum_{i<j} phi(r_ij) + sum_i F(rho_i),
!!     rho_i = sum_{j /= i} rho(r_ij),
!!
!! so the force on atom `i` couples to `F'` evaluated at every neighbour's
!! density as well as at its own. Three passes turn that into a gather:
!! densities first, then `F` and `F'` for all atoms, then the force. By the
!! time the force loop runs, `F'(rho_j)` is a plain array lookup, and atom
!! `i` can sum the whole force acting on it while writing only `f(:, i)`:
!!
!!     f_i = sum_j [ phi'(r_ij) + (F'_i + F'_j) rho'(r_ij) ] rhat_ij,
!!
!! with `rhat_ij` the unit vector from `i` towards `j`. This is where EAM
!! closes and a bond-order gather does not: the neighbour-dependent factor
!! is one scalar per atom, not a sum that has to be rebuilt by walking
!! atom `j`'s own neighbour list.
module rgpot_eam_al
   use iso_fortran_env, only: real32
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: eam_al_params_t, eam_al_energy_forces

   !> Order of the embedding polynomial.
   integer, parameter :: embed_order = 8

   !> Fraction of `rcut**2` at which every pair term is dropped outright.
   !!
   !! `phi` and `rho` are shifted to vanish at `rcut`, but their slopes are
   !! not, so a pair sitting on the edge would still exert force. The
   !! kernel truncates a hair inside instead and zeroes the whole pair
   !! there, which is what the aluminium parameters were fitted against.
   real(wp), parameter :: truncation_fraction = 0.9999_wp

   !> Double-exponential EAM parameters.
   !!
   !! The numbers written in EON r2549 `potinit.f` are default-real
   !! literals. A double-precision build still stores the single-precision
   !! rounding of each one. These defaults are that rounding. Keeping the
   !! extra printed digits moves the four-atom cluster by 1.76e-6 eV.
   type :: eam_al_params_t
      !> Gauge shift `g` shared by the pair and embedding terms.
      real(wp) :: g_transform = real(-0.60647886749203, real32)
      !> Pair amplitudes (eV) and decay rates (1/Angstrom).
      real(wp) :: pair_amp_a = real(2294.3609145535, real32)
      real(wp) :: pair_decay_a = real(3.0205380362464, real32)
      real(wp) :: pair_amp_b = real(-192.06894637533, real32)
      real(wp) :: pair_decay_b = real(1.5102690181232, real32)
      !> Overall scale on the atomic electron density.
      real(wp) :: density_scale = real(0.87928364657088, real32)
      !> Density decay rates (1/Angstrom) and the second term's weight.
      real(wp) :: density_decay_a = real(3.3068828537934, real32)
      real(wp) :: density_decay_b = real(6.6137657075868, real32)
      real(wp) :: density_weight_b = real(512.0, real32)
      !> Power of `r` multiplying the density's exponentials.
      integer :: density_power = 6
      !> Embedding coefficients `c_k` in `F(rho) = g rho + sum_k c_k rho^k`.
      real(wp) :: embed_coeff(embed_order) = [ &
                  real(4.4572157051836, real32), &
                  real(193.21775368064, real32), &
                  real(-1173.6502684704, real32), &
                  real(4203.3500116196, real32), &
                  real(-8785.1680280827, real32), &
                  real(10632.994102532, real32), &
                  real(-6921.0410455328, real32), &
                  real(1875.2698365752, real32)]
      !> Pair cutoff (Angstrom). `7.1` as a default-real literal.
      real(wp) :: rcut = real(7.1, real32)
   contains
      procedure :: cutoff => eam_al_cutoff
   end type eam_al_params_t

contains

   !> Interaction cutoff, the radius the neighbour table is built at.
   pure function eam_al_cutoff(self) result(rcut)
      class(eam_al_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = self%rcut
   end function eam_al_cutoff

   !> Radius inside `rcut` beyond which every pair term is zero.
   pure function truncation_radius(par) result(r)
      type(eam_al_params_t), intent(in) :: par
      real(wp) :: r

      r = sqrt(truncation_fraction)*par%rcut
   end function truncation_radius

   !> Unshifted pair double exponential `A e^{-a r} + B e^{-b r}`.
   pure function pair_shape(par, r) result(v)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: v

      v = par%pair_amp_a*exp(-par%pair_decay_a*r) &
          + par%pair_amp_b*exp(-par%pair_decay_b*r)
   end function pair_shape

   !> Unshifted density shape `r^eta (e^{-beta_a r} + w e^{-beta_b r})`.
   pure function density_shape(par, r) result(v)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: v

      v = r**par%density_power &
          *(exp(-par%density_decay_a*r) &
            + par%density_weight_b*exp(-par%density_decay_b*r))
   end function density_shape

   !> `x**n` for a small non-negative integer `n` by repeated squaring, so
   !! no libgcc `__powidf2` call sits in the pair loops.
   pure function ipow(x, n) result(y)
      real(wp), intent(in) :: x
      integer, intent(in) :: n
      real(wp) :: y

      real(wp) :: base
      integer :: e

      y = 1.0_wp
      base = x
      e = n
      do while (e > 0)
         if (iand(e, 1) == 1) y = y*base
         base = base*base
         e = ishft(e, -1)
      end do
   end function ipow

   !> `F(rho)` and `F'(rho)` in Horner form,
   !! `F = rho (g + c_1 + rho (c_2 + rho (c_3 + ...)))`.
   pure subroutine embed_horner(par, rho, f, dfdrho)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp), intent(out) :: f, dfdrho

      integer :: k
      real(wp) :: p, dp

      ! p(rho) = sum_k c_k rho^(k-1); F = g rho + rho p; F' = g + p + rho p'.
      p = par%embed_coeff(embed_order)
      dp = 0.0_wp
      do k = embed_order - 1, 1, -1
         dp = dp*rho + p
         p = p*rho + par%embed_coeff(k)
      end do
      f = rho*(par%g_transform + p)
      dfdrho = par%g_transform + p + rho*dp
   end subroutine embed_horner

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   !!
   !! Per ordered neighbour entry the density pass evaluates the two
   !! density exponentials once and keeps `rho(r)` and `rho'(r)`; the force
   !! pass adds the two pair exponentials. The shifts at `rcut` are
   !! evaluated once per call. scripts/validation/eam_al.py checks these
   !! forms, the Horner embedding and the gather force against the closed
   !! form in the module header.
   subroutine eam_al_energy_forces(positions, cell, par, table, energy, &
                                   forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, i, nent
      real(wp), allocatable :: rho(:), dembed(:), e_pair(:), e_embed(:)
      real(wp), allocatable :: dens(:), ddens(:)
      real(wp) :: rt, pair_shift, dens_shift

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp

      call table%build(positions, cell, par%cutoff(), status, errmsg)
      if (status /= 0) return

      rt = truncation_radius(par)
      pair_shift = pair_shape(par, par%rcut)
      dens_shift = density_shape(par, par%rcut)

      nent = max(table%row(natoms + 1_ip) - 1_ip, 1_ip)
      allocate (rho(natoms), dembed(natoms), e_pair(natoms), e_embed(natoms))
      allocate (dens(nent), ddens(nent))

      ! Pass one: each atom sums the density its own neighbours put at its
      ! site, keeping rho(r) and rho'(r) of every entry for pass three.
      do concurrent(i=1:natoms)
         call atom_density(par, table, i, rt, dens_shift, dens, ddens, rho(i))
      end do

      ! Pass two: F and F' for every atom. F' stands as its own pass
      ! because the force loop reads F' of the neighbours, not only of the
      ! atom it is summing for.
      do concurrent(i=1:natoms)
         call embed_horner(par, rho(i), e_embed(i), dembed(i))
      end do

      ! Pass three: each atom sums the whole force acting on it and writes
      ! only its own column.
      do concurrent(i=1:natoms)
         call atom_contribution(par, table, i, rt, pair_shift, dens, ddens, &
                                dembed, e_pair(i), forces(:, i))
      end do

      energy = sum(e_pair) + sum(e_embed)
   end subroutine eam_al_energy_forces

   !> Total electron density at atom `i`; stores `rho(r)` and `rho'(r)` of
   !! each of its entries (zero past the truncation radius).
   pure subroutine atom_density(par, table, i, rt, dens_shift, dens, ddens, &
                                rho_i)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp), intent(in) :: rt, dens_shift
      real(wp), intent(inout) :: dens(:), ddens(:)
      real(wp), intent(out) :: rho_i

      integer(ip) :: s
      real(wp) :: r, ea, eb, rpm1, rp

      rho_i = 0.0_wp
      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         if (r >= rt) then
            dens(s) = 0.0_wp
            ddens(s) = 0.0_wp
            cycle
         end if
         ea = exp(-par%density_decay_a*r)
         eb = par%density_weight_b*exp(-par%density_decay_b*r)
         rpm1 = ipow(r, par%density_power - 1)
         rp = rpm1*r
         dens(s) = par%density_scale*(rp*(ea + eb) - dens_shift)
         ddens(s) = par%density_scale &
                    *(real(par%density_power, wp)*rpm1*(ea + eb) &
                      - rp*(par%density_decay_a*ea + par%density_decay_b*eb))
         rho_i = rho_i + dens(s)
      end do
   end subroutine atom_density

   !> Atom `i`'s half of the pair energy and the total force acting on it.
   pure subroutine atom_contribution(par, table, i, rt, pair_shift, dens, &
                                     ddens, dembed, e_i, f_i)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp), intent(in) :: rt, pair_shift
      real(wp), intent(in) :: dens(:), ddens(:), dembed(:)
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: s, j
      real(wp) :: r, rhat(3), dedr, pa, pb, phi, dphi

      e_i = 0.0_wp
      f_i = 0.0_wp

      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         if (r >= rt) cycle
         j = table%idx(s)
         rhat = table%vec(:, s)/r
         pa = par%pair_amp_a*exp(-par%pair_decay_a*r)
         pb = par%pair_amp_b*exp(-par%pair_decay_b*r)

         ! Pair energy with the -2 g rho(r) gauge term, and its slope.
         phi = pa + pb - pair_shift - 2.0_wp*par%g_transform*dens(s)
         dphi = -(par%pair_decay_a*pa + par%pair_decay_b*pb) &
                - 2.0_wp*par%g_transform*ddens(s)

         ! Half the bond energy, since the neighbour's own pass takes the
         ! other half.
         e_i = e_i + 0.5_wp*phi

         ! The whole radial force this bond exerts on i: the pair slope
         ! plus the density slope weighted by both ends' embedding
         ! derivatives.
         dedr = dphi + (dembed(i) + dembed(j))*ddens(s)
         f_i = f_i + dedr*rhat
      end do
   end subroutine atom_contribution

end module rgpot_eam_al
