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

   !> Electron density a neighbour at distance `r` puts at an atom's site.
   !!
   !! Shifted so it vanishes at `rcut`, then scaled.
   elemental function density(par, r) result(rho)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: rho

      rho = 0.0_wp
      if (r >= truncation_radius(par)) return

      rho = par%density_scale &
            *(density_shape(par, r) - density_shape(par, par%rcut))
   end function density

   !> Radial derivative of `density`.
   !!
   !! `d/dr [r^eta (e^{-beta_a r} + w e^{-beta_b r})]` splits into the term
   !! from the power and the term from the two exponentials; the constant
   !! shift drops out. The expression also covers `eta == 0`, where the
   !! first term vanishes on its own.
   elemental function density_deriv(par, r) result(drho)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: drho

      real(wp) :: decay_a, decay_b, sum_exp, sum_decayed_exp

      drho = 0.0_wp
      if (r >= truncation_radius(par)) return

      decay_a = exp(-par%density_decay_a*r)
      decay_b = par%density_weight_b*exp(-par%density_decay_b*r)
      sum_exp = decay_a + decay_b
      sum_decayed_exp = par%density_decay_a*decay_a &
                        + par%density_decay_b*decay_b

      drho = par%density_scale &
             *(real(par%density_power, wp)*r**(par%density_power - 1)*sum_exp &
               - r**par%density_power*sum_decayed_exp)
   end function density_deriv

   !> Pair energy of one bond of length `r`.
   !!
   !! The `-2 g rho(r)` term is the gauge partner of the `+g rho_i` term in
   !! `F`; each unordered pair carries it once and each atom's embedding
   !! picks it up twice, so the two cancel over the whole cell.
   elemental function pair_energy(par, r) result(phi)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: phi

      phi = 0.0_wp
      if (r >= truncation_radius(par)) return

      phi = pair_shape(par, r) - pair_shape(par, par%rcut) &
            - 2.0_wp*par%g_transform*density(par, r)
   end function pair_energy

   !> Radial derivative of `pair_energy`.
   elemental function pair_deriv(par, r) result(dphi)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dphi

      dphi = 0.0_wp
      if (r >= truncation_radius(par)) return

      dphi = -(par%pair_amp_a*par%pair_decay_a*exp(-par%pair_decay_a*r) &
               + par%pair_amp_b*par%pair_decay_b*exp(-par%pair_decay_b*r)) &
             - 2.0_wp*par%g_transform*density_deriv(par, r)
   end function pair_deriv

   !> Embedding energy `F(rho) = g rho + sum_k c_k rho^k`.
   elemental function embedding(par, rho) result(e)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: e

      integer :: k

      e = par%g_transform*rho
      do k = 1, embed_order
         e = e + par%embed_coeff(k)*rho**k
      end do
   end function embedding

   !> Derivative of the embedding function with respect to the density.
   elemental function embedding_deriv(par, rho) result(dfdrho)
      type(eam_al_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: dfdrho

      integer :: k

      dfdrho = par%g_transform + par%embed_coeff(1)
      do k = 2, embed_order
         dfdrho = dfdrho + par%embed_coeff(k)*real(k, wp)*rho**(k - 1)
      end do
   end function embedding_deriv

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
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

      integer(ip) :: natoms, i
      real(wp), allocatable :: rho(:), dembed(:), e_pair(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp

      call table%build(positions, cell, par%cutoff(), status, errmsg)
      if (status /= 0) return

      allocate (rho(natoms), dembed(natoms), e_pair(natoms))

      ! Pass one: each atom sums the density its own neighbours put at its
      ! site, so the iterations write disjoint slots.
      do concurrent(i=1:natoms)
         rho(i) = atom_density(par, table, i)
      end do

      ! Pass two: the embedding derivative for every atom. It stands as its
      ! own pass because the force loop reads `F'` of the neighbours, not
      ! only of the atom it is summing for.
      dembed = embedding_deriv(par, rho)

      ! Pass three: each atom sums the whole force acting on it and writes
      ! only its own column.
      do concurrent(i=1:natoms)
         call atom_contribution(par, table, i, dembed, e_pair(i), forces(:, i))
      end do

      energy = sum(e_pair) + sum(embedding(par, rho))
   end subroutine eam_al_energy_forces

   !> Total electron density at atom `i`.
   pure function atom_density(par, table, i) result(rho_i)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp) :: rho_i

      integer(ip) :: s

      rho_i = 0.0_wp
      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         rho_i = rho_i + density(par, table%dist(s))
      end do
   end function atom_density

   !> Atom `i`'s half of the pair energy and the total force acting on it.
   pure subroutine atom_contribution(par, table, i, dembed, e_i, f_i)
      type(eam_al_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp), intent(in) :: dembed(:)
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: s, j
      real(wp) :: r, rhat(3), dedr

      e_i = 0.0_wp
      f_i = 0.0_wp

      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         j = table%idx(s)
         r = table%dist(s)
         rhat = table%vec(:, s)/r

         ! Half the bond energy, since the neighbour's own pass takes the
         ! other half.
         e_i = e_i + 0.5_wp*pair_energy(par, r)

         ! The whole radial force this bond exerts on i: the pair slope
         ! plus the density slope weighted by both ends' embedding
         ! derivatives.
         dedr = pair_deriv(par, r) &
                + (dembed(i) + dembed(j))*density_deriv(par, r)
         f_i = f_i + dedr*rhat
      end do
   end subroutine atom_contribution

end module rgpot_eam_al
