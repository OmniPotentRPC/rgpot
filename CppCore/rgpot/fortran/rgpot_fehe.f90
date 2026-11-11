! MIT License
! Copyright 2023--present rgpot developers
!
! Fe-He embedded-atom potential. The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/FeHe), BSD-3-Clause,
! copyright the eOn Development Team. The iron part is Ackland's 2004
! many-body a-Fe fit; the helium part adds a second, s-band density channel
! that couples Fe to He.
!
!> Fe-He embedded-atom potential in gather form.
!!
!! The energy has three species-dependent pair terms (Fe-Fe, Fe-He, He-He)
!! and *two* independent embedding channels:
!!
!!   * a d-band density `rho_d`, fed only by Fe-Fe pairs and embedded only
!!     on Fe atoms through `-sqrt(rho_d) + ...`;
!!   * an s-band density `rho_s`, fed only by Fe-He pairs and embedded on
!!     both species through `xhep(9) sqrt(rho_s) + ...`.
!!
!! He-He pairs feed neither density, so a pure-helium configuration reduces
!! to the bare HFD-B dimer potential.
!!
!! An embedded-atom sum closes as a gather in three passes, because the
!! force on atom `i` needs `dF/drho` at *both* ends of every pair:
!!
!!   1. per-atom densities from the full neighbour list;
!!   2. per-atom embedding energy and `dF/drho` for both channels;
!!   3. per-atom force gather, reading `dF/drho` of `i` and of each `j`.
!!
!! Each pass writes only slot `i` of its output arrays, so all three run
!! under `do concurrent`. Energy accrues to per-atom arrays and reduces
!! afterwards.
module rgpot_fehe
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: fehe_params_t, fehe_energy_forces

   !> Internal species labels. The C boundary speaks atomic numbers; these
   !! never leave the module.
   integer(ip), parameter :: species_fe = 1_ip
   integer(ip), parameter :: species_he = 2_ip

   !> Atomic numbers this potential covers.
   integer(ip), parameter :: z_fe = 26_ip
   integer(ip), parameter :: z_he = 2_ip

   !> Bohr radius (Angstrom), sets the s-band decay length.
   real(wp), parameter :: bohr_radius = 0.529177210818181818_wp
   !> Boltzmann constant (J/K) and the electronvolt (J), which together turn
   !! the helium dimer well depth from kelvin into eV.
   real(wp), parameter :: boltzmann_j_per_k = 1.380658e-23_wp
   real(wp), parameter :: electron_volt_j = 1.602177e-19_wp

   !> Densities below this embed to zero. Both embedding functions carry a
   !! `sqrt(rho)` term whose derivative diverges at the origin, so atoms
   !! with an empty channel take the limit rather than the expression.
   real(wp), parameter :: rho_floor = 1.0e-35_wp

   !> Every coefficient of the Fe-He model, one place.
   !!
   !! Names map onto the legacy source as follows: `fe_*` come from the
   !! inline Ackland-2004 expressions in `feforce.f`, `fehe_*` and
   !! `sband_emb_*` from `xhep(1:8)` and `xhep(9:11)`, `sband_amp` and
   !! `sband_decay` from `dNs` and `psi`, and `he_*` from the `herm`, `c6`,
   !! `c8`, `c10`, `heaa`, `hea`, `heb`, `hed`, `hee` block.
   type :: fehe_params_t
      !> Fe-Fe pair cutoff (Angstrom), `rcutp_fe`.
      real(wp) :: rcut_fe_pair = 5.3_wp
      !> Fe-Fe d-band density cutoff, `rcutr_fe`.
      real(wp) :: rcut_fe_rho = 4.2_wp
      !> Fe-He pair cutoff, `rcut_he`.
      real(wp) :: rcut_fehe_pair = 3.9028_wp
      !> Fe-He s-band density cutoff, `rcut_her`.
      real(wp) :: rcut_fehe_rho = 4.10_wp
      !> He-He pair cutoff, the bare `5.4` of the legacy dimer branch.
      real(wp) :: rcut_he_pair = 5.4_wp

      !> Fe-Fe screened Coulomb below `fe_bridge_lo`: `scale/r` times a
      !! four-term exponential screening function.
      real(wp) :: fe_zbl_scale = 9.7342365892908e+03_wp
      real(wp) :: fe_zbl_weight(4) = [ &
                  0.18180_wp, 0.50990_wp, 0.28020_wp, 0.02817_wp]
      real(wp) :: fe_zbl_decay(4) = [ &
                  2.8616724320005e+01_wp, 8.4267310396064e+00_wp, &
                  3.6030244464156e+00_wp, 1.8028536321603e+00_wp]

      !> Fe-Fe bridge between the screened Coulomb and the knot spline:
      !! `exp(c1 + c2 r + c3 r^2 + c4 r^3)`.
      real(wp) :: fe_bridge(4) = [ &
                  7.4122709384068e+00_wp, -6.4180690713367e-01_wp, &
                  -2.6043547961722e+00_wp, 6.2625393931230e-01_wp]
      real(wp) :: fe_bridge_lo = 1.0000_wp
      real(wp) :: fe_bridge_hi = 2.0500_wp

      !> Fe-Fe cubic knot spline above `fe_bridge_hi`,
      !! `sum_k a_k max(r_k - r, 0)^3`.
      real(wp) :: fe_pair_knot(13) = [ &
                  2.2_wp, 2.3_wp, 2.4_wp, 2.5_wp, 2.6_wp, 2.7_wp, 2.8_wp, &
                  3.0_wp, 3.3_wp, 3.7_wp, 4.2_wp, 4.7_wp, 5.3_wp]
      real(wp) :: fe_pair_coeff(13) = [ &
                  -2.7444805994228e+01_wp, 1.5738054058489e+01_wp, &
                  2.2077118733936e+00_wp, -2.4989799053251e+00_wp, &
                  4.2099676494795e+00_wp, -7.7361294129713e-01_wp, &
                  8.0656414937789e-01_wp, -2.3194358924605e+00_wp, &
                  2.6577406128280e+00_wp, -1.0260416933564e+00_wp, &
                  3.5018615891957e-01_wp, -5.8531821042271e-02_wp, &
                  -3.0458824556234e-03_wp]

      !> Fe-Fe d-band density, same cubic knot form.
      real(wp) :: fe_rho_knot(3) = [2.4_wp, 3.2_wp, 4.2_wp]
      real(wp) :: fe_rho_coeff(3) = [ &
                  1.1686859407970e+01_wp, -1.4710740098830e-02_wp, &
                  4.7193527075943e-01_wp]

      !> Fe d-band embedding, `-sqrt(rho) + q rho^2 + t rho^4`.
      real(wp) :: fe_emb_quadratic = -6.7314115586063e-04_wp
      real(wp) :: fe_emb_quartic = 7.6514905604792e-08_wp

      !> Fe-He pair term, cubic knot spline, `xhep(1:8)`.
      real(wp) :: fehe_knot(8) = [ &
                  1.5440_wp, 1.6155_wp, 1.6896_wp, 1.8017_wp, &
                  2.0482_wp, 2.3816_wp, 3.5067_wp, 3.9028_wp]
      real(wp) :: fehe_coeff(8) = [ &
                  559.804426025391_wp, -45.916354995728_wp, &
                  35.550312671661_wp, 164.319865173340_wp, &
                  -1.727464405060_wp, 0.106771826237_wp, &
                  0.073715269849_wp, 0.038235287677_wp]

      !> s-band density carried by an Fe-He pair,
      !! `amp r^3 exp(-2 decay r)`.
      real(wp) :: sband_amp = 20.0_wp
      real(wp) :: sband_decay = 1.5312426703208_wp/bohr_radius

      !> s-band embedding, `a sqrt(rho) + b rho^2 + c rho^4`, identical for
      !! both species: the legacy `embes` writes the same branch twice.
      real(wp) :: sband_emb_sqrt = 0.220813420485_wp
      real(wp) :: sband_emb_quadratic = 1.367508764130_wp
      real(wp) :: sband_emb_quartic = 3.382256025271_wp

      !> He-He HFD-B dimer: hard-core `aa exp(-alpha x + beta x^2)` less a
      !! damped dispersion series in `x = r / rm`.
      real(wp) :: he_rm = 2.9683_wp
      real(wp) :: he_c6 = 1.35186623_wp
      real(wp) :: he_c8 = 0.41495143_wp
      real(wp) :: he_c10 = 0.17151143_wp
      real(wp) :: he_aa = 186924.404_wp
      real(wp) :: he_alpha = 10.5717543_wp
      real(wp) :: he_beta = -2.07758779_wp
      !> Damping switches off above this reduced separation.
      real(wp) :: he_damp = 1.438_wp
      !> Well-depth scale (eV), `10.956 k_B` in the legacy source.
      real(wp) :: he_eps = 10.956_wp*boltzmann_j_per_k/electron_volt_j
   contains
      procedure :: cutoff => fehe_cutoff
   end type fehe_params_t

contains

   !> Largest of the five cutoffs, which is what the neighbour table needs.
   !! Every pair term re-checks its own cutoff.
   pure function fehe_cutoff(self) result(rcut)
      class(fehe_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = max(self%rcut_fe_pair, self%rcut_fe_rho, self%rcut_fehe_pair, &
                 self%rcut_fehe_rho, self%rcut_he_pair)
   end function fehe_cutoff

   ! ------------------------------------------------------------------
   ! Pair terms. Each returns zero outside its own cutoff, so a caller may
   ! evaluate any of them at any separation the table produced.
   !
   ! `max(knot - r, 0)` reproduces the legacy `HH(knot - r) * (knot - r)`
   ! exactly, including the closed-from-above convention at `r == knot`,
   ! without a Heaviside call or a branch.
   ! ------------------------------------------------------------------

   !> Fe-Fe pair energy (eV).
   elemental function pair_fe_fe(par, r) result(v)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: v

      v = 0.0_wp
      if (r > par%rcut_fe_pair) return

      if (r < par%fe_bridge_lo) then
         v = par%fe_zbl_scale/r &
             *sum(par%fe_zbl_weight*exp(-par%fe_zbl_decay*r))
      else if (r < par%fe_bridge_hi) then
         v = exp(par%fe_bridge(1) + par%fe_bridge(2)*r &
                 + par%fe_bridge(3)*r*r + par%fe_bridge(4)*r*r*r)
      else
         v = sum(par%fe_pair_coeff*max(par%fe_pair_knot - r, 0.0_wp)**3)
      end if
   end function pair_fe_fe

   !> `dV/dr` of `pair_fe_fe`.
   elemental function dpair_fe_fe(par, r) result(dv)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dv

      real(wp) :: screen(4), expo

      dv = 0.0_wp
      if (r > par%rcut_fe_pair) return

      if (r < par%fe_bridge_lo) then
         screen = par%fe_zbl_weight*exp(-par%fe_zbl_decay*r)
         dv = -par%fe_zbl_scale/(r*r)*sum(screen) &
              - par%fe_zbl_scale/r*sum(par%fe_zbl_decay*screen)
      else if (r < par%fe_bridge_hi) then
         expo = exp(par%fe_bridge(1) + par%fe_bridge(2)*r &
                    + par%fe_bridge(3)*r*r + par%fe_bridge(4)*r*r*r)
         dv = expo*(par%fe_bridge(2) + 2.0_wp*par%fe_bridge(3)*r &
                    + 3.0_wp*par%fe_bridge(4)*r*r)
      else
         dv = -3.0_wp &
              *sum(par%fe_pair_coeff*max(par%fe_pair_knot - r, 0.0_wp)**2)
      end if
   end function dpair_fe_fe

   !> Fe-He pair energy (eV).
   elemental function pair_fe_he(par, r) result(v)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: v

      v = 0.0_wp
      if (r > par%rcut_fehe_pair) return
      v = sum(par%fehe_coeff*max(par%fehe_knot - r, 0.0_wp)**3)
   end function pair_fe_he

   !> `dV/dr` of `pair_fe_he`.
   elemental function dpair_fe_he(par, r) result(dv)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dv

      dv = 0.0_wp
      if (r > par%rcut_fehe_pair) return
      dv = -3.0_wp*sum(par%fehe_coeff*max(par%fehe_knot - r, 0.0_wp)**2)
   end function dpair_fe_he

   !> He-He pair energy (eV), HFD-B form in the reduced separation
   !! `x = r / rm`. The dispersion series is damped below `he_damp`.
   elemental function pair_he_he(par, r) result(v)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: v

      real(wp) :: x, dispersion, core, damp

      v = 0.0_wp
      if (r > par%rcut_he_pair) return

      x = r/par%he_rm
      dispersion = par%he_c6/x**6 + par%he_c8/x**8 + par%he_c10/x**10
      core = par%he_aa*exp(-par%he_alpha*x + par%he_beta*x*x)

      damp = 1.0_wp
      if (x < par%he_damp) damp = exp(-(par%he_damp/x - 1.0_wp)**2)

      v = par%he_eps*(core - damp*dispersion)
   end function pair_he_he

   !> `dV/dr` of `pair_he_he`.
   elemental function dpair_he_he(par, r) result(dv)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dv

      real(wp) :: x, dispersion, ddispersion, core, damp, ddamp

      dv = 0.0_wp
      if (r > par%rcut_he_pair) return

      x = r/par%he_rm
      dispersion = par%he_c6/x**6 + par%he_c8/x**8 + par%he_c10/x**10
      ddispersion = -(6.0_wp*par%he_c6/x**7 + 8.0_wp*par%he_c8/x**9 &
                      + 10.0_wp*par%he_c10/x**11)
      core = par%he_aa*exp(-par%he_alpha*x + par%he_beta*x*x)

      damp = 1.0_wp
      ddamp = 0.0_wp
      if (x < par%he_damp) then
         damp = exp(-(par%he_damp/x - 1.0_wp)**2)
         ddamp = damp*2.0_wp*par%he_damp*(par%he_damp/x - 1.0_wp)/(x*x)
      end if

      ! Chain rule through x, hence the division by rm.
      dv = par%he_eps*(core*(-par%he_alpha + 2.0_wp*par%he_beta*x) &
                       - ddamp*dispersion - damp*ddispersion)/par%he_rm
   end function dpair_he_he

   ! ------------------------------------------------------------------
   ! Density channels.
   ! ------------------------------------------------------------------

   !> d-band density an Fe neighbour contributes at separation `r`.
   elemental function density_dband(par, r) result(rho)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: rho

      rho = 0.0_wp
      if (r > par%rcut_fe_rho) return
      rho = sum(par%fe_rho_coeff*max(par%fe_rho_knot - r, 0.0_wp)**3)
   end function density_dband

   !> `drho/dr` of `density_dband`.
   elemental function ddensity_dband(par, r) result(drho)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: drho

      drho = 0.0_wp
      if (r > par%rcut_fe_rho) return
      drho = -3.0_wp &
             *sum(par%fe_rho_coeff*max(par%fe_rho_knot - r, 0.0_wp)**2)
   end function ddensity_dband

   !> s-band density an unlike neighbour contributes at separation `r`.
   elemental function density_sband(par, r) result(rho)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: rho

      rho = 0.0_wp
      if (r > par%rcut_fehe_rho) return
      rho = par%sband_amp*r**3*exp(-2.0_wp*par%sband_decay*r)
   end function density_sband

   !> `drho/dr` of `density_sband`.
   elemental function ddensity_sband(par, r) result(drho)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: drho

      drho = 0.0_wp
      if (r > par%rcut_fehe_rho) return
      drho = par%sband_amp*r**2*exp(-2.0_wp*par%sband_decay*r) &
             *(3.0_wp - 2.0_wp*par%sband_decay*r)
   end function ddensity_sband

   ! ------------------------------------------------------------------
   ! Embedding.
   ! ------------------------------------------------------------------

   !> d-band embedding energy. Only Fe embeds; He carries no d-band density
   !! in the first place, so the two statements agree.
   elemental function embed_dband(par, species, rho) result(f)
      type(fehe_params_t), intent(in) :: par
      integer(ip), intent(in) :: species
      real(wp), intent(in) :: rho
      real(wp) :: f

      f = 0.0_wp
      if (species /= species_fe) return
      if (rho < rho_floor) return
      f = -sqrt(rho) + par%fe_emb_quadratic*rho**2 + par%fe_emb_quartic*rho**4
   end function embed_dband

   !> `dF/drho` of `embed_dband`.
   elemental function dembed_dband(par, species, rho) result(df)
      type(fehe_params_t), intent(in) :: par
      integer(ip), intent(in) :: species
      real(wp), intent(in) :: rho
      real(wp) :: df

      df = 0.0_wp
      if (species /= species_fe) return
      if (rho < rho_floor) return
      df = -0.5_wp/sqrt(rho) + 2.0_wp*par%fe_emb_quadratic*rho &
           + 4.0_wp*par%fe_emb_quartic*rho**3
   end function dembed_dband

   !> s-band embedding energy, shared by both species.
   elemental function embed_sband(par, rho) result(f)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: f

      f = 0.0_wp
      if (rho < rho_floor) return
      f = par%sband_emb_sqrt*sqrt(rho) + par%sband_emb_quadratic*rho**2 &
          + par%sband_emb_quartic*rho**4
   end function embed_sband

   !> `dF/drho` of `embed_sband`.
   elemental function dembed_sband(par, rho) result(df)
      type(fehe_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: df

      df = 0.0_wp
      if (rho < rho_floor) return
      df = 0.5_wp*par%sband_emb_sqrt/sqrt(rho) &
           + 2.0_wp*par%sband_emb_quadratic*rho &
           + 4.0_wp*par%sband_emb_quartic*rho**3
   end function dembed_sband

   ! ------------------------------------------------------------------
   ! Species dispatch. Three ordered pairs collapse to three cases because
   ! every term is symmetric under exchange.
   ! ------------------------------------------------------------------

   !> Pair energy and `dV/dr` for the species combination of `si` and `sj`.
   pure subroutine pair_terms(par, si, sj, r, v, dvdr)
      type(fehe_params_t), intent(in) :: par
      integer(ip), intent(in) :: si, sj
      real(wp), intent(in) :: r
      real(wp), intent(out) :: v, dvdr

      if (si == species_fe .and. sj == species_fe) then
         v = pair_fe_fe(par, r)
         dvdr = dpair_fe_fe(par, r)
      else if (si == species_he .and. sj == species_he) then
         v = pair_he_he(par, r)
         dvdr = dpair_he_he(par, r)
      else
         v = pair_fe_he(par, r)
         dvdr = dpair_fe_he(par, r)
      end if
   end subroutine pair_terms

   !> Density this pair contributes to each of its two atoms. Fe-Fe feeds
   !! the d-band, Fe-He the s-band, He-He neither.
   pure subroutine pair_densities(par, si, sj, r, rho_d, rho_s)
      type(fehe_params_t), intent(in) :: par
      integer(ip), intent(in) :: si, sj
      real(wp), intent(in) :: r
      real(wp), intent(out) :: rho_d, rho_s

      rho_d = 0.0_wp
      rho_s = 0.0_wp

      if (si == species_fe .and. sj == species_fe) then
         rho_d = density_dband(par, r)
      else if (si /= sj) then
         rho_s = density_sband(par, r)
      end if
   end subroutine pair_densities

   !> Radial derivatives matching `pair_densities`.
   pure subroutine pair_density_derivs(par, si, sj, r, drho_d, drho_s)
      type(fehe_params_t), intent(in) :: par
      integer(ip), intent(in) :: si, sj
      real(wp), intent(in) :: r
      real(wp), intent(out) :: drho_d, drho_s

      drho_d = 0.0_wp
      drho_s = 0.0_wp

      if (si == species_fe .and. sj == species_fe) then
         drho_d = ddensity_dband(par, r)
      else if (si /= sj) then
         drho_s = ddensity_sband(par, r)
      end if
   end subroutine pair_density_derivs

   ! ------------------------------------------------------------------
   ! The three passes.
   ! ------------------------------------------------------------------

   !> Pass one: both densities seen by atom `i`.
   subroutine atom_densities(par, table, species, i, rho_d, rho_s)
      type(fehe_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: species(:)
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: rho_d, rho_s

      integer(ip) :: s, j, ni
      integer(ip), allocatable :: idx(:)
      real(wp), allocatable :: vec(:, :), dist(:)
      real(wp) :: pair_d, pair_s

      rho_d = 0.0_wp
      rho_s = 0.0_wp
      call table%load(i, idx, vec, dist, ni)

      do s = 1_ip, ni
         j = idx(s)
         call pair_densities(par, species(i), species(j), dist(s), &
                             pair_d, pair_s)
         rho_d = rho_d + pair_d
         rho_s = rho_s + pair_s
      end do
   end subroutine atom_densities

   !> Pass three: half of each pair energy atom `i` takes part in, and the
   !! whole force acting on it.
   !!
   !! The force on `i` from neighbour `j` is
   !! `(dV/dr + (dFd_i + dFd_j) drho_d/dr + (dFs_i + dFs_j) drho_s/dr)`
   !! along the unit vector pointing from `i` towards `j`.
   subroutine atom_contribution(par, table, species, dfd, dfs, i, e_i, f_i)
      type(fehe_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: species(:)
      real(wp), intent(in) :: dfd(:), dfs(:)
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: s, j, ni
      integer(ip), allocatable :: idx(:)
      real(wp), allocatable :: vec(:, :), dist(:)
      real(wp) :: r, u_ij(3), v, dvdr, drho_d, drho_s, radial

      e_i = 0.0_wp
      f_i = 0.0_wp
      call table%load(i, idx, vec, dist, ni)

      do s = 1_ip, ni
         j = idx(s)
         r = dist(s)
         u_ij = vec(:, s)/r

         call pair_terms(par, species(i), species(j), r, v, dvdr)
         call pair_density_derivs(par, species(i), species(j), r, drho_d, &
                                  drho_s)

         e_i = e_i + 0.5_wp*v
         radial = dvdr + (dfd(i) + dfd(j))*drho_d + (dfs(i) + dfs(j))*drho_s
         f_i = f_i + radial*u_ij
      end do
   end subroutine atom_contribution

   !> Turn atomic numbers into internal species labels.
   subroutine map_species(atomic_numbers, species, status, errmsg)
      integer(ip), intent(in) :: atomic_numbers(:)
      integer(ip), intent(out) :: species(:)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: i
      character(len=64) :: detail

      status = 0
      errmsg = ""

      do i = 1_ip, int(size(atomic_numbers), ip)
         select case (atomic_numbers(i))
         case (z_fe)
            species(i) = species_fe
         case (z_he)
            species(i) = species_he
         case default
            write (detail, '(i0,a,i0)') atomic_numbers(i), " on atom ", i
            status = 2
            errmsg = "rgpot_fehe: unsupported atomic number "// &
                     trim(detail)//"; this potential covers iron (26) "// &
                     "and helium (2) only"
            return
         end select
      end do
   end subroutine map_species

   !> Energy and forces for `positions` (3 x natoms) of the species named by
   !! `atomic_numbers`, in `cell`.
   subroutine fehe_energy_forces(positions, atomic_numbers, cell, par, table, &
                                 energy, forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      real(wp), intent(in) :: cell(3, 3)
      type(fehe_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, i, k, norder
      integer(ip), allocatable :: order(:)
      integer(ip), allocatable :: species(:)
      real(wp), allocatable :: rho_d(:), rho_s(:), dfd(:), dfs(:)
      real(wp), allocatable :: e_embed(:), e_pair(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp
      status = 0
      errmsg = ""

      if (int(size(atomic_numbers), ip) /= natoms) then
         status = 1
         errmsg = "rgpot_fehe: atomic_numbers and positions disagree on the "// &
                  "atom count"
         return
      end if

      allocate (species(natoms))
      call map_species(atomic_numbers, species, status, errmsg)
      if (status /= 0) return

      call table%build(positions, cell, par%cutoff(), status, errmsg)
      if (status /= 0) return

      allocate (rho_d(natoms), rho_s(natoms), dfd(natoms), dfs(natoms))
      allocate (e_embed(natoms), e_pair(natoms))

      call table%visit_order(order, norder)
      if (norder > 0_ip) then
         do k = 1_ip, norder
            i = order(k)
            call table%note_center(i)
            call atom_densities(par, table, species, i, rho_d(i), rho_s(i))
         end do
      else
         do i = 1_ip, natoms
            call atom_densities(par, table, species, i, rho_d(i), rho_s(i))
         end do
      end if

      do concurrent(i=1:natoms)
         e_embed(i) = embed_dband(par, species(i), rho_d(i)) &
                      + embed_sband(par, rho_s(i))
         dfd(i) = dembed_dband(par, species(i), rho_d(i))
         dfs(i) = dembed_sband(par, rho_s(i))
      end do

      if (norder > 0_ip) then
         do k = 1_ip, norder
            i = order(k)
            call table%note_center(i)
            call atom_contribution(par, table, species, dfd, dfs, i, e_pair(i), &
                                   forces(:, i))
         end do
      else
         do i = 1_ip, natoms
            call atom_contribution(par, table, species, dfd, dfs, i, e_pair(i), &
                                   forces(:, i))
         end do
      end if

      energy = sum(e_pair) + sum(e_embed)
   end subroutine fehe_energy_forces

end module rgpot_fehe
