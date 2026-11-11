! MIT License
! Copyright 2023--present rgpot developers
!
! Environment-dependent interatomic potential (EDIP) for silicon. The
! kernel descends from eOn (https://github.com/TheochemUI/eOn,
! client/potentials/EDIP), BSD-3-Clause, copyright the eOn Development
! Team. The physics is M. Z. Bazant's EDIP routine, 1997 copyright
! M. Z. Bazant and Harvard University, as adapted by S. Goedecker in 2002.

!> EDIP silicon in gather form.
!!
!! EDIP gives atom `i` a pair sum whose strength depends on a coordination
!! number `Z_i`, plus a three-body sum over the neighbour pairs `i`
!! centres:
!!
!!   `E_i = sum_j V2(r_ij, Z_i) + sum_{j<k} V3(r_ij, r_ik, theta_jik, Z_i)`
!!
!! `Z_i` counts neighbours through a cutoff function `f(r)` that falls from
!! one to zero between `c` and `b`, so every term of `E_i` also pushes on
!! the atoms that set `Z_i`. The published form is a scatter with three
!! force channels writing into other atoms' slots, which serialises the
!! atom loop.
!!
!! The gather closes because both environment dependencies collapse to one
!! scalar per atom, `Z_i` and `dE_i/dZ_i`. With those known the whole force
!! on `i` is a sum over `i`'s own neighbourhood:
!!
!!   * two-body, `[dV2/dr(r_ij, Z_i) + dV2/dr(r_ij, Z_j)] * u_ij`;
!!   * coordination, `[dE_i/dZ_i + dE_j/dZ_j] * df/dr(r_ij) * u_ij`;
!!   * three-body, `i` as the centre of `(j, i, k)` and as an end of every
!!     triplet centred on a neighbour `m`, the two roles of `rgpot_sw`.
!!
!! Evaluation is therefore two `do concurrent` passes: one over the
!! per-atom scalars (`Z`, energy, `dE/dZ`), one over forces. Neither writes
!! outside the atom it is indexed by.
!!
!! A triplet whose two ends are the same atom reached through different
!! periodic images is dropped, in the centre role and the end role alike,
!! so the force stays the exact gradient of the energy reported here.
module rgpot_edip
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: edip_params_t, edip_energy_forces

   !> EDIP parameters. Defaults are the Bazant-Kaxiras-Justo silicon set as
   !! hard-coded in the eOn kernel.
   type :: edip_params_t
      real(wp) :: cap_a = 5.6714030_wp   !< Repulsive pair prefactor (eV).
      real(wp) :: cap_b = 2.0002804_wp   !< Pair length scale (Angstrom).
      real(wp) :: rh = 1.2085196_wp      !< Pair power-law exponent.
      real(wp) :: a = 3.1213820_wp       !< Pair cutoff (Angstrom).
      real(wp) :: sig = 0.5774108_wp     !< Pair cutoff decay (Angstrom).
      real(wp) :: lam = 1.4533108_wp     !< Three-body strength (eV).
      real(wp) :: gam = 1.1247945_wp     !< Three-body radial decay (Angstrom).
      real(wp) :: b = 3.1213820_wp       !< Coordination cutoff, outer (Angstrom).
      real(wp) :: c = 2.5609104_wp       !< Coordination cutoff, inner (Angstrom).
      real(wp) :: delta = 78.7590539_wp  !< Angular anharmonicity scale.
      real(wp) :: mu = 0.6966326_wp      !< Angular width vs coordination.
      real(wp) :: qo = 312.1341346_wp    !< Angular width at zero coordination.
      real(wp) :: palp = 1.4074424_wp    !< Bond-order prefactor.
      real(wp) :: bet = 0.0070975_wp     !< Bond-order decay in `Z**2`.
      real(wp) :: alp = 3.1083847_wp     !< Coordination cutoff sharpness.
      real(wp) :: u1 = -0.165799_wp      !< Angular minimum, constant term.
      real(wp) :: u2 = 32.557_wp         !< Angular minimum, amplitude.
      real(wp) :: u3 = 0.286198_wp       !< Angular minimum, shape.
      real(wp) :: u4 = 0.66_wp           !< Angular minimum, decay in `Z`.
   contains
      procedure :: cutoff => edip_cutoff
   end type edip_params_t

   !> Combinations of `edip_params_t` the kernels reuse, formed once per
   !! evaluation so the inner loops carry no `sqrt` or division of constants.
   type :: edip_const_t
      real(wp) :: bg        !< Three-body radial cutoff, equal to `a`.
      real(wp) :: eta       !< `delta / qo`.
      real(wp) :: qort      !< `sqrt(qo)`.
      real(wp) :: muhalf    !< `mu / 2`.
      real(wp) :: u5        !< `u2 * u4`.
      real(wp) :: bmc       !< `b - c`.
      real(wp) :: cmbinv    !< `1 / (c - b)`.
   end type edip_const_t

   !> Exponential arguments below this underflow to zero; guarding keeps
   !! `exp` out of its denormal range for pairs sitting at a cutoff.
   real(wp), parameter :: exp_floor = -300.0_wp

contains

   !> Interaction cutoff: the widest of the pair and coordination ranges.
   pure function edip_cutoff(self) result(rcut)
      class(edip_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = max(self%a, self%b)
   end function edip_cutoff

   !> Derived constants for one parameter set.
   pure function edip_constants(par) result(cst)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t) :: cst

      cst%bg = par%a
      cst%eta = par%delta/par%qo
      cst%qort = sqrt(par%qo)
      cst%muhalf = par%mu*0.5_wp
      cst%u5 = par%u2*par%u4
      cst%bmc = par%b - par%c
      cst%cmbinv = 1.0_wp/(par%c - par%b)
   end function edip_constants

   !> Coordination cutoff `f(r)` and `df/dr`: one inside `c`, zero outside
   !! `b`, and a smooth exponential switch between them.
   pure subroutine coord_term(par, cst, r, fz, dfz)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      real(wp), intent(in) :: r
      real(wp), intent(out) :: fz, dfz

      real(wp) :: xinv, xinv3, den, temp1

      fz = 0.0_wp
      dfz = 0.0_wp
      if (r >= par%b .or. r >= cst%bg) return

      if (r < par%c) then
         fz = 1.0_wp
         return
      end if

      xinv = cst%bmc/(r - par%c)
      xinv3 = xinv*xinv*xinv
      den = 1.0_wp/(1.0_wp - xinv3)
      temp1 = par%alp*den
      if (temp1 > exp_floor) then
         fz = exp(temp1)
         dfz = fz*temp1*den*3.0_wp*xinv3*xinv*cst%cmbinv
      end if
   end subroutine coord_term

   !> Radial pieces of the pair term at `r`: the cutoff envelope `t0`, the
   !! power law `t1`, and the two derivative factors `t2`, `t3`.
   pure subroutine pair_radial(par, r, t0, t1, t2, t3)
      type(edip_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp), intent(out) :: t0, t1, t2, t3

      real(wp) :: rmainv, arg

      rmainv = 1.0_wp/(r - par%a)
      arg = par%sig*rmainv

      t0 = 0.0_wp
      if (arg > exp_floor) t0 = par%cap_a*exp(arg)
      t1 = (par%cap_b/r)**par%rh
      t2 = par%rh/r
      t3 = par%sig*rmainv*rmainv
   end subroutine pair_radial

   !> Bond order `p(Z)` and `dp/dZ`.
   pure subroutine bond_order(par, z, pz, dp)
      type(edip_params_t), intent(in) :: par
      real(wp), intent(in) :: z
      real(wp), intent(out) :: pz, dp

      real(wp) :: temp0

      temp0 = par%bet*z
      pz = par%palp*exp(-temp0*z)
      dp = -2.0_wp*temp0*pz
   end subroutine bond_order

   !> Inverse angular width and angular minimum at coordination `z`, with
   !! the derivative of each in `z`.
   pure subroutine angular_shape(par, cst, z, winv, dwinv, tau, dtau)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      real(wp), intent(in) :: z
      real(wp), intent(out) :: winv, dwinv, tau, dtau

      real(wp) :: temp0

      winv = cst%qort*exp(-cst%muhalf*z)
      dwinv = -cst%muhalf*winv
      temp0 = exp(-par%u4*z)
      tau = par%u1 + par%u2*temp0*(par%u3 - temp0)
      dtau = cst%u5*temp0*(2.0_wp*temp0 - par%u3)
   end subroutine angular_shape

   !> Angular function `H(l, Z)` and `dH/dx`, where `x = (l + tau) * winv`.
   pure subroutine angular_term(par, cst, winv, tau, lcos, h, dhdx)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      real(wp), intent(in) :: winv, tau, lcos
      real(wp), intent(out) :: h, dhdx

      real(wp) :: temp0, x

      x = (lcos + tau)*winv
      temp0 = exp(-x*x)
      h = par%lam*(1.0_wp - temp0 + cst%eta*x*x)
      dhdx = 2.0_wp*par%lam*x*(temp0 + cst%eta)
   end subroutine angular_term

   !> Three-body radial factor `g(r)` and `dg/dr` for one leg.
   pure subroutine leg_radial(par, cst, r, g, dg)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      real(wp), intent(in) :: r
      real(wp), intent(out) :: g, dg

      real(wp) :: rmbinv, temp1

      g = 0.0_wp
      dg = 0.0_wp
      if (r >= cst%bg) return

      rmbinv = 1.0_wp/(r - cst%bg)
      temp1 = par%gam*rmbinv
      if (temp1 > exp_floor) then
         g = exp(temp1)
         dg = -rmbinv*temp1*g
      end if
   end subroutine leg_radial

   !> Gradient of one triplet's energy with respect to the position of the
   !! leg atom `self`, the triplet being centred on a shared atom.
   !!
   !! `u_self` and `u_other` point from the centre towards each end, and
   !! `dhdl` is `dH/dx * winv`.
   pure function leg_gradient(g_self, dg_self, u_self, rinv_self, &
                              g_other, u_other, lcos, h, dhdl) result(grad)
      real(wp), intent(in) :: g_self, dg_self, u_self(3), rinv_self
      real(wp), intent(in) :: g_other, u_other(3), lcos, h, dhdl
      real(wp) :: grad(3)

      grad = dg_self*g_other*h*u_self &
             + g_self*g_other*dhdl*(u_other - lcos*u_self)*rinv_self
   end function leg_gradient

   !> Three-body legs of atom `i`: every neighbour inside `bg`, with its
   !! radial factor, unit vector, inverse distance, and atom index.
   subroutine collect_legs(par, cst, table, i, maxnbr, nleg, &
                           g, dg, u, rinv, who)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: i, maxnbr
      integer(ip), intent(out) :: nleg
      real(wp), intent(out) :: g(maxnbr), dg(maxnbr), u(3, maxnbr), rinv(maxnbr)
      integer(ip), intent(out) :: who(maxnbr)

      integer(ip) :: s, ni
      integer(ip), allocatable :: idx(:)
      real(wp), allocatable :: vec(:, :), dist(:)
      real(wp) :: r

      nleg = 0_ip
      call table%load(i, idx, vec, dist, ni)
      do s = 1_ip, ni
         r = dist(s)
         if (r >= cst%bg) cycle
         if (nleg >= maxnbr) exit
         nleg = nleg + 1_ip
         call leg_radial(par, cst, r, g(nleg), dg(nleg))
         u(:, nleg) = vec(:, s)/r
         rinv(nleg) = 1.0_wp/r
         who(nleg) = idx(s)
      end do
   end subroutine collect_legs

   !> Coordination, energy, and `dE/dZ` of atom `i`.
   !!
   !! Every quantity here depends on atom `i`'s own neighbourhood only, so
   !! the iterations of the calling loop are independent.
   subroutine atom_terms(par, cst, table, i, maxnbr, z, e_i, dedz)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: i, maxnbr
      real(wp), intent(out) :: z, e_i, dedz

      real(wp) :: g(maxnbr), dg(maxnbr), u(3, maxnbr), rinv(maxnbr)
      integer(ip) :: who(maxnbr)
      integer(ip) :: s, nleg, nj, nk, ni
      integer(ip), allocatable :: idx(:)
      real(wp), allocatable :: vec(:, :), dist(:)
      real(wp) :: r, t0, t1, t2, t3, fz, dfz, pz, dp
      real(wp) :: winv, dwinv, tau, dtau, lcos, h, dhdx, dxdz, gg

      e_i = 0.0_wp
      dedz = 0.0_wp

      z = 0.0_wp
      call table%load(i, idx, vec, dist, ni)
      do s = 1_ip, ni
         call coord_term(par, cst, dist(s), fz, dfz)
         z = z + fz
      end do

      call bond_order(par, z, pz, dp)

      ! Pair term V2(r, Z) and the share of dE/dZ it carries.
      do s = 1_ip, ni
         r = dist(s)
         if (r >= par%a) cycle
         call pair_radial(par, r, t0, t1, t2, t3)
         e_i = e_i + (t1 - pz)*t0
         dedz = dedz - dp*t0
      end do

      call collect_legs(par, cst, table, i, maxnbr, nleg, g, dg, u, rinv, who)
      call angular_shape(par, cst, z, winv, dwinv, tau, dtau)

      ! Three-body term over the unordered pairs of legs i centres.
      do nj = 1_ip, nleg - 1_ip
         do nk = nj + 1_ip, nleg
            if (who(nj) == who(nk)) cycle
            lcos = dot_product(u(:, nj), u(:, nk))
            call angular_term(par, cst, winv, tau, lcos, h, dhdx)
            gg = g(nj)*g(nk)
            e_i = e_i + gg*h
            dxdz = dwinv*(lcos + tau) + winv*dtau
            dedz = dedz + gg*dhdx*dxdz
         end do
      end do
   end subroutine atom_terms

   !> Total force on atom `i`, given `Z` and `dE/dZ` of every atom.
   subroutine atom_force(par, cst, table, i, maxnbr, z, dedz, f_i)
      type(edip_params_t), intent(in) :: par
      type(edip_const_t), intent(in) :: cst
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: i, maxnbr
      real(wp), intent(in) :: z(:), dedz(:)
      real(wp), intent(out) :: f_i(3)

      real(wp) :: g(maxnbr), dg(maxnbr), u(3, maxnbr), rinv(maxnbr)
      integer(ip) :: who(maxnbr)
      integer(ip) :: s, sn, j, m, n, nleg, nj, nk, ni, nm
      integer(ip), allocatable :: idx(:), midx(:)
      real(wp), allocatable :: vec(:, :), mvec(:, :), dist(:), mdist(:)
      real(wp) :: r, rim, rmn, uij(3), u_mi(3), u_mn(3)
      real(wp) :: t0, t1, t2, t3, fz, dfz, pz_i, pz_j, dp_i, dp_j
      real(wp) :: winv, dwinv, tau, dtau, lcos, h, dhdx
      real(wp) :: g_self, dg_self, g_other, dg_other

      f_i = 0.0_wp
      call bond_order(par, z(i), pz_i, dp_i)
      call table%load(i, idx, vec, dist, ni)

      ! Pair and coordination channels. Both close because the pair term
      ! needs the bond order of each end and the coordination term needs
      ! dE/dZ of each end, and both are already known per atom.
      do s = 1_ip, ni
         j = idx(s)
         r = dist(s)
         uij = vec(:, s)/r

         if (r < par%a) then
            call pair_radial(par, r, t0, t1, t2, t3)
            call bond_order(par, z(j), pz_j, dp_j)
            f_i = f_i - t0*(t1*t2 + (t1 - pz_i)*t3)*uij &
                  - t0*(t1*t2 + (t1 - pz_j)*t3)*uij
         end if

         call coord_term(par, cst, r, fz, dfz)
         f_i = f_i + (dedz(i) + dedz(j))*dfz*uij
      end do

      ! Three-body, i at the centre of every unordered pair of its legs.
      call collect_legs(par, cst, table, i, maxnbr, nleg, g, dg, u, rinv, who)
      call angular_shape(par, cst, z(i), winv, dwinv, tau, dtau)

      do nj = 1_ip, nleg - 1_ip
         do nk = nj + 1_ip, nleg
            if (who(nj) == who(nk)) cycle
            lcos = dot_product(u(:, nj), u(:, nk))
            call angular_term(par, cst, winv, tau, lcos, h, dhdx)
            f_i = f_i &
                  + leg_gradient(g(nj), dg(nj), u(:, nj), rinv(nj), &
                                 g(nk), u(:, nk), lcos, h, dhdx*winv) &
                  + leg_gradient(g(nk), dg(nk), u(:, nk), rinv(nk), &
                                 g(nj), u(:, nj), lcos, h, dhdx*winv)
         end do
      end do

      ! Three-body, i at an end of every triplet centred on a neighbour m.
      do s = 1_ip, ni
         m = idx(s)
         rim = dist(s)
         if (rim >= cst%bg) cycle

         ! Vector m -> i is the reverse of the stored i -> m vector.
         u_mi = -vec(:, s)/rim
         call leg_radial(par, cst, rim, g_self, dg_self)
         call angular_shape(par, cst, z(m), winv, dwinv, tau, dtau)

         call table%load(m, midx, mvec, mdist, nm)
         do sn = 1_ip, nm
            n = midx(sn)
            if (n == i) cycle
            rmn = mdist(sn)
            if (rmn >= cst%bg) cycle
            u_mn = mvec(:, sn)/rmn
            call leg_radial(par, cst, rmn, g_other, dg_other)
            lcos = dot_product(u_mi, u_mn)
            call angular_term(par, cst, winv, tau, lcos, h, dhdx)
            f_i = f_i - leg_gradient(g_self, dg_self, u_mi, 1.0_wp/rim, &
                                     g_other, u_mn, lcos, h, dhdx*winv)
         end do
      end do
   end subroutine atom_force

   !> Perpendicular width of the cell along each lattice vector.
   pure function cell_widths(cell) result(w)
      real(wp), intent(in) :: cell(3, 3)
      real(wp) :: w(3)

      real(wp) :: face(3, 3), vol, area
      integer :: k

      face(:, 1) = cross(cell(:, 2), cell(:, 3))
      face(:, 2) = cross(cell(:, 3), cell(:, 1))
      face(:, 3) = cross(cell(:, 1), cell(:, 2))
      vol = abs(dot_product(cell(:, 1), face(:, 1)))

      w = 0.0_wp
      do k = 1, 3
         area = norm2(face(:, k))
         if (area > 0.0_wp) w(k) = vol/area
      end do
   end function cell_widths

   !> Vector cross product.
   pure function cross(p, q) result(r)
      real(wp), intent(in) :: p(3), q(3)
      real(wp) :: r(3)

      r(1) = p(2)*q(3) - p(3)*q(2)
      r(2) = p(3)*q(1) - p(1)*q(3)
      r(3) = p(1)*q(2) - p(2)*q(1)
   end function cross

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   subroutine edip_energy_forces(positions, cell, par, table, energy, forces, &
                                 status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(edip_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      type(edip_const_t) :: cst
      integer(ip) :: natoms, i, maxnbr, k, norder
      integer(ip), allocatable :: order(:)
      real(wp) :: rcut, widths(3)
      real(wp), allocatable :: e_atom(:), z(:), dedz(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp
      status = 0
      errmsg = ""
      rcut = par%cutoff()

      ! One cell layer has to hold the cutoff, the precondition the legacy
      ! kernel enforced by binning into cells of at least the cutoff width.
      widths = cell_widths(cell)
      if (minval(widths) < rcut) then
         status = 1
         errmsg = "rgpot_edip: cell is thinner than the interaction cutoff"
         return
      end if

      call table%build(positions, cell, rcut, status, errmsg)
      if (status /= 0) return

      cst = edip_constants(par)

      maxnbr = 1_ip
      call table%visit_order(order, norder)
      if (norder > 0_ip) then
         block
            integer(ip) :: ni
            integer(ip), allocatable :: idx(:)
            real(wp), allocatable :: vec(:, :), dist(:)
            do k = 1_ip, norder
               call table%note_center(order(k))
               call table%load(order(k), idx, vec, dist, ni)
               maxnbr = max(maxnbr, ni)
            end do
         end block
      else
         do i = 1_ip, natoms
            maxnbr = max(maxnbr, table%row(i + 1_ip) - table%row(i))
         end do
      end if

      allocate (e_atom(natoms), source=0.0_wp)
      allocate (z(natoms), source=0.0_wp)
      allocate (dedz(natoms), source=0.0_wp)

      ! Pass one: per-atom scalars. Each iteration reads its own
      ! neighbourhood and writes its own slots.
      if (norder > 0_ip) then
         do k = 1_ip, norder
            i = order(k)
            call table%note_center(i)
            call atom_terms(par, cst, table, i, maxnbr, z(i), e_atom(i), dedz(i))
         end do
      else
         do i = 1_ip, natoms
            call atom_terms(par, cst, table, i, maxnbr, z(i), e_atom(i), dedz(i))
         end do
      end if

      ! Pass two: forces. Each iteration reads the pass-one arrays and
      ! writes only its own force column.
      if (norder > 0_ip) then
         do k = 1_ip, norder
            i = order(k)
            call table%note_center(i)
            call atom_force(par, cst, table, i, maxnbr, z, dedz, forces(:, i))
         end do
      else
         do i = 1_ip, natoms
            call atom_force(par, cst, table, i, maxnbr, z, dedz, forces(:, i))
         end do
      end if

      energy = sum(e_atom)
   end subroutine edip_energy_forces

end module rgpot_edip
