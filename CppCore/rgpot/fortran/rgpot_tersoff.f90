! MIT License
! Copyright 2023--present rgpot developers
!
! Tersoff silicon. The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/Tersoff), BSD-3-Clause,
! copyright the eOn Development Team.

!> Tersoff bond-order potential in gather form.
!!
!! Every directed pair `(i, j)` carries a bond order `b_ij` set by a
!! coordination sum `zeta_ij` over the neighbours `k` of `i`. The pair
!! therefore couples to a whole shell of third atoms, and its gradient
!! splits into terms landing on `i`, on `j`, and on each `k`. That is why
!! a bond-order kernel resists the one-pass gather a pair potential
!! allows: the `k` terms need `dE_ij/dzeta_ij`, which is not known until
!! the whole shell has been summed.
!!
!! Splitting the evaluation into three passes over the full neighbour
!! table removes the obstacle:
!!
!!   1. `zeta` for every directed pair, from the neighbours of the pair's
!!      first atom;
!!   2. the per-pair scalars the gradient needs, `dE_ij/dr_ij` at fixed
!!      bond order and `dE_ij/dzeta_ij`;
!!   3. the gather, in which atom `a` collects every term that lands on
!!      it and writes only `forces(:, a)`.
!!
!! Pass 3 closes because pass 2 leaves `dE_ij/dzeta_ij` stored per pair.
!! Atom `a` then plays three roles, all reachable from its own row of the
!! table:
!!
!!   * owner of the pairs `(a, j)`, taking their radial term and
!!     `dzeta_aj/dr_a`;
!!   * far end of the pairs `(m, a)` for `m` in `a`'s row, reached through
!!     the reverse entry, taking their radial term and `dzeta_ma/dr_a`;
!!   * third atom of the pairs `(m, j)` for `m` in `a`'s row and `j` in
!!     `m`'s, taking `dzeta_mj/dr_a`.
!!
!! The last two roles share one walk over `m`'s row and one cosine per
!! step, so the gather costs `O(natoms * nn^2)` like the two passes before
!! it. All three passes run under `do concurrent`; none of them scatters.
module rgpot_tersoff
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: tersoff_params_t, tersoff_energy_forces

   !> Tersoff parameters. Defaults are eOn's silicon set, Tersoff's Si(C).
   type :: tersoff_params_t
      real(wp) :: a = 1.8308e+3_wp      !< Repulsive prefactor (eV).
      real(wp) :: b = 4.7118e+2_wp      !< Attractive prefactor (eV).
      real(wp) :: lambda1 = 2.4799_wp   !< Repulsive decay (1/Angstrom).
      real(wp) :: lambda2 = 1.7322_wp   !< Attractive decay (1/Angstrom).
      real(wp) :: beta = 1.1e-6_wp      !< Bond-order coordination weight.
      real(wp) :: n = 7.8734e-1_wp      !< Bond-order exponent.
      real(wp) :: c = 1.0039e+5_wp      !< Angular numerator.
      real(wp) :: d = 1.6217e+1_wp      !< Angular width.
      real(wp) :: h = -5.9825e-1_wp     !< Preferred cosine, negated.
      real(wp) :: r = 2.7_wp            !< Taper start (Angstrom).
      real(wp) :: s = 3.0_wp            !< Taper end and cutoff (Angstrom).
      real(wp) :: chi = 1.0_wp          !< Bond-order scale, unity for one species.
   contains
      procedure :: cutoff => tersoff_cutoff
   end type tersoff_params_t

   !> Below this coordination the bond order is its `zeta -> 0` limit
   !! `chi`, and its derivative is dropped: `zeta**(n - 1)` has a negative
   !! exponent and diverges at the origin while the energy stays finite.
   real(wp), parameter :: zeta_floor = 1.0e-15_wp

   !> Two entries name the same bond when they name the same atoms and
   !! their vectors cancel. Distinct periodic images of one atom sit at
   !! least a cell edge apart, so the test needs only to survive rounding.
   real(wp), parameter :: bond_match_tol = 1.0e-8_wp

   real(wp), parameter :: pi = acos(-1.0_wp)

contains

   !> Interaction cutoff, the outer taper radius `S`.
   pure function tersoff_cutoff(self) result(rcut)
      class(tersoff_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = self%s
   end function tersoff_cutoff

   !> Cutoff taper and its radial derivative: unity inside `R`, a cosine
   !! shoulder to zero at `S`, zero beyond.
   pure subroutine cutoff_taper(par, rr, fc, dfc)
      type(tersoff_params_t), intent(in) :: par
      real(wp), intent(in) :: rr
      real(wp), intent(out) :: fc, dfc

      real(wp) :: width, phase

      if (rr <= par%r) then
         fc = 1.0_wp
         dfc = 0.0_wp
      else if (rr >= par%s) then
         fc = 0.0_wp
         dfc = 0.0_wp
      else
         width = par%s - par%r
         phase = pi*(rr - par%r)/width
         fc = 0.5_wp + 0.5_wp*cos(phase)
         dfc = -pi/(2.0_wp*width)*sin(phase)
      end if
   end subroutine cutoff_taper

   !> Repulsive and attractive pair functions with their radial
   !! derivatives. The attractive branch carries its own sign.
   pure subroutine pair_functions(par, rr, frep, dfrep, fatt, dfatt)
      type(tersoff_params_t), intent(in) :: par
      real(wp), intent(in) :: rr
      real(wp), intent(out) :: frep, dfrep, fatt, dfatt

      frep = par%a*exp(-par%lambda1*rr)
      dfrep = -par%lambda1*frep
      fatt = -par%b*exp(-par%lambda2*rr)
      dfatt = -par%lambda2*fatt
   end subroutine pair_functions

   !> Angular function `g(cos theta)` and its derivative in the cosine.
   pure subroutine angular(par, cosine, gval, dgval)
      type(tersoff_params_t), intent(in) :: par
      real(wp), intent(in) :: cosine
      real(wp), intent(out) :: gval, dgval

      real(wp) :: c2, d2, gap, denom

      c2 = par%c*par%c
      d2 = par%d*par%d
      gap = par%h - cosine
      denom = d2 + gap*gap

      gval = 1.0_wp + c2/d2 - c2/denom
      dgval = -2.0_wp*c2*gap/(denom*denom)
   end subroutine angular

   !> Bond order `b(zeta)` and `db/dzeta`.
   pure subroutine bond_order(par, zeta, bij, dbij)
      type(tersoff_params_t), intent(in) :: par
      real(wp), intent(in) :: zeta
      real(wp), intent(out) :: bij, dbij

      real(wp) :: bz_n, base

      if (zeta < zeta_floor) then
         bij = par%chi
         dbij = 0.0_wp
      else
         bz_n = (par%beta*zeta)**par%n
         base = 1.0_wp + bz_n
         bij = par%chi/base**(0.5_wp/par%n)
         dbij = -par%chi*0.5_wp*par%beta**par%n*zeta**(par%n - 1.0_wp) &
                /base**((0.5_wp + par%n)/par%n)
      end if
   end subroutine bond_order

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   subroutine tersoff_energy_forces(positions, cell, par, table, energy, &
                                    forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(tersoff_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, nentries, i, s
      real(wp) :: rcut
      real(wp), allocatable :: e_atom(:), zeta(:), e_pair(:), f_rad(:)
      real(wp), allocatable :: c_zeta(:)
      integer(ip), allocatable :: rev(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp
      rcut = par%cutoff()

      call table%build(positions, cell, rcut, status, errmsg)
      if (status /= 0) return

      if (table%streaming) then
         call tersoff_stream(par, table, natoms, energy, forces, status, errmsg)
         return
      end if

      nentries = table%row(natoms + 1_ip) - 1_ip
      allocate (e_atom(natoms), source=0.0_wp)
      allocate (zeta(nentries), e_pair(nentries), f_rad(nentries))
      allocate (c_zeta(nentries), rev(nentries))

      ! Each atom resolves the reverse of its own bonds, so the map is
      ! written in disjoint slices.
      do concurrent(i=1:natoms)
         call atom_reverse(table, i, &
                           rev(table%row(i):table%row(i + 1_ip) - 1_ip))
      end do

      if (any(rev == 0_ip)) then
         status = 2
         errmsg = "rgpot_tersoff: a pair in the neighbour table has no reverse"
         return
      end if

      ! Pass 1: coordination sum of every pair the atom owns.
      do concurrent(i=1:natoms)
         call atom_zeta(par, table, i, &
                        zeta(table%row(i):table%row(i + 1_ip) - 1_ip))
      end do

      ! Pass 2: per-pair scalars, each entry depending only on its own
      ! length and coordination.
      do concurrent(s=1:nentries)
         call pair_scalars(par, table%dist(s), zeta(s), e_pair(s), f_rad(s), &
                           c_zeta(s))
      end do

      ! Pass 3: the gather. Each atom writes one force column and one
      ! energy slot.
      do concurrent(i=1:natoms)
         call atom_contribution(par, table, e_pair, f_rad, c_zeta, rev, i, &
                                e_atom(i), forces(:, i))
      end do

      energy = sum(e_atom)
   end subroutine tersoff_energy_forces

   !> For each bond leaving `i`, the entry of the neighbour's row that
   !! carries the same bond in the opposite direction, or zero if the
   !! table holds none.
   !!
   !! A periodic cell can list one atom several times in a single row,
   !! once per image, and those entries are distinct bonds. Matching the
   !! atom index alone would confuse them, so the vectors must cancel too.
   pure subroutine atom_reverse(table, i, rev_i)
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      integer(ip), intent(out) :: rev_i(:)

      integer(ip) :: sj, st, j, base

      base = table%row(i) - 1_ip

      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         j = table%idx(sj)
         rev_i(sj - base) = 0_ip
         do st = table%row(j), table%row(j + 1_ip) - 1_ip
            if (table%idx(st) /= i) cycle
            if (maxval(abs(table%vec(:, st) + table%vec(:, sj))) &
                > bond_match_tol) cycle
            rev_i(sj - base) = st
            exit
         end do
      end do
   end subroutine atom_reverse

   !> Coordination sum `zeta_ij` of every pair `i` owns, over the other
   !! neighbours of `i`.
   pure subroutine atom_zeta(par, table, i, zeta_i)
      type(tersoff_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: zeta_i(:)

      integer(ip) :: sj, sk, base
      real(wp) :: u_ij(3), u_ik(3), fc_k, dfc_k, gval, dgval, acc

      base = table%row(i) - 1_ip

      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         u_ij = table%vec(:, sj)/table%dist(sj)
         acc = 0.0_wp
         do sk = table%row(i), table%row(i + 1_ip) - 1_ip
            ! The bond itself is excluded, but a second image of the same
            ! atom is a legitimate third leg.
            if (sk == sj) cycle
            u_ik = table%vec(:, sk)/table%dist(sk)
            call cutoff_taper(par, table%dist(sk), fc_k, dfc_k)
            call angular(par, dot_product(u_ij, u_ik), gval, dgval)
            acc = acc + fc_k*gval
         end do
         zeta_i(sj - base) = acc
      end do
   end subroutine atom_zeta

   !> Pair energy, its radial derivative at fixed bond order, and its
   !! derivative in the coordination sum.
   pure subroutine pair_scalars(par, rij, zeta, e_ij, f_ij, c_ij)
      type(tersoff_params_t), intent(in) :: par
      real(wp), intent(in) :: rij, zeta
      real(wp), intent(out) :: e_ij, f_ij, c_ij

      real(wp) :: fc, dfc, frep, dfrep, fatt, dfatt, bij, dbij

      call cutoff_taper(par, rij, fc, dfc)
      call pair_functions(par, rij, frep, dfrep, fatt, dfatt)
      call bond_order(par, zeta, bij, dbij)

      e_ij = fc*(frep + bij*fatt)
      f_ij = dfc*(frep + bij*fatt) + fc*(dfrep + bij*dfatt)
      c_ij = fc*fatt*dbij
   end subroutine pair_scalars

   !> Energy owned by atom `i` and the total force acting on it.
   !!
   !! The energy of a directed pair is halved because the table lists both
   !! directions. The gradient is halved for the same reason and negated
   !! once at the end.
   pure subroutine atom_contribution(par, table, e_pair, f_rad, c_zeta, rev, &
                                     i, e_i, f_i)
      type(tersoff_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      real(wp), intent(in) :: e_pair(:), f_rad(:), c_zeta(:)
      integer(ip), intent(in) :: rev(:)
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: sj, sk, sm, st, m, back
      real(wp) :: rij, rik, rim, rmt
      real(wp) :: u_ij(3), u_ik(3), u_mi(3), u_mt(3)
      real(wp) :: grad(3), dcos_j(3), dcos_k(3), dcos_i(3)
      real(wp) :: cosine, gval, dgval, fc_k, dfc_k, fc_i, dfc_i, c_back

      e_i = 0.0_wp
      grad = 0.0_wp

      ! Role one: the pairs `i` owns. Their radial term acts on `i`, and
      ! so does the coordination sum, through every third leg.
      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         rij = table%dist(sj)
         u_ij = table%vec(:, sj)/rij
         e_i = e_i + 0.5_wp*e_pair(sj)

         ! Radial terms of both directed pairs on this bond. The bond
         ! orders differ, so the reverse entry carries its own scalar.
         grad = grad - (f_rad(sj) + f_rad(rev(sj)))*u_ij

         do sk = table%row(i), table%row(i + 1_ip) - 1_ip
            if (sk == sj) cycle
            rik = table%dist(sk)
            u_ik = table%vec(:, sk)/rik
            cosine = dot_product(u_ij, u_ik)
            call cutoff_taper(par, rik, fc_k, dfc_k)
            call angular(par, cosine, gval, dgval)

            dcos_j = (u_ik - cosine*u_ij)/rij
            dcos_k = (u_ij - cosine*u_ik)/rik

            ! Each leg is a function of differences alone, so what it
            ! gives the centre is what it takes from the two ends.
            grad = grad - c_zeta(sj)*(fc_k*dgval*dcos_j &
                                      + dfc_k*gval*u_ik + fc_k*dgval*dcos_k)
         end do
      end do

      ! Roles two and three: the pairs owned by a neighbour `m`. Walking
      ! `m`'s row once serves both, since the cosine at `m` between the
      ! bond to `i` and the bond of the entry is shared.
      do sm = table%row(i), table%row(i + 1_ip) - 1_ip
         m = table%idx(sm)
         rim = table%dist(sm)
         u_mi = -table%vec(:, sm)/rim
         back = rev(sm)
         c_back = c_zeta(back)
         call cutoff_taper(par, rim, fc_i, dfc_i)

         do st = table%row(m), table%row(m + 1_ip) - 1_ip
            if (st == back) cycle
            rmt = table%dist(st)
            u_mt = table%vec(:, st)/rmt
            cosine = dot_product(u_mi, u_mt)
            call angular(par, cosine, gval, dgval)
            call cutoff_taper(par, rmt, fc_k, dfc_k)

            dcos_i = (u_mt - cosine*u_mi)/rim

            ! Role two: `i` is the far end of the pair `(m, i)`, and this
            ! entry is one of the third legs of its coordination sum.
            grad = grad + c_back*fc_k*dgval*dcos_i

            ! Role three: `i` is a third leg of the pair `(m, idx(st))`.
            grad = grad + c_zeta(st)*(dfc_i*gval*u_mi + fc_i*dgval*dcos_i)
         end do
      end do

      f_i = -0.5_wp*grad
   end subroutine atom_contribution

   !> Same gather as the stored table, one atom of bonds at a time.
   subroutine tersoff_stream(par, table, natoms, energy, forces, status, errmsg)
      type(tersoff_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      integer(ip), intent(in) :: natoms
      real(wp), intent(inout) :: energy
      real(wp), intent(inout) :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: i, a, b, m, ni, nj, back, st, k, norder
      integer(ip), allocatable :: order(:)
      integer(ip), allocatable :: idx(:), jdx(:)
      real(wp), allocatable :: vec(:, :), jvec(:, :), dist(:), jdist(:)
      real(wp), allocatable :: zeta_i(:), e_pair(:), f_rad(:), c_zeta(:)
      real(wp), allocatable :: zeta_m(:), f_rad_m(:), c_zeta_m(:)
      real(wp) :: e_i, grad(3), u_ij(3), u_ik(3), u_mi(3), u_mt(3)
      real(wp) :: rij, rik, rim, rmt, cosine
      real(wp) :: gval, dgval, fc_k, dfc_k, fc_i, dfc_i
      real(wp) :: dcos_j(3), dcos_k(3), dcos_i(3), unused

      status = 0
      errmsg = ""
      energy = 0.0_wp
      forces = 0.0_wp

      call table%visit_order(order, norder)
      do k = 1_ip, norder
         i = order(k)
         call table%note_center(i)
         call table%load(i, idx, vec, dist, ni)
         if (allocated(zeta_i)) then
            if (size(zeta_i) < ni) deallocate (zeta_i, e_pair, f_rad, c_zeta)
         end if
         if (.not. allocated(zeta_i)) then
            allocate (zeta_i(max(ni, 1_ip)), e_pair(max(ni, 1_ip)))
            allocate (f_rad(max(ni, 1_ip)), c_zeta(max(ni, 1_ip)))
         end if

         do a = 1_ip, ni
            u_ij = vec(:, a)/dist(a)
            zeta_i(a) = 0.0_wp
            do b = 1_ip, ni
               if (b == a) cycle
               u_ik = vec(:, b)/dist(b)
               call cutoff_taper(par, dist(b), fc_k, dfc_k)
               call angular(par, dot_product(u_ij, u_ik), gval, dgval)
               zeta_i(a) = zeta_i(a) + fc_k*gval
            end do
            call pair_scalars(par, dist(a), zeta_i(a), e_pair(a), f_rad(a), &
                              c_zeta(a))
         end do

         e_i = 0.0_wp
         grad = 0.0_wp
         do a = 1_ip, ni
            e_i = e_i + 0.5_wp*e_pair(a)
            rij = dist(a)
            u_ij = vec(:, a)/rij
            do b = 1_ip, ni
               if (b == a) cycle
               rik = dist(b)
               u_ik = vec(:, b)/rik
               cosine = dot_product(u_ij, u_ik)
               call cutoff_taper(par, rik, fc_k, dfc_k)
               call angular(par, cosine, gval, dgval)
               dcos_j = (u_ik - cosine*u_ij)/rij
               dcos_k = (u_ij - cosine*u_ik)/rik
               grad = grad - c_zeta(a)*(fc_k*dgval*dcos_j &
                                        + dfc_k*gval*u_ik + fc_k*dgval*dcos_k)
            end do
         end do

         do a = 1_ip, ni
            m = idx(a)
            rij = dist(a)
            u_ij = vec(:, a)/rij
            call table%load(m, jdx, jvec, jdist, nj)
            if (allocated(zeta_m)) then
               if (size(zeta_m) < nj) deallocate (zeta_m, f_rad_m, c_zeta_m)
            end if
            if (.not. allocated(zeta_m)) then
               allocate (zeta_m(max(nj, 1_ip)), f_rad_m(max(nj, 1_ip)))
               allocate (c_zeta_m(max(nj, 1_ip)))
            end if
            do st = 1_ip, nj
               u_mi = jvec(:, st)/jdist(st)
               zeta_m(st) = 0.0_wp
               do b = 1_ip, nj
                  if (b == st) cycle
                  u_mt = jvec(:, b)/jdist(b)
                  call cutoff_taper(par, jdist(b), fc_k, dfc_k)
                  call angular(par, dot_product(u_mi, u_mt), gval, dgval)
                  zeta_m(st) = zeta_m(st) + fc_k*gval
               end do
               call pair_scalars(par, jdist(st), zeta_m(st), unused, f_rad_m(st), &
                                 c_zeta_m(st))
            end do

            back = 0_ip
            do st = 1_ip, nj
               if (jdx(st) /= i) cycle
               if (maxval(abs(jvec(:, st) + vec(:, a))) > bond_match_tol) cycle
               back = st
               exit
            end do
            if (back == 0_ip) then
               status = 2
               errmsg = "rgpot_tersoff: a pair in the neighbour table has no reverse"
               return
            end if

            grad = grad - (f_rad(a) + f_rad_m(back))*u_ij
            rim = dist(a)
            u_mi = -vec(:, a)/rim
            call cutoff_taper(par, rim, fc_i, dfc_i)
            do st = 1_ip, nj
               if (st == back) cycle
               rmt = jdist(st)
               u_mt = jvec(:, st)/rmt
               cosine = dot_product(u_mi, u_mt)
               call angular(par, cosine, gval, dgval)
               call cutoff_taper(par, rmt, fc_k, dfc_k)
               dcos_i = (u_mt - cosine*u_mi)/rim
               grad = grad + c_zeta_m(back)*fc_k*dgval*dcos_i
               grad = grad + c_zeta_m(st)*(dfc_i*gval*u_mi + fc_i*dgval*dcos_i)
            end do
         end do

         forces(:, i) = -0.5_wp*grad
         energy = energy + e_i
      end do
   end subroutine tersoff_stream

end module rgpot_tersoff
