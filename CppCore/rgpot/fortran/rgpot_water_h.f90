! MIT License
! Copyright 2023--present rgpot developers
!
! The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/Water_H/potH_H2O.f),
! BSD-3-Clause, copyright the eOn Development Team. The functional form is
! Andersson's potH-H2O surface, J. Chem. Phys. 124, 064715 (2006).

!> One H atom moving in the field of water molecules.
!!
!! The legacy kernel takes three distances measured from the lone H atom
!! (to the oxygen, and to each of the two hydrogens of one water) and
!! returns an energy plus the three radial derivatives. Each of the three
!! distances enters through its own independent terms: a damped `-C6/r^6`
!! dispersion on every leg, a Born-Mayer repulsion on every leg, and a
!! Morse well on the oxygen leg alone. No term couples two distances.
!!
!! Structure: a star gather over sites, not a neighbour-table pair sum.
!!
!! Three properties of the expression pick that shape.
!!
!!   * There is one centre. Every interaction has the lone H at one end,
!!     so the sum is `natoms - 1` legs, not `O(natoms^2)` pairs.
!!   * The legs are independent and the two hydrogen legs share a single
!!     parameter set, so which water a hydrogen belongs to cancels out of
!!     the total. Grouping sites into molecules changes nothing, which is
!!     why this is a site loop rather than a molecule loop.
!!   * There is no cutoff. Every site interacts with the lone H at its
!!     nearest periodic image, however far away. A vesin table built at
!!     some finite radius would silently truncate the sum, so the minimum
!!     image is taken here instead.
!!
!! Each site writes only its own force, and the lone H takes the negated
!! sum of the legs, so the site loop runs under `do concurrent` and the
!! reduction happens once outside it.
!!
!! This module carries the H-water term. Water-water TIP4P electrostatics,
!! the oxygen Lennard-Jones, and the CCL quartic intramolecular restraints
!! are the water potential's own physics and enter through its module.
!!
!! Units: the published parameters are kJ/mol and Angstrom, and are held
!! that way. `energy_scale` converts the result to the eV/Angstrom that
!! every rgpot entry point reports.
module rgpot_water_h
   use rgpot_kinds, only: wp, ip, c_double, c_int, c_size_t
   implicit none
   private

   public :: water_h_params_t, water_h_energy_forces, water_h_classify

#ifdef RGPOT_HAS_FAST_NLIST
   interface
      !> Minimum-image wrap from the minimage static library.
      !!
      !! Fractional rounding. Exact for an orthorhombic cell, and for a
      !! skewed cell when the wrapped length sits strictly inside half
      !! the smallest face altitude (Smith, CCP5, 1989). Outside that
      !! bound the 26-image scan below is the nearest image.
      function rgpot_fast_wrap(cell, diffs, n, out) bind(c, name="rgpot_fast_wrap") &
         result(status)
         import :: c_double, c_int, c_size_t
         implicit none
         real(c_double), intent(in) :: cell(*)
         real(c_double), intent(in) :: diffs(*)
         integer(c_size_t), value :: n
         real(c_double), intent(out) :: out(*)
         integer(c_int) :: status
      end function rgpot_fast_wrap
   end interface
#endif

   !> Atomic numbers this potential accepts.
   integer(ip), parameter, public :: z_hydrogen = 1_ip
   integer(ip), parameter, public :: z_oxygen = 8_ip

   !> potH-H2O parameters. Defaults are the published set, carried over
   !! from the legacy `BLOCK DATA` and the quantities `setup()` derived
   !! from it.
   type :: water_h_params_t
      !> Dispersion damping reference radius, H to oxygen (Angstrom).
      real(wp) :: rm_ho = 4.04741_wp
      !> Dispersion damping reference radius, H to hydrogen (Angstrom).
      real(wp) :: rm_hh = 2.19954_wp
      !> Dispersion coefficient, H to oxygen (kJ/mol Angstrom^6).
      real(wp) :: c6_ho = 2216.76_wp
      !> Dispersion coefficient, H to hydrogen (kJ/mol Angstrom^6).
      real(wp) :: c6_hh = 30.2562_wp
      !> Born-Mayer prefactor, H to oxygen (kJ/mol).
      real(wp) :: a_ho = 4158.88_wp
      !> Born-Mayer decay, H to oxygen (1/Angstrom).
      real(wp) :: b_ho = 2.63805_wp
      !> Born-Mayer prefactor, H to hydrogen (kJ/mol).
      real(wp) :: a_hh = 5563.57_wp
      !> Born-Mayer decay, H to hydrogen (1/Angstrom).
      real(wp) :: b_hh = 4.27352_wp
      !> Morse well depth on the oxygen leg (kJ/mol).
      real(wp) :: de = 287.598_wp
      !> Morse range parameter on the oxygen leg (1/Angstrom).
      real(wp) :: beta = 4.73771_wp
      !> Morse equilibrium separation on the oxygen leg (Angstrom).
      real(wp) :: r_eq = 0.816286_wp
      !> Damping switches on below `damp_scale * rm`.
      real(wp) :: damp_scale = 1.28_wp
      !> kJ/mol to eV.
      real(wp) :: energy_scale = 1.0_wp/96.485336_wp
   contains
      procedure :: damp_onset_ho => water_h_damp_onset_ho
      procedure :: damp_onset_hh => water_h_damp_onset_hh
   end type water_h_params_t

   !> Separations below this leave the radial derivative undefined, so a
   !! leg this short is reported as a fault rather than evaluated.
   real(wp), parameter :: min_separation = 1.0e-8_wp

contains

   !> Distance below which the oxygen leg's dispersion is damped.
   pure function water_h_damp_onset_ho(self) result(r_damp)
      class(water_h_params_t), intent(in) :: self
      real(wp) :: r_damp

      r_damp = self%damp_scale*self%rm_ho
   end function water_h_damp_onset_ho

   !> Distance below which a hydrogen leg's dispersion is damped.
   pure function water_h_damp_onset_hh(self) result(r_damp)
      class(water_h_params_t), intent(in) :: self
      real(wp) :: r_damp

      r_damp = self%damp_scale*self%rm_hh
   end function water_h_damp_onset_hh

   !> Damped dispersion `-D(r) C6 / r^6` and its radial derivative.
   !!
   !! The damping function `D = exp(-(r_damp/r - 1)^2)` reaches one at
   !! `r_damp` with a vanishing slope, so the two branches join smoothly.
   elemental subroutine dispersion(c6, r_damp, r, energy, dedr)
      real(wp), intent(in) :: c6, r_damp, r
      real(wp), intent(out) :: energy, dedr

      real(wp) :: damp, ddamp, inv_r6

      if (r >= r_damp) then
         damp = 1.0_wp
         ddamp = 0.0_wp
      else
         damp = exp(-(r_damp/r - 1.0_wp)**2)
         ddamp = 2.0_wp*(r_damp**2/r**3 - r_damp/r**2)*damp
      end if

      inv_r6 = r**(-6)
      energy = -damp*c6*inv_r6
      dedr = -ddamp*c6*inv_r6 + 6.0_wp*damp*c6*inv_r6/r
   end subroutine dispersion

   !> Born-Mayer repulsion `A exp(-B r)` and its radial derivative.
   elemental subroutine repulsion(a, b, r, energy, dedr)
      real(wp), intent(in) :: a, b, r
      real(wp), intent(out) :: energy, dedr

      energy = a*exp(-b*r)
      dedr = -b*energy
   end subroutine repulsion

   !> Morse well and its radial derivative.
   elemental subroutine morse(de, beta, r_eq, r, energy, dedr)
      real(wp), intent(in) :: de, beta, r_eq, r
      real(wp), intent(out) :: energy, dedr

      real(wp) :: dr, decay, decay_squared

      dr = r - r_eq
      decay = exp(-beta*dr)
      decay_squared = exp(-2.0_wp*beta*dr)

      energy = de*(decay_squared - 2.0_wp*decay)
      dedr = -2.0_wp*beta*de*(decay_squared - decay)
   end subroutine morse

   !> Energy and radial derivative of the lone H against one oxygen.
   pure subroutine oxygen_leg(par, r, energy, dedr)
      type(water_h_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp), intent(out) :: energy, dedr

      real(wp) :: e_disp, d_disp, e_rep, d_rep, e_morse, d_morse

      call dispersion(par%c6_ho, par%damp_onset_ho(), r, e_disp, d_disp)
      call repulsion(par%a_ho, par%b_ho, r, e_rep, d_rep)
      call morse(par%de, par%beta, par%r_eq, r, e_morse, d_morse)

      energy = e_disp + e_rep + e_morse
      dedr = d_disp + d_rep + d_morse
   end subroutine oxygen_leg

   !> Energy and radial derivative of the lone H against one water
   !! hydrogen. The Morse well belongs to the oxygen leg alone.
   pure subroutine hydrogen_leg(par, r, energy, dedr)
      type(water_h_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp), intent(out) :: energy, dedr

      real(wp) :: e_disp, d_disp, e_rep, d_rep

      call dispersion(par%c6_hh, par%damp_onset_hh(), r, e_disp, d_disp)
      call repulsion(par%a_hh, par%b_hh, r, e_rep, d_rep)

      energy = e_disp + e_rep
      dedr = d_disp + d_rep
   end subroutine hydrogen_leg

   !> Inverse of the cell matrix by cofactors.
   !!
   !! `invertible` is false for a singular cell, which includes the zero
   !! cell a caller hands over for an isolated cluster; the leg vectors
   !! are then taken as they stand, with no periodic image search.
   pure subroutine invert_cell(cell, inv, invertible)
      real(wp), intent(in) :: cell(3, 3)
      real(wp), intent(out) :: inv(3, 3)
      logical, intent(out) :: invertible

      real(wp) :: det, extent

      inv(1, 1) = cell(2, 2)*cell(3, 3) - cell(2, 3)*cell(3, 2)
      inv(2, 1) = cell(2, 3)*cell(3, 1) - cell(2, 1)*cell(3, 3)
      inv(3, 1) = cell(2, 1)*cell(3, 2) - cell(2, 2)*cell(3, 1)
      inv(1, 2) = cell(1, 3)*cell(3, 2) - cell(1, 2)*cell(3, 3)
      inv(2, 2) = cell(1, 1)*cell(3, 3) - cell(1, 3)*cell(3, 1)
      inv(3, 2) = cell(1, 2)*cell(3, 1) - cell(1, 1)*cell(3, 2)
      inv(1, 3) = cell(1, 2)*cell(2, 3) - cell(1, 3)*cell(2, 2)
      inv(2, 3) = cell(1, 3)*cell(2, 1) - cell(1, 1)*cell(2, 3)
      inv(3, 3) = cell(1, 1)*cell(2, 2) - cell(1, 2)*cell(2, 1)

      det = cell(1, 1)*inv(1, 1) + cell(1, 2)*inv(2, 1) + cell(1, 3)*inv(3, 1)
      extent = maxval(abs(cell))

      invertible = abs(det) > 1.0e-12_wp*max(extent**3, 1.0_wp)
      if (invertible) then
         inv = inv/det
      else
         inv = 0.0_wp
      end if
   end subroutine invert_cell

   !> Shortest of the central image and the surrounding shell of 26.
   !!
   !! The scan itself is `rgpot_water_h_image`. Rounding the fractional
   !! coordinates lands in the primitive cell, and on a skewed cell that
   !! image can be longer than a neighbour, so the shell is scanned and
   !! the shortest vector is kept. One routine serves the kernel and the
   !! tests: a Fortran matmul and a second C++ copy are not the same
   !! rounding on every compiler.
   function nearest_image(cell, inv_cell, d) result(shortest)
      real(wp), intent(in) :: cell(3, 3), inv_cell(3, 3), d(3)
      real(wp) :: shortest(3)

      interface
         subroutine rgpot_water_h_image(cell, d, out) &
            bind(c, name="rgpot_water_h_image")
            import :: c_double
            real(c_double), intent(in) :: cell(3, 3)
            real(c_double), intent(in) :: d(3)
            real(c_double), intent(out) :: out(3)
         end subroutine rgpot_water_h_image
      end interface

      ! The inverse is formed inside the C++ scan. `inv_cell` stays so the
      ! callers that already computed one do not grow a second argument list.
      call rgpot_water_h_image(cell, d, shortest)
   end function nearest_image

#ifdef RGPOT_HAS_FAST_NLIST
   !> Perpendicular face altitudes, and whether the cell is orthorhombic.
   pure subroutine face_widths(cell, widths, ortho)
      real(wp), intent(in) :: cell(3, 3)
      real(wp), intent(out) :: widths(3)
      logical, intent(out) :: ortho

      real(wp) :: a(3), b(3), cvec(3), bc(3), ca(3), ab(3)
      real(wp) :: det, nb, nc, na

      a = cell(:, 1)
      b = cell(:, 2)
      cvec = cell(:, 3)
      bc(1) = b(2)*cvec(3) - b(3)*cvec(2)
      bc(2) = b(3)*cvec(1) - b(1)*cvec(3)
      bc(3) = b(1)*cvec(2) - b(2)*cvec(1)
      ca(1) = cvec(2)*a(3) - cvec(3)*a(2)
      ca(2) = cvec(3)*a(1) - cvec(1)*a(3)
      ca(3) = cvec(1)*a(2) - cvec(2)*a(1)
      ab(1) = a(2)*b(3) - a(3)*b(2)
      ab(2) = a(3)*b(1) - a(1)*b(3)
      ab(3) = a(1)*b(2) - a(2)*b(1)
      det = dot_product(a, bc)
      nb = norm2(bc)
      nc = norm2(ca)
      na = norm2(ab)
      if (nb > 0.0_wp .and. nc > 0.0_wp .and. na > 0.0_wp) then
         widths(1) = abs(det)/nb
         widths(2) = abs(det)/nc
         widths(3) = abs(det)/na
      else
         widths = 0.0_wp
      end if
      ortho = cell(1, 2) == 0.0_wp .and. cell(1, 3) == 0.0_wp .and. &
              cell(2, 1) == 0.0_wp .and. cell(2, 3) == 0.0_wp .and. &
              cell(3, 1) == 0.0_wp .and. cell(3, 2) == 0.0_wp
   end subroutine face_widths

   !> Batch wrap. Orthorhombic cells, and skewed legs inside the Smith
   !! bound, take ``rgpot_fast_wrap``. Every other leg takes the 26-image
   !! scan, which is also the path when the batch wrap fails.
   subroutine wrap_legs(cell, inv_cell, n, raw, wrapped)
      real(wp), intent(in) :: cell(3, 3), inv_cell(3, 3)
      integer(ip), intent(in) :: n
      real(wp), intent(in) :: raw(:, :)
      real(wp), intent(out) :: wrapped(:, :)

      integer(c_int) :: wstatus
      integer(ip) :: i
      real(wp) :: widths(3), room, span
      logical :: ortho

      call face_widths(cell, widths, ortho)
      if (ortho) then
         wstatus = rgpot_fast_wrap(cell, raw, int(n, c_size_t), wrapped)
         if (wstatus == 0_c_int) return
      else
         room = 0.5_wp*minval(widths) - 1.0e-12_wp
         if (room > 0.0_wp) then
            wstatus = rgpot_fast_wrap(cell, raw, int(n, c_size_t), wrapped)
            if (wstatus == 0_c_int) then
               do i = 1_ip, n
                  span = dot_product(wrapped(:, i), wrapped(:, i))
                  if (.not. (span < room*room)) then
                     wrapped(:, i) = nearest_image(cell, inv_cell, raw(:, i))
                  end if
               end do
               return
            end if
         end if
      end if
      do i = 1_ip, n
         wrapped(:, i) = nearest_image(cell, inv_cell, raw(:, i))
      end do
   end subroutine wrap_legs
#endif

   !> One leg: the energy it contributes and the force it puts on the site.
   !!
   !! `d` is the separation already reduced to the minimum image when the
   !! cell is periodic. `force` acts on the site; the lone H receives its
   !! negative. `usable` is false when the site sits on top of the lone H,
   !! where the radial derivative has no direction.
   pure subroutine site_leg(par, d, oxygen, energy, force, usable)
      type(water_h_params_t), intent(in) :: par
      real(wp), intent(in) :: d(3)
      logical, intent(in) :: oxygen
      real(wp), intent(out) :: energy
      real(wp), intent(out) :: force(3)
      logical, intent(out) :: usable

      real(wp) :: r, dedr

      energy = 0.0_wp
      force = 0.0_wp

      r = norm2(d)

      usable = r > min_separation
      if (.not. usable) return

      if (oxygen) then
         call oxygen_leg(par, r, energy, dedr)
      else
         call hydrogen_leg(par, r, energy, dedr)
      end if

      energy = par%energy_scale*energy
      dedr = par%energy_scale*dedr
      force = -dedr*d/r
   end subroutine site_leg

   !> Identify the lone H and mark which of the remaining atoms are oxygen.
   !!
   !! Atomic numbers alone cannot separate the lone H from the hydrogens
   !! bound in water, because both are Z=1. The eOn kernel this ports
   !! resolves that by position: the lone H is the last atom. That
   !! convention is enforced here rather than guessed at, together with
   !! the composition it implies, `n_hydrogen == 2 * n_oxygen + 1`.
   subroutine water_h_classify(atomic_numbers, lone_h, is_oxygen, status, errmsg)
      integer(ip), intent(in) :: atomic_numbers(:)
      integer(ip), intent(out) :: lone_h
      logical, allocatable, intent(out) :: is_oxygen(:)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, n_oxygen, n_hydrogen

      status = 0
      errmsg = ""
      lone_h = 0_ip
      natoms = int(size(atomic_numbers), ip)

      if (natoms < 4_ip) then
         status = 3
         errmsg = "rgpot_water_h: one water plus one H needs at least 4 atoms, got " &
                  //itoa(natoms)
         return
      end if

      if (any(atomic_numbers /= z_hydrogen .and. atomic_numbers /= z_oxygen)) then
         status = 3
         errmsg = "rgpot_water_h: only hydrogen (Z=1) and oxygen (Z=8) atoms belong "// &
                  "in a water plus H system"
         return
      end if

      if (atomic_numbers(natoms) /= z_hydrogen) then
         status = 3
         errmsg = "rgpot_water_h: the lone H must be the last atom, but atom " &
                  //itoa(natoms)//" has Z="//itoa(atomic_numbers(natoms))
         return
      end if

      n_oxygen = int(count(atomic_numbers == z_oxygen), ip)
      n_hydrogen = natoms - n_oxygen

      if (n_oxygen < 1_ip) then
         status = 3
         errmsg = "rgpot_water_h: no oxygen atoms, so there is no water for the H "// &
                  "atom to move in"
         return
      end if

      if (n_hydrogen /= 2_ip*n_oxygen + 1_ip) then
         status = 3
         errmsg = "rgpot_water_h: composition is not whole water molecules plus one "// &
                  "H: "//itoa(n_oxygen)//" O and "//itoa(n_hydrogen)//" H, where " &
                  //itoa(2_ip*n_oxygen + 1_ip)//" H are needed"
         return
      end if

      lone_h = natoms
      allocate (is_oxygen(natoms))
      is_oxygen = atomic_numbers == z_oxygen
   end subroutine water_h_classify

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   subroutine water_h_energy_forces(positions, atomic_numbers, cell, par, &
                                    energy, forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      real(wp), intent(in) :: cell(3, 3)
      type(water_h_params_t), intent(in) :: par
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, i, lone_h, slot, nleg
      real(wp) :: inv_cell(3, 3)
      logical :: periodic
      logical, allocatable :: is_oxygen(:), usable(:)
      real(wp), allocatable :: e_site(:), leg_force(:, :)
      real(wp), allocatable :: raw(:, :), wrapped(:, :)
      integer(ip), allocatable :: leg_of(:)

      status = 0
      errmsg = ""
      energy = 0.0_wp
      forces = 0.0_wp
      natoms = int(size(positions, 2), ip)

      if (size(positions, 1) /= 3 .or. size(forces, 1) /= 3) then
         status = 1
         errmsg = "rgpot_water_h: positions and forces must be shaped [3, natoms]"
         return
      end if

      if (int(size(forces, 2), ip) /= natoms) then
         status = 1
         errmsg = "rgpot_water_h: forces must have one column per atom"
         return
      end if

      if (int(size(atomic_numbers), ip) /= natoms) then
         status = 1
         errmsg = "rgpot_water_h: atomic_numbers must have one entry per atom"
         return
      end if

      call water_h_classify(atomic_numbers, lone_h, is_oxygen, status, errmsg)
      if (status /= 0) return

      call invert_cell(cell, inv_cell, periodic)

      allocate (e_site(natoms), source=0.0_wp)
      allocate (leg_force(3, natoms), source=0.0_wp)
      allocate (usable(natoms), source=.true.)

      nleg = natoms - 1_ip
      allocate (raw(3, nleg), wrapped(3, nleg), leg_of(nleg))
      slot = 0_ip
      do i = 1_ip, natoms
         if (i == lone_h) cycle
         slot = slot + 1_ip
         leg_of(slot) = i
         raw(:, slot) = positions(:, i) - positions(:, lone_h)
      end do
      if (periodic) then
#ifdef RGPOT_HAS_FAST_NLIST
         call wrap_legs(cell, inv_cell, slot, raw, wrapped)
#else
         do i = 1_ip, slot
            wrapped(:, i) = nearest_image(cell, inv_cell, raw(:, i))
         end do
#endif
      else
         wrapped = raw
      end if

      ! Every site owns one leg and writes only its own column, so the
      ! iterations touch disjoint memory. The lone H's force is the one
      ! quantity that reduces across sites, and it is summed below rather
      ! than accumulated here, because `do concurrent` gained reduction
      ! clauses only in Fortran 2023.
      do concurrent(i=1:slot)
         call site_leg(par, wrapped(:, i), is_oxygen(leg_of(i)), e_site(leg_of(i)), &
                       leg_force(:, leg_of(i)), usable(leg_of(i)))
      end do

      if (.not. all(usable)) then
         status = 2
         errmsg = "rgpot_water_h: an atom coincides with the lone H atom, leaving "// &
                  "the radial derivative without a direction"
         return
      end if

      energy = sum(e_site)
      forces = leg_force
      forces(:, lone_h) = -sum(leg_force, dim=2)
   end subroutine water_h_energy_forces

   !> Decimal text for an index or a count, for the error messages.
   function itoa(value) result(text)
      integer(ip), intent(in) :: value
      character(len=:), allocatable :: text

      character(len=16) :: buffer

      write (buffer, '(i0)') value
      text = trim(buffer)
   end function itoa

end module rgpot_water_h
