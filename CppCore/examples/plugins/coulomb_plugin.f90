! rgpot plugin: Coulomb potential in Fortran 2003+ (bind(C)).
!
! Demonstrates implementing an rgpot plugin in modern Fortran using
! ISO_C_BINDING. The potential computes pairwise Coulomb interactions
! between point charges (charge = atomic_number for simplicity).
!
! E = sum_{i<j} q_i * q_j / r_ij  (in eV*angstrom units)
! F_i = -dE/dr_i
!
! Build:
!   gfortran -shared -fPIC -o rgpot_coulomb.so coulomb_plugin.f90 \
!     -I../../../include
!
! The plugin header types are re-declared locally (Fortran cannot
! #include a C header), matching the layout of rgpot_plugin.h.

module coulomb_mod
  use iso_c_binding
  implicit none

  ! Coulomb constant: e^2 / (4*pi*eps0) in eV*angstrom
  real(c_double), parameter :: KE = 14.3996448915d0

contains

  subroutine coulomb_forces(n, pos, atmnrs, cell, energy, forces) bind(C)
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in) :: pos(3, n)
    integer(c_int), intent(in) :: atmnrs(n)
    real(c_double), intent(in) :: cell(3, 3)
    real(c_double), intent(out) :: energy
    real(c_double), intent(out) :: forces(3, n)

    integer :: i, j
    real(c_double) :: dx, dy, dz, r, inv_r, qi, qj, fij

    energy = 0.0d0
    forces = 0.0d0

    do i = 1, n - 1
      qi = dble(atmnrs(i))
      do j = i + 1, n
        qj = dble(atmnrs(j))
        dx = pos(1, i) - pos(1, j)
        dy = pos(2, i) - pos(2, j)
        dz = pos(3, i) - pos(3, j)
        r = sqrt(dx*dx + dy*dy + dz*dz)
        if (r < 1.0d-10) cycle
        inv_r = 1.0d0 / r

        energy = energy + KE * qi * qj * inv_r

        ! F_i = KE * qi * qj * (r_i - r_j) / r^3
        fij = KE * qi * qj * inv_r * inv_r * inv_r
        forces(1, i) = forces(1, i) + fij * dx
        forces(2, i) = forces(2, i) + fij * dy
        forces(3, i) = forces(3, i) + fij * dz
        forces(1, j) = forces(1, j) - fij * dx
        forces(2, j) = forces(2, j) - fij * dy
        forces(3, j) = forces(3, j) - fij * dz
      end do
    end do
  end subroutine

end module coulomb_mod
