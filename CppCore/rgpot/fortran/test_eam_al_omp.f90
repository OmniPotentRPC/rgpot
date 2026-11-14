! MIT License
! Copyright 2023--present rgpot developers

!> The threaded EAM-Al entry matches the serial entry, at one thread and at
!! two, including the forces.
program test_eam_al_omp
   use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double, c_int64_t
   use, intrinsic :: iso_fortran_env, only: int64
   use omp_lib
   use rgpot_kinds, only: wp
   implicit none
   interface
      function ws_new() bind(c, name="rgpot_fortran_workspace_new")
         import :: c_ptr
         type(c_ptr) :: ws_new
      end function
      function force_ws(h, n, p, c, f, e) bind(c, name="rgpot_eam_al_force_ws")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: n
         real(c_double), intent(in) :: p(*), c(*)
         real(c_double), intent(out) :: f(*), e
         integer(c_int) :: force_ws
      end function
      function force_omp(h, n, p, c, f, e) bind(c, name="rgpot_eam_al_force_ws_omp")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: n
         real(c_double), intent(in) :: p(*), c(*)
         real(c_double), intent(out) :: f(*), e
         integer(c_int) :: force_omp
      end function
   end interface
   integer, parameter :: n = 64
   real(wp) :: pos(3, n), fser(3, n), fpar(3, n), cell(9), eser, epar
   type(c_ptr) :: h
   integer :: st, threads_now
   h = ws_new()
   call fcc(pos, cell)
   threads_now = omp_get_max_threads()
   call omp_set_num_threads(1)
   st = force_ws(h, n, pos, cell, fser, eser)
   if (st /= 0) error stop "serial failed"
   st = force_omp(h, n, pos, cell, fpar, epar)
   if (st /= 0) error stop "omp one thread failed"
   if (eser /= epar .or. any(fser /= fpar)) error stop "one thread differs"
   call omp_set_num_threads(2)
   st = force_omp(h, n, pos, cell, fpar, epar)
   if (st /= 0) error stop "omp two threads failed"
   if (eser /= epar .or. any(fser /= fpar)) error stop "two threads differ"
   call omp_set_num_threads(threads_now)
   print *, "test_eam_al_omp: ok"
contains
   subroutine fcc(p, c)
      real(wp), intent(out) :: p(:, :), c(9)
      integer :: ix, iy, iz, b, count, ncell
      real(wp) :: a
      real(wp), parameter :: basis(3, 4) = reshape([0.0_wp, 0.0_wp, 0.0_wp, &
         0.0_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.0_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.0_wp], [3, 4])
      a = 4.05_wp
      ncell = 4
      c = 0
      c(1) = ncell*a
      c(5) = ncell*a
      c(9) = ncell*a
      count = 0
      do ix = 0, ncell - 1
         do iy = 0, ncell - 1
            do iz = 0, ncell - 1
               do b = 1, 4
                  count = count + 1
                  if (count > size(p, 2)) return
                  p(:, count) = (real([ix, iy, iz], wp) + basis(:, b))*a
               end do
            end do
         end do
      end do
   end subroutine fcc
end program
