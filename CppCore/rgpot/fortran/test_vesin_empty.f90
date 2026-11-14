! MIT License
! Copyright 2023--present rgpot developers

!> The vendored vesin Fortran interface rebuilt between systems with and
!! without pairs: an empty system, one atom, and two atoms past the
!! cutoff, met first, after a pair, and before one. Every requested array
!! stays associated with `length` columns, so a loop bounded by `size`
!! reads only what vesin stored.
program test_vesin_empty
   use, intrinsic :: iso_c_binding, only: c_double, c_size_t
   use vesin, only: NeighborList, VesinAutoAlgorithm, VesinBruteForce, VesinCellList
   implicit none

   real(c_double) :: box(3, 3)
   real(c_double) :: none(3, 0), one(3, 1), apart(3, 2), close(3, 2)
   integer :: algorithm, shell, skin, cell, g
   integer, parameter :: algorithms(3) = [VesinAutoAlgorithm, VesinBruteForce, VesinCellList]
   logical :: failed

   failed = .false.
   box = 0.0_c_double
   box(1, 1) = 30.0_c_double
   box(2, 2) = 30.0_c_double
   box(3, 3) = 30.0_c_double
   one = reshape([5.0_c_double, 5.0_c_double, 5.0_c_double], [3, 1])
   apart = reshape([5.0_c_double, 5.0_c_double, 5.0_c_double, &
                    15.0_c_double, 5.0_c_double, 5.0_c_double], [3, 2])
   close = reshape([5.0_c_double, 5.0_c_double, 5.0_c_double, &
                    6.0_c_double, 5.0_c_double, 5.0_c_double], [3, 2])

   do algorithm = 1, 3
      do shell = 0, 1
         do skin = 0, 1
            do cell = 0, 1
               do g = 1, 3
                  call sequence(algorithms(algorithm), shell == 1, real(skin, c_double), &
                                cell == 1, g)
               end do
            end do
         end do
      end do
   end do

   if (failed) then
      error stop 1
   end if
   print '(a)', "vesin Fortran interface: zero-pair rebuilds OK"

contains

   subroutine sequence(alg, full, skin_width, periodic, geometry)
      integer, intent(in) :: alg, geometry
      logical, intent(in) :: full, periodic
      real(c_double), intent(in) :: skin_width

      type(NeighborList) :: list
      integer(c_size_t) :: pairs

      pairs = 1_c_size_t
      if (full) pairs = 2_c_size_t

      ! First call with zero pairs, then pairs, then zero pairs again.
      list = NeighborList(cutoff=2.0_c_double, full=full, algorithm=alg, n_threads=1, &
                          skin=skin_width, return_shifts=.true., &
                          return_distances=.true., return_vectors=.true.)
      call compute(list, geometry, periodic, 0_c_size_t)
      call compute(list, 4, periodic, pairs)
      call compute(list, geometry, periodic, 0_c_size_t)
      call compute(list, geometry, periodic, 0_c_size_t)
      call compute(list, 4, periodic, pairs)
      call list%free()
   end subroutine sequence

   subroutine compute(list, geometry, periodic, expected)
      type(NeighborList), intent(inout) :: list
      integer, intent(in) :: geometry
      logical, intent(in) :: periodic
      integer(c_size_t), intent(in) :: expected

      integer :: status, k
      integer(c_size_t) :: i, j

      select case (geometry)
      case (1)
         call list%compute(none, box, [periodic, periodic, periodic], status)
      case (2)
         call list%compute(one, box, [periodic, periodic, periodic], status)
      case (3)
         call list%compute(apart, box, [periodic, periodic, periodic], status)
      case default
         call list%compute(close, box, [periodic, periodic, periodic], status)
      end select

      if (status /= 0) then
         print '(a, i0, 2a)', "geometry ", geometry, ": ", list%errmsg
         failed = .true.
         return
      end if
      if (list%length /= expected) then
         print '(a, i0, a, i0)', "geometry ", geometry, ": length ", list%length
         failed = .true.
         return
      end if
      if (.not. (associated(list%pairs) .and. associated(list%shifts) .and. &
                 associated(list%distances) .and. associated(list%vectors))) then
         print '(a, i0, a)', "geometry ", geometry, ": a requested array is not associated"
         failed = .true.
         return
      end if
      if (size(list%pairs, 1) /= 2 .or. size(list%pairs, 2) /= expected .or. &
          size(list%shifts, 1) /= 3 .or. size(list%shifts, 2) /= expected .or. &
          size(list%vectors, 1) /= 3 .or. size(list%vectors, 2) /= expected .or. &
          size(list%distances) /= expected) then
         print '(a, i0, a)', "geometry ", geometry, ": array shapes do not match the length"
         failed = .true.
         return
      end if
      do k = 1, size(list%pairs, 2)
         i = list%pairs(1, k)
         j = list%pairs(2, k)
         if (i == j .or. i > 1_c_size_t .or. j > 1_c_size_t .or. &
             any(list%shifts(:, k) /= 0) .or. &
             abs(list%distances(k) - 1.0_c_double) > 0.0_c_double .or. &
             abs(abs(list%vectors(1, k)) - 1.0_c_double) > 0.0_c_double) then
            print '(a, i0, a, i0)', "geometry ", geometry, ": bad pair ", k
            failed = .true.
         end if
      end do
   end subroutine compute

end program test_vesin_empty
