! MIT License
! Copyright 2023--present rgpot developers

!> Full neighbour tables in CSR form, built with rgpot's neighbour query.
!!
!! Every ported potential evaluates forces as a *gather*: atom `i` sums
!! the whole force acting on it from its own neighbours, and writes only
!! `f(:, i)`. Iterations of the atom loop then touch disjoint memory, so
!! they run under `do concurrent`. That needs a **full** list (both `i->j`
!! and `j->i`), with the pair vectors and distances already folded, so
!! the kernels never redo minimum image arithmetic.
!!
!! The search is `rgpot_nlist_fortran_build`. A positive skin keeps the
!! candidate list in an `rgpot_nlist_builder` and reuses it while every
!! atom has moved by at most half the skin. Each call still writes
!! vectors for the positions it was given and keeps only pairs strictly
!! inside the cutoff, which is the Verlet filter. The builder stores
!! those pairs by central atom and then by partner, so the table is
!! filled with one forward copy. The write buffer stays
!! on the table, so a later call that fits in it walks those candidates
!! once. The first call, and a call that outgrows the buffer, counts and
!! then fills. A call with the same positions, cell, cutoff, skin, and
!! periodicity keeps the table it already built.
!!
!! Cell convention: C callers hand over nine doubles, row-major, one cell
!! vector per row. Fortran reads that same memory column-major, so a
!! `real(wp) :: cell(3, 3)` dummy sees one cell vector per *column*.
!! `vec(3, npairs)` is that same layout: column `p` is `r_j - r_i`.
module rgpot_neighbors
   use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, c_null_ptr, &
                                          c_null_char, c_char, c_int, c_double, &
                                          c_size_t, c_int32_t, c_associated
   use rgpot_kinds, only: wp, ip, lp
   implicit none
   private

   public :: neighbor_table_t

   !> Full neighbour list: for atom `i`, entries `row(i) : row(i+1) - 1`
   !! index into `idx`, `vec`, and `dist`.
   type :: neighbor_table_t
      !> Row starts, `natoms + 1` entries, one-based.
      integer(ip), allocatable :: row(:)
      !> Neighbour atom index for each entry, one-based.
      integer(ip), allocatable :: idx(:)
      !> `r_j - r_i` with the periodic shift folded in, `3 x nentries`.
      real(wp), allocatable :: vec(:, :)
      !> `|r_j - r_i|` for each entry.
      real(wp), allocatable :: dist(:)
      !> Atom count the table was built for.
      integer(ip) :: natoms = 0_ip
      !> Cutoff the table was built at.
      real(wp), private :: cutoff = -1.0_wp
      !> Verlet skin the builder caches with.
      real(wp), private :: skin = -1.0_wp
      !> Builder kept across calls. Reused while atoms stay within half the skin.
      type(c_ptr), private :: builder = c_null_ptr
      !> Scratch for the C fill. Kept so a warm call walks the pairs once.
      !! The builder writes central atom, then partner, so this scratch is
      !! already the row order the kernel sums.
      integer(c_int), allocatable, private :: scratch_i(:), scratch_j(:)
      real(c_double), allocatable, private :: scratch_v(:, :), scratch_d(:)
      !> Geometry of the table now stored. A later call with the same bits
      !! does not walk or copy again.
      real(wp), allocatable, private :: ref_pos(:, :)
      real(wp), private :: ref_cell(3, 3) = 0.0_wp
      integer(c_int32_t), private :: ref_pbc(3) = 0_c_int32_t
      logical, private :: have_geom = .false.
   contains
      procedure :: build => neighbor_table_build
      procedure :: count_for => neighbor_table_count_for
      procedure :: release => neighbor_table_release
   end type neighbor_table_t

   interface
      function rgpot_nlist_fortran_open() bind(C, name="rgpot_nlist_fortran_open")
         import :: c_ptr
         implicit none
         type(c_ptr) :: rgpot_nlist_fortran_open
      end function rgpot_nlist_fortran_open

      subroutine rgpot_nlist_fortran_close(list) bind(C, name="rgpot_nlist_fortran_close")
         import :: c_ptr
         implicit none
         type(c_ptr), value :: list
      end subroutine rgpot_nlist_fortran_close

      function rgpot_nlist_fortran_build(list, positions, natoms, box9, cutoff, skin, &
                                         periodic, first, second, vector, distance, cap, &
                                         count, reused) &
         bind(C, name="rgpot_nlist_fortran_build")
         import :: c_ptr, c_double, c_size_t, c_int32_t, c_int
         implicit none
         type(c_ptr), value :: list
         real(c_double), intent(in) :: positions(*)
         integer(c_size_t), value :: natoms
         real(c_double), intent(in) :: box9(*)
         real(c_double), value :: cutoff, skin
         integer(c_int32_t), intent(in) :: periodic(*)
         type(c_ptr), value :: first, second, vector, distance
         integer(c_size_t), value :: cap
         integer(c_size_t), intent(out) :: count
         integer(c_int32_t), intent(out) :: reused
         integer(c_int32_t) :: rgpot_nlist_fortran_build
      end function rgpot_nlist_fortran_build

      function rgpot_nlist_strerror(code) bind(C, name="rgpot_nlist_strerror")
         import :: c_ptr, c_int32_t
         implicit none
         integer(c_int32_t), value :: code
         type(c_ptr) :: rgpot_nlist_strerror
      end function rgpot_nlist_strerror
   end interface

contains

   !> Rebuild the table for `positions` (3 x natoms) within `cutoff`.
   !!
   !! `periodic` defaults to fully periodic. `status` is zero on success;
   !! on failure the table is left empty and `errmsg` describes the fault.
   !! `skin` defaults to a tenth of the cutoff. The builder reuses its
   !! candidates while every atom has moved by at most half of that skin.
   subroutine neighbor_table_build(self, positions, cell, cutoff, status, &
                                   errmsg, periodic, skin)
      class(neighbor_table_t), intent(inout) :: self
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      real(wp), intent(in) :: cutoff
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg
      logical, intent(in), optional :: periodic(3)
      real(wp), intent(in), optional :: skin

      integer(c_int32_t) :: pbc(3), rc, reused
      integer(c_size_t) :: natoms_c, needed, cap
      real(wp) :: use_skin
      ! RGPOT_NLIST_SHORT. A short buffer still reports the full count.
      integer, parameter :: nlist_short = 6
      integer :: p, np
      integer(ip) :: i, j, natoms, slot, cursor
      type(c_ptr) :: msg

      status = 0
      errmsg = ""
      natoms = int(size(positions, 2), ip)

      if (size(positions, 1) /= 3) then
         status = 1
         errmsg = "rgpot_neighbors: positions must be shaped [3, natoms]"
         return
      end if

      pbc = 1_c_int32_t
      if (present(periodic)) then
         where (.not. periodic) pbc = 0_c_int32_t
      end if

      use_skin = 0.1_wp*cutoff
      if (present(skin)) use_skin = skin

      ! Zero displacement, bitwise: the stored vectors and distances are
      ! already the ones a fresh fill would write. A move inside the skin
      ! still changes those distances, so that path rebuilds.
      if (self%have_geom .and. allocated(self%row) .and. self%natoms == natoms .and. &
          transfer(self%cutoff, 0_lp) == transfer(cutoff, 0_lp) .and. &
          transfer(self%skin, 0_lp) == transfer(use_skin, 0_lp) .and. &
          all(self%ref_pbc == pbc) .and. same_bits(self%ref_cell, cell) .and. &
          same_bits(self%ref_pos, positions)) then
         status = 0
         return
      end if
      self%have_geom = .false.

      if (.not. c_associated(self%builder)) then
         self%builder = rgpot_nlist_fortran_open()
         if (.not. c_associated(self%builder)) then
            status = 7
            errmsg = "rgpot_neighbors: the neighbour builder could not be allocated"
            return
         end if
      end if

      natoms_c = int(natoms, c_size_t)
      needed = 0_c_size_t
      cap = 0_c_size_t
      if (allocated(self%scratch_i)) cap = int(size(self%scratch_i), c_size_t)
      ! A buffer that already holds the pairs is filled in one walk. The
      ! first call, or a call that outgrows it, counts and then fills.
      if (cap > 0_c_size_t) then
         rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                        real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                        loc_ints(self%scratch_i), loc_ints(self%scratch_j), &
                                        loc_vecs(self%scratch_v), loc_reals(self%scratch_d), &
                                        cap, needed, reused)
      else
         rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                        real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                        c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
                                        0_c_size_t, needed, reused)
      end if
      if (rc /= 0_c_int32_t .and. rc /= nlist_short) then
         status = int(rc)
         msg = rgpot_nlist_strerror(rc)
         errmsg = "rgpot_neighbors: "//c_string(msg)
         return
      end if

      np = int(needed)
      self%natoms = natoms
      self%cutoff = cutoff
      self%skin = use_skin

      if ((cap == 0_c_size_t .or. rc == nlist_short) .and. np > 0) then
         if (allocated(self%scratch_i)) then
            deallocate (self%scratch_i, self%scratch_j, self%scratch_v, self%scratch_d)
         end if
         allocate (self%scratch_i(np), self%scratch_j(np), self%scratch_v(3, np), &
                   self%scratch_d(np))
         cap = int(np, c_size_t)
         rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                        real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                        loc_ints(self%scratch_i), loc_ints(self%scratch_j), &
                                        loc_vecs(self%scratch_v), loc_reals(self%scratch_d), &
                                        cap, needed, reused)
         if (rc /= 0_c_int32_t) then
            status = int(rc)
            msg = rgpot_nlist_strerror(rc)
            errmsg = "rgpot_neighbors: "//c_string(msg)
            return
         end if
         np = int(needed)
      end if

      ! The C list is ordered by central atom, then partner. Dropping an
      ! atom paired with itself and writing the rest in that order is the
      ! row layout the force sum uses.
      if (allocated(self%row)) then
         if (size(self%row) /= natoms + 1) deallocate (self%row)
      end if
      if (.not. allocated(self%row)) allocate (self%row(natoms + 1))
      if (allocated(self%idx)) then
         if (size(self%idx) < np) deallocate (self%idx, self%vec, self%dist)
      end if
      if (.not. allocated(self%idx)) then
         allocate (self%idx(max(np, 1)))
         allocate (self%vec(3, max(np, 1)))
         allocate (self%dist(max(np, 1)))
      end if
      slot = 0
      cursor = 1
      do p = 1, np
         i = int(self%scratch_i(p), ip) + 1_ip
         j = int(self%scratch_j(p), ip) + 1_ip
         if (i == j) cycle
         do while (cursor <= i)
            self%row(cursor) = slot + 1_ip
            cursor = cursor + 1
         end do
         slot = slot + 1
         self%idx(slot) = j
         self%vec(:, slot) = self%scratch_v(:, p)
         self%dist(slot) = self%scratch_d(p)
      end do
      do while (cursor <= natoms + 1)
         self%row(cursor) = slot + 1_ip
         cursor = cursor + 1
      end do

      if (.not. allocated(self%ref_pos) .or. size(self%ref_pos, 2) /= natoms) then
         if (allocated(self%ref_pos)) deallocate (self%ref_pos)
         allocate (self%ref_pos(3, natoms))
      end if
      self%ref_pos = positions
      self%ref_cell = cell
      self%ref_pbc = pbc
      self%have_geom = .true.
   end subroutine neighbor_table_build

   !> Free the builder; the table can be built again afterwards.
   subroutine neighbor_table_release(self)
      class(neighbor_table_t), intent(inout) :: self

      if (c_associated(self%builder)) call rgpot_nlist_fortran_close(self%builder)
      self%builder = c_null_ptr
      self%cutoff = -1.0_wp
      self%skin = -1.0_wp
      self%have_geom = .false.
   end subroutine neighbor_table_release

   !> Number of neighbours atom `i` owns.
   pure function neighbor_table_count_for(self, i) result(n)
      class(neighbor_table_t), intent(in) :: self
      integer(ip), intent(in) :: i
      integer(ip) :: n

      n = self%row(i + 1_ip) - self%row(i)
   end function neighbor_table_count_for


   !> An allocatable component cannot carry TARGET. The dummy can, and
   !! c_loc of that dummy is the address the C fill writes into.
   !> True when every element has the same bits, including signed zero.
   pure function same_bits(a, b) result(eq)
      real(wp), intent(in) :: a(:, :), b(:, :)
      logical :: eq

      integer :: i, j
      integer(lp) :: xa, xb

      eq = size(a, 1) == size(b, 1) .and. size(a, 2) == size(b, 2)
      if (.not. eq) return
      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            xa = transfer(a(i, j), 0_lp)
            xb = transfer(b(i, j), 0_lp)
            if (xa /= xb) then
               eq = .false.
               return
            end if
         end do
      end do
   end function same_bits

   function loc_ints(a) result(p)
      integer(c_int), intent(inout), target :: a(:)
      type(c_ptr) :: p
      p = c_loc(a)
   end function loc_ints

   function loc_reals(a) result(p)
      real(c_double), intent(inout), target :: a(:)
      type(c_ptr) :: p
      p = c_loc(a)
   end function loc_reals

   function loc_vecs(a) result(p)
      real(c_double), intent(inout), target :: a(:, :)
      type(c_ptr) :: p
      p = c_loc(a)
   end function loc_vecs

   function c_string(ptr) result(text)
      type(c_ptr), intent(in) :: ptr
      character(len=:), allocatable :: text
      character(kind=c_char), pointer :: raw(:)
      integer :: n, k

      if (.not. c_associated(ptr)) then
         text = "unknown neighbour error"
         return
      end if
      call c_f_pointer(ptr, raw, [4096])
      n = 0
      do while (raw(n + 1) /= c_null_char .and. n < 4096)
         n = n + 1
      end do
      allocate (character(len=n) :: text)
      do k = 1, n
         text(k:k) = raw(k)
      end do
   end function c_string

end module rgpot_neighbors
