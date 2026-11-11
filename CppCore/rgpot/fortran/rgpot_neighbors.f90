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
!! inside the cutoff, which is the Verlet filter.
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

   !> One x-plane of directed pairs. Indices stay zero-based so a reused
   !! plane is not shifted twice.
   type :: plane_dump
      integer :: ix = -2
      integer(ip) :: n = 0_ip
      ! Pointers so the C walk can write the buffer in place. Allocatable
      ! components cannot carry the target attribute.
      integer(ip), pointer :: ia(:) => null()
      integer(ip), pointer :: ja(:) => null()
      real(wp), pointer :: vec(:, :) => null()
      real(wp), pointer :: dist(:) => null()
      !> Atoms that own a row in this plane, and that row in CSR order.
      integer(ip) :: nown = 0_ip
      integer(ip), pointer :: atoms(:) => null()
      integer(ip), pointer :: off(:) => null()
      integer(ip), pointer :: nbr(:) => null()
      real(wp), pointer :: gvec(:, :) => null()
      real(wp), pointer :: gdist(:) => null()
   end type plane_dump

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
      !> When true, build keeps the cell bins and load asks for one atom.
      logical :: streaming = .false.
      !> x-plane whose three-plane row cache is current. -2 means empty.
      integer, private :: hold_ix = -2
      !> Grid width the row cache was filled against.
      integer, private :: hold_nx = 0
      !> Up to three x-planes. A slide reuses the two that are still needed.
      type(plane_dump), private :: hold_raw(3)
      !> Positions and cell the stored rows were written for. An unchanged
      !! geometry keeps those rows instead of writing them again.
      real(wp), allocatable, private :: held_pos(:, :)
      real(wp), private :: held_cell(3, 3) = 0.0_wp
      !> Which of the three plane buffers holds atom i's row. 0 if none.
      integer(ip), allocatable, private :: hold_home(:)
      !> x-plane of atom i, or -1 until that plane has been fetched.
      integer, allocatable, private :: hold_plane(:)
   contains
      procedure :: build => neighbor_table_build
      procedure :: count_for => neighbor_table_count_for
      procedure :: release => neighbor_table_release
      procedure :: load => neighbor_load
      procedure :: set_stream => neighbor_table_set_stream
      procedure :: pair_slots => neighbor_table_slots
      procedure :: visit_order => neighbor_visit_order
      procedure :: note_center => neighbor_note_center
      procedure :: grid => neighbor_grid
      procedure :: plane_pairs => neighbor_plane_pairs
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

      function rgpot_nlist_fortran_stream(list, positions, natoms, box9, cutoff, periodic) &
         bind(C, name="rgpot_nlist_fortran_stream")
         import :: c_ptr, c_double, c_size_t, c_int32_t
         implicit none
         type(c_ptr), value :: list
         real(c_double), intent(in) :: positions(*)
         integer(c_size_t), value :: natoms
         real(c_double), intent(in) :: box9(*)
         real(c_double), value :: cutoff
         integer(c_int32_t), intent(in) :: periodic(*)
         integer(c_int32_t) :: rgpot_nlist_fortran_stream
      end function rgpot_nlist_fortran_stream

      function rgpot_nlist_fortran_atom(list, atom, idx, vector, distance, cap, count) &
         bind(C, name="rgpot_nlist_fortran_atom")
         import :: c_ptr, c_double, c_size_t, c_int32_t
         implicit none
         type(c_ptr), value :: list
         integer(c_int32_t), value :: atom
         type(c_ptr), value :: idx
         type(c_ptr), value :: vector
         type(c_ptr), value :: distance
         integer(c_size_t), value :: cap
         integer(c_size_t), intent(out) :: count
         integer(c_int32_t) :: rgpot_nlist_fortran_atom
      end function rgpot_nlist_fortran_atom

      function rgpot_nlist_fortran_order(list, atoms, cap, count) &
         bind(C, name="rgpot_nlist_fortran_order")
         import :: c_ptr, c_size_t, c_int32_t
         implicit none
         type(c_ptr), value :: list
         type(c_ptr), value :: atoms
         integer(c_size_t), value :: cap
         integer(c_size_t), intent(out) :: count
         integer(c_int32_t) :: rgpot_nlist_fortran_order
      end function rgpot_nlist_fortran_order

      function rgpot_nlist_fortran_center(list, atom) &
         bind(C, name="rgpot_nlist_fortran_center")
         import :: c_ptr, c_int32_t
         implicit none
         type(c_ptr), value :: list
         integer(c_int32_t), value :: atom
         integer(c_int32_t) :: rgpot_nlist_fortran_center
      end function rgpot_nlist_fortran_center

      function rgpot_nlist_fortran_grid(list, nx, ny, nz) &
         bind(C, name="rgpot_nlist_fortran_grid")
         import :: c_ptr, c_int32_t
         implicit none
         type(c_ptr), value :: list
         integer(c_int32_t), intent(out) :: nx, ny, nz
         integer(c_int32_t) :: rgpot_nlist_fortran_grid
      end function rgpot_nlist_fortran_grid

      function rgpot_nlist_fortran_plane_of(list, atom) &
         bind(C, name="rgpot_nlist_fortran_plane_of")
         import :: c_ptr, c_int32_t
         implicit none
         type(c_ptr), value :: list
         integer(c_int32_t), value :: atom
         integer(c_int32_t) :: rgpot_nlist_fortran_plane_of
      end function rgpot_nlist_fortran_plane_of

      function rgpot_nlist_fortran_plane(list, ix, first, second, vector, distance, cap, count) &
         bind(C, name="rgpot_nlist_fortran_plane")
         import :: c_ptr, c_int32_t, c_size_t
         implicit none
         type(c_ptr), value :: list
         integer(c_int32_t), value :: ix
         type(c_ptr), value :: first, second, vector, distance
         integer(c_size_t), value :: cap
         integer(c_size_t), intent(out) :: count
         integer(c_int32_t) :: rgpot_nlist_fortran_plane
      end function rgpot_nlist_fortran_plane
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
      integer(c_int), allocatable, target :: ifirst(:), isecond(:)
      real(c_double), allocatable, target :: cvec(:, :), cdist(:)
      real(wp) :: use_skin
      integer :: p, np, q, nkept
      integer(ip) :: i, j, natoms
      integer(ip), allocatable :: fill(:), by_j(:), jstart(:)
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

      if (.not. c_associated(self%builder)) then
         self%builder = rgpot_nlist_fortran_open()
         if (.not. c_associated(self%builder)) then
            status = 7
            errmsg = "rgpot_neighbors: the neighbour builder could not be allocated"
            return
         end if
      end if

      natoms_c = int(natoms, c_size_t)
      if (self%streaming) then
         rc = rgpot_nlist_fortran_stream(self%builder, positions, natoms_c, cell, &
                                         real(cutoff, c_double), pbc)
         if (rc /= 0_c_int32_t) then
            status = int(rc)
            msg = rgpot_nlist_strerror(rc)
            errmsg = "rgpot_neighbors: "//c_string(msg)
            return
         end if
         self%natoms = natoms
         self%cutoff = cutoff
         if (allocated(self%idx)) deallocate (self%idx, self%vec, self%dist)
         if (allocated(self%row)) deallocate (self%row)
         self%hold_ix = -2
         self%hold_nx = 0
         self%hold_raw%ix = -2
         self%hold_raw%n = 0_ip
         if (allocated(self%hold_home)) self%hold_home = 0_ip
         if (allocated(self%hold_plane)) self%hold_plane = -1
         return
      end if
      needed = 0_c_size_t
      rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                     real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                     c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
                                     0_c_size_t, needed, reused)
      if (rc /= 0_c_int32_t) then
         status = int(rc)
         msg = rgpot_nlist_strerror(rc)
         errmsg = "rgpot_neighbors: "//c_string(msg)
         return
      end if

      np = int(needed)
      if (reused /= 0_c_int32_t .and. allocated(self%idx) .and. allocated(self%held_pos) .and. &
          self%natoms == natoms .and. self%cutoff == cutoff .and. &
          same_geometry(self%held_pos, positions, self%held_cell, cell)) then
         return
      end if
      self%natoms = natoms
      self%cutoff = cutoff
      self%skin = use_skin

      if (np > 0) then
         allocate (ifirst(np), isecond(np), cvec(3, np), cdist(np))
         cap = int(np, c_size_t)
         rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                        real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                        c_loc(ifirst), c_loc(isecond), c_loc(cvec), c_loc(cdist), &
                                        cap, needed, reused)
         if (rc /= 0_c_int32_t) then
            status = int(rc)
            msg = rgpot_nlist_strerror(rc)
            errmsg = "rgpot_neighbors: "//c_string(msg)
            return
         end if
         np = int(needed)
      end if

      if (allocated(self%row)) then
         if (size(self%row) /= natoms + 1) deallocate (self%row)
      end if
      if (.not. allocated(self%row)) allocate (self%row(natoms + 1))
      allocate (fill(natoms), source=0_ip)

      ! Rows ordered by partner index, so a reused candidate list and a
      ! fresh build sum forces in one order.
      allocate (by_j(max(np, 1)), jstart(natoms + 1))
      jstart = 0_ip
      self%row = 0_ip
      do p = 1, np
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         if (i == j) cycle
         self%row(i) = self%row(i) + 1_ip
         jstart(j) = jstart(j) + 1_ip
      end do

      call prefix_sum(self%row)
      call prefix_sum(jstart)

      nkept = int(self%row(natoms + 1) - 1_ip)
      do p = 1, np
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         if (i == j) cycle
         by_j(jstart(j)) = p
         jstart(j) = jstart(j) + 1_ip
      end do

      if (allocated(self%idx)) then
         if (size(self%idx) < self%row(natoms + 1) - 1_ip) then
            deallocate (self%idx, self%vec, self%dist)
         end if
      end if
      if (.not. allocated(self%idx)) then
         allocate (self%idx(max(int(self%row(natoms + 1)) - 1, 1)))
         allocate (self%vec(3, max(int(self%row(natoms + 1)) - 1, 1)))
         allocate (self%dist(max(int(self%row(natoms + 1)) - 1, 1)))
      end if

      do q = 1, nkept
         p = int(by_j(q))
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         associate (slot => self%row(i) + fill(i))
            self%idx(slot) = j
            self%vec(:, slot) = cvec(:, p)
            self%dist(slot) = cdist(p)
         end associate
         fill(i) = fill(i) + 1_ip
      end do
      self%held_pos = positions
      self%held_cell = cell
   end subroutine neighbor_table_build

   !> True when `a` and `b` and the two cells are the same numbers.
   pure function same_geometry(a, b, ca, cb) result(yes)
      real(wp), intent(in) :: a(:, :), b(:, :), ca(3, 3), cb(3, 3)
      logical :: yes

      integer :: i, k

      yes = .false.
      if (size(a, 1) /= size(b, 1) .or. size(a, 2) /= size(b, 2)) return
      do i = 1, size(a, 2)
         do k = 1, size(a, 1)
            if (a(k, i) /= b(k, i)) return
         end do
      end do
      do i = 1, 3
         do k = 1, 3
            if (ca(k, i) /= cb(k, i)) return
         end do
      end do
      yes = .true.
   end function same_geometry

   !> Free the builder; the table can be built again afterwards.
   subroutine neighbor_table_release(self)
      class(neighbor_table_t), intent(inout) :: self

      if (c_associated(self%builder)) call rgpot_nlist_fortran_close(self%builder)
      self%builder = c_null_ptr
      self%cutoff = -1.0_wp
      self%skin = -1.0_wp
   end subroutine neighbor_table_release

   !> Ask the next build to stream (`enabled`) or to store the table.
   subroutine neighbor_table_set_stream(self, enabled)
      class(neighbor_table_t), intent(inout) :: self
      logical, intent(in) :: enabled

      self%streaming = enabled
   end subroutine neighbor_table_set_stream

   !> Stored pair count. Zero while streaming, because no index is kept.
   function neighbor_table_slots(self) result(n)
      class(neighbor_table_t), intent(in) :: self
      integer(lp) :: n

      n = 0_lp
      if (self%streaming .or. .not. allocated(self%row)) return
      if (self%natoms < 1_ip) return
      n = int(self%row(self%natoms + 1_ip) - 1_ip, lp)
   end function neighbor_table_slots

   !> Neighbours of atom `i` (one-based), partner index then geometry.
   !!
   !! A stored table copies its row. A stream copies the row from the
   !! x-plane cache, which a centre slides by one plane at a time.
   subroutine neighbor_load(self, atom, idx, vec, dist, n)
      class(neighbor_table_t), intent(inout) :: self
      integer(ip), intent(in) :: atom
      integer(ip), allocatable, intent(inout), target :: idx(:)
      real(wp), allocatable, intent(inout), target :: vec(:, :)
      real(wp), allocatable, intent(inout), target :: dist(:)
      integer(ip), intent(out) :: n

      integer(ip) :: s, slot, at

      n = 0_ip
      if (atom < 1_ip .or. atom > self%natoms) return

      if (self%streaming) then
         if (self%hold_ix < 0 .or. .not. allocated(self%hold_home)) then
            call neighbor_note_center(self, atom)
         end if
         if (.not. allocated(self%hold_home)) return
         slot = self%hold_home(atom)
         if (slot == 0_ip) then
            if (.not. plane_covered(self%hold_ix, &
                                    int(rgpot_nlist_fortran_plane_of(self%builder, &
                                                                     int(atom - 1_ip, c_int32_t))), &
                                    self%hold_nx)) then
               call neighbor_note_center(self, atom)
            end if
            slot = self%hold_home(atom)
         end if
         if (slot < 1_ip .or. slot > 3_ip) return
         if (.not. associated(self%hold_raw(slot)%off)) return
         n = self%hold_raw(slot)%off(atom + 1_ip) - self%hold_raw(slot)%off(atom)
         call grow_load(idx, vec, dist, n)
         do s = 1_ip, n
            at = self%hold_raw(slot)%off(atom) + s - 1_ip
            idx(s) = self%hold_raw(slot)%nbr(at)
            vec(:, s) = self%hold_raw(slot)%gvec(:, at)
            dist(s) = self%hold_raw(slot)%gdist(at)
         end do
         return
      end if

      if (.not. allocated(self%row)) return
      n = self%row(atom + 1_ip) - self%row(atom)
      call grow_load(idx, vec, dist, n)
      do s = 1_ip, n
         slot = self%row(atom) + s - 1_ip
         idx(s) = self%idx(slot)
         vec(:, s) = self%vec(:, slot)
         dist(s) = self%dist(slot)
      end do
   end subroutine neighbor_load

   subroutine grow_load(idx, vec, dist, n)
      integer(ip), allocatable, intent(inout) :: idx(:)
      real(wp), allocatable, intent(inout) :: vec(:, :)
      real(wp), allocatable, intent(inout) :: dist(:)
      integer(ip), intent(in) :: n

      integer(ip) :: need

      need = max(n, 1_ip)
      if (allocated(idx)) then
         if (size(idx) < need) deallocate (idx, vec, dist)
      end if
      if (.not. allocated(idx)) then
         allocate (idx(need), vec(3, need), dist(need))
      end if
   end subroutine grow_load

   !> Cell-major atom indices when streaming. `norder` is 0 for a stored table,
   !! and the caller then walks atoms `1 .. natoms` in index order.
   subroutine neighbor_visit_order(self, order, norder)
      class(neighbor_table_t), intent(in) :: self
      integer(ip), allocatable, intent(inout) :: order(:)
      integer(ip), intent(out) :: norder

      integer(c_int32_t) :: rc
      integer(c_size_t) :: needed, cap
      integer(ip), allocatable, target :: raw(:)
      integer(ip) :: k

      norder = 0_ip
      if (.not. self%streaming .or. .not. c_associated(self%builder)) return
      needed = 0_c_size_t
      rc = rgpot_nlist_fortran_order(self%builder, c_null_ptr, 0_c_size_t, needed)
      if (rc /= 0_c_int32_t) return
      norder = int(needed, ip)
      if (norder < 1_ip) return
      allocate (raw(norder))
      cap = int(norder, c_size_t)
      rc = rgpot_nlist_fortran_order(self%builder, c_loc(raw), cap, needed)
      if (rc /= 0_c_int32_t) then
         norder = 0_ip
         return
      end if
      if (allocated(order)) then
         if (size(order) < norder) deallocate (order)
      end if
      if (.not. allocated(order)) allocate (order(norder))
      do k = 1_ip, norder
         order(k) = raw(k) + 1_ip
      end do
   end subroutine neighbor_visit_order

   !> Slide the stream window onto the cells around `atom`.
   subroutine neighbor_note_center(self, atom)
      class(neighbor_table_t), intent(inout) :: self
      integer(ip), intent(in) :: atom

      integer(c_int32_t) :: ix

      ix = -1_c_int32_t
      if (.not. self%streaming .or. atom < 1_ip) return
      if (.not. c_associated(self%builder)) return
      if (allocated(self%hold_plane)) then
         if (atom <= size(self%hold_plane)) ix = int(self%hold_plane(atom), c_int32_t)
      end if
      if (ix < 0_c_int32_t) then
         ix = rgpot_nlist_fortran_plane_of(self%builder, int(atom - 1_ip, c_int32_t))
      end if
      if (ix < 0_c_int32_t .or. int(ix) == self%hold_ix) return
      call hold_refill(self, int(ix))
   end subroutine neighbor_note_center

   !> True when `ix` is one of the three planes cached around `center`.
   pure function plane_covered(center, ix, nx) result(yes)
      integer, intent(in) :: center, ix, nx
      logical :: yes

      integer :: d, p

      yes = .false.
      if (center < 0 .or. nx < 1 .or. ix < 0) return
      do d = -1, 1
         p = center + d
         if (p < 0) p = p + nx
         if (p >= nx) p = p - nx
         if (p == ix) yes = .true.
      end do
   end function plane_covered

   !> Pull one x-plane into `dump`. A buffer that is already large enough
   !! is filled in one call; a short one is grown and filled once more.
   subroutine fetch_raw(self, ix, dump)
      class(neighbor_table_t), intent(in) :: self
      integer, intent(in) :: ix
      type(plane_dump), intent(inout) :: dump

      integer(c_int32_t) :: rc
      integer(c_size_t) :: needed, cap
      integer(ip) :: n, room

      dump%ix = -2
      dump%n = 0_ip
      if (.not. associated(dump%ia)) then
         allocate (dump%ia(64), dump%ja(64), dump%vec(3, 64), dump%dist(64))
      end if
      do
         room = int(size(dump%ia), ip)
         cap = int(room, c_size_t)
         needed = 0_c_size_t
         rc = rgpot_nlist_fortran_plane(self%builder, int(ix, c_int32_t), &
                                        c_loc(dump%ia), c_loc(dump%ja), c_loc(dump%vec), &
                                        c_loc(dump%dist), cap, needed)
         n = int(needed, ip)
         if (n < 0_ip) return
         if (n <= room) exit
         deallocate (dump%ia, dump%ja, dump%vec, dump%dist)
         allocate (dump%ia(n), dump%ja(n), dump%vec(3, n), dump%dist(n))
      end do
      if (rc /= 0_c_int32_t) return
      dump%n = n
      dump%ix = ix
      call group_plane(self%natoms, dump)
   end subroutine fetch_raw

   !> One plane's pairs, grouped by the central atom. Done once per plane.
   subroutine group_plane(natoms, dump)
      integer(ip), intent(in) :: natoms
      type(plane_dump), intent(inout) :: dump

      integer(ip), allocatable :: cnt(:)
      integer(ip) :: q, i, base, k, n

      n = dump%n
      dump%nown = 0_ip
      allocate (cnt(natoms), source=0_ip)
      do q = 1_ip, n
         i = dump%ia(q) + 1_ip
         if (i >= 1_ip .and. i <= natoms) cnt(i) = cnt(i) + 1_ip
      end do
      k = 0_ip
      do i = 1_ip, natoms
         if (cnt(i) > 0_ip) k = k + 1_ip
      end do
      if (associated(dump%atoms)) then
         if (size(dump%atoms) < max(k, 1_ip)) deallocate (dump%atoms)
      end if
      if (.not. associated(dump%atoms)) allocate (dump%atoms(max(k, 1_ip)))
      if (associated(dump%off)) then
         if (size(dump%off) /= natoms + 1_ip) deallocate (dump%off)
      end if
      if (.not. associated(dump%off)) allocate (dump%off(natoms + 1_ip))
      dump%off(1) = 1_ip
      base = 0_ip
      do i = 1_ip, natoms
         dump%off(i + 1_ip) = dump%off(i) + cnt(i)
         if (cnt(i) > 0_ip) then
            base = base + 1_ip
            dump%atoms(base) = i
         end if
      end do
      dump%nown = base
      if (associated(dump%nbr)) then
         if (size(dump%nbr) < max(n, 1_ip)) deallocate (dump%nbr, dump%gvec, dump%gdist)
      end if
      if (.not. associated(dump%nbr)) then
         allocate (dump%nbr(max(n, 1_ip)), dump%gvec(3, max(n, 1_ip)), dump%gdist(max(n, 1_ip)))
      end if
      cnt = dump%off(1:natoms)
      do q = 1_ip, n
         i = dump%ia(q) + 1_ip
         if (i < 1_ip .or. i > natoms) cycle
         base = cnt(i)
         dump%nbr(base) = dump%ja(q) + 1_ip
         dump%gvec(:, base) = dump%vec(:, q)
         dump%gdist(base) = dump%dist(q)
         cnt(i) = base + 1_ip
      end do
   end subroutine group_plane

   subroutine drop_home(self, s)
      class(neighbor_table_t), intent(inout) :: self
      integer, intent(in) :: s

      integer(ip) :: k, i

      if (.not. allocated(self%hold_home)) return
      if (.not. associated(self%hold_raw(s)%atoms)) return
      do k = 1_ip, self%hold_raw(s)%nown
         i = self%hold_raw(s)%atoms(k)
         if (i >= 1_ip .and. i <= self%natoms) then
            if (self%hold_home(i) == int(s, ip)) self%hold_home(i) = 0_ip
            if (allocated(self%hold_plane)) self%hold_plane(i) = -1
         end if
      end do
   end subroutine drop_home

   !> Rows for the three x-planes around `ix`, so a three-body term can
   !! read a neighbour without another cell walk. Planes already held are
   !! not fetched again.
   subroutine hold_refill(self, ix)
      class(neighbor_table_t), intent(inout) :: self
      integer, intent(in) :: ix

      integer(c_int32_t) :: cnx, cny, cnz, rc
      integer :: d, p, np, s, t, planes(3), slot_of(3)
      integer(ip) :: i, k
      logical :: keep(3), seen

      rc = rgpot_nlist_fortran_grid(self%builder, cnx, cny, cnz)
      if (rc /= 0_c_int32_t) return
      self%hold_nx = int(cnx)
      np = 0
      do d = -1, 1
         p = ix + d
         if (p < 0) p = p + self%hold_nx
         if (p >= self%hold_nx) p = p - self%hold_nx
         if (p < 0 .or. p >= self%hold_nx) cycle
         seen = .false.
         if (np > 0) seen = any(planes(1:np) == p)
         if (.not. seen) then
            np = np + 1
            planes(np) = p
         end if
      end do

      if (allocated(self%hold_home)) then
         if (size(self%hold_home) /= self%natoms) deallocate (self%hold_home)
      end if
      if (.not. allocated(self%hold_home)) allocate (self%hold_home(self%natoms), source=0_ip)
      if (allocated(self%hold_plane)) then
         if (size(self%hold_plane) /= self%natoms) deallocate (self%hold_plane)
      end if
      if (.not. allocated(self%hold_plane)) allocate (self%hold_plane(self%natoms), source=-1)

      keep = .false.
      slot_of = 0
      do p = 1, np
         do s = 1, 3
            if (self%hold_raw(s)%ix == planes(p)) then
               slot_of(p) = s
               keep(s) = .true.
               exit
            end if
         end do
      end do
      do p = 1, np
         if (slot_of(p) /= 0) cycle
         s = 0
         do t = 1, 3
            if (.not. keep(t)) then
               s = t
               exit
            end if
         end do
         if (s == 0) return
         call drop_home(self, s)
         call fetch_raw(self, planes(p), self%hold_raw(s))
         if (self%hold_raw(s)%ix /= planes(p)) return
         do k = 1_ip, self%hold_raw(s)%nown
            i = self%hold_raw(s)%atoms(k)
            self%hold_home(i) = int(s, ip)
            self%hold_plane(i) = self%hold_raw(s)%ix
         end do
         keep(s) = .true.
         slot_of(p) = s
      end do
      self%hold_ix = ix
   end subroutine hold_refill

   subroutine neighbor_grid(self, nx, ny, nz, stat)
      class(neighbor_table_t), intent(in) :: self
      integer, intent(out) :: nx, ny, nz, stat

      integer(c_int32_t) :: cnx, cny, cnz, rc

      nx = 0
      ny = 0
      nz = 0
      stat = 1
      if (.not. c_associated(self%builder)) return
      rc = rgpot_nlist_fortran_grid(self%builder, cnx, cny, cnz)
      stat = int(rc)
      if (rc /= 0_c_int32_t) return
      nx = int(cnx)
      ny = int(cny)
      nz = int(cnz)
   end subroutine neighbor_grid

   !> Directed pairs whose central atom lies in x-plane `ix`. Indices are
   !! one-based. The same arrays are grown and reused by the caller.
   subroutine neighbor_plane_pairs(self, ix, ia, ja, vec, dist, n, stat)
      class(neighbor_table_t), intent(in) :: self
      integer, intent(in) :: ix
      integer(ip), allocatable, intent(inout), target :: ia(:), ja(:)
      real(wp), allocatable, intent(inout), target :: vec(:, :), dist(:)
      integer(ip), intent(out) :: n
      integer, intent(out) :: stat

      integer(c_int32_t) :: rc
      integer(c_size_t) :: needed, cap
      integer(ip) :: q, room

      n = 0_ip
      stat = 1
      if (.not. c_associated(self%builder)) return
      if (.not. allocated(ia)) allocate (ia(64), ja(64), vec(3, 64), dist(64))
      do
         room = int(size(ia), ip)
         cap = int(room, c_size_t)
         needed = 0_c_size_t
         rc = rgpot_nlist_fortran_plane(self%builder, int(ix, c_int32_t), &
                                        c_loc(ia), c_loc(ja), c_loc(vec), c_loc(dist), &
                                        cap, needed)
         n = int(needed, ip)
         if (n < 0_ip) then
            n = 0_ip
            stat = 1
            return
         end if
         if (n <= room) exit
         deallocate (ia, ja, vec, dist)
         allocate (ia(max(n, 1_ip)), ja(max(n, 1_ip)), vec(3, max(n, 1_ip)), dist(max(n, 1_ip)))
      end do
      stat = int(rc)
      if (rc /= 0_c_int32_t) then
         n = 0_ip
         return
      end if
      do q = 1_ip, n
         ia(q) = ia(q) + 1_ip
         ja(q) = ja(q) + 1_ip
      end do
   end subroutine neighbor_plane_pairs

   !> Number of neighbours atom `i` owns.
   pure function neighbor_table_count_for(self, i) result(n)
      class(neighbor_table_t), intent(in) :: self
      integer(ip), intent(in) :: i
      integer(ip) :: n

      n = self%row(i + 1_ip) - self%row(i)
   end function neighbor_table_count_for

   !> Turn per-atom counts into one-based row starts in place.
   pure subroutine prefix_sum(row)
      integer(ip), intent(inout) :: row(:)

      integer :: k
      integer(ip) :: running, held

      running = 1_ip
      do k = 1, size(row)
         held = row(k)
         row(k) = running
         running = running + held
      end do
   end subroutine prefix_sum

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
