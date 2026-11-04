! MIT License
! Copyright 2023--present rgpot developers

!> Full neighbour tables in CSR form, built with vesin.
!!
!! Every ported potential evaluates forces as a *gather*: atom `i` sums
!! the whole force acting on it from its own neighbours, and writes only
!! `f(:, i)`. Iterations of the atom loop then touch disjoint memory, so
!! they run under `do concurrent`. That needs a **full** list (both `i->j`
!! and `j->i`), which this type provides along with the pair vectors and
!! distances vesin already computed, so the kernels never redo minimum
!! image arithmetic.
!!
!! Cell convention: C callers hand over nine doubles, row-major, one cell
!! vector per row. Fortran reads that same memory column-major, so a
!! `real(wp) :: cell(3, 3)` dummy sees one cell vector per *column*, which
!! is exactly what vesin expects. No transpose anywhere.
module rgpot_neighbors
   use rgpot_kinds, only: wp, ip, lp
   use vesin, only: NeighborList
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
      !> Verlet skin the calculator caches with.
      real(wp), private :: skin = -1.0_wp
      !> Calculator kept across calls so vesin reuses its buffers.
      type(NeighborList), private :: nl
      logical, private :: nl_ready = .false.
   contains
      procedure :: build => neighbor_table_build
      procedure :: count_for => neighbor_table_count_for
      procedure :: release => neighbor_table_release
   end type neighbor_table_t

contains

   !> Rebuild the table for `positions` (3 x natoms) within `cutoff`.
   !!
   !! `periodic` defaults to fully periodic. `status` is zero on success;
   !! on failure the table is left empty and `errmsg` describes the fault.
   subroutine neighbor_table_build(self, positions, cell, cutoff, status, &
                                   errmsg, periodic, skin)
      class(neighbor_table_t), intent(inout) :: self
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      real(wp), intent(in) :: cutoff
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg
      logical, intent(in), optional :: periodic(3)
      !> Verlet skin handed to vesin, which then caches its topology
      !! between calls and rebuilds only once an atom has moved more than
      !! half of it. Defaults to a tenth of the cutoff.
      real(wp), intent(in), optional :: skin

      logical :: pbc(3)
      real(wp) :: use_skin
      integer :: vstatus, p, np, q, nkept
      integer(ip) :: i, j, natoms
      integer(ip), allocatable :: fill(:), by_j(:), jstart(:)

      status = 0
      errmsg = ""
      natoms = int(size(positions, 2), ip)

      if (size(positions, 1) /= 3) then
         status = 1
         errmsg = "rgpot_neighbors: positions must be shaped [3, natoms]"
         return
      end if

      pbc = .true.
      if (present(periodic)) pbc = periodic

      use_skin = 0.1_wp*cutoff
      if (present(skin)) use_skin = skin

      ! Exact comparison on purpose: the question is whether the caller
      ! passed the same numbers as last time, not whether the geometry is
      ! close, so a tolerance would keep a calculator built for a different
      ! cutoff.
      if (.not. self%nl_ready .or. self%cutoff /= cutoff &
          .or. self%skin /= use_skin) then
         ! Release the previous calculator's C buffers before replacing it;
         ! within one cutoff the calculator is reused so vesin recycles them.
         if (self%nl_ready) call self%nl%free()
         ! Unsorted: the CSR passes below order every row by partner index
         ! in linear time, which vesin's global sort would do in n log n.
         ! F2, F8: pin vesin to one thread. The default (zero) follows
         ! OMP_NUM_THREADS and a process-global pool, so the candidate
         ! list would depend on how many threads the process has.
         self%nl = NeighborList(cutoff=cutoff, full=.true., sorted=.false., &
                                skin=use_skin, n_threads=1, &
                                return_distances=.true., return_vectors=.true.)
         self%nl_ready = .true.
         self%cutoff = cutoff
         self%skin = use_skin
      end if

      call self%nl%compute(positions, cell, periodic=pbc, status=vstatus)
      if (vstatus /= 0) then
         status = vstatus
         errmsg = "rgpot_neighbors: vesin build failed: "//self%nl%errmsg
         return
      end if

      np = int(self%nl%length)
      self%natoms = natoms

      if (allocated(self%row)) then
         if (size(self%row) /= natoms + 1) deallocate (self%row)
      end if
      if (.not. allocated(self%row)) allocate (self%row(natoms + 1))
      allocate (fill(natoms), source=0_ip)

      ! Two stable counting sorts give rows ordered by partner: first the
      ! pairs by j, then that order scattered by i. Each row then lists its
      ! partners in ascending j whatever order vesin produced (a cached
      ! topology or a rebuild), so the force sums run in one order for a
      ! given geometry. Self-image pairs carry no direction usable by a
      ! gather kernel and are dropped.
      allocate (by_j(np), jstart(natoms + 1))
      jstart = 0_ip
      self%row = 0_ip
      do p = 1, np
         i = int(self%nl%pairs(1, p), ip) + 1_ip
         j = int(self%nl%pairs(2, p), ip) + 1_ip
         if (i == j) cycle
         self%row(i) = self%row(i) + 1_ip
         jstart(j) = jstart(j) + 1_ip
      end do

      call prefix_sum(self%row)
      call prefix_sum(jstart)

      nkept = int(self%row(natoms + 1) - 1_ip)
      do p = 1, np
         i = int(self%nl%pairs(1, p), ip) + 1_ip
         j = int(self%nl%pairs(2, p), ip) + 1_ip
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

      ! Scatter each pair, taken in ascending j, into its owner's slice.
      do q = 1, nkept
         p = int(by_j(q))
         i = int(self%nl%pairs(1, p), ip) + 1_ip
         j = int(self%nl%pairs(2, p), ip) + 1_ip
         associate (slot => self%row(i) + fill(i))
            self%idx(slot) = j
            self%vec(:, slot) = self%nl%vectors(:, p)
            self%dist(slot) = self%nl%distances(p)
         end associate
         fill(i) = fill(i) + 1_ip
      end do
   end subroutine neighbor_table_build

   !> Free vesin's buffers; the table can be built again afterwards.
   subroutine neighbor_table_release(self)
      class(neighbor_table_t), intent(inout) :: self

      if (self%nl_ready) call self%nl%free()
      self%nl_ready = .false.
      self%cutoff = -1.0_wp
      self%skin = -1.0_wp
   end subroutine neighbor_table_release

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

end module rgpot_neighbors
