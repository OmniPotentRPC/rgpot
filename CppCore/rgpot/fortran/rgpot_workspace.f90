! MIT License
! Copyright 2023--present rgpot developers

!> Per-caller state of the Fortran potentials: one neighbour table and one
!! error message, behind an opaque C pointer.
!!
!! Each C++ potential instance owns one workspace, so two instances keep
!! separate vesin tables (their Verlet skins track their own geometries)
!! and can evaluate on separate threads at once. The `_ws` entry points of
!! the kernels take a workspace; the older entry points without it share
!! one module-level workspace and stay process-serial.
module rgpot_workspace
   use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, &
                                          c_associated, c_null_ptr, c_char, &
                                          c_int, c_null_char
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: rgpot_workspace_t, workspace_from_c, ws_set_error, ws_clear_error
   public :: rgpot_fortran_workspace_new, rgpot_fortran_workspace_free
   public :: rgpot_fortran_workspace_error

   integer, parameter :: max_len = 512

   type :: rgpot_workspace_t
      !> Neighbour table the kernel rebuilds or reuses for this caller.
      type(neighbor_table_t) :: table
      !> Last failure message of this caller.
      character(len=max_len) :: message = ""
      integer :: message_len = 0
   end type rgpot_workspace_t

contains

   !> A fresh workspace, or NULL when allocation fails.
   function rgpot_fortran_workspace_new() result(handle) &
      bind(c, name="rgpot_fortran_workspace_new")
      type(c_ptr) :: handle

      type(rgpot_workspace_t), pointer :: ws
      integer :: stat

      handle = c_null_ptr
      allocate (ws, stat=stat)
      if (stat /= 0) return
      handle = c_loc(ws)
   end function rgpot_fortran_workspace_new

   !> Release a workspace from rgpot_fortran_workspace_new. NULL is a no-op.
   subroutine rgpot_fortran_workspace_free(handle) &
      bind(c, name="rgpot_fortran_workspace_free")
      type(c_ptr), value :: handle

      type(rgpot_workspace_t), pointer :: ws

      if (.not. c_associated(handle)) return
      call c_f_pointer(handle, ws)
      call ws%table%release()
      deallocate (ws)
   end subroutine rgpot_fortran_workspace_free

   !> Copy the workspace's last message into `buffer` as a NUL-terminated C
   !! string; returns the characters written, excluding the terminator.
   function rgpot_fortran_workspace_error(handle, buffer, buffer_len) &
      result(written) bind(c, name="rgpot_fortran_workspace_error")
      type(c_ptr), value :: handle
      character(kind=c_char), intent(out) :: buffer(*)
      integer(c_int), value, intent(in) :: buffer_len
      integer(c_int) :: written

      type(rgpot_workspace_t), pointer :: ws
      integer :: k, room

      written = 0_c_int
      if (buffer_len > 0_c_int) buffer(1) = c_null_char
      if (.not. c_associated(handle)) return
      call c_f_pointer(handle, ws)
      room = int(buffer_len) - 1
      written = int(min(ws%message_len, max(room, 0)), c_int)
      do k = 1, int(written)
         buffer(k) = ws%message(k:k)
      end do
      if (buffer_len > 0_c_int) buffer(int(written) + 1) = c_null_char
   end function rgpot_fortran_workspace_error

   !> The workspace behind a C handle (unassociated for NULL).
   function workspace_from_c(handle) result(ws)
      type(c_ptr), intent(in) :: handle
      type(rgpot_workspace_t), pointer :: ws

      nullify (ws)
      if (c_associated(handle)) call c_f_pointer(handle, ws)
   end function workspace_from_c

   subroutine ws_set_error(ws, text)
      type(rgpot_workspace_t), intent(inout) :: ws
      character(len=*), intent(in) :: text

      ws%message_len = min(len_trim(text), max_len)
      ws%message = text(1:ws%message_len)
   end subroutine ws_set_error

   subroutine ws_clear_error(ws)
      type(rgpot_workspace_t), intent(inout) :: ws

      ws%message = ""
      ws%message_len = 0
   end subroutine ws_clear_error

end module rgpot_workspace
