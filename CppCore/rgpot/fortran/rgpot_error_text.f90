! MIT License
! Copyright 2023--present rgpot developers

!> Fixed-storage construction of error messages.
!!
!! Appending into a caller-local buffer avoids deferred-length character
!! concatenation temporaries on concurrent error paths.
module rgpot_error_text
   use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_f_pointer, &
                                          c_null_char, c_ptr
   use rgpot_kinds, only: ip
   implicit none
   private

   public :: error_text_t

   integer, parameter :: max_error_len = 512

   type :: error_text_t
      character(len=max_error_len), private :: buffer = ""
      integer, private :: used = 0
   contains
      procedure :: append => error_text_append
      procedure :: append_c => error_text_append_c
      procedure :: append_int => error_text_append_int
      procedure :: finish => error_text_finish
   end type error_text_t

contains

   !> Append as much of `text` as fits.
   subroutine error_text_append(self, text)
      class(error_text_t), intent(inout) :: self
      character(len=*), intent(in) :: text

      integer :: count

      count = min(len(text), max_error_len - self%used)
      if (count <= 0) return
      self%buffer(self%used + 1:self%used + count) = text(1:count)
      self%used = self%used + count
   end subroutine error_text_append

   !> Append a NUL-terminated C string, or a fallback for a null pointer.
   subroutine error_text_append_c(self, ptr)
      class(error_text_t), intent(inout) :: self
      type(c_ptr), intent(in) :: ptr

      character(kind=c_char), pointer :: raw(:)
      integer :: k, room

      if (.not. c_associated(ptr)) then
         call self%append("unknown neighbour error")
         return
      end if

      room = max_error_len - self%used
      if (room <= 0) return
      call c_f_pointer(ptr, raw, [room])
      do k = 1, room
         if (raw(k) == c_null_char) exit
         self%used = self%used + 1
         self%buffer(self%used:self%used) = raw(k)
      end do
   end subroutine error_text_append_c

   !> Append the decimal representation of an integer.
   subroutine error_text_append_int(self, value)
      class(error_text_t), intent(inout) :: self
      integer(ip), intent(in) :: value

      character(len=32) :: digits
      integer :: count

      write (digits, '(i0)') value
      count = len_trim(digits)
      call self%append(digits(1:count))
   end subroutine error_text_append_int

   !> Move the accumulated text into an allocatable result.
   subroutine error_text_finish(self, text)
      class(error_text_t), intent(in) :: self
      character(len=:), allocatable, intent(out) :: text

      if (self%used > 0) then
         text = self%buffer(1:self%used)
      else
         text = ""
      end if
   end subroutine error_text_finish

end module rgpot_error_text
