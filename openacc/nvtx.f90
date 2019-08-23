! Fortran bindings for a small subset of the NVIDIA Tools Extensions library
module nvtx
 use iso_c_binding
 public :: nvtxrangepusha, nvtxrangepop
 public :: nvtxrangepushaargb
 interface
   ! Annotate the timeline with a message
   ! Parameters:
   ! * string : the message in a string format
   subroutine nvtxrangepusha(string) bind(C, name="nvtxRangePushA")
     use iso_c_binding , only : c_char
     character(kind=c_char) :: string(*)
   end subroutine nvtxrangepusha

   ! Annotate the timeline with both a message and an ARGB color
   ! Parameters:
   ! * string : the message in a string format
   ! * argb  : the color in argb format (example: Z'FF880000'
   subroutine nvtxrangepushaargb(string,argb) bind(C, name="_nvtxRangePushAARGB")
     use iso_c_binding , only : c_char, c_int
     character(kind=c_char) :: string(*)
     integer(kind=c_int), value :: argb
   end subroutine nvtxrangepushaargb

   ! Pop the last range off the stack
   subroutine nvtxrangepop() bind(C, name="nvtxRangePop")
   end subroutine

   ! Place a mark on the timeline with a message
   ! Parameters:
   ! * string : the message in a string format
   ! NOT YET EXPOSED
   subroutine nvtxMarkA(string) bind(C, name="nvtxMarkA")
     use iso_c_binding , only : c_char
     character(kind=c_char) :: string(*)
   end subroutine

   ! Name an OS thread
   ! NOT YET EXPOSED
   subroutine nvtxNameOsThread(tid, string) bind(C, name="nvtxNameOsThread")
     use iso_c_binding , only : c_int, c_char
     integer(kind=c_int) :: tid
     character(kind=c_char) :: string(*)
   end subroutine
 end interface
end module nvtx
