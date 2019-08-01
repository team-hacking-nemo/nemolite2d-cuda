module timer_test_utils_mod
  !> Module holding utility routines to aid in testing of dl_timer
  use iso_c_binding
  implicit none

  interface
     function fsleep(nsecs) bind(c)
       import :: C_INT
       integer(C_INT) :: nsecs, fsleep
     end function fsleep
  end interface

end module timer_test_utils_mod
