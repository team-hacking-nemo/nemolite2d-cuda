!> Simple program to test the granularity/overhead of the timing API
!! This code should be run pinned to a single core and for a duration
!! sufficient to ensure the OS has migrated other processes off the
!! chosen core.
PROGRAM timer_test_granularity
  use dl_timer

  integer :: time0
  integer :: istep
  integer, parameter :: nstep = 50000

  !--------------------------------------------------------------
  ! Initialisation

  call timer_init()

  !--------------------------------------------------------------
  ! Time-stepping

  do istep = 1, nstep
     call timer_start(time0, label='Empty time-step')
     ! Absolutely no work here!
     call timer_stop(time0)
  end do

  !---------------------------------------------------------------
  ! Finalise

  call timer_report()

END PROGRAM timer_test_granularity
