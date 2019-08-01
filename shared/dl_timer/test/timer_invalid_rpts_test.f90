!> Simple program to test the timing API
PROGRAM timer_test
  use dl_timer
  use dl_timer_constants_mod

  integer :: time0
  integer :: istep
  integer, parameter :: nstep = 1000

  real(r_def) :: mysum
  !--------------------------------------------------------------
  ! Initialisation

  call timer_init()
  mysum = 0.0d0

  ! Attempt to register a timer with an invalid number of implicit
  ! repeats
  call timer_start(time0, label='Time-stepping', num_repeats=0_i_def64)

  do istep = 1, nstep
     mysum = mysum + sqrt(5.0d0*istep*istep)
  end do

  call timer_stop(time0)

  !---------------------------------------------------------------
  ! Finalise

  ! Output the result to prevent the compiler optimising-out the loop
  write (*,*) 'Fake checksum = ', mysum

  call timer_report()

END PROGRAM timer_test
