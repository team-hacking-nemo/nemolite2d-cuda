!> Simple program to test the timing API
PROGRAM timer_test
  use dl_timer
  use dl_timer_constants_mod

  integer :: timer0, timer1
  integer :: istep
  integer(i_def64), parameter :: nstep = 1000

  real(r_def) :: mysum
  !--------------------------------------------------------------
  ! Initialisation

  call timer_init()

  ! Example of registering a timer for future use
  call timer_register(timer1, label='Single step')

  mysum = 0.0d0

  !--------------------------------------------------------------
  ! Time-stepping

  call timer_start(timer0, label='Time-stepping', num_repeats=nstep)

  do istep = 1, nstep
     call timer_start(timer1)
     mysum = mysum + sqrt(5.0d0*istep*istep)
     call timer_stop(timer1)
  end do

  call timer_stop(timer0)

  !---------------------------------------------------------------
  ! Finalise

  ! Output the result to prevent the compiler optimising-out the loop
  write (*,*) 'Fake checksum = ', mysum

  call timer_report()

END PROGRAM timer_test
