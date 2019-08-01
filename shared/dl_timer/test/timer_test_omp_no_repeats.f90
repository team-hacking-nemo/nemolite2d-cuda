!> Simple program to test the timing API
PROGRAM timer_test
  use dl_timer
  implicit none

  integer, parameter :: r_def = KIND(1.0d0)

  integer :: time0, time1
  integer :: istep, j
  integer, parameter :: nstep = 1000
  integer :: nloops

  real(r_def) :: mysum
  !--------------------------------------------------------------
  ! Initialisation

  call timer_init()
  mysum = 0.0d0
  nloops = 200

  !--------------------------------------------------------------
  ! Time-stepping

  call timer_start(time0, label='Time-stepping')

  !$omp parallel default(none) private(istep, mysum, j, time1) shared(nloops)
  do istep = 1, nstep

     call timer_start(time1, label='Fake section')
     !$omp do
     do j = 1, nloops
        mysum = mysum + sqrt(5.0d0*istep*istep)
     end do
     !$omp end do nowait
     call timer_stop(time1)
     !$omp barrier

  end do
  !$omp end parallel

  call timer_stop(time0)

  !---------------------------------------------------------------
  ! Finalise

  ! Output the result to prevent the compiler optimising-out the loop
  write (*,*) 'Fake checksum = ', mysum

  call timer_report()

END PROGRAM timer_test
