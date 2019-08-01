!> Simple program to test the timing API with a program
!! parallelised using OpenMP threading.
PROGRAM timer_test
  use dl_timer
  use dl_timer_constants_mod
!$ use omp_lib
  implicit none

  integer :: time0, time1
  integer :: istep, j
  integer(i_def64), parameter :: nstep = 1000
  integer :: nloops, nloops_private
  integer :: myid = 1

  real(r_def) :: mysum
  !--------------------------------------------------------------
  ! Initialisation

  call timer_init()
  mysum = 0.0d0
  nloops = 100

  !--------------------------------------------------------------
  ! Time-stepping

  call timer_start(time0, label='Time-stepping', num_repeats=nstep)

  !$omp parallel default(none) private(istep, mysum, j, time1, myid, nloops_private) &
  !$omp          shared(nloops)
  !$ myid = omp_get_thread_num()
  nloops_private = nloops * (myid+1)
  do istep = 1, nstep

     call timer_start(time1, label='Fake section')
     !$omp do
     do j = 1, nloops_private
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
