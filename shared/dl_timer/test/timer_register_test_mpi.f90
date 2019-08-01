!> Simple program to test the timing API
PROGRAM timer_test
  use mpi
  use dl_timer
  use dl_timer_constants_mod

  integer :: timer0, timer1
  integer :: istep
  integer(i_def64), parameter :: nstep = 1000

  integer numtasks, rank, ierr, j

  real(r_def) :: mysum
  !--------------------------------------------------------------
  ! Initialisation

  ! Initialize the MPI library:
  call MPI_INIT(ierr)
  if (ierr .ne. MPI_SUCCESS) then
     print *,'Error starting MPI program. Terminating.'
     call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if

  ! Get the number of processors this job is using
  call MPI_COMM_SIZE(MPI_COMM_WORLD, numtasks, ierr)

  ! Get the rank of the processor this thread is running on
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

  call timer_init()

  ! Register multiple timers for future use
  call timer_register(timer1, label='Single step')
  call timer_register(timer0, label='Time-stepping', num_repeats=nstep)
  
  mysum = 0.0d0

  ! Artificially create a load imbalance
  nloops = 200*(rank+1)

  !--------------------------------------------------------------
  ! Time-stepping

  call timer_start(timer0)

  do istep = 1, nstep
     call timer_start(timer1)
     do j = 1, nloops
        mysum = mysum + 5.0d0*(istep+j)**3.214_r_def
     end do
     call timer_stop(timer1)
  end do

  call timer_stop(timer0)

  !---------------------------------------------------------------
  ! Finalise

  ! Output the result to prevent the compiler optimising-out the loop
  write (*,*) 'Fake checksum = ', mysum

  call timer_report()

  call mpi_finalize(ierr)
  
END PROGRAM timer_test
