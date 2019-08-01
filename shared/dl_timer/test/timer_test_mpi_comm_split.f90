!> Simple MPI program to test the timing API when not all ranks
!! enter a given timed region
PROGRAM timer_test
  use dl_timer_constants_mod
  use dl_timer
  use mpi
  implicit none

  integer :: time0
  integer(i_def64), parameter :: nstep = 5

  real(r_def) :: mysum

  integer numtasks, rank, ierr, j

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

  ! Initialise the timer library (must be done *after* MPI has been
  ! initialised)
  call timer_init()

  mysum = 0.0d0

  !--------------------------------------------------------------
  ! Time-stepping

  call timer_start(time0, label='Time-stepping', num_repeats=nstep)

  ! Don't do time-stepping on odd ranks so that not all PEs enter the
  ! 'fake section' region
  if (mod(rank,2) == 0) call step()

  call timer_stop(time0)

  !---------------------------------------------------------------
  ! Finalise

  ! Output the result to prevent the compiler optimising-out the loop
  write (*,*) 'Fake checksum = ', mysum

  call timer_report()

  ! Tell the MPI library to release all resources it is using:
  call MPI_FINALIZE(ierr)

contains

  !--------------------------------------------------------------
  subroutine step()
    use timer_test_utils_mod, only: fsleep
    implicit none
    ! Locals
    integer :: istep, time1, nloops, ret
    
    ! Artificially create a load imbalance
    nloops = 200*(rank+1)

    do istep = 1, nstep

       call timer_start(time1, label='Fake section')
       do j = 1, nloops
          mysum = mysum + sqrt(5.0d0*istep*istep)
       end do
       ret = fsleep(1)
       call timer_stop(time1)

    end do

  end subroutine step
  
END PROGRAM timer_test
