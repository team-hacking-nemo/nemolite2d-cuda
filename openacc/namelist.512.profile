!-----------------------------------------------------------------------
&namctl     !   parameters for a example setup
!-----------------------------------------------------------------------
   jpiglo      =     514               !  number of columns of model grid
   jpjglo      =     514               !  number of rows of model grid
   jphgr_msh   =       1               !  type of grid (0: read in a data file; 1: setup with following parameters)
   dx          =    1000.0             !  grid size in x direction (m)
   dy          =    1000.0             !  grid size in y direction (m)
   dep_const   =     100.0             !  constant depth (m)
   nit000      =       1               !  first time step
   nitend      =      10               !43200               !  end time step
   irecord     =      10               ! 180               !  intervals to save results
   rdt         =      20.0             !  size of time step (second) 
   cbfr        =       0.00015          !  bottom friction coefficeint
   visc        =       0.1             !  horizontal kinematic viscosity coefficient 
/
