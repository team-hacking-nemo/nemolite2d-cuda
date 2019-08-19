!> Module holding basic KIND parameters
MODULE kind_params_mod
  use, intrinsic::iso_c_binding, only : c_float, c_double
  IMPLICIT none

  PUBLIC

  !> Douple precision kind parameter
  INTEGER, PARAMETER :: wp = c_float

END MODULE kind_params_mod
