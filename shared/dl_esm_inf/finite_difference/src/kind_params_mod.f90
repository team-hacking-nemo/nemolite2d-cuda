!> Module holding basic KIND parameters
MODULE kind_params_mod
  use, intrinsic::iso_c_binding, only : c_float, c_double
  IMPLICIT none

  PUBLIC
  INTEGER, PARAMETER :: sp = c_float
  INTEGER, PARAMETER :: dp = c_double
  INTEGER, PARAMETER :: wp = dp

END MODULE kind_params_mod
