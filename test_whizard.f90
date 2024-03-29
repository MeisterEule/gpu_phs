program test
  use gpu_phs_whizard_interface
  implicit none
 
  integer, parameter :: n_channels = 1
  integer, parameter :: n_trees = 2
  integer, parameter :: n_groves = 3
  integer, parameter :: n_x = 4
  integer, parameter :: n_in = 5
  integer, parameter :: n_out = 5

  call whizard_set_particle_structure (n_channels, n_trees, n_groves, n_x, n_in, n_out)
  
  
end program test
