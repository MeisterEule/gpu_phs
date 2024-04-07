program test
  use gpu_phs_whizard_interface
  implicit none
 
  integer, parameter :: n_channels = 1
  integer, parameter :: n_trees = 1
  integer, parameter :: n_groves = 1
  integer, parameter :: n_x = 2
  integer, parameter :: n_in = 2
  integer, parameter :: n_out = 2

  integer, parameter :: n_prt_out = 2**n_out - 1

  integer, dimension(n_prt_out) :: map_ids
  real(kind=8), dimension(n_prt_out) :: masses
  real(kind=8), dimension(n_prt_out) :: widths
  integer :: c, i

  call whizard_set_particle_structure (n_channels, n_trees, n_groves, n_x, n_in, n_out)
  call whizard_init_mappings()
  do c = 1, n_channels
     do i = 1, n_prt_out
        map_ids(i) = 2 * c + 3 * i
        masses(i) = 1.2345 * c + 5.4321 * i
        widths(i) = 5.4321 * c + 1.2345 * i
     end do
     call whizard_fill_mapping (c, map_ids, masses, widths)
  end do
  call whizard_show_module() 
  
end program test
