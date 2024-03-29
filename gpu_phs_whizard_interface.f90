module gpu_phs_whizard_interface
   use iso_c_binding
   interface
      subroutine whizard_set_particle_structure (n_channels, n_trees, n_groves, n_x, n_in, n_out) &
         bind (C, name='c_whizard_set_particle_structure')
         import c_int
         integer(kind=c_int), intent(in) :: n_channels, n_trees, n_groves, n_x, n_in, n_out
      end subroutine whizard_set_particle_structure
      !!!subroutine whizard_init_phs (n_channels, channels, 
      !!!subroutine whizard_gen_phs_all (sqrts, n_channels, channels, &
      !!!           x, factors, volumes, oks, momenta) bind (C, name='c_whizard_gen_phs_all')
      !!!   import c_double, c_int, c_bool
      !!!   integer(kind=c_int), intent(in) :: n_channels
      !!!   integer(kind=c_int), dimension(*), intent(in) :: channels
      !!!   real(kind=c_double), intent(in) :: sqrts
      !!!   real(kind=c_double), dimension(*), intent(in) :: x
      !!!   real(kind=c_double), dimension(*), intent(out) :: factors
      !!!   real(kind=c_double), dimension(*), intent(out) :: volumes
      !!!   integer(kind=c_bool), dimension(*), intent(out) :: oks 
      !!!   real(kind=c_double), dimension(*), intent(out) :: momenta
      !!!end subroutine whizard_gen_phs_all
   end interface
end module gpu_phs_whizard_interface
