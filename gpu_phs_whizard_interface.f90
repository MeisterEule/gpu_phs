module gpu_phs_whizard_interface
   use iso_c_binding
   interface
      subroutine whizard_set_particle_structure (n_channels, n_trees, n_groves, n_x, n_in, n_out) &
         bind (C, name='c_whizard_set_particle_structure')
         import c_int
         integer(kind=c_int), intent(in) :: n_channels, n_trees, n_groves, n_x, n_in, n_out
      end subroutine whizard_set_particle_structure

      subroutine whizard_init_mappings (n_channels) &
         bind (C, name='c_whizard_init_mappings') 
         import c_int
         integer(kind=c_int), intent(in) :: n_channels
      end subroutine whizard_init_mappings

      subroutine whizard_fill_mapping (channel, map_ids, masses, widths) &
         bind (C, name='c_whizard_fill_mapping')
         import c_int, c_double
         integer(kind=c_int), intent(in) :: channel
         integer(kind=c_int), dimension(*), intent(in) :: map_ids
         real(kind=c_double), dimension(*), intent(in) :: masses
         real(kind=c_double), dimension(*), intent(in) :: widths
      end subroutine whizard_fill_mapping
   end interface
end module gpu_phs_whizard_interface
