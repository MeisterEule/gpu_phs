! WHIZARD 3.1.4.1 Nov 08 2023
!
! Copyright (C) 1999-2024 by
!     Wolfgang Kilian <kilian@physik.uni-siegen.de>
!     Thorsten Ohl <ohl@physik.uni-wuerzburg.de>
!     Juergen Reuter <juergen.reuter@desy.de>
!
!     with contributions from
!     cf. main AUTHORS file
!
! WHIZARD is free software; you can redistribute it and/or modify it
! under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2, or (at your option)
! any later version.
!
! WHIZARD is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program; if not, write to the Free Software
! Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This file has been stripped of most comments.  For documentation, refer
! to the source 'whizard.nw'

module gpu_phs_whizard_interface
  use iso_c_binding !NODEP!
  implicit none
  interface
      subroutine whizard_set_particle_structure (n_channels, n_trees, n_groves, n_x, n_in, n_out) &
         bind (C, name='c_whizard_set_particle_structure')
         import c_int
         integer(kind=c_int), intent(in) :: n_channels, n_trees, n_groves, n_x, n_in, n_out
      end subroutine whizard_set_particle_structure

      subroutine whizard_set_flavors (masses, widths) &
         bind (C, name='c_whizard_set_flavors')
         import c_double
         real(kind=c_double), intent(in), dimension(*) :: masses
         real(kind=c_double), intent(in), dimension(*) :: widths
      end subroutine whizard_set_flavors

      subroutine whizard_init_mappings () &
         bind (C, name='c_whizard_init_mappings') 
      end subroutine whizard_init_mappings
      subroutine whizard_fill_mapping (channel, map_ids, masses, widths) &
         bind (C, name='c_whizard_fill_mapping')
         import c_int, c_double
         integer(kind=c_int), intent(in) :: channel
         integer(kind=c_int), dimension(*), intent(in) :: map_ids
         real(kind=c_double), dimension(*), intent(in) :: masses
         real(kind=c_double), dimension(*), intent(in) :: widths
      end subroutine whizard_fill_mapping
      subroutine whizard_init_tree_structures () &
         bind (C, name='c_whizard_init_tree_structures')
      end subroutine whizard_init_tree_structures

      subroutine whizard_fill_tree_structure (channel, daughters1, daughters2, has_children, contains_friends) &
         bind (C, name='c_whizard_fill_tree_structure')
         import c_int
         integer, intent(in) :: channel
         integer, intent(in), dimension(*) :: daughters1
         integer, intent(in), dimension(*) :: daughters2
         integer, intent(in), dimension(*) :: has_children
         integer, intent(in) :: contains_friends
      end subroutine whizard_fill_tree_structure
      subroutine whizard_show_module () &
         bind(C, name='c_whizard_show_module')
      end subroutine whizard_show_module
      subroutine whizard_init_channel_ids (batch_size, n_channels) &
         bind(C, name='c_whizard_init_channel_ids')
         import c_int
         integer(kind=c_int), intent(in) :: batch_size
         integer(kind=c_int), intent(in) :: n_channels
      end subroutine whizard_init_channel_ids

     subroutine whizard_set_threads (msq_threads, cb_threads, ab_threads) &
        bind(C, name='c_whizard_set_threads')
        import c_int
        integer(kind=c_int), intent(in) :: msq_threads
        integer(kind=c_int), intent(in) :: cb_threads
        integer(kind=c_int), intent(in) :: ab_threads
     end subroutine whizard_set_threads

    subroutine whizard_init_gpu_phs (sqrts) &
        bind(C, name='c_whizard_init_gpu_phs')
        import c_double
        real(kind=c_double), intent(in) :: sqrts
    end subroutine whizard_init_gpu_phs

    subroutine whizard_gen_phs_from_x_gpu (n_events, n_channels, n_x, x, &
                                           factors, volumes, oks, p) &
       bind(C, name='c_whizard_gen_phs_from_x_gpu')
       import c_int, c_bool, c_double
       integer(kind=c_int), intent(in) :: n_events
       integer(kind=c_int), intent(in) :: n_channels
       integer(kind=c_int), intent(in) :: n_x
       real(kind=c_double), dimension(*), intent(in) :: x
       real(kind=c_double), dimension(*), intent(out) :: factors
       real(kind=c_double), dimension(*), intent(inout) :: volumes
       logical(kind=c_bool), dimension(*), intent(out) :: oks
       real(kind=c_double), dimension(*), intent(inout) :: p
    end subroutine whizard_gen_phs_from_x_gpu 

  end interface
end module gpu_phs_whizard_interface

