!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Definition of disperson types for DFT calculations
!> \author JGH (20.10.2008)
! **************************************************************************************************
MODULE qs_dispersion_types

   USE eeq_input,                       ONLY: eeq_solver_type
   USE input_section_types,             ONLY: section_vals_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type,&
                                              release_neighbor_list_sets
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_dispersion_types'

! **************************************************************************************************
   INTEGER, PARAMETER                       :: dftd2_pp = 1
   INTEGER, PARAMETER                       :: dftd3_pp = 2
   INTEGER, PARAMETER                       :: dftd4_pp = 3

   TYPE qs_dispersion_type
      INTEGER                                :: TYPE = -1
      INTEGER                                :: pp_type = -1
      INTEGER                                :: nl_type = -1
      CHARACTER(LEN=default_string_length)   :: ref_functional = ""
      REAL(KIND=dp)                          :: scaling = -1.0_dp
      REAL(KIND=dp)                          :: rc_disp = -1.0_dp
      REAL(KIND=dp)                          :: rc_d4 = -1.0_dp
      REAL(KIND=dp)                          :: rc_cn = -1.0_dp
      REAL(KIND=dp)                          :: exp_pre = -1.0_dp
      TYPE(section_vals_type), POINTER       :: dftd_section => NULL()
      LOGICAL                                :: verbose = .FALSE. !extended output
      CHARACTER(LEN=default_string_length)   :: parameter_file_name = ""
      CHARACTER(LEN=default_string_length)   :: kernel_file_name = ""
      !charges
      LOGICAL                                :: ext_charges = .FALSE.
      REAL(KIND=dp), DIMENSION(:), POINTER   :: charges => NULL() !charges for D4
      REAL(KIND=dp), DIMENSION(:), POINTER   :: dcharges => NULL() !derivatives of D4 energy wrt charges
      TYPE(eeq_solver_type)                  :: eeq_sparam ! parameters for EEQ solver
      !DFT-D3 global parameters
      INTEGER                                :: max_elem = -1 !elements parametrized
      INTEGER                                :: maxc = -1 !max coordination number references per element
      REAL(KIND=dp)                          :: k1 = -1.0_dp, k2 = -1.0_dp, k3 = -1.0_dp !ad hoc parameters
      REAL(KIND=dp)                          :: alp = -1.0_dp !ad hoc parameters
      REAL(KIND=dp)                          :: s6 = -1.0_dp, s8 = -1.0_dp, sr6 = -1.0_dp !scaling parameters
      REAL(KIND=dp)                          :: a1 = -1.0_dp, a2 = -1.0_dp !BJ scaling parameters
      REAL(KIND=dp)                          :: eps_cn = -1.0_dp
      LOGICAL                                :: d4_reference_code = .FALSE. !Use D4 calculation from ext. library
      LOGICAL                                :: d4_debug = .FALSE. !Output debug information for D4
      LOGICAL                                :: doabc = .FALSE. !neglect C9 terms
      LOGICAL                                :: c9cnst = .FALSE. !use constant c9 terms
      LOGICAL                                :: lrc = .FALSE. !calculate a long range correction
      LOGICAL                                :: srb = .FALSE. !calculate a short range bond correction
      REAL(KIND=dp), DIMENSION(4)            :: srb_params = -1.0_dp ! parameters for SRB (s,g,t1,t2)
      REAL(KIND=dp)                          :: s9 = -1.0_dp !scale the many-body dispersion energy (default=1.0), dftd4
      TYPE(neighbor_list_set_p_type), &
         DIMENSION(:), POINTER               :: sab_vdw => NULL(), sab_cn => NULL() ! neighborlists for pair interactions
      REAL(KIND=dp), DIMENSION(:, :, :, :, :), POINTER &
         :: c6ab => NULL()
      INTEGER, DIMENSION(:), POINTER         :: maxci => NULL()
      REAL(KIND=dp), DIMENSION(:, :), POINTER :: r0ab => NULL()
      REAL(KIND=dp), DIMENSION(:), POINTER   :: rcov => NULL() !covalent radii
      REAL(KIND=dp), DIMENSION(:), POINTER   :: eneg => NULL() !electronegativity
      REAL(KIND=dp), DIMENSION(:), POINTER   :: r2r4 => NULL() !atomic <r^2>/<r^4> values
      INTEGER                                :: cnfun = 1 ! CN function to be used
      REAL(KIND=dp), DIMENSION(:), POINTER   :: cn => NULL()
      TYPE(cn_kind_list), DIMENSION(:), POINTER  :: cnkind => NULL()
      TYPE(cn_atom_list), DIMENSION(:), POINTER  :: cnlist => NULL()
      ! KG molecular corrections
      LOGICAL                                :: domol = .FALSE.
      REAL(KIND=dp)                          :: kgc8 = -1.0_dp !s8 scaling parameter
      !vdW-DF variables
      REAL(KIND=dp)                          :: pw_cutoff = -1.0_dp
      !parameters for the rVV10 functional
      REAL(KIND=dp)                          :: b_value = -1.0_dp, c_value = -1.0_dp, scale_rvv10 = -1.0_dp
      INTEGER                                :: nqs = -1, nr_points = -1
      !! The number of q points and radial points
      !! used in generating the kernel phi(q1*r, q2*r)
      !! (see DION 14-16 and SOLER 3)
      REAL(KIND=dp)                          :: r_max = -1.0_dp, q_cut = -1.0_dp, q_min = -1.0_dp, dk = -1.0_dp
      !! The maximum value of r, the maximum and minimum
      !! values of q and the k-space spacing of grid points.
      !! Note that, during a vdW run, values of q0 found
      !! larger than q_cut will be saturated (SOLER 6-7) to
      !! q_cut
      REAL(KIND=dp), DIMENSION(:), POINTER    :: q_mesh => NULL() !! The values of all the q points used
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER &
         :: kernel => NULL() !! A matrix holding the Fourier transformed kernel function
      !! for each pair of q values.  The ordering is
      !! kernel(k_point, q1_value, q2_value)
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER &
         :: d2phi_dk2 => NULL() !! A matrix holding the second derivatives of the above
      !! kernel matrix at each of the q points.  Stored as
      !! d2phi_dk2(k_point, q1_value, q2_value)
      REAL(KIND=dp), DIMENSION(:, :), POINTER  :: d2y_dx2 => NULL() !! 2nd derivatives of q_mesh for interpolation
      INTEGER, DIMENSION(:, :), POINTER  :: d3_exclude_pair => NULL()
      INTEGER  :: nd3_exclude_pair = -1
   END TYPE qs_dispersion_type

   TYPE qs_atom_dispersion_type
      INTEGER                                :: TYPE = -1
      LOGICAL                                :: defined = .FALSE.
      REAL(KIND=dp)                          :: vdw_radii = -1.0_dp !van der Waals radii
      REAL(KIND=dp)                          :: c6 = -1.0_dp !c6 coefficients
   END TYPE qs_atom_dispersion_type

   TYPE cn_kind_list
      REAL(KIND=dp)                          :: cnum = -1.0_dp
      INTEGER                                :: kind = -1
   END TYPE cn_kind_list
   TYPE cn_atom_list
      REAL(KIND=dp)                          :: cnum = -1.0_dp
      INTEGER                                :: natom = -1
      INTEGER, DIMENSION(:), POINTER          :: atom => NULL()
   END TYPE cn_atom_list

! **************************************************************************************************

   PUBLIC :: qs_atom_dispersion_type, qs_dispersion_type
   PUBLIC :: dftd2_pp, dftd3_pp, dftd4_pp
   PUBLIC :: qs_dispersion_release

! **************************************************************************************************
CONTAINS
! **************************************************************************************************
!> \brief ...
!> \param dispersion_env ...
! **************************************************************************************************
   SUBROUTINE qs_dispersion_release(dispersion_env)

      TYPE(qs_dispersion_type), POINTER                  :: dispersion_env

      INTEGER                                            :: i

      IF (ASSOCIATED(dispersion_env)) THEN
         ! DFT-D3 arrays
         IF (ASSOCIATED(dispersion_env%maxci)) THEN
            DEALLOCATE (dispersion_env%maxci)
         END IF
         IF (ASSOCIATED(dispersion_env%c6ab)) THEN
            DEALLOCATE (dispersion_env%c6ab)
         END IF
         IF (ASSOCIATED(dispersion_env%r0ab)) THEN
            DEALLOCATE (dispersion_env%r0ab)
         END IF
         IF (ASSOCIATED(dispersion_env%rcov)) THEN
            DEALLOCATE (dispersion_env%rcov)
         END IF
         IF (ASSOCIATED(dispersion_env%eneg)) THEN
            DEALLOCATE (dispersion_env%eneg)
         END IF
         IF (ASSOCIATED(dispersion_env%r2r4)) THEN
            DEALLOCATE (dispersion_env%r2r4)
         END IF
         IF (ASSOCIATED(dispersion_env%cn)) THEN
            DEALLOCATE (dispersion_env%cn)
         END IF
         IF (ASSOCIATED(dispersion_env%cnkind)) THEN
            DEALLOCATE (dispersion_env%cnkind)
         END IF
         IF (ASSOCIATED(dispersion_env%cnlist)) THEN
            DO i = 1, SIZE(dispersion_env%cnlist)
               DEALLOCATE (dispersion_env%cnlist(i)%atom)
            END DO
            DEALLOCATE (dispersion_env%cnlist)
         END IF
         ! vdD-DF
         IF (ASSOCIATED(dispersion_env%q_mesh)) THEN
            DEALLOCATE (dispersion_env%q_mesh)
         END IF
         IF (ASSOCIATED(dispersion_env%kernel)) THEN
            DEALLOCATE (dispersion_env%kernel)
         END IF
         IF (ASSOCIATED(dispersion_env%d2phi_dk2)) THEN
            DEALLOCATE (dispersion_env%d2phi_dk2)
         END IF
         IF (ASSOCIATED(dispersion_env%d2y_dx2)) THEN
            DEALLOCATE (dispersion_env%d2y_dx2)
         END IF
         IF (ASSOCIATED(dispersion_env%d3_exclude_pair)) THEN
            DEALLOCATE (dispersion_env%d3_exclude_pair)
         END IF
         ! neighborlists
         CALL release_neighbor_list_sets(dispersion_env%sab_vdw)
         CALL release_neighbor_list_sets(dispersion_env%sab_cn)
         ! charges
         IF (ASSOCIATED(dispersion_env%charges)) THEN
            DEALLOCATE (dispersion_env%charges)
         END IF
         IF (ASSOCIATED(dispersion_env%dcharges)) THEN
            DEALLOCATE (dispersion_env%dcharges)
         END IF

         DEALLOCATE (dispersion_env)

      END IF

   END SUBROUTINE qs_dispersion_release

END MODULE qs_dispersion_types

