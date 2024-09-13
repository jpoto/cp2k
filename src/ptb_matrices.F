!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2024 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Compute pTB matrices
!> Reference: S. Grimme, M. Müller, and A. Hansen, 
!>            J. Chem. Phys., vol. 158, no. 12, p. 124111, (2023)
!>            DOI: 10.1063/5.0137838.
!> \author JVP (09.2024)
! *************************************************************************************************
MODULE ptb_matrices

   USE message_passing,                 ONLY: mp_para_env_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE eeq_charge_model,                ONLY: eeq_charges
   USE kinds,                           ONLY: dp

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_matrices'

! *** Public data types ***

   PUBLIC :: build_ptb_matrices

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param para_env ...
!> \param calculate_forces ...
! **************************************************************************************************
   SUBROUTINE build_ptb_matrices(qs_env, para_env, cn_cutoff)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mp_para_env_type), POINTER                    :: para_env
      REAL(KIND=dp), INTENT(IN)                          :: cn_cutoff

      REAL(KIND=dp), DIMENSION(:), ALLOCATABLE           :: charges
      REAL(KIND=dp)                                      :: enshift_type
      INTEGER                                            :: handle

      CHARACTER(LEN=*), PARAMETER :: routineN = 'build_ptb_matrices'

      CALL timeset(routineN, handle)

      !getting partial charges
      CALL eeq_charges(qs_env, charges, enshift_type, cn_cutoff)

      

      MARK_USED(para_env)

      CALL timestop(handle)

   END SUBROUTINE build_ptb_matrices

END MODULE ptb_matrices