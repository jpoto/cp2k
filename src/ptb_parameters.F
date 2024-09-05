!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2024 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Read pTB parameters.
!> \author JVP (08.2024)
! **************************************************************************************************
MODULE ptb_parameters

   USE message_passing,                 ONLY: mp_para_env_type
   USE ptb_types,                       ONLY: ptb_atom_type

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_parameters'

! *** Public data types ***

   PUBLIC :: ptb_parameters_setup

CONTAINS


! **************************************************************************************************
!> \brief ...
!> \param param ...
!> \param parameter_file_path ...
!> \param parameter_file_name ...
!> \param para_env ...
! **************************************************************************************************
   SUBROUTINE ptb_parameters_setup(param, parameter_file_path, parameter_file_name, &
                                  para_env)

      TYPE(ptb_atom_type), POINTER                       :: param
      CHARACTER(LEN=*), INTENT(IN)                       :: parameter_file_path, parameter_file_name
      TYPE(mp_para_env_type), POINTER                    :: para_env

   END SUBROUTINE ptb_parameters_setup

END MODULE ptb_parameters

