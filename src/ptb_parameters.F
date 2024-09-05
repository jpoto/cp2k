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

   USE cp_parser_methods,               ONLY: parser_get_next_line,&
                                              parser_get_object
   USE cp_parser_types,                 ONLY: cp_parser_type,&
                                              parser_create
   USE kinds,                           ONLY: default_string_length
   USE message_passing,                 ONLY: mp_para_env_type
   USE periodic_table,                  ONLY: nelem,&
                                              ptable
   USE ptb_types,                       ONLY: max_elem,&
                                              ptb_atom_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_parameters'

! *** Public data types ***

   PUBLIC :: ptb_parameters_init

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param param ...
!> \param element_symbol ...
!> \param parameter_file_path ...
!> \param parameter_file_name ...
!> \param para_env ...
! **************************************************************************************************
   SUBROUTINE ptb_parameters_init(param, element_symbol, parameter_file_path, parameter_file_name, &
                                  para_env)

      TYPE(ptb_atom_type), POINTER                       :: param
      CHARACTER(LEN=2), INTENT(IN)                       :: element_symbol
      CHARACTER(LEN=*), INTENT(IN)                       :: parameter_file_path, parameter_file_name
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CHARACTER(len=default_string_length)               :: filename, hline
      INTEGER                                            :: ielem, k
      LOGICAL                                            :: at_end, found
      TYPE(cp_parser_type)                               :: parser

      filename = ADJUSTL(TRIM(parameter_file_path))//ADJUSTL(TRIM(parameter_file_name))

      !identify element
      ielem = 999
      DO k = 1, nelem
         IF (element_symbol == ptable(k)%symbol) THEN
            ielem = k
            EXIT
         END IF
      END DO

      IF (ielem > max_elem) CPABORT('Element not available for pTB')

      !parse informaiton from file
      CALL parser_create(parser, filename, para_env=para_env)
      found = .FALSE.
      DO
         at_end = .FALSE.
         CALL parser_get_next_line(parser, 1, at_end)
         IF (at_end) EXIT
         CALL parser_get_object(parser, hline)
         WRITE (*, *) hline
      END DO

   END SUBROUTINE ptb_parameters_init

END MODULE ptb_parameters

