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
MODULE ptb_types

   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type
   USE cp_output_handling,              ONLY: cp_p_file,&
                                              cp_print_key_unit_nr,&
                                              cp_print_key_finished_output
   USE input_section_types,             ONLY: section_vals_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   ! *** Global parameters ***

   INTEGER, PARAMETER, PRIVATE :: max_elem = 86
   INTEGER, PARAMETER, PRIVATE :: max_shell = 7
   INTEGER, PARAMETER, PRIVATE :: max_prim = 5

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_types'

! **************************************************************************************************
   TYPE ptb_atom_type
      ! PRIVATE
      CHARACTER(LEN=default_string_length)   :: typ = ""
      CHARACTER(LEN=default_string_length)   :: aname = ""
      CHARACTER(LEN=2)                       :: symbol = ""
      LOGICAL                                :: defined = .FALSE.
      INTEGER                                :: z = -1 !atomic number
      REAL(KIND=dp)                          :: zeff = -1.0_dp !effective core charge
      INTEGER                                :: natorb = -1 !number of orbitals
      INTEGER                                :: lmax = -1 !max angular momentum

      ! shell specific parameters
      INTEGER                                :: nshell = -1 !number of orbital shells
      INTEGER, DIMENSION(max_shell)          :: nval = -1 ! n-quantum number of shell i
      INTEGER, DIMENSION(max_shell)          :: lval = -1 ! l-quantum number of shell i
      REAL(KIND=dp), DIMENSION(max_shell)    :: kappa = -1.0_dp


   END TYPE ptb_atom_type

! *** Public data types ***

   PUBLIC :: ptb_atom_type, get_ptb_atom_param, write_ptb_atom_param

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param xtb_parameter ...
! **************************************************************************************************
   SUBROUTINE allocate_ptb_atom_param(ptb_parameter)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter

      IF (ASSOCIATED(ptb_parameter)) &
         CALL deallocate_ptb_atom_param(ptb_parameter)

      ALLOCATE (ptb_parameter)

      ptb_parameter%defined = .FALSE.
      ptb_parameter%aname = ""
      ptb_parameter%symbol = ""
      ptb_parameter%z = -1
      ptb_parameter%zeff = -1.0_dp
      ptb_parameter%natorb = 0
      ptb_parameter%lmax = -1
      ptb_parameter%nshell = 0
      ptb_parameter%nval = 0
      ptb_parameter%lval = 0
      ptb_parameter%kappa = 0.0_dp

   END SUBROUTINE allocate_ptb_atom_param

! **************************************************************************************************
!> \brief ...
!> \param xtb_parameter ...
! **************************************************************************************************
   SUBROUTINE deallocate_ptb_atom_param(ptb_parameter)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter

      CPASSERT(ASSOCIATED(ptb_parameter))
      DEALLOCATE (ptb_parameter)

   END SUBROUTINE deallocate_ptb_atom_param


! **************************************************************************************************
!> \brief ...
!> \param xtb_parameter ...
!> \param aname ...
!> \param defined ...
!> \param kappa ...
! **************************************************************************************************
   SUBROUTINE get_ptb_atom_param(ptb_parameter, aname, defined, kappa &
                                 )

      TYPE(ptb_atom_type), POINTER                               :: ptb_parameter
      CHARACTER(LEN=default_string_length), &       
         INTENT(OUT), OPTIONAL                                   :: aname
      LOGICAL, INTENT(OUT), OPTIONAL                             :: defined
      REAL(KIND=dp), DIMENSION(max_shell), INTENT(OUT), OPTIONAL :: kappa

      CPASSERT(ASSOCIATED(ptb_parameter))

      IF (PRESENT(aname)) aname = ptb_parameter%aname
      IF (PRESENT(defined)) defined = ptb_parameter%defined

      IF (PRESENT(kappa)) kappa = ptb_parameter%kappa

   END SUBROUTINE get_ptb_atom_param


! **************************************************************************************************
!> \brief ...
!> \param ptb_parameter ...
!> \param subsys_section ...
! **************************************************************************************************
   SUBROUTINE write_ptb_atom_param(ptb_parameter, subsys_section)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter
      TYPE(section_vals_type), POINTER                   :: subsys_section

      CHARACTER(LEN=default_string_length)               :: aname
      INTEGER                                            :: io_unit
      LOGICAL                                            :: defined
      REAL(KIND=dp), DIMENSION(max_shell)                :: kappa
      TYPE(cp_logger_type), POINTER                      :: logger

      NULLIFY (logger)
      logger => cp_get_default_logger()
      IF (ASSOCIATED(ptb_parameter)) THEN

         io_unit = cp_print_key_unit_nr(logger, subsys_section, "PRINT%KINDS", &
                                        extension=".Log")

         IF (io_unit > 0) THEN
            CALL get_ptb_atom_param(ptb_parameter, aname=aname, defined=defined, kappa=kappa)

            WRITE (UNIT=io_unit, FMT="(/,A,T67,A14)") " pTB  parameters: ", TRIM(aname)
            IF (defined) THEN

            ELSE
               WRITE (UNIT=io_unit, FMT="(T55,A)") "Parameters are not defined"
            END IF
         END IF
         CALL cp_print_key_finished_output(io_unit, logger, subsys_section, "PRINT%KINDS")
      END IF

   END SUBROUTINE write_ptb_atom_param

END MODULE ptb_types

