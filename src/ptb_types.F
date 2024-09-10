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
   USE cp_output_handling,              ONLY: cp_print_key_finished_output,&
                                              cp_print_key_unit_nr
   USE input_section_types,             ONLY: section_vals_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   ! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_types'

   INTEGER, PARAMETER, PUBLIC :: max_elem = 86
   INTEGER, PARAMETER, PUBLIC :: max_shell = 7
   INTEGER, PARAMETER, PUBLIC :: max_prim = 5

! **************************************************************************************************
   TYPE ptb_atom_type
      ! PRIVATE
      CHARACTER(LEN=default_string_length)   :: typ = ""
      CHARACTER(LEN=2)                       :: symbol = ""
      LOGICAL                                :: defined = .FALSE.
      INTEGER                                :: z = -1 !atomic number

      ! shell specific parameters
      INTEGER                                       :: nshell = -1 !number of orbital shells
      REAL(KIND=dp)                                 :: valence = -1.0_dp !valance / missing charge

      INTEGER, DIMENSION(max_shell)                 :: ind = -1  !number of prim.
      INTEGER, DIMENSION(max_shell)                 :: nprim = -1  !number of prim.
      INTEGER, DIMENSION(max_shell)                 :: lval = -1 ! l-quantum number of shell i

      REAL(KIND=dp), DIMENSION(max_shell, max_prim) :: kappa = -1.0_dp
      REAL(KIND=dp), DIMENSION(max_shell, max_prim) :: coef = -1.0_dp

   END TYPE ptb_atom_type

! *** Public data types ***

   PUBLIC :: ptb_atom_type, get_ptb_atom_param, write_ptb_atom_param, &
             allocate_ptb_atom_param, deallocate_ptb_atom_param

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param ptb_parameter ...
! **************************************************************************************************
   SUBROUTINE allocate_ptb_atom_param(ptb_parameter)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter

      IF (ASSOCIATED(ptb_parameter)) &
         CALL deallocate_ptb_atom_param(ptb_parameter)

      ALLOCATE (ptb_parameter)

      ptb_parameter%typ = "not defined"
      ptb_parameter%symbol = ""
      ptb_parameter%defined = .FALSE.
      ptb_parameter%z = -1

      ptb_parameter%nshell = 0
      ptb_parameter%valence = -1.0_dp

      ptb_parameter%ind = 0
      ptb_parameter%nprim = 0
      ptb_parameter%lval = 0

      ptb_parameter%kappa = 0.0_dp
      ptb_parameter%coef = 0.0_dp

   END SUBROUTINE allocate_ptb_atom_param

! **************************************************************************************************
!> \brief ...
!> \param ptb_parameter ...
! **************************************************************************************************
   SUBROUTINE deallocate_ptb_atom_param(ptb_parameter)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter

      CPASSERT(ASSOCIATED(ptb_parameter))
      DEALLOCATE (ptb_parameter)

   END SUBROUTINE deallocate_ptb_atom_param

! **************************************************************************************************
!> \brief ...
!> \param ptb_parameter ...
!> \param typ ...
!> \param symbol ...
!> \param defined ...
!> \param z ...
!> \param nshell ...
!> \param valence ...
!> \param ind ...
!> \param nprim ...
!> \param lval ...
!> \param kappa ...
!> \param coef ...
! **************************************************************************************************
   SUBROUTINE get_ptb_atom_param(ptb_parameter, typ, symbol, defined, z, &
                                 nshell, valence, ind, nprim, lval, kappa, coef)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter
      CHARACTER(LEN=default_string_length), &
         INTENT(OUT), OPTIONAL                           :: typ
      CHARACTER(LEN=2), INTENT(OUT), OPTIONAL            :: symbol
      LOGICAL, INTENT(OUT), OPTIONAL                     :: defined
      INTEGER, INTENT(OUT), OPTIONAL                     :: z, nshell
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: valence
      INTEGER, DIMENSION(max_shell), INTENT(OUT), &
         OPTIONAL                                        :: ind, nprim, lval
      REAL(KIND=dp), DIMENSION(max_shell, max_prim), &
         INTENT(OUT), OPTIONAL                           :: kappa, coef

      CPASSERT(ASSOCIATED(ptb_parameter))

      IF (PRESENT(typ)) typ = ptb_parameter%typ
      IF (PRESENT(symbol)) symbol = ptb_parameter%symbol
      IF (PRESENT(defined)) defined = ptb_parameter%defined
      IF (PRESENT(z)) z = ptb_parameter%z

      IF (PRESENT(nshell)) nshell = ptb_parameter%nshell
      IF (PRESENT(valence)) valence = ptb_parameter%valence

      IF (PRESENT(ind)) ind = ptb_parameter%ind
      IF (PRESENT(nprim)) nprim = ptb_parameter%nprim
      IF (PRESENT(lval)) lval = ptb_parameter%lval

      IF (PRESENT(kappa)) kappa = ptb_parameter%kappa
      IF (PRESENT(coef)) coef = ptb_parameter%coef

   END SUBROUTINE get_ptb_atom_param

! **************************************************************************************************
!> \brief ...
!> \param ptb_parameter ...
!> \param subsys_section ...
! **************************************************************************************************
   SUBROUTINE write_ptb_atom_param(ptb_parameter, subsys_section)

      TYPE(ptb_atom_type), POINTER                       :: ptb_parameter
      TYPE(section_vals_type), POINTER                   :: subsys_section

      INTEGER                                            :: io_unit, ishell
      TYPE(cp_logger_type), POINTER                      :: logger

      NULLIFY (logger)
      logger => cp_get_default_logger()
      IF (ASSOCIATED(ptb_parameter)) THEN

         io_unit = cp_print_key_unit_nr(logger, subsys_section, "PRINT%KINDS", &
                                        extension=".Log")

         IF (io_unit > 0) THEN

            WRITE (UNIT=io_unit, FMT="(/,A,T67,A14)") " pTB  parameters for ", ptb_parameter%symbol
            IF (ptb_parameter%defined) THEN
               WRITE (UNIT=io_unit, FMT="(T16,A,T71,A)") "typ                 ", ptb_parameter%typ
               WRITE (UNIT=io_unit, FMT="(T16,A,T71,I10)") "element number      ", ptb_parameter%z
               WRITE (UNIT=io_unit, FMT="(T16,A,T71,I10)") "number of shells    ", ptb_parameter%nshell
               WRITE (UNIT=io_unit, FMT="(T16,A,T71,E14.5)") "valence electrons   ", ptb_parameter%valence

               WRITE (UNIT=io_unit, FMT="(T16,A,T71)", advance="no") "primitves per shell "
               WRITE (UNIT=io_unit, FMT="(7I10)") ptb_parameter%nprim(1:ptb_parameter%nshell)
               WRITE (UNIT=io_unit, FMT="(T16,A,T71)", advance="no") "ang. mom. of the shell "
               WRITE (UNIT=io_unit, FMT="(7I10)") ptb_parameter%lval(1:ptb_parameter%nshell)
               WRITE (UNIT=io_unit, FMT="(T16,A)") "  exponents:     "
               DO ishell = 1, ptb_parameter%nshell
                  WRITE (UNIT=io_unit, FMT="(T20,7E14.5)") ptb_parameter%kappa(ishell, 1:max_prim)
               END DO
               WRITE (UNIT=io_unit, FMT="(T16,A)") "  coefficents:   "
               DO ishell = 1, ptb_parameter%nshell
                  WRITE (UNIT=io_unit, FMT="(T20,7E14.5)") ptb_parameter%coef(ishell, 1:max_prim)
               END DO
            ELSE
               WRITE (UNIT=io_unit, FMT="(T55,A)") "Parameters are not defined"
            END IF
         END IF
         CALL cp_print_key_finished_output(io_unit, logger, subsys_section, "PRINT%KINDS")
      END IF

   END SUBROUTINE write_ptb_atom_param

END MODULE ptb_types

