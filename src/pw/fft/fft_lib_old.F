!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
MODULE fft_lib_old

   USE fft_plan,                        ONLY: fft_plan_type
   USE fftsg_lib,                       ONLY: fftsg_do_cleanup,&
                                              fftsg_do_init,&
                                              fftsg_get_lengths
   USE fftw3_lib,                       ONLY: fftw3_do_cleanup,&
                                              fftw3_do_init,&
                                              fftw3_get_lengths
#include "../../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_lib_old'

   PUBLIC :: fft_do_cleanup, fft_do_init, fft_get_lengths, fft_create_plan_3d
   PUBLIC :: fft_create_plan_1d, fft_library, fft_destroy_plan

CONTAINS
! **************************************************************************************************
!> \brief Interface to FFT libraries
!> \param fftlib ...
!> \return ...
!> \par History
!>      IAB 09-Jan-2009 : Modified to use fft_plan_type
!>                        (c) The Numerical Algorithms Group (NAG) Ltd, 2009 on behalf of the HECToR project
!> \author JGH
! **************************************************************************************************
   FUNCTION fft_library(fftlib) RESULT(flib)

      CHARACTER(len=*), INTENT(IN)                       :: fftlib
      INTEGER                                            :: flib

      SELECT CASE (fftlib)
      CASE DEFAULT
         flib = -1
      CASE ("FFTSG")
         flib = 1
      CASE ("FFTW3")
         flib = 3
      END SELECT

   END FUNCTION fft_library

! **************************************************************************************************
!> \brief ...
!> \param fft_type ...
!> \param wisdom_file ...
! **************************************************************************************************
   SUBROUTINE fft_do_init(fft_type, wisdom_file)
      INTEGER, INTENT(IN)                                :: fft_type
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file

      SELECT CASE (fft_type)
      CASE DEFAULT
         CPABORT("fft_do_init")
      CASE (1)
         CALL fftsg_do_init()
      CASE (3)
         CALL fftw3_do_init(wisdom_file)
      END SELECT

   END SUBROUTINE fft_do_init

! **************************************************************************************************
!> \brief ...
!> \param fft_type ...
!> \param wisdom_file ...
!> \param ionode ...
! **************************************************************************************************
   SUBROUTINE fft_do_cleanup(fft_type, wisdom_file, ionode)
      INTEGER, INTENT(IN)                                :: fft_type
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file
      LOGICAL, INTENT(IN)                                :: ionode

      SELECT CASE (fft_type)
      CASE DEFAULT
         CPABORT("fft_do_cleanup")
      CASE (1)
         CALL fftsg_do_cleanup()
      CASE (3)
         CALL fftw3_do_cleanup(wisdom_file, ionode)
      END SELECT

   END SUBROUTINE fft_do_cleanup

! **************************************************************************************************
!> \brief ...
!> \param fft_type ...
!> \param DATA ...
!> \param max_length ...
! **************************************************************************************************
   SUBROUTINE fft_get_lengths(fft_type, DATA, max_length)

      INTEGER, INTENT(IN)                                :: fft_type
      INTEGER, DIMENSION(*)                              :: DATA
      INTEGER, INTENT(INOUT)                             :: max_length

      SELECT CASE (fft_type)
      CASE DEFAULT
         CPABORT("fft_get_lengths")
      CASE (1)
         CALL fftsg_get_lengths(DATA, max_length)
      CASE (3)
         CALL fftw3_get_lengths(DATA, max_length)
      END SELECT

   END SUBROUTINE fft_get_lengths

! **************************************************************************************************

! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param fft_type ...
!> \param fft_in_place ...
!> \param fsign ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE fft_create_plan_3d(plan, fft_type, fft_in_place, fsign, n)

      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan
      INTEGER, INTENT(IN)                                :: fft_type
      LOGICAL, INTENT(IN)                                :: fft_in_place
      INTEGER, INTENT(IN)                                :: fsign
      INTEGER, DIMENSION(3), INTENT(IN)                  :: n

      plan%fft_type = fft_type
      plan%fsign = fsign
      plan%fft_in_place = fft_in_place
      plan%n_3d = n
!$    plan%need_alt_plan = .FALSE.

      ! Planning only needed for FFTW3
      IF (fft_type == 3) THEN
         plan%valid = .TRUE.
      END IF

   END SUBROUTINE fft_create_plan_3d

! **************************************************************************************************

! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param fft_type ...
!> \param fsign ...
!> \param trans ...
!> \param n ...
!> \param m ...
! **************************************************************************************************
   SUBROUTINE fft_create_plan_1d(plan, fft_type, fsign, trans, n, m)
      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan
      INTEGER, INTENT(IN)                                :: fft_type, fsign
      LOGICAL, INTENT(IN)                                :: trans
      INTEGER, INTENT(IN)                                :: n, m

      plan%fft_type = fft_type
      plan%fsign = fsign
      plan%trans = trans
      plan%n = n
      plan%m = m
!$    plan%need_alt_plan = .FALSE.

      ! Planning only needed for FFTW3
      IF ((fft_type == 3) .AND. (n*m /= 0)) THEN
         plan%valid = .TRUE.
      ELSE
         plan%valid = .FALSE.
      END IF

   END SUBROUTINE fft_create_plan_1d

! **************************************************************************************************
!> \brief ...
!> \param plan ...
! **************************************************************************************************
   SUBROUTINE fft_destroy_plan(plan)
      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan

! Planning only needed for FFTW3

      IF (plan%valid .AND. plan%fft_type == 3) THEN
         plan%valid = .FALSE.
      END IF

   END SUBROUTINE fft_destroy_plan
END MODULE fft_lib_old
