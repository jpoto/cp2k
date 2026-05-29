!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

MODULE qs_dispersion_s_dftd3

   USE kinds, ONLY: default_string_length, &
                    dp
   USE message_passing, ONLY: mp_para_env_type

#if defined(__S_DFTD3)
   USE dftd3_data_r4r2, ONLY: get_r4r2_val
   USE dftd3_data_vdwrad, ONLY: get_vdw_rad
   USE dftd3_param, ONLY: d3_param, &
                          get_rational_damping, &
                          get_zero_damping, &
                          error_type
   USE dftd3_reference, ONLY: init_reference_c6, &
                              get_c6
   USE mctc_data, ONLY: get_covalent_rad
#endif

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_dispersion_s_dftd3'

   PUBLIC :: dftd3_param_from_library, dftd3_functional_supported

CONTAINS

#if defined(__S_DFTD3)
! **************************************************************************************************
!> \brief Check if a functional is supported by the s-dftd3 library
!> \param pp_type ...
!> \param ref_functional ...
!> \param found ...
! **************************************************************************************************
   SUBROUTINE dftd3_functional_supported(pp_type, ref_functional, found)
      INTEGER, INTENT(IN)                                :: pp_type
      CHARACTER(LEN=*), INTENT(IN)                       :: ref_functional
      LOGICAL, INTENT(OUT)                               :: found

      CHARACTER(LEN=default_string_length)               :: func_name
      TYPE(d3_param)                                     :: d3params
      TYPE(error_type)                                   :: lib_error

      func_name = ref_functional
      d3params = d3_param()
      lib_error = error_type()
      found = .TRUE.

      SELECT CASE (pp_type)
      CASE (2)
         CALL get_zero_damping(d3params, func_name, lib_error)
      CASE (3)
         CALL get_rational_damping(d3params, func_name, lib_error)
      CASE DEFAULT
         d3params = d3_param()
      END SELECT

      IF (lib_error%has_failed()) found = .FALSE.

   END SUBROUTINE dftd3_functional_supported

! **************************************************************************************************
!> \brief ...
!> \param c6ab ...
!> \param maxci ...
!> \param r0ab ...
!> \param rcov ...
!> \param r2r4 ...
!> \param pp_type ...
!> \param ref_functional ...
!> \param s6 ...
!> \param s8 ...
!> \param a1 ...
!> \param a2 ...
!> \param sr6 ...
!> \param para_env ...
!> \param error ...
! **************************************************************************************************
   SUBROUTINE dftd3_param_from_library(c6ab, maxci, r0ab, rcov, r2r4, &
                                       pp_type, ref_functional, &
                                       s6, s8, a1, a2, sr6, &
                                       para_env, error)

      REAL(KIND=dp), DIMENSION(:, :, :, :, :), &
         INTENT(INOUT)                                   :: c6ab
      INTEGER, DIMENSION(:), INTENT(INOUT)               :: maxci
      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: r0ab
      REAL(KIND=dp), DIMENSION(:), INTENT(INOUT)         :: rcov, r2r4
      INTEGER, INTENT(IN)                                :: pp_type
      CHARACTER(LEN=*), INTENT(IN)                       :: ref_functional
      REAL(KIND=dp), INTENT(INOUT)                       :: s6, s8, a1, a2, sr6
      TYPE(mp_para_env_type), INTENT(IN), POINTER        :: para_env
      CHARACTER(LEN=*), INTENT(OUT), OPTIONAL            :: error

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dftd3_param_from_library'

      CHARACTER(LEN=default_string_length)               :: func_name
      INTEGER                                            :: handle, i, islice, iz, jref, jz, nelem, &
                                                            nref
      REAL(KIND=dp)                                      :: c6_val, r0_val, r4r2, rcov_val
      REAL(KIND=dp), DIMENSION(94)                       :: r0ab_tmp, r2r4_tmp, rcov_tmp
      TYPE(d3_param)                                     :: d3params
      TYPE(error_type), ALLOCATABLE                      :: lib_error

      CALL timeset(routineN, handle)

      CALL init_reference_c6()

      nref = 7
      nelem = MIN(94, SIZE(maxci))

      maxci(:) = 0
      maxci(:nelem) = nref

      IF (para_env%is_source()) THEN
         DO iz = 1, nelem
            rcov_val = get_covalent_rad(iz)
            rcov_tmp(iz) = rcov_val

            r4r2 = get_r4r2_val(iz)
            r2r4_tmp(iz) = r4r2

            IF (r4r2 > 0.0_dp) THEN
               r0_val = get_vdw_rad(iz)
               r0ab_tmp(iz) = r0_val
            ELSE
               r0ab_tmp(iz) = 0.0_dp
            END IF
         END DO

         c6ab = 0.0_dp
         DO jref = 1, nref
            DO jz = 1, nelem
               DO i = 1, nref
                  DO iz = 1, nelem
                     c6_val = get_c6(i, jref, iz, jz)
                     DO islice = 1, SIZE(c6ab, 5)
                        c6ab(iz, jz, i, jref, islice) = c6_val
                     END DO
                  END DO
               END DO
            END DO
         END DO
      END IF

      CALL para_env%bcast(r2r4_tmp)
      CALL para_env%bcast(r0ab_tmp)
      CALL para_env%bcast(rcov_tmp)

      IF (.NOT. para_env%is_source()) THEN
         c6ab = 0.0_dp
         DO jref = 1, nref
            DO jz = 1, nelem
               DO i = 1, nref
                  DO iz = 1, nelem
                     c6_val = get_c6(i, jref, iz, jz)
                     DO islice = 1, SIZE(c6ab, 5)
                        c6ab(iz, jz, i, jref, islice) = c6_val
                     END DO
                  END DO
               END DO
            END DO
         END DO
      END IF

      r2r4(:nelem) = r2r4_tmp(:nelem)
      rcov(:nelem) = rcov_tmp(:nelem)
      r0ab = 0.0_dp
      DO iz = 1, nelem
         DO jz = 1, nelem
            r0ab(iz, jz) = r0ab_tmp(iz) + r0ab_tmp(jz)
         END DO
      END DO

      func_name = ref_functional
      d3params = d3_param()

      IF (para_env%is_source()) THEN
         SELECT CASE (pp_type)
         CASE (2)
            CALL get_zero_damping(d3params, func_name, lib_error)
         CASE (3)
            CALL get_rational_damping(d3params, func_name, lib_error)
         CASE DEFAULT
            d3params = d3_param()
         END SELECT
      END IF

      CALL para_env%bcast(func_name)
      CALL para_env%bcast(d3params%s6)
      CALL para_env%bcast(d3params%s8)
      CALL para_env%bcast(d3params%a1)
      CALL para_env%bcast(d3params%a2)
      CALL para_env%bcast(d3params%rs6)

      IF (PRESENT(error)) error = ""

      s6 = d3params%s6
      s8 = d3params%s8
      a1 = d3params%a1
      a2 = d3params%a2
      sr6 = d3params%rs6

      CALL timestop(handle)

   END SUBROUTINE dftd3_param_from_library

#else

! **************************************************************************************************
!> \brief ...
!> \param c6ab ...
!> \param maxci ...
!> \param r0ab ...
!> \param rcov ...
!> \param r2r4 ...
!> \param pp_type ...
!> \param ref_functional ...
!> \param s6 ...
!> \param s8 ...
!> \param a1 ...
!> \param a2 ...
!> \param sr6 ...
!> \param para_env ...
!> \param error ...
! **************************************************************************************************
   SUBROUTINE dftd3_param_from_library(c6ab, maxci, r0ab, rcov, r2r4, &
                                       pp_type, ref_functional, &
                                       s6, s8, a1, a2, sr6, &
                                       para_env, error)

      REAL(KIND=dp), DIMENSION(:, :, :, :, :), &
         INTENT(INOUT)                                   :: c6ab
      INTEGER, DIMENSION(:), INTENT(INOUT)               :: maxci
      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: r0ab
      REAL(KIND=dp), DIMENSION(:), INTENT(INOUT)         :: rcov, r2r4
      INTEGER, INTENT(IN)                                :: pp_type
      CHARACTER(LEN=*), INTENT(IN)                       :: ref_functional
      REAL(KIND=dp), INTENT(INOUT)                       :: s6, s8, a1, a2, sr6
      TYPE(mp_para_env_type), INTENT(IN), POINTER        :: para_env
      CHARACTER(LEN=*), INTENT(OUT), OPTIONAL            :: error

      CPABORT("s-dftd3 library not compiled in")

   END SUBROUTINE dftd3_param_from_library

#endif

END MODULE qs_dispersion_s_dftd3
