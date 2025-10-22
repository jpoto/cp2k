!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Fortran API for the grid package, which is written in C.
!> \author Ole Schuett
! **************************************************************************************************
MODULE fft_api
   USE ISO_C_BINDING,                   ONLY: C_BOOL,&
                                              C_CHAR,&
                                              C_INT,&
                                              C_LOC,&
                                              C_NULL_CHAR,&
                                              C_PTR
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_api'

   PUBLIC :: fft_library_init, fft_library_finalize
   PUBLIC :: fft_print_timing_report

   ! Keep it synchronized with fft_api_F.c
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_DEFAULT = 1
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_FFTW = 2
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_REFERENCE = 3
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_ESTIMATE = 11
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_MEASURE = 12
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_PATIENT = 13
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE = 14

CONTAINS

! **************************************************************************************************
!> \brief Initialize FFT library
!> \param backend ...
!> \param planning_mode ...
!> \param use_fftw_mpi ...
!> \param wisdom_file ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_init(backend, planning_mode, use_fftw_mpi, wisdom_file)
      INTEGER, INTENT(IN)                                :: backend, planning_mode
      LOGICAL, INTENT(IN)                                :: use_fftw_mpi
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file

      CHARACTER(KIND=C_CHAR, LEN=1), ALLOCATABLE, &
         DIMENSION(:), TARGET                            :: wisdom_file_c
      INTEGER                                            :: i
      INTERFACE
         SUBROUTINE fft_library_init_c(backend, planning_mode, use_fftw_mpi, wisdom_file) BIND(C, name="fft_library_init_F")
            IMPORT :: C_INT, C_BOOL, C_PTR
            INTEGER(C_INT), VALUE :: backend, planning_mode
            LOGICAL(C_BOOL), VALUE :: use_fftw_mpi
            TYPE(C_PTR), VALUE :: wisdom_file
         END SUBROUTINE fft_library_init_c
      END INTERFACE

      ALLOCATE (wisdom_file_c(LEN_TRIM(wisdom_file) + 1))
      DO i = 1, LEN_TRIM(wisdom_file)
         wisdom_file_c(i) = wisdom_file(i:i)
      END DO
      wisdom_file_c(LEN_TRIM(wisdom_file) + 1) = C_NULL_CHAR

      CALL fft_library_init_c(backend, planning_mode, LOGICAL(use_fftw_mpi, C_BOOL), C_LOC(wisdom_file_c))

      DEALLOCATE (wisdom_file_c)

   END SUBROUTINE fft_library_init

! **************************************************************************************************
!> \brief Finalize FFT library
!> \param wisdom_file ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_finalize(wisdom_file)
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file

      CHARACTER(KIND=C_CHAR, LEN=1), ALLOCATABLE, &
         DIMENSION(:), TARGET                            :: wisdom_file_c
      INTEGER                                            :: i
      INTERFACE
         SUBROUTINE fft_library_finalize_c(wisdom_file) BIND(C, name="fft_library_finalize_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: wisdom_file
         END SUBROUTINE fft_library_finalize_c
      END INTERFACE

      ALLOCATE (wisdom_file_c(LEN_TRIM(wisdom_file) + 1))
      DO i = 1, LEN_TRIM(wisdom_file)
         wisdom_file_c(i) = wisdom_file(i:i)
      END DO
      wisdom_file_c(LEN_TRIM(wisdom_file) + 1) = C_NULL_CHAR

      CALL fft_library_finalize_c(C_LOC(wisdom_file_c))

      DEALLOCATE (wisdom_file_c)

   END SUBROUTINE fft_library_finalize

! **************************************************************************************************
!> \brief Print FFT library statistics
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_print_timing_report()

      INTERFACE
         SUBROUTINE fft_print_timing_report_c() &
            BIND(C, name="fft_print_timing_report_F")
         END SUBROUTINE fft_print_timing_report_c
      END INTERFACE

      CALL fft_print_timing_report_c()

   END SUBROUTINE fft_print_timing_report

END MODULE fft_api
