!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Fortran API for the grid package, which is written in C.
!> \author Ole Schuett
! **************************************************************************************************
MODULE fft_api
   USE ISO_C_BINDING,                   ONLY: &
        C_ASSOCIATED, C_BOOL, C_CHAR, C_DOUBLE, C_DOUBLE_COMPLEX, C_F_POINTER, C_INT, C_LOC, C_NULL_CHAR, &
        C_NULL_PTR, C_PTR
   USE kinds,                           ONLY: dp
   USE message_passing,                 ONLY: mp_comm_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_api'

   PUBLIC :: fft_library_init, fft_library_finalize
   PUBLIC :: fft_library_acc_init, fft_library_acc_finalize
   PUBLIC :: fft_print_timing_report
   PUBLIC :: fft_backend_in_use, fft_use_mpi, fft_has_guru_interface
   PUBLIC :: fft_allocate, fft_free
   PUBLIC :: fft_grid_type, fft_create_grid, fft_create_grid_from_reference, fft_free_grid
   PUBLIC :: fft_grid_get_npts_global, fft_grid_get_comm
   PUBLIC :: fft_grid_get_proc2local_rs, fft_grid_get_proc2local_ms, fft_grid_get_proc2local_gs
   PUBLIC :: fft_1d_fw_local, fft_1d_bw_local
   PUBLIC :: fft_2d_fw_local, fft_2d_bw_local
   PUBLIC :: fft_3d_fw_local, fft_3d_bw_local
   PUBLIC :: fft_1d_fw_local_inplace, fft_1d_bw_local_inplace
   PUBLIC :: fft_2d_fw_local_inplace, fft_2d_bw_local_inplace
   PUBLIC :: fft_3d_fw_local_inplace, fft_3d_bw_local_inplace

   ! Keep it synchronized with fft_api_F.c
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_DEFAULT = 1
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_FFTW = 2
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_REFERENCE = 3
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_ESTIMATE = 11
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_MEASURE = 12
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_PATIENT = 13
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE = 14

   INTERFACE fft_allocate
      MODULE PROCEDURE :: fft_allocate_real_1
      MODULE PROCEDURE :: fft_allocate_real_2
      MODULE PROCEDURE :: fft_allocate_real_3
      MODULE PROCEDURE :: fft_allocate_complex_1
      MODULE PROCEDURE :: fft_allocate_complex_2
      MODULE PROCEDURE :: fft_allocate_complex_3
   END INTERFACE fft_allocate

   INTERFACE fft_free
      MODULE PROCEDURE :: fft_free_real_1
      MODULE PROCEDURE :: fft_free_real_2
      MODULE PROCEDURE :: fft_free_real_3
      MODULE PROCEDURE :: fft_free_complex_1
      MODULE PROCEDURE :: fft_free_complex_2
      MODULE PROCEDURE :: fft_free_complex_3
   END INTERFACE fft_free

   INTERFACE
      SUBROUTINE fft_allocate_double_c(length, buffer) BIND(C, name="fft_allocate_double")
         IMPORT :: C_PTR, C_INT
         INTEGER(C_INT), VALUE :: length
         TYPE(C_PTR) :: buffer
      END SUBROUTINE fft_allocate_double_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_allocate_complex_c(length, buffer) BIND(C, name="fft_allocate_complex")
         IMPORT :: C_PTR, C_INT
         INTEGER(C_INT), VALUE :: length
         TYPE(C_PTR) :: buffer
      END SUBROUTINE fft_allocate_complex_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_free_double_c(buffer) BIND(C, name="fft_free_double")
         IMPORT :: C_PTR
         TYPE(C_PTR), VALUE :: buffer
      END SUBROUTINE fft_free_double_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_free_complex_c(buffer) BIND(C, name="fft_free_complex")
         IMPORT :: C_PTR
         TYPE(C_PTR), VALUE :: buffer
      END SUBROUTINE fft_free_complex_c
   END INTERFACE

   TYPE fft_grid_type
      PRIVATE
      TYPE(C_PTR) :: fft_grid = C_NULL_PTR
   END TYPE fft_grid_type

CONTAINS

! **************************************************************************************************
!> \brief Initialize FFT library
!> \param backend ...
!> \param planning_mode ...
!> \param use_fftw_mpi ...
!> \param use_guru_interface ...
!> \param wisdom_file ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_init(backend, planning_mode, use_fftw_mpi, use_guru_interface, wisdom_file)
      INTEGER, INTENT(IN)                                :: backend, planning_mode
      LOGICAL, INTENT(IN)                                :: use_fftw_mpi, use_guru_interface
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file

      CHARACTER(KIND=C_CHAR, LEN=1), ALLOCATABLE, &
         DIMENSION(:), TARGET                            :: wisdom_file_c
      INTEGER                                            :: i
      INTERFACE
         SUBROUTINE fft_library_init_c(backend, planning_mode, use_fftw_mpi, use_guru_interface, &
                                       wisdom_file) BIND(C, name="fft_library_init_F")
            IMPORT :: C_INT, C_BOOL, C_PTR
            INTEGER(C_INT), VALUE :: backend, planning_mode
            LOGICAL(C_BOOL), VALUE :: use_fftw_mpi, use_guru_interface
            TYPE(C_PTR), VALUE :: wisdom_file
         END SUBROUTINE fft_library_init_c
      END INTERFACE

      ALLOCATE (wisdom_file_c(LEN_TRIM(wisdom_file) + 1))
      DO i = 1, LEN_TRIM(wisdom_file)
         wisdom_file_c(i) = wisdom_file(i:i)
      END DO
      wisdom_file_c(LEN_TRIM(wisdom_file) + 1) = C_NULL_CHAR

      CALL fft_library_init_c(backend, planning_mode, LOGICAL(use_fftw_mpi, C_BOOL), &
                              LOGICAL(use_guru_interface, C_BOOL), C_LOC(wisdom_file_c))

      DEALLOCATE (wisdom_file_c)

   END SUBROUTINE fft_library_init

! **************************************************************************************************
!> \brief Initialize FFT library
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_acc_init()
      INTERFACE
         SUBROUTINE fft_init_acc_lib_c() BIND(C, name="fft_init_acc_lib")
         END SUBROUTINE fft_init_acc_lib_c
      END INTERFACE

      CALL fft_init_acc_lib_c()

   END SUBROUTINE fft_library_acc_init

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
!> \brief Finalize FFT library
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_acc_finalize()
      INTERFACE
         SUBROUTINE fft_library_acc_finalize_c() BIND(C, name="fft_finalize_acc_lib")
         END SUBROUTINE fft_library_acc_finalize_c
      END INTERFACE

      CALL fft_library_acc_finalize_c()

   END SUBROUTINE fft_library_acc_finalize

! **************************************************************************************************
!> \brief Print FFT library statistics
!> \param threshold ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_print_timing_report(threshold)
      REAL(KIND=dp), INTENT(IN)                          :: threshold

      INTERFACE
         SUBROUTINE fft_print_timing_report_c(threshold) &
            BIND(C, name="fft_print_timing_report_F")
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), VALUE :: threshold
         END SUBROUTINE fft_print_timing_report_c
      END INTERFACE

      CALL fft_print_timing_report_c(threshold)

   END SUBROUTINE fft_print_timing_report

! **************************************************************************************************
!> \brief Inquire the library backend in use
!> \return ...
!> \author Frederick Stein
! **************************************************************************************************
   INTEGER FUNCTION fft_backend_in_use()

      INTERFACE
         INTEGER(C_INT) FUNCTION fft_backend_in_use_c() &
            BIND(C, name="fft_backend_in_use_F")
            IMPORT :: C_INT
         END FUNCTION fft_backend_in_use_c
      END INTERFACE

      fft_backend_in_use = fft_backend_in_use_c()

   END FUNCTION fft_backend_in_use

! **************************************************************************************************
!> \brief Inquire whether the library backend supports MPI
!> \return ...
!> \author Frederick Stein
! **************************************************************************************************
   LOGICAL FUNCTION fft_use_mpi()

      INTERFACE
         LOGICAL(C_BOOL) FUNCTION fft_use_mpi_c() &
            BIND(C, name="fft_use_mpi_F")
            IMPORT :: C_BOOL
         END FUNCTION fft_use_mpi_c
      END INTERFACE

      fft_use_mpi = fft_use_mpi_c()

   END FUNCTION fft_use_mpi

! **************************************************************************************************
!> \brief Inquire whether a guru interface is available
!> \return ...
!> \author Frederick Stein
! **************************************************************************************************
   LOGICAL FUNCTION fft_has_guru_interface()

      INTERFACE
         LOGICAL(C_BOOL) FUNCTION fft_has_guru_interface_c() &
            BIND(C, name="fft_has_guru_interface_F")
            IMPORT :: C_BOOL
         END FUNCTION fft_has_guru_interface_c
      END INTERFACE

      fft_has_guru_interface = fft_has_guru_interface_c()

   END FUNCTION fft_has_guru_interface

! **************************************************************************************************
!> \brief Allocate a complex array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_complex_1(buffer, length)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:), &
         POINTER                                         :: buffer
      INTEGER, INTENT(IN)                                :: length

      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_complex_c(INT(length, C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer, [length])

   END SUBROUTINE fft_allocate_complex_1

! **************************************************************************************************
!> \brief Allocate a complex array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_complex_2(buffer, length)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer
      INTEGER, DIMENSION(2), INTENT(IN)                  :: length

      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_complex_c(INT(PRODUCT(length), C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer, length)

   END SUBROUTINE fft_allocate_complex_2

! **************************************************************************************************
!> \brief Allocate a complex array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_complex_3(buffer, length)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer
      INTEGER, DIMENSION(3), INTENT(IN)                  :: length

      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_complex_c(INT(PRODUCT(length), C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer, length)

   END SUBROUTINE fft_allocate_complex_3

! **************************************************************************************************
!> \brief Allocate a double array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_real_1(buffer, length)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:), POINTER   :: buffer
      INTEGER, INTENT(IN)                                :: length

      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_double_c(INT(length, C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer, [length])

   END SUBROUTINE fft_allocate_real_1

! **************************************************************************************************
!> \brief Allocate a double array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_real_2(buffer, length)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer
      INTEGER, DIMENSION(2), INTENT(IN)                  :: length

      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:), POINTER   :: buffer_1d
      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_double_c(INT(PRODUCT(length), C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer_1d, [PRODUCT(length)])

      buffer(1:length(1), 1:length(2)) => buffer_1d

   END SUBROUTINE fft_allocate_real_2

! **************************************************************************************************
!> \brief Allocate a double array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_allocate_real_3(buffer, length)

      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer
      INTEGER, DIMENSION(3), INTENT(IN)                  :: length

      TYPE(C_PTR)                                        :: buffer_c

      buffer_c = C_NULL_PTR

      CALL fft_allocate_double_c(INT(PRODUCT(length), C_INT), buffer_c)

      CALL C_F_POINTER(buffer_c, buffer, length)

   END SUBROUTINE fft_allocate_real_3

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_1(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_1

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_2(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_2

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_3(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_3

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_1(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:), POINTER   :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_1

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_2(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_2

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_3(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_3

! **************************************************************************************************
!> \brief Performs a complex-to-complex 1D forwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out)
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(OUT)        :: grid_out

      INTERFACE
         SUBROUTINE fft_1d_fw_local_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out) &
            BIND(C, name="fft_1d_fw_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid_in
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(OUT) :: grid_out
         END SUBROUTINE fft_1d_fw_local_c
      END INTERFACE

      CALL fft_1d_fw_local_c(fft_size, number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                             LOGICAL(transpose_gs, C_BOOL), grid_in, grid_out)

   END SUBROUTINE fft_1d_fw_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 1D backwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out)
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*)                     :: grid_in, grid_out

      INTERFACE
         SUBROUTINE fft_1d_bw_local_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out) &
            BIND(C, name="fft_1d_bw_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: grid_in, grid_out
         END SUBROUTINE fft_1d_bw_local_c
      END INTERFACE

      CALL fft_1d_bw_local_c(fft_size, number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                             LOGICAL(transpose_gs, C_BOOL), grid_in, grid_out)

   END SUBROUTINE fft_1d_bw_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 2D forwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out)
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(OUT)        :: grid_out

      INTERFACE
         SUBROUTINE fft_2d_fw_local_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out) &
            BIND(C, name="fft_2d_fw_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid_in
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(OUT) :: grid_out
         END SUBROUTINE fft_2d_fw_local_c
      END INTERFACE

      CALL fft_2d_fw_local_c([fft_size(2), fft_size(1)], number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                             LOGICAL(transpose_gs, C_BOOL), grid_in, grid_out)

   END SUBROUTINE fft_2d_fw_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 2D backwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out)
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*)                     :: grid_in, grid_out

      INTERFACE
         SUBROUTINE fft_2d_bw_local_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in, grid_out) &
            BIND(C, name="fft_2d_bw_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: grid_in, grid_out
         END SUBROUTINE fft_2d_bw_local_c
      END INTERFACE

      CALL fft_2d_bw_local_c([fft_size(2), fft_size(1)], number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                             LOGICAL(transpose_gs, C_BOOL), grid_in, grid_out)

   END SUBROUTINE fft_2d_bw_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 3D forwards FFT
!> \param fft_size ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_fw_local(fft_size, grid_in, grid_out)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(OUT)        :: grid_out

      INTERFACE
         SUBROUTINE fft_3d_fw_local_c(fft_size, grid_in, grid_out) &
            BIND(C, name="fft_3d_fw_local_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(3) :: fft_size
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid_in
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(OUT) :: grid_out
         END SUBROUTINE fft_3d_fw_local_c
      END INTERFACE

      CALL fft_3d_fw_local_c([fft_size(3), fft_size(2), fft_size(1)], grid_in, grid_out)

   END SUBROUTINE fft_3d_fw_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 3D backwards FFT
!> \param fft_size ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_bw_local(fft_size, grid_in, grid_out)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*)                     :: grid_in, grid_out

      INTERFACE
         SUBROUTINE fft_3d_bw_local_c(fft_size, grid_in, grid_out) &
            BIND(C, name="fft_3d_bw_local_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(3) :: fft_size
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: grid_in, grid_out
         END SUBROUTINE fft_3d_bw_local_c
      END INTERFACE

      CALL fft_3d_bw_local_c([fft_size(3), fft_size(2), fft_size(1)], grid_in, grid_out)

   END SUBROUTINE fft_3d_bw_local

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 1D forwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_fw_local_inplace(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid)
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid

      INTERFACE
         SUBROUTINE fft_1d_fw_local_inplace_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid) &
            BIND(C, name="fft_1d_fw_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_1d_fw_local_inplace_c
      END INTERFACE

      CALL fft_1d_fw_local_inplace_c(fft_size, number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                                     LOGICAL(transpose_gs, C_BOOL), grid)

   END SUBROUTINE fft_1d_fw_local_inplace

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 1D backwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_bw_local_inplace(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid)
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*)                     :: grid

      INTERFACE
         SUBROUTINE fft_1d_bw_local_inplace_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid) &
            BIND(C, name="fft_1d_bw_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: grid
         END SUBROUTINE fft_1d_bw_local_inplace_c
      END INTERFACE

      CALL fft_1d_bw_local_inplace_c(fft_size, number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                                     LOGICAL(transpose_gs, C_BOOL), grid)

   END SUBROUTINE fft_1d_bw_local_inplace

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 2D forwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_fw_local_inplace(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid)
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid

      INTERFACE
         SUBROUTINE fft_2d_fw_local_inplace_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid) &
            BIND(C, name="fft_2d_fw_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_2d_fw_local_inplace_c
      END INTERFACE

      CALL fft_2d_fw_local_inplace_c([fft_size(2), fft_size(1)], number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                                     LOGICAL(transpose_gs, C_BOOL), grid)

   END SUBROUTINE fft_2d_fw_local_inplace

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 2D backwards FFT
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_rs ...
!> \param transpose_gs ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_bw_local_inplace(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid)
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_rs, transpose_gs
      COMPLEX(KIND=dp), DIMENSION(*)                     :: grid

      INTERFACE
         SUBROUTINE fft_2d_bw_local_inplace_c(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid) &
            BIND(C, name="fft_2d_bw_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_rs, transpose_gs
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_2d_bw_local_inplace_c
      END INTERFACE

      CALL fft_2d_bw_local_inplace_c([fft_size(2), fft_size(1)], number_of_ffts, LOGICAL(transpose_rs, C_BOOL), &
                                     LOGICAL(transpose_gs, C_BOOL), grid)

   END SUBROUTINE fft_2d_bw_local_inplace

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 3D forwards FFT
!> \param fft_size ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_fw_local_inplace(fft_size, grid)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid

      INTERFACE
         SUBROUTINE fft_3d_fw_local_inplace_c(fft_size, grid) &
            BIND(C, name="fft_3d_fw_local_inplace_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_3d_fw_local_inplace_c
      END INTERFACE

      CALL fft_3d_fw_local_inplace_c([fft_size(3), fft_size(2), fft_size(1)], grid)

   END SUBROUTINE fft_3d_fw_local_inplace

! **************************************************************************************************
!> \brief Performs an in-place complex-to-complex 3D backwards FFT
!> \param fft_size ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_bw_local_inplace(fft_size, grid)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: grid

      INTERFACE
         SUBROUTINE fft_3d_bw_local_inplace_c(fft_size, grid) &
            BIND(C, name="fft_3d_bw_local_inplace_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_3d_bw_local_inplace_c
      END INTERFACE

      CALL fft_3d_bw_local_inplace_c([fft_size(3), fft_size(2), fft_size(1)], grid)

   END SUBROUTINE fft_3d_bw_local_inplace

! **************************************************************************************************
!> \brief Create a FFT grid layout anew
!> \param fft_grid ...
!> \param fft_size ...
!> \param comm ...
!> \param dh_inv ...
!> \param use_halfspace ...
! **************************************************************************************************
   SUBROUTINE fft_create_grid(fft_grid, fft_size, comm, dh_inv, use_halfspace, cutoff, pgrid_guess, external_local_bounds)
      TYPE(fft_grid_type), INTENT(INOUT)                 :: fft_grid
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      INTEGER, INTENT(IN), DIMENSION(2), OPTIONAL        :: pgrid_guess
      INTEGER, INTENT(IN), DIMENSION(2, 3), OPTIONAL     :: external_local_bounds

      CLASS(mp_comm_type), INTENT(IN) :: comm
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN) :: dh_inv
      LOGICAL, INTENT(IN) :: use_halfspace
      INTEGER(C_INT), DIMENSION(2), TARGET :: pgrid_guess_c
      INTEGER(C_INT), DIMENSION(2, 3), TARGET :: external_local_bounds_c
      TYPE(C_PTR) :: pgrid_guess_ptr, external_local_bounds_ptr

      INTERFACE
         SUBROUTINE fft_create_grid_c(layout, comm, fft_size, dh_inv, use_halfspace, cutoff, pgrid_guess, external_local_bounds) &
            BIND(C, name="fft_create_grid_F")
            IMPORT :: C_INT, C_DOUBLE, C_PTR, C_BOOL
            TYPE(C_PTR), INTENT(INOUT) :: layout
            INTEGER(C_INT), VALUE :: comm
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            REAL(C_DOUBLE), DIMENSION(3, 3), INTENT(IN) :: dh_inv
            LOGICAL(C_BOOL), VALUE :: use_halfspace
            REAL(C_DOUBLE), VALUE :: cutoff
            TYPE(C_PTR), VALUE :: pgrid_guess
            TYPE(C_PTR), VALUE :: external_local_bounds
         END SUBROUTINE fft_create_grid_c
      END INTERFACE

      IF (C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("The grid layout must be unassociated before creation")

      pgrid_guess_ptr = C_NULL_PTR
      IF (PRESENT(pgrid_guess)) THEN
         pgrid_guess_c = INT(pgrid_guess, C_INT)
         pgrid_guess_ptr = C_LOC(pgrid_guess_c)
      END IF

      external_local_bounds_ptr = C_NULL_PTR
      IF (PRESENT(external_local_bounds)) THEN
         external_local_bounds_c = INT(external_local_bounds, C_INT)
         external_local_bounds_ptr = C_LOC(external_local_bounds_c)
      END IF

      CALL fft_create_grid_c(fft_grid%fft_grid, INT(comm%get_handle(), C_INT), INT(fft_size, C_INT), &
                           REAL(dh_inv, C_DOUBLE), LOGICAL(use_halfspace, C_BOOL), REAL(cutoff, C_DOUBLE), &
                           pgrid_guess_ptr, external_local_bounds_ptr)

      IF (.NOT. C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("Grid creation failed")

   END SUBROUTINE fft_create_grid

! **************************************************************************************************
!> \brief Create a FFT grid layout from a reference grid
!> \param fft_grid ...
!> \param fft_size ...
!> \param fft_grid_ref ...
! **************************************************************************************************
   SUBROUTINE fft_create_grid_from_reference(fft_grid, fft_size, cutoff, external_local_bounds, fft_grid_ref)
      TYPE(fft_grid_type), INTENT(INOUT) :: fft_grid
      INTEGER, INTENT(IN), DIMENSION(3) :: fft_size
      REAL(C_DOUBLE), INTENT(IN) :: cutoff
      INTEGER, INTENT(IN), DIMENSION(2, 3), OPTIONAL     :: external_local_bounds
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid_ref

      INTEGER(C_INT), DIMENSION(2, 3), TARGET :: external_local_bounds_c
      TYPE(C_PTR) :: external_local_bounds_ptr

      INTERFACE
         SUBROUTINE fft_create_grid_from_reference_c(layout, fft_size, cutoff, external_local_bounds, layout_ref) &
            BIND(C, name="fft_create_grid_from_reference_F")
            IMPORT :: C_INT, C_PTR, C_DOUBLE
            TYPE(C_PTR), INTENT(INOUT) :: layout
            INTEGER(C_INT), DIMENSION(3) :: fft_size
            REAL(C_DOUBLE), VALUE :: cutoff
            TYPE(C_PTR), VALUE :: external_local_bounds
            TYPE(C_PTR), VALUE :: layout_ref
         END SUBROUTINE fft_create_grid_from_reference_c
      END INTERFACE

      IF (C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("The grid must be unassociated before creation")
      IF (.NOT. C_ASSOCIATED(fft_grid_ref%fft_grid)) &
         CPABORT("The reference grid must be associated")

      external_local_bounds_ptr = C_NULL_PTR
      IF (PRESENT(external_local_bounds)) THEN
         external_local_bounds_c = INT(external_local_bounds, C_INT)
         external_local_bounds_ptr = C_LOC(external_local_bounds_c)
      END IF

      CALL fft_create_grid_from_reference_c(fft_grid%fft_grid, INT(fft_size, C_INT), REAL(cutoff, C_DOUBLE), &
                                            external_local_bounds_ptr, fft_grid_ref%fft_grid)

      IF (.NOT. C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("The grid creation failed")

   END SUBROUTINE fft_create_grid_from_reference

! **************************************************************************************************
!> \brief Free a FFT grid layout
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_free_grid(fft_grid)
      TYPE(fft_grid_type), INTENT(INOUT) :: fft_grid

      INTERFACE
         SUBROUTINE fft_free_grid_c(layout) &
            BIND(C, name="fft_free_grid_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
         END SUBROUTINE fft_free_grid_c
      END INTERFACE

      CALL fft_free_grid_c(fft_grid%fft_grid)

      fft_grid%fft_grid = C_NULL_PTR

   END SUBROUTINE fft_free_grid

! **************************************************************************************************
!> \brief Get the number of global points of a grid type
!> \param fft_grid ...
! **************************************************************************************************
   FUNCTION fft_grid_get_npts_global(fft_grid) RESULT(npts_global)
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid
      INTEGER, DIMENSION(3) :: npts_global

      INTEGER(C_INT), DIMENSION(3) :: npts_global_c

      INTERFACE
         SUBROUTINE fft_grid_get_npts_global_c(layout, npts_global) &
            BIND(C, name="fft_grid_get_npts_global_F")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE :: layout
            INTEGER(C_INT), DIMENSION(*) :: npts_global
         END SUBROUTINE fft_grid_get_npts_global_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      CALL fft_grid_get_npts_global_c(fft_grid%fft_grid, npts_global_c)

      npts_global = INT(npts_global_c)

   END FUNCTION fft_grid_get_npts_global

! **************************************************************************************************
!> \brief Get the communicator of a grid type
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_comm(fft_grid, comm)
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid
      CLASS(mp_comm_type) :: comm

      INTEGER(C_INT) :: comm_handle
      TYPE(mp_comm_type) :: my_comm

      INTERFACE
         SUBROUTINE fft_grid_get_comm_c(layout, comm) &
            BIND(C, name="fft_grid_get_comm_F")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE :: layout
            INTEGER(C_INT), INTENT(OUT) :: comm
         END SUBROUTINE fft_grid_get_comm_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      CALL fft_grid_get_comm_c(fft_grid%fft_grid, comm_handle)

      CALL my_comm%set_handle(INT(comm_handle), 2)
      CALL comm%from_dup(my_comm)

   END SUBROUTINE fft_grid_get_comm

! **************************************************************************************************
!> \brief Get the mapping of the process ID to the local bounds in real space
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_rs(fft_grid, proc2local_rs, lower_bounds)
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(INOUT) :: proc2local_rs
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL :: lower_bounds

      INTEGER :: proc, dir
      TYPE(mp_comm_type) :: comm
      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE :: proc2local_rs_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_rs_c(layout, proc2local_rs) &
            BIND(C, name="fft_grid_get_proc2local_rs_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE  :: proc2local_rs
         END SUBROUTINE fft_grid_get_proc2local_rs_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE(proc2local_rs_c(SIZE(proc2local_rs)))

      CALL fft_grid_get_proc2local_rs_c(fft_grid%fft_grid, C_LOC(proc2local_rs_c))

      CALL fft_grid_get_comm(fft_grid, comm)

      IF (PRESENT(lower_bounds)) THEN
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_rs(1, dir, proc) = lower_bounds(dir)+INT(proc2local_rs_c(6*proc+2*dir))
               proc2local_rs(2, dir, proc) = lower_bounds(dir)+INT(proc2local_rs_c(6*proc+2*dir)+proc2local_rs_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      ELSE
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_rs(1, dir, proc) = INT(proc2local_rs_c(6*proc+2*dir))
               proc2local_rs(2, dir, proc) = INT(proc2local_rs_c(6*proc+2*dir)+proc2local_rs_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      END IF

      CALL comm%free()

   END SUBROUTINE fft_grid_get_proc2local_rs

! **************************************************************************************************
!> \brief Get the mapping of the process ID to the local bounds in mixed space
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_ms(fft_grid, proc2local_ms, lower_bounds)
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(INOUT) :: proc2local_ms
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL :: lower_bounds

      INTEGER :: proc, dir
      TYPE(mp_comm_type) :: comm
      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE :: proc2local_ms_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_ms_c(layout, proc2local_ms) &
            BIND(C, name="fft_grid_get_proc2local_ms_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE  :: proc2local_ms
         END SUBROUTINE fft_grid_get_proc2local_ms_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE(proc2local_ms_c(SIZE(proc2local_ms)))

      CALL fft_grid_get_proc2local_ms_c(fft_grid%fft_grid, C_LOC(proc2local_ms_c))

      CALL fft_grid_get_comm(fft_grid, comm)

      IF (PRESENT(lower_bounds)) THEN
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_ms(1, dir, proc) = lower_bounds(dir)+INT(proc2local_ms_c(6*proc+2*dir))
               proc2local_ms(2, dir, proc) = lower_bounds(dir)+INT(proc2local_ms_c(6*proc+2*dir)+proc2local_ms_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      ELSE
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_ms(1, dir, proc) = INT(proc2local_ms_c(6*proc+2*dir))
               proc2local_ms(2, dir, proc) = INT(proc2local_ms_c(6*proc+2*dir)+proc2local_ms_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      END IF

      CALL comm%free()

   END SUBROUTINE fft_grid_get_proc2local_ms

! **************************************************************************************************
!> \brief Get the mapping of the process ID to the local bounds in reciprocal space
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_gs(fft_grid, proc2local_gs, lower_bounds)
      TYPE(fft_grid_type), INTENT(IN) :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(INOUT) :: proc2local_gs
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL :: lower_bounds

      INTEGER :: proc, dir
      TYPE(mp_comm_type) :: comm
      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE :: proc2local_gs_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_gs_c(layout, proc2local_gs) &
            BIND(C, name="fft_grid_get_proc2local_gs_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE  :: proc2local_gs
         END SUBROUTINE fft_grid_get_proc2local_gs_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE(proc2local_gs_c(SIZE(proc2local_gs)))

      CALL fft_grid_get_proc2local_gs_c(fft_grid%fft_grid, C_LOC(proc2local_gs_c))

      CALL fft_grid_get_comm(fft_grid, comm)

      IF (PRESENT(lower_bounds)) THEN
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_gs(1, dir, proc) = lower_bounds(dir)+INT(proc2local_gs_c(6*proc+2*dir))
               proc2local_gs(2, dir, proc) = lower_bounds(dir)+INT(proc2local_gs_c(6*proc+2*dir)+proc2local_gs_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      ELSE
         DO proc = 0, comm%num_pe-1
            DO dir = 1, 3
               proc2local_gs(1, dir, proc) = INT(proc2local_gs_c(6*proc+2*dir))
               proc2local_gs(2, dir, proc) = INT(proc2local_gs_c(6*proc+2*dir)+proc2local_gs_c(6*proc+2*dir+1)-1)
            END DO
         END DO
      END IF

      CALL comm%free()

   END SUBROUTINE fft_grid_get_proc2local_gs

END MODULE fft_api
