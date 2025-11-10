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
                                              C_DOUBLE,&
                                              C_DOUBLE_COMPLEX,&
                                              C_INT,&
                                              C_LOC,&
                                              C_NULL_CHAR,&
                                              C_PTR
   USE kinds,                           ONLY: dp
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_api'

   PUBLIC :: fft_library_init, fft_library_finalize
   PUBLIC :: fft_library_acc_init, fft_library_acc_finalize
   PUBLIC :: fft_print_timing_report
   PUBLIC :: fft_backend_in_use, fft_use_mpi, fft_has_guru_interface
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
            INTEGER(C_INT), DIMENSION(3) :: fft_size
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
            INTEGER(C_INT), DIMENSION(3) :: fft_size
            COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*), INTENT(INOUT) :: grid
         END SUBROUTINE fft_3d_bw_local_inplace_c
      END INTERFACE

      CALL fft_3d_bw_local_inplace_c([fft_size(3), fft_size(2), fft_size(1)], grid)

   END SUBROUTINE fft_3d_bw_local_inplace

END MODULE fft_api
