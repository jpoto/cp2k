!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Includes all necessary routines, functions and parameters from
!>        libxc. Provides CP2K routines/functions where the LibXC calling list
!>        is version dependent (>=4.0.3). The naming convention for such
!>        routines/functions is xc_f03_XXX --> 'xc_libxc_wrap_XXX'. All version
!>        independent routines/functions are just bypassed to higher level
!>        module file 'xc_libxc'.
!>
!> \par History
!>      08.2015 created [A. Gloess (agloess)]
!>      01.2018 refactoring [A. Gloess (agloess)]
!>      10.2018/04.2019 added hyb_mgga [S. Simko, included by F. Stein]
!> \author A. Gloess (agloess)
! **************************************************************************************************
MODULE xc_libxc_wrap
#if defined (__LIBXC)
#include <xc_version.h>
! check for LibXC version - updated for LibXC 7.x
#if (XC_MAJOR_VERSION < 7)
   This version of CP2K requires libxc version 7.x or above for full functionality.
   Some features may not be available with older versions.
#else
   USE kinds, ONLY: dp
   USE xc_f03_lib_m, ONLY: xc_f03_func_end, &
                           xc_f03_func_init, &
                           xc_f03_functional_get_name, &
                           xc_f03_func_set_ext_params, &
                           xc_f03_func_set_dens_threshold, &
                           xc_f03_func_set_tau_threshold, &
                           xc_f03_version_string, &
                           xc_f03_reference, &
                           xc_f03_reference_doi, &
                           xc_f03_functional_get_number, &
                           xc_f03_available_functional_numbers, &
                           xc_f03_available_functional_names, &
                           xc_f03_maximum_name_length, &
                           xc_f03_number_of_functionals, &
                           !
                           xc_f03_gga_exc, &
                           xc_f03_gga_exc_vxc, &
                           xc_f03_gga_exc_vxc_fxc, &
                           xc_f03_gga_fxc, &
                           xc_f03_gga_vxc, &
                           xc_f03_gga_vxc_fxc, &
                           !
                           xc_f03_func_get_info, &
                           xc_f03_func_info_get_family, &
                           xc_f03_func_info_get_kind, &
                           xc_f03_func_info_get_name, &
                           xc_f03_func_info_get_references, &
                           xc_f03_func_info_get_flags, &
                           xc_f03_func_info_get_n_ext_params, &
                           xc_f03_func_info_get_ext_params_name, &
                           xc_f03_func_info_get_ext_params_default_value, &
                           xc_f03_func_info_get_ext_params_description, &
                           !
                           xc_f03_func_reference_get_ref, &
                           xc_f03_func_reference_get_doi, &
                           !
                           xc_f03_lda_exc_vxc_fxc_kxc, &
                           xc_f03_lda_exc, &
                           xc_f03_lda_exc_vxc, &
                           xc_f03_lda_exc_vxc_fxc, &
                           xc_f03_lda_fxc, &
                           xc_f03_lda_kxc, &
                           xc_f03_lda_vxc, &
                           !
                           xc_f03_mgga_exc_vxc_fxc, &
                           xc_f03_mgga_exc, &
                           xc_f03_mgga_exc_vxc, &
                           xc_f03_mgga_fxc, &
                           xc_f03_mgga_vxc, &
                           xc_f03_mgga_vxc_fxc, &
                           !
                           xc_f03_func_t, &
                           xc_f03_func_info_t, &
                           xc_f03_func_reference_t, &
                           !
                           XC_FAMILY_LDA, &
                           XC_FAMILY_GGA, &
                           XC_FAMILY_MGGA, &
                           XC_FAMILY_HYB_LDA, &
                           XC_FAMILY_HYB_GGA, &
                           XC_FAMILY_HYB_MGGA, &
                           !
                           XC_UNPOLARIZED, &
                           XC_POLARIZED, &
                           !
                           XC_EXCHANGE, &
                           XC_CORRELATION, &
                           XC_EXCHANGE_CORRELATION, &
                           XC_KINETIC, &
                           !
                           XC_FLAGS_NEEDS_LAPLACIAN, &
                           XC_FLAGS_HAVE_EXC, &
                           XC_FLAGS_DEVELOPMENT, &
                           XC_FLAGS_ON_HOST, &
                           XC_FLAGS_ON_DEVICE, &
                           xc_f03_func_info_set_default_flags

#include "../base/base_uses.f90"
   USE ISO_C_BINDING, ONLY: C_PTR, C_NULL_PTR, C_SIZE_T, C_ASSOCIATED, C_LOC
#endif
   IMPLICIT NONE
   PRIVATE
#if defined (__LIBXC)

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_libxc_wrap'

   CHARACTER(LEN=*), PARAMETER, PUBLIC :: libxc_version = XC_VERSION

   PUBLIC :: xc_f03_func_t, xc_f03_func_info_t
   PUBLIC :: xc_f03_func_init, xc_f03_func_end, xc_f03_func_set_ext_params
   PUBLIC :: xc_f03_functional_get_name, xc_f03_available_functional_numbers, xc_f03_maximum_name_length, &
             xc_f03_number_of_functionals, xc_f03_available_functional_names
   PUBLIC :: xc_f03_func_get_info, xc_f03_func_info_get_family, xc_f03_func_info_get_kind, &
             xc_f03_func_info_get_name, xc_f03_func_info_get_ext_params_name, &
             xc_f03_func_info_get_ext_params_description, xc_f03_func_info_get_ext_params_default_value, &
             xc_f03_func_info_get_n_ext_params
   PUBLIC :: xc_f03_gga_exc, xc_f03_gga_exc_vxc, xc_f03_gga_exc_vxc_fxc, xc_f03_gga_fxc, &
             xc_f03_gga_vxc, xc_f03_gga_vxc_fxc
   PUBLIC :: xc_f03_lda_exc_vxc_fxc_kxc, &
             xc_f03_lda_exc, xc_f03_lda_exc_vxc, xc_f03_lda_exc_vxc_fxc, &
             xc_f03_lda_fxc, xc_f03_lda_kxc, xc_f03_lda_vxc
   PUBLIC :: xc_f03_mgga_exc_vxc_fxc, &
             xc_f03_mgga_exc, xc_f03_mgga_exc_vxc, xc_f03_mgga_fxc, &
             xc_f03_mgga_vxc, xc_f03_mgga_vxc_fxc

   PUBLIC :: XC_FAMILY_LDA, XC_FAMILY_GGA, XC_FAMILY_MGGA, &
             XC_FAMILY_HYB_LDA, XC_FAMILY_HYB_GGA, XC_FAMILY_HYB_MGGA

   PUBLIC :: XC_UNPOLARIZED, XC_POLARIZED

   PUBLIC :: XC_EXCHANGE, XC_CORRELATION, XC_EXCHANGE_CORRELATION, XC_KINETIC
   PUBLIC :: XC_FLAGS_ON_HOST, XC_FLAGS_ON_DEVICE

   ! Create aliases for backward compatibility with LibXC 5.x interface
   PUBLIC :: xc_f03_lda, xc_f03_mgga

   ! Backend selection constants (similar to SPLA)
   INTEGER, PARAMETER, PUBLIC :: do_libxc_cpu = 1, &
                                 do_libxc_gpu = 2, &
                                 do_libxc_safe = 3

   ! Device buffer type for CUDA memory management
   TYPE, PUBLIC :: libxc_device_buffer_type
      TYPE(C_PTR) :: ptr = C_NULL_PTR
      INTEGER(KIND=C_SIZE_T) :: size = 0
   END TYPE libxc_device_buffer_type

! wrappers for routines
   PUBLIC :: xc_libxc_wrap_info_refs, &
             xc_libxc_wrap_version, &
             xc_libxc_wrap_library_reference, &
             xc_libxc_wrap_info_needs_laplace, &
             xc_libxc_wrap_info_no_exc, &
             xc_libxc_wrap_set_thresholds, &
             xc_libxc_wrap_functional_get_number, &
             xc_cuda_wrapper_init, &
             xc_cuda_wrapper_cleanup, &
             xc_cuda_wrapper_is_available, &
             xc_libxc_wrap_init_cuda_safe, &
             xc_libxc_wrap_is_under_development, &
             xc_libxc_get_reference_length, &
             xc_libxc_check_functional, &
             libxc_set_backend, libxc_get_backend, &
             libxc_func_init_wrapper, libxc_func_end_wrapper, &
             libxc_set_library, &
             libxc_is_cuda, &
             libxc_device_buffer_alloc, libxc_device_buffer_free, &
             libxc_host_to_device, libxc_device_to_host, &
             libxc_host_to_device_2d, libxc_device_to_host_2d, &
             libxc_assert_device_buffer, &
             libxc_device_buffer_get_ptr, &
             libxc_buffer_create, libxc_buffer_free, &
             libxc_buffer_h2d, libxc_buffer_d2h, &
             libxc_gga_exc_vxc_wrapper, libxc_gga_exc_wrapper, &
             libxc_lda_exc_vxc_wrapper, libxc_lda_exc_wrapper, &
             libxc_mgga_exc_vxc_wrapper, libxc_mgga_exc_wrapper, &
             libxc_mgga_vxc_wrapper, &
             libxc_lda_exc_vxc_spin_wrapper, libxc_gga_exc_vxc_spin_wrapper, &
             libxc_mgga_exc_vxc_spin_wrapper, &
             libxc_gga_exc_vxc_cuda_wrapper, libxc_mgga_exc_vxc_cuda_wrapper, &
             libxc_lda_exc_vxc_device, libxc_lda_exc_device, &
             libxc_gga_exc_device, libxc_gga_exc_vxc_device, &
             libxc_gga_vxc_device, &
             libxc_mgga_exc_device, libxc_mgga_exc_vxc_device, &
             libxc_mgga_vxc_device, &
             libxc_lda_exc_vxc_spin_device, libxc_gga_exc_vxc_spin_device, &
             libxc_mgga_exc_vxc_spin_device

   ! Backend selection for CPU vs GPU Libxc
   INTEGER, PARAMETER :: LIBXC_BACKEND_CPU = 1
   INTEGER, PARAMETER :: LIBXC_BACKEND_GPU = 2
   INTEGER :: libxc_current_backend = LIBXC_BACKEND_CPU
   INTEGER :: libxc_selected_library = do_libxc_cpu

   ! Alias definitions for backward compatibility
   PROCEDURE(xc_f03_lda_exc_vxc_fxc_kxc), POINTER :: xc_f03_lda => xc_f03_lda_exc_vxc_fxc_kxc
   PROCEDURE(xc_f03_mgga_exc_vxc_fxc), POINTER :: xc_f03_mgga => xc_f03_mgga_exc_vxc_fxc

   ! Interface declarations for CUDA wrapper functions
   INTERFACE
      FUNCTION xc_cuda_wrapper_init() BIND(C, name="xc_cuda_wrapper_init")
         USE ISO_C_BINDING, ONLY: C_INT
      INTEGER(KIND=C_INT)                                :: xc_cuda_wrapper_init

      END FUNCTION xc_cuda_wrapper_init

      SUBROUTINE xc_cuda_wrapper_cleanup() BIND(C, name="xc_cuda_wrapper_cleanup")
      END SUBROUTINE xc_cuda_wrapper_cleanup

      FUNCTION xc_cuda_wrapper_is_available() BIND(C, name="xc_cuda_wrapper_is_available")
         USE ISO_C_BINDING, ONLY: C_INT
      INTEGER(KIND=C_INT)                                :: xc_cuda_wrapper_is_available

      END FUNCTION xc_cuda_wrapper_is_available

      ! Interface for CUDA wrapper functions
      FUNCTION xc_cuda_wrapper_gga_exc_vxc_c(func, np, rho, exc, vrho, vsigma, work) BIND(C, name='xc_cuda_wrapper_gga_exc_vxc')
         USE ISO_C_BINDING
      TYPE(C_PTR), VALUE                                 :: func
      INTEGER(C_INT), VALUE                              :: np
      TYPE(C_PTR), VALUE                                 :: rho, exc, vrho, vsigma, work
      INTEGER(C_INT)                                     :: xc_cuda_wrapper_gga_exc_vxc_c

      END FUNCTION xc_cuda_wrapper_gga_exc_vxc_c

       FUNCTION xc_cuda_wrapper_mgga_exc_vxc_c(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau, work) BIND(C, name='xc_cuda_wrapper_mgga_exc_vxc')
         USE ISO_C_BINDING
      TYPE(C_PTR), VALUE                                 :: func
      INTEGER(C_INT), VALUE                              :: np
      TYPE(C_PTR), VALUE                                 :: rho, sigma, lapl, tau, exc, vrho, &
                                                            vsigma, vlapl, vtau, work
      INTEGER(C_INT)                                     :: xc_cuda_wrapper_mgga_exc_vxc_c

      END FUNCTION xc_cuda_wrapper_mgga_exc_vxc_c
   END INTERFACE

CONTAINS

   ! Backend selection functions
! **************************************************************************************************
!> \brief ...
!> \param backend ...
! **************************************************************************************************
   SUBROUTINE libxc_set_backend(backend)
      INTEGER, INTENT(IN)                                :: backend

      SELECT CASE (backend)
      CASE (LIBXC_BACKEND_CPU)
         libxc_current_backend = LIBXC_BACKEND_CPU
      CASE (LIBXC_BACKEND_GPU)
#if defined (__OFFLOAD_CUDA)
         libxc_current_backend = LIBXC_BACKEND_GPU
#else
         CPABORT("GPU backend requested but CUDA Libxc not available")
#endif
      CASE DEFAULT
         CPABORT("Invalid Libxc backend specified")
      END SELECT
   END SUBROUTINE libxc_set_backend

! **************************************************************************************************
!> \brief Set the libxc library to use (CPU or GPU), similar to SPLA backend selection
!> \param library The library to use (do_libxc_cpu or do_libxc_gpu)
! **************************************************************************************************
   SUBROUTINE libxc_set_library(library)
      INTEGER, INTENT(IN)                                :: library

      SELECT CASE (library)
      CASE (do_libxc_cpu)
         libxc_selected_library = do_libxc_cpu
#if defined (__OFFLOAD_CUDA)
      CASE (do_libxc_gpu)
         libxc_selected_library = do_libxc_gpu
      CASE (do_libxc_safe)
         libxc_selected_library = do_libxc_safe
#else
      CASE (do_libxc_gpu)
         CPABORT("GPU library requested but CUDA Libxc not available")
      CASE (do_libxc_safe)
         libxc_selected_library = do_libxc_safe
#endif
      CASE DEFAULT
         CPABORT("Invalid Libxc library specified")
      END SELECT

      ! Automatically set the backend based on library selection
      IF (libxc_selected_library == do_libxc_gpu) THEN
         libxc_current_backend = LIBXC_BACKEND_GPU
      ELSE IF (libxc_selected_library == do_libxc_safe) THEN
         libxc_current_backend = LIBXC_BACKEND_CPU
      ELSE
         libxc_current_backend = LIBXC_BACKEND_CPU
      END IF
   END SUBROUTINE libxc_set_library

! **************************************************************************************************
!> \brief ...
!> \return ...
! **************************************************************************************************
   FUNCTION libxc_get_backend() RESULT(backend)
      INTEGER                                            :: backend

      backend = libxc_current_backend
   END FUNCTION libxc_get_backend

! **************************************************************************************************
!> \brief ...
! **************************************************************************************************
   SUBROUTINE xc_libxc_wrap_init_cuda_safe()
      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_init_cuda_safe', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      ! Set default flags based on CUDA availability
#if defined (__OFFLOAD_CUDA)
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
      libxc_selected_library = do_libxc_gpu
      libxc_current_backend = LIBXC_BACKEND_GPU
#else
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      libxc_selected_library = do_libxc_cpu
      libxc_current_backend = LIBXC_BACKEND_CPU
#endif

      CALL timestop(handle)
   END SUBROUTINE xc_libxc_wrap_init_cuda_safe

! **************************************************************************************************
!> \brief Provides the reference(s) for this functional.
!> \param xc_info func_info object of the functional
!> \return upper bound for the length of the reference string
!> \author F. Stein
! **************************************************************************************************
   FUNCTION xc_libxc_get_reference_length(xc_info) RESULT(length)

      TYPE(xc_f03_func_info_t), INTENT(IN)               :: xc_info
      INTEGER                                            :: length

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_get_reference_length'
      INTEGER, PARAMETER                                 :: maxlen = 67

      CHARACTER(LEN=1024)                                :: doi_string, ref_string
      CHARACTER(LEN=128)                                 :: descr_string
      INTEGER                                            :: handle, i, i_ref, i_ref_old, n_params
      TYPE(xc_f03_func_reference_t)                      :: xc_ref

      CALL timeset(routineN, handle)

      ! We are counting the number of necessary lines by carrying out a dry run of xc_libxc_wrap_info_refs
      i_ref = 0
      i_ref_old = -1
      length = 0
      DO WHILE (i_ref >= 0 .AND. i_ref /= i_ref_old)
         ! information about functional references
         xc_ref = xc_f03_func_info_get_references(xc_info, i_ref)
         ref_string = xc_f03_func_reference_get_ref(xc_ref)
         doi_string = xc_f03_func_reference_get_doi(xc_ref)
         length = length + LEN_TRIM(ref_string) + LEN_TRIM(doi_string) + 11
         IF (MOD(length, maxlen) /= 0) length = length + maxlen - MOD(length, maxlen)
         ! information about (optional) external parameters
         n_params = xc_f03_func_info_get_n_ext_params(xc_info)
         IF (n_params > 0) THEN
            length = length + maxlen
         END IF
         DO i = 1, n_params
            descr_string = xc_f03_func_info_get_ext_params_description(xc_info, i - 1)
            length = length + LEN_TRIM(descr_string) + 3
            IF (MOD(length, maxlen) /= 0) length = length + maxlen - MOD(length, maxlen)
         END DO
         i_ref_old = i_ref
      END DO
      ! two additional lines for spin polarization, scaling factor and buffer
      length = length + 2*maxlen

      CALL timestop(handle)

   END FUNCTION xc_libxc_get_reference_length

! **************************************************************************************************
!> \brief Provides the reference(s) for this functional.
!> \param xc_info ...
!> \param polarized ...
!> \param sc ...
!> \param reference ...
!>
!> \author A. Gloess (agloess)
! **************************************************************************************************
   SUBROUTINE xc_libxc_wrap_info_refs(xc_info, polarized, sc, reference)
      TYPE(xc_f03_func_info_t), INTENT(IN)               :: xc_info
      INTEGER, INTENT(IN)                                :: polarized
      REAL(KIND=dp), INTENT(IN)                          :: sc
      CHARACTER(LEN=*), INTENT(OUT)                      :: reference

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_info_refs'
      INTEGER, PARAMETER                                 :: maxlen = 67

      CHARACTER(LEN=1028)                                :: doi_string, ref_string
      CHARACTER(LEN=128)                                 :: descr_string
      CHARACTER(LEN=2067)                                :: tmp_string
      INTEGER                                            :: empty, first, handle, i, i_ref, &
                                                            i_ref_old, idx, last, n_params
      TYPE(xc_f03_func_reference_t)                      :: xc_ref

! conservative estimate of the necessary length: 2*1028+11=2067

      CALL timeset(routineN, handle)

      i_ref = 0
      i_ref_old = -1
      idx = 1
      first = 1
      DO WHILE (i_ref >= 0 .AND. i_ref /= i_ref_old)
         ! information about functional references
         xc_ref = xc_f03_func_info_get_references(xc_info, i_ref)
         ref_string = xc_f03_func_reference_get_ref(xc_ref)
         doi_string = xc_f03_func_reference_get_doi(xc_ref)
         WRITE (tmp_string, '(a1,i1,a2,a,a7,a)') '[', idx, '] ', &
            TRIM(ref_string), ', doi: ', TRIM(doi_string)
         last = first + LEN_TRIM(tmp_string) - 1
         reference(first:last) = TRIM(tmp_string)
         first = last + 1
         empty = last + (maxlen - 1) - MOD(last - 1, maxlen)
         ! fill up line with 'spaces'
         IF (empty /= last) THEN
            reference(first:empty) = ' '
            first = empty + 1
         END IF
         ! information about (optional) external parameters
         n_params = xc_f03_func_info_get_n_ext_params(xc_info)
         IF (n_params > 0) THEN
            reference(first:first + maxlen - 1) = 'Optional external parameters:'//REPEAT(' ', maxlen - 28)
            first = first + maxlen
         END IF
         DO i = 1, n_params
            descr_string = xc_f03_func_info_get_ext_params_description(xc_info, i - 1)
            last = first + LEN_TRIM(descr_string) - 1 + 3
            reference(first:last) = ' * '//TRIM(descr_string)
            first = last + 1
            empty = last + (maxlen - 1) - MOD(last - 1, maxlen)
            ! fill up line with 'spaces'

            IF (empty /= last) THEN
               reference(first:empty) = ' '
               first = empty + 1
            END IF
         END DO
         idx = idx + 1
         i_ref_old = i_ref
      END DO
      SELECT CASE (polarized)
      CASE (XC_UNPOLARIZED)
         WRITE (tmp_string, "('{scale=',f5.3,', spin-unpolarized}')") sc
      CASE (XC_POLARIZED)
         WRITE (tmp_string, "('{scale=',f5.3,', spin-polarized}')") sc
      CASE default
         CPABORT("Unsupported value for variable 'polarized'.")
      END SELECT
      last = first + LEN_TRIM(tmp_string) - 1
      reference(first:last) = TRIM(tmp_string)
      first = last + 1
      ! fill with 'spaces'
      reference(first:LEN(reference)) = ' '

      IF (last > LEN(reference)) &
         CPABORT("Faulty reference length.")

      CALL timestop(handle)

   END SUBROUTINE xc_libxc_wrap_info_refs

! **************************************************************************************************
!> \brief Provides the version of the LibXC library CP2K is running against.
!> \param version version string reported by the library at run time
!> \param compiled_version version of the headers CP2K was compiled against
!> \author A. Gloess (agloess)
!> \note The run-time and compile-time versions differ whenever CP2K is executed
!>       against a shared LibXC other than the one it was built with. Reporting
!>       the run-time version is what actually describes the numbers produced.
!>
! **************************************************************************************************
   SUBROUTINE xc_libxc_wrap_version(version, compiled_version)
      CHARACTER(LEN=*), INTENT(OUT)                      :: version
      CHARACTER(LEN=*), INTENT(OUT), OPTIONAL            :: compiled_version

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_version'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CALL xc_f03_version_string(version)
      IF (PRESENT(compiled_version)) compiled_version = TRIM(libxc_version)

      CALL timestop(handle)

   END SUBROUTINE xc_libxc_wrap_version

! **************************************************************************************************
!> \brief Provides the citation LibXC asks to be given for the library itself.
!> \param reference bibliographic reference
!> \param doi digital object identifier of that reference
!> \author S. Lehtola
!> \note Taken from the library at run time rather than hard-coded, so that the
!>       citation stays correct when LibXC updates its own self-citation.
!>
! **************************************************************************************************
   SUBROUTINE xc_libxc_wrap_library_reference(reference, doi)
      CHARACTER(LEN=*), INTENT(OUT)                      :: reference, doi

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_library_reference'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CALL xc_f03_reference(reference)
      CALL xc_f03_reference_doi(doi)

      CALL timestop(handle)

   END SUBROUTINE xc_libxc_wrap_library_reference

! **************************************************************************************************
!> \brief Applies the CP2K screening cutoffs to a LibXC functional object.
!> \param xc_func LibXC functional object
!> \param xc_info info object belonging to xc_func
!> \param epsilon_rho density cutoff (XC%DENSITY_CUTOFF)
!> \param epsilon_tau kinetic energy density cutoff (XC%TAU_CUTOFF)
!> \author S. Lehtola
!> \note LibXC screens every point against the density threshold internally, which
!>       is what allows CP2K to hand it whole blocks of points instead of filtering
!>       point by point. The sigma threshold is deliberately left at LibXC's own
!>       default: CP2K has no cutoff of its own on the gradient here, and raising
!>       it in step with the density cutoff would floor sigma for points with a
!>       small gradient that used to be evaluated as given.
!>
! **************************************************************************************************
   SUBROUTINE xc_libxc_wrap_set_thresholds(xc_func, xc_info, epsilon_rho, epsilon_tau)
      TYPE(xc_f03_func_t), INTENT(INOUT)                 :: xc_func
      TYPE(xc_f03_func_info_t), INTENT(IN)               :: xc_info
      REAL(KIND=dp), INTENT(IN)                          :: epsilon_rho, epsilon_tau

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_set_thresholds'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CALL xc_f03_func_set_dens_threshold(xc_func, epsilon_rho)

      SELECT CASE (xc_f03_func_info_get_family(xc_info))
      CASE (XC_FAMILY_MGGA, XC_FAMILY_HYB_MGGA)
         CALL xc_f03_func_set_tau_threshold(xc_func, epsilon_tau)
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE xc_libxc_wrap_set_thresholds

! **************************************************************************************************
!> \brief Checks existence of functional in LibXC
!> \param func_string ...
!> \return ...
!> \author F. Stein
!> \note Remove prefix to keep compatibility, functionals can be specified (in
!>       LIBXC section) as:
!>       GGA_X_...  or  XC_GGA_X_...
!>       Starting from version 2.2.0 both name conventions are allowed, before
!>       the 'XC_' prefix was necessary.
!>
! **************************************************************************************************
   LOGICAL FUNCTION xc_libxc_check_functional(func_string) RESULT(exists)
      CHARACTER(LEN=*), INTENT(IN)                       :: func_string

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_check_functional'

      INTEGER                                            :: func_id, handle

      CALL timeset(routineN, handle)

      IF (func_string(1:3) == "XC_") THEN
         func_id = xc_f03_functional_get_number(func_string(4:LEN_TRIM(func_string)))
      ELSE
         func_id = xc_f03_functional_get_number(func_string(1:LEN_TRIM(func_string)))
      END IF

      exists = .TRUE.
      IF (func_id == -1) exists = .FALSE.

      CALL timestop(handle)

   END FUNCTION xc_libxc_check_functional

! **************************************************************************************************
!> \brief Provides the functional ID.
!> \param func_string ...
!> \return ...
!> \author A. Gloess (agloess)
!> \note Remove prefix to keep compatibility, functionals can be specified (in
!>       LIBXC section) as:
!>       GGA_X_...  or  XC_GGA_X_...
!>       Starting from version 2.2.0 both name conventions are allowed, before
!>       the 'XC_' prefix was necessary.
!>
! **************************************************************************************************
   INTEGER FUNCTION xc_libxc_wrap_functional_get_number(func_string) RESULT(func_id)
      CHARACTER(LEN=*), INTENT(IN)                       :: func_string

      CHARACTER(LEN=*), PARAMETER :: routineN = 'xc_libxc_wrap_functional_get_number'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      IF (func_string(1:3) == "XC_") THEN
         func_id = xc_f03_functional_get_number(func_string(4:LEN_TRIM(func_string)))
      ELSE
         func_id = xc_f03_functional_get_number(func_string(1:LEN_TRIM(func_string)))
      END IF
      IF (func_id == -1) THEN
         CPABORT(TRIM(func_string)//": wrong functional name")
      END IF

      CALL timestop(handle)

   END FUNCTION xc_libxc_wrap_functional_get_number

! **************************************************************************************************
!> \brief Wrapper to test wether functional is considered under development in Libxc
!> \param xc_info ...
!>
!> \return ...
!> \author F. Stein (fstein93)
! **************************************************************************************************
   LOGICAL FUNCTION xc_libxc_wrap_is_under_development(xc_info)
      TYPE(xc_f03_func_info_t)                           :: xc_info

      IF (IAND(xc_f03_func_info_get_flags(xc_info), XC_FLAGS_DEVELOPMENT) == XC_FLAGS_DEVELOPMENT) THEN
         xc_libxc_wrap_is_under_development = .TRUE.
      ELSE
         xc_libxc_wrap_is_under_development = .FALSE.
      END IF

   END FUNCTION xc_libxc_wrap_is_under_development

! **************************************************************************************************
!> \brief Wrapper for functionals that need the Laplacian, all others can use
!>        a dummy array.
!> \param xc_info info object of an already initialized functional
!>
!> \return ...
!> \author A. Gloess (agloess)
!> \note Reads the flag off an info object the caller already holds. Creating a
!>       throw-away functional object just to look at one flag used to cost an
!>       extra xc_f03_func_init/xc_f03_func_end pair per evaluated batch.
! **************************************************************************************************
   LOGICAL FUNCTION xc_libxc_wrap_info_needs_laplace(xc_info)
      ! Only some MGGA functionals needs the laplacian
      TYPE(xc_f03_func_info_t), INTENT(IN)               :: xc_info

      IF (IAND(xc_f03_func_info_get_flags(xc_info), XC_FLAGS_NEEDS_LAPLACIAN) == XC_FLAGS_NEEDS_LAPLACIAN) THEN
         xc_libxc_wrap_info_needs_laplace = .TRUE.
      ELSE
         xc_libxc_wrap_info_needs_laplace = .FALSE.
      END IF

   END FUNCTION xc_libxc_wrap_info_needs_laplace

! **************************************************************************************************
!> \brief Whether the functional does not provide the energy density.
!> \param xc_info info object of an already initialized functional
!>
!> \return .TRUE. when LibXC cannot supply Exc for this functional
!> \author S. Lehtola
! **************************************************************************************************
   LOGICAL FUNCTION xc_libxc_wrap_info_no_exc(xc_info)
      TYPE(xc_f03_func_info_t), INTENT(IN)               :: xc_info

      IF (IAND(xc_f03_func_info_get_flags(xc_info), XC_FLAGS_HAVE_EXC) == XC_FLAGS_HAVE_EXC) THEN
         xc_libxc_wrap_info_no_exc = .FALSE.
      ELSE
         xc_libxc_wrap_info_no_exc = .TRUE.
      END IF

   END FUNCTION xc_libxc_wrap_info_no_exc

   ! Backend-aware wrapper functions for key Libxc operations
   ! These functions will dispatch to either CPU or CUDA implementations
   ! based on the current backend setting

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param functional ...
!> \param nspin ...
! **************************************************************************************************
   SUBROUTINE libxc_func_init_wrapper(func, functional, nspin)
      TYPE(xc_f03_func_t), INTENT(OUT)                   :: func
      INTEGER, INTENT(IN)                                :: functional
      INTEGER, INTENT(IN), OPTIONAL                      :: nspin

      SELECT CASE (libxc_current_backend)
      CASE (LIBXC_BACKEND_CPU)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
         IF (PRESENT(nspin)) THEN
            CALL xc_f03_func_init(func, functional, nspin)
         ELSE
            CALL xc_f03_func_init(func, functional, XC_UNPOLARIZED)
         END IF
      CASE (LIBXC_BACKEND_GPU)
#if defined (__OFFLOAD_CUDA)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         IF (PRESENT(nspin)) THEN
            CALL xc_f03_func_init(func, functional, nspin)
         ELSE
            CALL xc_f03_func_init(func, functional, XC_UNPOLARIZED)
         END IF
#else
         CPABORT("GPU backend requested but CUDA Libxc not available")
#endif
      END SELECT
   END SUBROUTINE libxc_func_init_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
! **************************************************************************************************
   SUBROUTINE libxc_func_end_wrapper(func)
      TYPE(xc_f03_func_t), INTENT(INOUT)                 :: func

      SELECT CASE (libxc_current_backend)
      CASE (LIBXC_BACKEND_CPU)
         CALL xc_f03_func_end(func)
      CASE (LIBXC_BACKEND_GPU)
#if defined (__OFFLOAD_CUDA)
         ! For now, use CPU version until proper CUDA symbols are available
         CALL xc_f03_func_end(func)
         ! Directly call CUDA-enabled functions (same symbols, GPU-accelerated internally)
#else
         CPABORT("GPU backend requested but CUDA Libxc not available")
#endif
      END SELECT
   END SUBROUTINE libxc_func_end_wrapper

#endif
#endif

   ! Device buffer management functions
! **************************************************************************************************
!> \brief ...
!> \return ...
! **************************************************************************************************
   FUNCTION libxc_is_cuda() RESULT(is_cuda)
      LOGICAL                                            :: is_cuda

      is_cuda = (libxc_current_backend == LIBXC_BACKEND_GPU)
   END FUNCTION libxc_is_cuda

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE libxc_device_buffer_alloc(buffer, n)
      USE ISO_C_BINDING, ONLY: C_PTR, C_NULL_PTR, C_INT64_T
      USE offload_api, ONLY: offload_create_buffer, offload_buffer_type
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: buffer
      INTEGER, INTENT(IN)                                :: n

      TYPE(offload_buffer_type)                          :: offload_buffer

      buffer%size = INT(n, KIND=C_INT64_T)
      CALL offload_create_buffer(n, offload_buffer)
      ! Store the device pointer from the offload buffer
      buffer%ptr = offload_buffer%c_ptr
   END SUBROUTINE libxc_device_buffer_alloc

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE libxc_device_buffer_free(buffer)
      USE ISO_C_BINDING, ONLY: C_PTR, C_NULL_PTR
      USE offload_api, ONLY: offload_free_buffer, offload_buffer_type
      TYPE(libxc_device_buffer_type), INTENT(INOUT)      :: buffer

      TYPE(offload_buffer_type)                          :: offload_buffer

      ! Create a temporary offload buffer with the device pointer
      offload_buffer%c_ptr = buffer%ptr
      offload_buffer%host_buffer => NULL()

      CALL offload_free_buffer(offload_buffer)
      buffer%ptr = C_NULL_PTR
      buffer%size = 0
   END SUBROUTINE libxc_device_buffer_free

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param host ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE libxc_host_to_device(buffer, host, n)
      USE ISO_C_BINDING, ONLY: C_PTR, C_LOC
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_h2d
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: host
      INTEGER, INTENT(IN)                                :: n

      REAL(KIND=dp), DIMENSION(n), TARGET                :: temp_host
      TYPE(offload_buffer_type)                          :: offload_buffer

      ! Copy host data to a temporary array with proper bounds
      temp_host(:) = host(1:n)

      ! Create a temporary offload buffer with the device pointer
      offload_buffer%c_ptr = buffer%ptr
      offload_buffer%host_buffer => temp_host

      CALL offload_buffer_h2d(offload_buffer, n)
   END SUBROUTINE libxc_host_to_device

   ! Specialized transfer function for 2D host arrays (component, point) to 1D device buffers
   ! This handles the data layout conversion from CP2K format to Libxc format
! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param host_2d ...
!> \param ncomp ...
!> \param npoints ...
! **************************************************************************************************
   SUBROUTINE libxc_host_to_device_2d(buffer, host_2d, ncomp, npoints)
      USE ISO_C_BINDING, ONLY: C_PTR, C_LOC
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_h2d
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: host_2d
      INTEGER, INTENT(IN)                                :: ncomp, npoints

      INTEGER                                            :: i, idx, j
      REAL(KIND=dp), DIMENSION(ncomp*npoints), TARGET    :: temp_host
      TYPE(offload_buffer_type)                          :: offload_buffer

      ! Transpose from CP2K layout (component, point) to Libxc layout (interleaved)
      idx = 1
      DO i = 1, npoints
         DO j = 1, ncomp
            temp_host(idx) = host_2d(j, i)
            idx = idx + 1
         END DO
      END DO

      ! Create a temporary offload buffer with the device pointer
      offload_buffer%c_ptr = buffer%ptr
      offload_buffer%host_buffer => temp_host

      CALL offload_buffer_h2d(offload_buffer, ncomp*npoints)
   END SUBROUTINE libxc_host_to_device_2d

! **************************************************************************************************
!> \brief ...
!> \param host ...
!> \param buffer ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE libxc_device_to_host(host, buffer, n)
      USE ISO_C_BINDING, ONLY: C_PTR, C_LOC
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_d2h
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: host
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer
      INTEGER, INTENT(IN)                                :: n

      REAL(KIND=dp), DIMENSION(n), TARGET                :: temp_host
      TYPE(offload_buffer_type)                          :: offload_buffer

      ! Create a temporary offload buffer with the device pointer
      offload_buffer%c_ptr = buffer%ptr
      offload_buffer%host_buffer => temp_host

      CALL offload_buffer_d2h(offload_buffer, n)

      ! Copy data from temporary array to output
      host(1:n) = temp_host(:)
   END SUBROUTINE libxc_device_to_host

   ! Specialized transfer function for 1D device buffers to 2D host arrays (component, point)
   ! This handles the data layout conversion from Libxc format to CP2K format
! **************************************************************************************************
!> \brief ...
!> \param host_2d ...
!> \param buffer ...
!> \param ncomp ...
!> \param npoints ...
! **************************************************************************************************
   SUBROUTINE libxc_device_to_host_2d(host_2d, buffer, ncomp, npoints)
      USE ISO_C_BINDING, ONLY: C_PTR, C_LOC
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_d2h
      REAL(KIND=dp), DIMENSION(:, :), INTENT(OUT)        :: host_2d
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer
      INTEGER, INTENT(IN)                                :: ncomp, npoints

      INTEGER                                            :: i, idx, j
      REAL(KIND=dp), DIMENSION(ncomp*npoints), TARGET    :: temp_host
      TYPE(offload_buffer_type)                          :: offload_buffer

      ! Create a temporary offload buffer with the device pointer
      offload_buffer%c_ptr = buffer%ptr
      offload_buffer%host_buffer => temp_host

      CALL offload_buffer_d2h(offload_buffer, ncomp*npoints)

      ! Transpose from Libxc layout (interleaved) to CP2K layout (component, point)
      idx = 1
      DO i = 1, npoints
         DO j = 1, ncomp
            host_2d(j, i) = temp_host(idx)
            idx = idx + 1
         END DO
      END DO
   END SUBROUTINE libxc_device_to_host_2d

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   LOGICAL FUNCTION libxc_assert_device_buffer(buffer)
      USE ISO_C_BINDING, ONLY: C_PTR, C_NULL_PTR, C_ASSOCIATED
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer

      libxc_assert_device_buffer = .TRUE.
      IF (.NOT. C_ASSOCIATED(buffer%ptr)) THEN
         CPABORT("Device buffer pointer is not associated")
         libxc_assert_device_buffer = .FALSE.
      END IF
      IF (buffer%size <= 0) THEN
         CPABORT("Device buffer size is invalid")
         libxc_assert_device_buffer = .FALSE.
      END IF
   END FUNCTION libxc_assert_device_buffer

   ! Function to get the actual device pointer from a libxc_device_buffer_type
! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   FUNCTION libxc_device_buffer_get_ptr(buffer) RESULT(ptr)
      USE ISO_C_BINDING, ONLY: C_PTR
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: buffer
      TYPE(C_PTR)                                        :: ptr

      ptr = buffer%ptr
   END FUNCTION libxc_device_buffer_get_ptr

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE libxc_buffer_create(buffer, length)
      USE offload_api, ONLY: offload_buffer_type, offload_create_buffer
      TYPE(offload_buffer_type), INTENT(OUT)             :: buffer
      INTEGER, INTENT(IN)                                :: length

      CALL offload_create_buffer(length, buffer)
   END SUBROUTINE libxc_buffer_create

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE libxc_buffer_free(buffer)
      USE offload_api, ONLY: offload_buffer_type, offload_free_buffer
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer

      CALL offload_free_buffer(buffer)
   END SUBROUTINE libxc_buffer_free

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE libxc_buffer_h2d(buffer, length)
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_h2d
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer
      INTEGER, INTENT(IN)                                :: length

      CALL offload_buffer_h2d(buffer, length)
   END SUBROUTINE libxc_buffer_h2d

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE libxc_buffer_d2h(buffer, length)
      USE offload_api, ONLY: offload_buffer_type, offload_buffer_d2h
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer
      INTEGER, INTENT(IN)                                :: length

      CALL offload_buffer_d2h(buffer, length)
   END SUBROUTINE libxc_buffer_d2h

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_vxc_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_gga_exc_vxc(func, np, rho, sigma, exc, vrho, vsigma)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         CALL xc_f03_gga_exc_vxc(func, np, rho, sigma, exc, vrho, vsigma)
      END IF
   END SUBROUTINE libxc_gga_exc_vxc_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param exc ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_wrapper(func, np, rho, sigma, exc)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_gga_exc(func, np, rho, sigma, exc)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         CALL xc_f03_gga_exc(func, np, rho, sigma, exc)
      END IF
   END SUBROUTINE libxc_gga_exc_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param exc ...
!> \param vrho ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_vxc_wrapper(func, np, rho, exc, vrho)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_lda_exc_vxc(func, np, rho, exc, vrho)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         CALL xc_f03_lda_exc_vxc(func, np, rho, exc, vrho)
      END IF
   END SUBROUTINE libxc_lda_exc_vxc_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param exc ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_wrapper(func, np, rho, exc)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_lda_exc(func, np, rho, exc)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         CALL xc_f03_lda_exc(func, np, rho, exc)
      END IF
   END SUBROUTINE libxc_lda_exc_wrapper

   ! Device-aware LDA wrapper functions
   ! These functions accept libxc_device_buffer_type and call CUDA Libxc directly
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_vxc_device(func, rho_d, exc_d, vrho_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, rho_ptr, vrho_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr, c_vrho_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_lda_exc_vxc(func, np, rho_ptr, exc_ptr, vrho_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_lda_exc_vxc_device

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param exc_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_device(func, rho_d, exc_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, rho_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_lda_exc(func, np, rho_ptr, exc_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_lda_exc_device

   ! Device-aware GGA wrapper functions
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param exc_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_device(func, rho_d, sigma_d, exc_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, rho_ptr, sigma_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr, c_sigma_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_gga_exc(func, np, rho_ptr, sigma_ptr, exc_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_gga_exc_device

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_vxc_device(func, rho_d, sigma_d, exc_d, vrho_d, vsigma_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d, vsigma_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, rho_ptr, sigma_ptr, vrho_ptr, &
                                                            vsigma_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr, c_sigma_ptr, &
                                                            c_vrho_ptr, c_vsigma_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [INT(np)])

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_gga_exc_vxc(func, np, rho_ptr, sigma_ptr, exc_ptr, vrho_ptr, vsigma_ptr)
   END SUBROUTINE libxc_gga_exc_vxc_device

   ! Device-aware GGA vxc wrapper function (without exc)
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_vxc_device(func, rho_d, sigma_d, vrho_d, vsigma_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: vrho_d, vsigma_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: rho_ptr, sigma_ptr, vrho_ptr, vsigma_ptr
      TYPE(C_PTR)                                        :: c_rho_ptr, c_sigma_ptr, c_vrho_ptr, &
                                                            c_vsigma_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_gga_vxc(func, np, rho_ptr, sigma_ptr, vrho_ptr, vsigma_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_gga_vxc_device

   ! Device-aware MGGA wrapper functions
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param lapl_d ...
!> \param tau_d ...
!> \param exc_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_device(func, rho_d, sigma_d, lapl_d, tau_d, exc_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d, lapl_d, tau_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, lapl_ptr, rho_ptr, sigma_ptr, &
                                                            tau_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_lapl_ptr, c_rho_ptr, &
                                                            c_sigma_ptr, c_tau_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_lapl_ptr = libxc_device_buffer_get_ptr(lapl_d)
      c_tau_ptr = libxc_device_buffer_get_ptr(tau_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_lapl_ptr, lapl_ptr, [INT(np)])
      CALL C_F_POINTER(c_tau_ptr, tau_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_mgga_exc(func, np, rho_ptr, sigma_ptr, lapl_ptr, tau_ptr, exc_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_mgga_exc_device

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param lapl_d ...
!> \param tau_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param vlapl_d ...
!> \param vtau_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_vxc_device(func, rho_d, sigma_d, lapl_d, tau_d, exc_d, vrho_d, vsigma_d, vlapl_d, vtau_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d, lapl_d, tau_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d, vsigma_d, vlapl_d, vtau_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, lapl_ptr, rho_ptr, sigma_ptr, &
                                                            tau_ptr, vlapl_ptr, vrho_ptr, &
                                                            vsigma_ptr, vtau_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_lapl_ptr, c_rho_ptr, &
                                                            c_sigma_ptr, c_tau_ptr, c_vlapl_ptr, &
                                                            c_vrho_ptr, c_vsigma_ptr, c_vtau_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_lapl_ptr = libxc_device_buffer_get_ptr(lapl_d)
      c_tau_ptr = libxc_device_buffer_get_ptr(tau_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)
      c_vlapl_ptr = libxc_device_buffer_get_ptr(vlapl_d)
      c_vtau_ptr = libxc_device_buffer_get_ptr(vtau_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_lapl_ptr, lapl_ptr, [INT(np)])
      CALL C_F_POINTER(c_tau_ptr, tau_ptr, [INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_vlapl_ptr, vlapl_ptr, [INT(np)])
      CALL C_F_POINTER(c_vtau_ptr, vtau_ptr, [INT(np)])

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_mgga_exc_vxc(func, np, rho_ptr, sigma_ptr, lapl_ptr, tau_ptr, &
                               exc_ptr, vrho_ptr, vsigma_ptr, vlapl_ptr, vtau_ptr)
   END SUBROUTINE libxc_mgga_exc_vxc_device

   ! Device-aware MGGA vxc wrapper function (without exc)
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param lapl_d ...
!> \param tau_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param vlapl_d ...
!> \param vtau_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_vxc_device(func, rho_d, sigma_d, lapl_d, tau_d, vrho_d, vsigma_d, vlapl_d, vtau_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d, lapl_d, tau_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: vrho_d, vsigma_d, vlapl_d, vtau_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: lapl_ptr, rho_ptr, sigma_ptr, tau_ptr, &
                                                            vlapl_ptr, vrho_ptr, vsigma_ptr, &
                                                            vtau_ptr
      TYPE(C_PTR)                                        :: c_lapl_ptr, c_rho_ptr, c_sigma_ptr, &
                                                            c_tau_ptr, c_vlapl_ptr, c_vrho_ptr, &
                                                            c_vsigma_ptr, c_vtau_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_lapl_ptr = libxc_device_buffer_get_ptr(lapl_d)
      c_tau_ptr = libxc_device_buffer_get_ptr(tau_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)
      c_vlapl_ptr = libxc_device_buffer_get_ptr(vlapl_d)
      c_vtau_ptr = libxc_device_buffer_get_ptr(vtau_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_lapl_ptr, lapl_ptr, [INT(np)])
      CALL C_F_POINTER(c_tau_ptr, tau_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [INT(np)])
      CALL C_F_POINTER(c_vlapl_ptr, vlapl_ptr, [INT(np)])
      CALL C_F_POINTER(c_vtau_ptr, vtau_ptr, [INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_mgga_vxc(func, np, rho_ptr, sigma_ptr, lapl_ptr, tau_ptr, &
                           vrho_ptr, vsigma_ptr, vlapl_ptr, vtau_ptr)

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_mgga_vxc_device

   ! Device-aware spin-polarized LDA wrapper functions
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_vxc_spin_device(func, rho_d, exc_d, vrho_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: exc_ptr, rho_ptr, vrho_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr, c_vrho_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)

      ! Convert C pointers to Fortran pointers for array interface
      ! Note: Spin-polarized uses 2-component arrays (alpha, beta)
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [2, INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_lda_exc_vxc(func, np, rho_ptr(1, 1), exc_ptr(1), vrho_ptr(1, 1))

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_lda_exc_vxc_spin_device

   ! Device-aware spin-polarized GGA wrapper functions
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_vxc_spin_device(func, rho_d, sigma_d, exc_d, vrho_d, vsigma_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d, vsigma_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, sigma_ptr, vsigma_ptr
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: rho_ptr, vrho_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_rho_ptr, c_sigma_ptr, &
                                                            c_vrho_ptr, c_vsigma_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [3*INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [3*INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_gga_exc_vxc(func, np, rho_ptr(1, 1), sigma_ptr(1), exc_ptr(1), vrho_ptr(1, 1), vsigma_ptr(1))

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_gga_exc_vxc_spin_device

   ! Device-aware spin-polarized MGGA wrapper functions
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param rho_d ...
!> \param sigma_d ...
!> \param lapl_d ...
!> \param tau_d ...
!> \param exc_d ...
!> \param vrho_d ...
!> \param vsigma_d ...
!> \param vlapl_d ...
!> \param vtau_d ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_vxc_spin_device(func, rho_d, sigma_d, lapl_d, tau_d, exc_d, vrho_d, vsigma_d, vlapl_d, vtau_d, np)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      TYPE(libxc_device_buffer_type), INTENT(IN)         :: rho_d, sigma_d, lapl_d, tau_d
      TYPE(libxc_device_buffer_type), INTENT(OUT)        :: exc_d, vrho_d, vsigma_d, vlapl_d, vtau_d
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np

      REAL(KIND=dp), DIMENSION(:), POINTER               :: exc_ptr, lapl_ptr, sigma_ptr, tau_ptr, &
                                                            vsigma_ptr
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: rho_ptr, vlapl_ptr, vrho_ptr, vtau_ptr
      TYPE(C_PTR)                                        :: c_exc_ptr, c_lapl_ptr, c_rho_ptr, &
                                                            c_sigma_ptr, c_tau_ptr, c_vlapl_ptr, &
                                                            c_vrho_ptr, c_vsigma_ptr, c_vtau_ptr

      ! Get actual device pointers
      c_rho_ptr = libxc_device_buffer_get_ptr(rho_d)
      c_sigma_ptr = libxc_device_buffer_get_ptr(sigma_d)
      c_lapl_ptr = libxc_device_buffer_get_ptr(lapl_d)
      c_tau_ptr = libxc_device_buffer_get_ptr(tau_d)
      c_exc_ptr = libxc_device_buffer_get_ptr(exc_d)
      c_vrho_ptr = libxc_device_buffer_get_ptr(vrho_d)
      c_vsigma_ptr = libxc_device_buffer_get_ptr(vsigma_d)
      c_vlapl_ptr = libxc_device_buffer_get_ptr(vlapl_d)
      c_vtau_ptr = libxc_device_buffer_get_ptr(vtau_d)

      ! Convert C pointers to Fortran pointers for array interface
      CALL C_F_POINTER(c_rho_ptr, rho_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_sigma_ptr, sigma_ptr, [3*INT(np)])
      CALL C_F_POINTER(c_lapl_ptr, lapl_ptr, [2*INT(np)])
      CALL C_F_POINTER(c_tau_ptr, tau_ptr, [2*INT(np)])
      CALL C_F_POINTER(c_exc_ptr, exc_ptr, [INT(np)])
      CALL C_F_POINTER(c_vrho_ptr, vrho_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_vsigma_ptr, vsigma_ptr, [3*INT(np)])
      CALL C_F_POINTER(c_vlapl_ptr, vlapl_ptr, [2, INT(np)])
      CALL C_F_POINTER(c_vtau_ptr, vtau_ptr, [2, INT(np)])

      ! Set device execution flags for CUDA
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)

      ! Call CUDA Libxc directly with device pointers (as Fortran arrays)
      CALL xc_f03_mgga_exc_vxc(func, np, rho_ptr(1, 1), sigma_ptr(1), lapl_ptr(1), tau_ptr(1), &
                               exc_ptr(1), vrho_ptr(1, 1), vsigma_ptr(1), vlapl_ptr(1, 1), vtau_ptr(1, 1))

      ! Reset to host flags
      CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
   END SUBROUTINE libxc_mgga_exc_vxc_spin_device

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param lapl ...
!> \param tau ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
!> \param vlapl ...
!> \param vtau ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_vxc_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma, lapl, tau
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma, vlapl, vtau

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "MGGA exc_vxc: Using GPU backend, np =", np
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_mgga_exc_vxc(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         WRITE (*, *) "MGGA exc_vxc: Using CPU backend, np =", np
         CALL xc_f03_mgga_exc_vxc(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      END IF
   END SUBROUTINE libxc_mgga_exc_vxc_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param lapl ...
!> \param tau ...
!> \param exc ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_wrapper(func, np, rho, sigma, lapl, tau, exc)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma, lapl, tau
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "MGGA exc: Using GPU backend, np =", np
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_mgga_exc(func, np, rho, sigma, lapl, tau, exc)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         WRITE (*, *) "MGGA exc: Using CPU backend, np =", np
         CALL xc_f03_mgga_exc(func, np, rho, sigma, lapl, tau, exc)
      END IF
   END SUBROUTINE libxc_mgga_exc_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param lapl ...
!> \param tau ...
!> \param vrho ...
!> \param vsigma ...
!> \param vlapl ...
!> \param vtau ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_vxc_wrapper(func, np, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma, lapl, tau
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: vrho, vsigma, vlapl, vtau

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "MGGA vxc: Using GPU backend, np =", np
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_mgga_vxc(func, np, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         WRITE (*, *) "MGGA vxc: Using CPU backend, np =", np
         CALL xc_f03_mgga_vxc(func, np, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau)
      END IF
   END SUBROUTINE libxc_mgga_vxc_wrapper

   ! Fortran wrapper functions that call the CUDA wrapper with memory validation
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_vxc_cuda_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_LOC, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma

      INTEGER(C_INT)                                     :: result
      TYPE(C_PTR)                                        :: c_exc, c_func, c_rho, c_sigma, c_vrho, &
                                                            c_vsigma, c_work

      ! Initialize CUDA wrapper
      RESULT = xc_cuda_wrapper_init()
      IF (RESULT /= 0) THEN
         WRITE (*, *) "CUDA wrapper initialization failed, using CPU fallback"
         CALL libxc_gga_exc_vxc_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
         RETURN
      END IF

      ! Check if CUDA is available
      IF (xc_cuda_wrapper_is_available() == 0) THEN
         WRITE (*, *) "CUDA not available, using CPU fallback"
         CALL libxc_gga_exc_vxc_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
         RETURN
      END IF

      ! Get C pointers to Fortran arrays
      c_func = xc_f03_func_get_c_ptr(func)
      c_rho = C_LOC(rho)
      c_sigma = C_LOC(sigma)
      c_exc = C_LOC(exc)
      c_vrho = C_LOC(vrho)
      c_vsigma = C_LOC(vsigma)
      c_work = C_NULL_PTR

      ! Call CUDA wrapper with memory validation
      RESULT = xc_cuda_wrapper_gga_exc_vxc_c(c_func, INT(np, C_INT), c_rho, c_exc, c_vrho, c_vsigma, c_work)

      IF (RESULT == 0) THEN
         WRITE (*, *) "CUDA GGA operation completed successfully"
      ELSE
         WRITE (*, *) "CUDA GGA operation failed with code:", RESULT
         WRITE (*, *) "Falling back to CPU"
         CALL libxc_gga_exc_vxc_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
      END IF

      ! Cleanup CUDA wrapper
      CALL xc_cuda_wrapper_cleanup()
   END SUBROUTINE libxc_gga_exc_vxc_cuda_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param lapl ...
!> \param tau ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
!> \param vlapl ...
!> \param vtau ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_vxc_cuda_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      USE ISO_C_BINDING, ONLY: C_SIZE_T, C_PTR, C_LOC, C_F_POINTER
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma, lapl, tau
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma, vlapl, vtau

      INTEGER(C_INT)                                     :: result
      TYPE(C_PTR)                                        :: c_exc, c_func, c_lapl, c_rho, c_sigma, &
                                                            c_tau, c_vlapl, c_vrho, c_vsigma, &
                                                            c_vtau, c_work

      ! Initialize CUDA wrapper
      RESULT = xc_cuda_wrapper_init()
      IF (RESULT /= 0) THEN
         WRITE (*, *) "CUDA wrapper initialization failed, using CPU fallback"
         CALL libxc_mgga_exc_vxc_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
         RETURN
      END IF

      ! Check if CUDA is available
      IF (xc_cuda_wrapper_is_available() == 0) THEN
         WRITE (*, *) "CUDA not available, using CPU fallback"
         CALL libxc_mgga_exc_vxc_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
         RETURN
      END IF

      ! Get C pointers to Fortran arrays
      c_func = xc_f03_func_get_c_ptr(func)
      c_rho = C_LOC(rho)
      c_sigma = C_LOC(sigma)
      c_lapl = C_LOC(lapl)
      c_tau = C_LOC(tau)
      c_exc = C_LOC(exc)
      c_vrho = C_LOC(vrho)
      c_vsigma = C_LOC(vsigma)
      c_vlapl = C_LOC(vlapl)
      c_vtau = C_LOC(vtau)
      c_work = C_NULL_PTR

      ! Call CUDA wrapper with memory validation
        RESULT = xc_cuda_wrapper_mgga_exc_vxc_c(c_func, INT(np, C_INT), c_rho, c_sigma, c_lapl, c_tau, c_exc, c_vrho, c_vsigma, c_vlapl, c_vtau, c_work)

      IF (RESULT == 0) THEN
         WRITE (*, *) "CUDA MGGA operation completed successfully"
      ELSE
         WRITE (*, *) "CUDA MGGA operation failed with code:", RESULT
         WRITE (*, *) "Falling back to CPU"
         CALL libxc_mgga_exc_vxc_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      END IF

      ! Cleanup CUDA wrapper
      CALL xc_cuda_wrapper_cleanup()
   END SUBROUTINE libxc_mgga_exc_vxc_cuda_wrapper

   ! Spin-polarized wrapper functions that handle backend selection and memory translation
! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param exc ...
!> \param vrho ...
! **************************************************************************************************
   SUBROUTINE libxc_lda_exc_vxc_spin_wrapper(func, np, rho, exc, vrho)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "Spin-polarized LDA exc_vxc: Using GPU backend, np =", np
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_DEVICE)
         CALL xc_f03_lda_exc_vxc(func, np, rho, exc, vrho)
         CALL xc_f03_func_info_set_default_flags(XC_FLAGS_ON_HOST)
      ELSE
         WRITE (*, *) "Spin-polarized LDA exc_vxc: Using CPU backend, np =", np
         CALL xc_f03_lda_exc_vxc(func, np, rho, exc, vrho)
      END IF
   END SUBROUTINE libxc_lda_exc_vxc_spin_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
! **************************************************************************************************
   SUBROUTINE libxc_gga_exc_vxc_spin_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "Spin-polarized GGA exc_vxc: Using GPU backend with CUDA wrapper, np =", np
         CALL libxc_gga_exc_vxc_cuda_wrapper(func, np, rho, sigma, exc, vrho, vsigma)
      ELSE
         WRITE (*, *) "Spin-polarized GGA exc_vxc: Using CPU backend, np =", np
         CALL xc_f03_gga_exc_vxc(func, np, rho, sigma, exc, vrho, vsigma)
      END IF
   END SUBROUTINE libxc_gga_exc_vxc_spin_wrapper

! **************************************************************************************************
!> \brief ...
!> \param func ...
!> \param np ...
!> \param rho ...
!> \param sigma ...
!> \param lapl ...
!> \param tau ...
!> \param exc ...
!> \param vrho ...
!> \param vsigma ...
!> \param vlapl ...
!> \param vtau ...
! **************************************************************************************************
   SUBROUTINE libxc_mgga_exc_vxc_spin_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      USE ISO_C_BINDING, ONLY: C_SIZE_T
      TYPE(xc_f03_func_t), INTENT(IN)                    :: func
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: np
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho, sigma, lapl, tau
      REAL(KIND=dp), DIMENSION(*), INTENT(OUT)           :: exc, vrho, vsigma, vlapl, vtau

      IF (libxc_current_backend == LIBXC_BACKEND_GPU) THEN
         WRITE (*, *) "Spin-polarized MGGA exc_vxc: Using GPU backend with CUDA wrapper, np =", np
         CALL libxc_mgga_exc_vxc_cuda_wrapper(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      ELSE
         WRITE (*, *) "Spin-polarized MGGA exc_vxc: Using CPU backend, np =", np
         CALL xc_f03_mgga_exc_vxc(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau)
      END IF
   END SUBROUTINE libxc_mgga_exc_vxc_spin_wrapper

END MODULE xc_libxc_wrap
