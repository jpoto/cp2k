!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Device-only operations on plane-wave arrays for the fused G-space pipeline.
!>
!>        This module is deliberately free of any dependency on the `pw_gpu`
!>        Fortran module: the device kernels are reached through C bindings and
!>        all memory management goes through the offload library (`offload_api`).
!>        It is therefore safe to use from modules (SCF, Poisson) that must not
!>        pull in `pw_gpu`.
!>
!>        Residency contract: the *_c1d routines below operate on device mirrors
!>        only (no host traffic). The caller is responsible for making the
!>        inputs device-current (pw%ensure_device()) and for syncing back to the
!>        host (pw%sync_host()) only at real host consumers.
! **************************************************************************************************
MODULE pw_offload

   USE iso_c_binding, ONLY: C_ASSOCIATED, C_DOUBLE, C_INT, C_LOC, C_NULL_PTR, C_PTR, C_SIZE_T
   USE kinds, ONLY: dp
   USE offload_api, ONLY: offload_device_alloc, offload_device_d2d, offload_device_d2h
   USE pw_types, ONLY: pw_c1d_gs_type

#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pw_offload'

   PUBLIC :: pw_offload_available
   PUBLIC :: pw_offload_copy_c1d
   PUBLIC :: pw_offload_axpy_c1d
   PUBLIC :: pw_offload_multiply_c1d
   PUBLIC :: pw_offload_dot_c1d
   PUBLIC :: pw_offload_g0_c1d

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
   INTERFACE
      SUBROUTINE pw_gpu_multiply_c1d_dev_c(a, b, n) BIND(C, name="pw_gpu_multiply_c1d_dev")
         IMPORT :: C_INT, C_PTR
         TYPE(C_PTR), VALUE :: a, b
         INTEGER(KIND=C_INT), VALUE :: n
      END SUBROUTINE pw_gpu_multiply_c1d_dev_c

      SUBROUTINE pw_gpu_axpy_c1d_dev_c(y, x, a, b, n) BIND(C, name="pw_gpu_axpy_c1d_dev")
         IMPORT :: C_DOUBLE, C_INT, C_PTR
         TYPE(C_PTR), VALUE :: y, x
         REAL(KIND=C_DOUBLE), VALUE :: a, b
         INTEGER(KIND=C_INT), VALUE :: n
      END SUBROUTINE pw_gpu_axpy_c1d_dev_c

      FUNCTION pw_gpu_dot_c1d_dev_f(a, b, n) BIND(C, name="pw_gpu_dot_c1d_dev") RESULT(r)
         IMPORT :: C_DOUBLE, C_INT, C_PTR
         TYPE(C_PTR), VALUE :: a, b
         INTEGER(KIND=C_INT), VALUE :: n
         REAL(KIND=C_DOUBLE) :: r
      END FUNCTION pw_gpu_dot_c1d_dev_f
   END INTERFACE
#endif

CONTAINS

! **************************************************************************************************
!> \brief Whether the device-only operations are available (accelerator + PW offload).
! **************************************************************************************************
   PURE FUNCTION pw_offload_available() RESULT(res)
      LOGICAL :: res
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      res = .TRUE.
#else
      res = .FALSE.
#endif
   END FUNCTION pw_offload_available

! **************************************************************************************************
!> \brief Device-only copy: dst_device <- src_device (no host traffic).
! **************************************************************************************************
   SUBROUTINE pw_offload_copy_c1d(dst, src)
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: dst
      TYPE(pw_c1d_gs_type), INTENT(IN) :: src
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER(KIND=C_SIZE_T) :: nbytes
      CPASSERT(src%device_valid)
      IF (.NOT. ASSOCIATED(dst%array)) RETURN
      nbytes = INT(STORAGE_SIZE(dst%array)/8, KIND=C_SIZE_T)
      IF (nbytes <= 0) RETURN
      IF (.NOT. C_ASSOCIATED(dst%device_data)) dst%device_data = offload_device_alloc(nbytes)
      CALL offload_device_d2d(dst%device_data, src%device_data, nbytes)
      CALL dst%mark_device_current()
#else
      MARK_USED(dst)
      MARK_USED(src)
      CPABORT("Compiled without PW offloading.")
#endif
   END SUBROUTINE pw_offload_copy_c1d

! **************************************************************************************************
!> \brief Device-only axpy: y_device = a*x_device + b*y_device.
! **************************************************************************************************
   SUBROUTINE pw_offload_axpy_c1d(y, x, a, b)
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: y
      TYPE(pw_c1d_gs_type), INTENT(IN) :: x
      REAL(KIND=dp), INTENT(IN) :: a, b
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CPASSERT(x%device_valid)
      CPASSERT(C_ASSOCIATED(y%device_data))
      CALL pw_gpu_axpy_c1d_dev_c(y%device_data, x%device_data, &
                                 REAL(a, KIND=C_DOUBLE), REAL(b, KIND=C_DOUBLE), &
                                 INT(SIZE(y%array), KIND=C_INT))
      CALL y%mark_device_current()
#else
      MARK_USED(y)
      MARK_USED(x)
      MARK_USED(a)
      MARK_USED(b)
      CPABORT("Compiled without PW offloading.")
#endif
   END SUBROUTINE pw_offload_axpy_c1d

! **************************************************************************************************
!> \brief Device-only elementwise multiply: a_device *= b_device.
! **************************************************************************************************
   SUBROUTINE pw_offload_multiply_c1d(a, b)
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: a
      TYPE(pw_c1d_gs_type), INTENT(IN) :: b
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CPASSERT(a%device_valid)
      CPASSERT(b%device_valid)
      CALL pw_gpu_multiply_c1d_dev_c(a%device_data, b%device_data, INT(SIZE(a%array), KIND=C_INT))
      CALL a%mark_device_current()
#else
      MARK_USED(a)
      MARK_USED(b)
      CPABORT("Compiled without PW offloading.")
#endif
   END SUBROUTINE pw_offload_multiply_c1d

! **************************************************************************************************
!> \brief Device-only reduction Re<conj(a)|b> over all points (raw sum on host).
! **************************************************************************************************
   FUNCTION pw_offload_dot_c1d(a, b) RESULT(res)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a, b
      REAL(KIND=dp) :: res
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CPASSERT(a%device_valid)
      CPASSERT(b%device_valid)
      res = REAL(pw_gpu_dot_c1d_dev_f(a%device_data, b%device_data, &
                                      INT(SIZE(a%array), KIND=C_INT)), KIND=dp)
#else
      MARK_USED(a)
      MARK_USED(b)
      CPABORT("Compiled without PW offloading.")
#endif
   END FUNCTION pw_offload_dot_c1d

! **************************************************************************************************
!> \brief Reads the G=0 coefficient (first complex) of a device-resident array to the host.
! **************************************************************************************************
   FUNCTION pw_offload_g0_c1d(a) RESULT(g0)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a
      COMPLEX(KIND=dp), TARGET :: g0
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CPASSERT(a%device_valid)
      IF (ASSOCIATED(a%array)) THEN
         CALL offload_device_d2h(C_LOC(g0), a%device_data, INT(16, KIND=C_SIZE_T))
      ELSE
         g0 = CMPLX(0.0_dp, 0.0_dp, KIND=dp)
      END IF
#else
      MARK_USED(a)
      g0 = CMPLX(0.0_dp, 0.0_dp, KIND=dp)
#endif
   END FUNCTION pw_offload_g0_c1d

END MODULE pw_offload
