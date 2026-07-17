!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

PROGRAM kpoint_restart_transfer_TEST
   USE ieee_arithmetic,                 ONLY: ieee_quiet_nan,&
                                              ieee_value
   USE kinds,                           ONLY: dp
   USE kpoint_restart_transfer,         ONLY: &
        analyze_metric_density_dense, metric_projection_density_nonfinite, &
        metric_projection_density_nonhermitian, metric_projection_occupation_bounds, &
        metric_projection_overlap_ill_conditioned, metric_projection_overlap_nonfinite, &
        metric_projection_overlap_nonhermitian, project_metric_density_dense, &
        project_restart_occupations

   IMPLICIT NONE

   CALL test_rks_waterfill()
   CALL test_uks_waterfill()
   CALL test_shifted_weight_waterfill()
   CALL test_metric_projection()
   CALL test_ill_conditioned_overlap()
   CALL test_nonfinite_inputs()
   CALL test_nonhermitian_inputs()
   CALL test_infeasible_electron_target()
   CALL test_target_occupation_bounds()

CONTAINS

! **************************************************************************************************
!> \brief Check clipping to [0,2] and exact RKS electron conservation.
! **************************************************************************************************
   SUBROUTINE test_rks_waterfill()
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: count, residual
      REAL(KIND=dp), DIMENSION(2)                        :: weights
      REAL(KIND=dp), DIMENSION(3, 2)                     :: projected, raw

      raw(:, 1) = [-0.30_dp, 0.45_dp, 2.35_dp]
      raw(:, 2) = [0.10_dp, 1.15_dp, 2.25_dp]
      weights = [0.375_dp, 0.625_dp]
      CALL project_restart_occupations(raw, weights, 2.0_dp, 2.0_dp, projected, accepted, residual)
      count = weights(1)*SUM(projected(:, 1)) + weights(2)*SUM(projected(:, 2))
      IF (.NOT. accepted .OR. MINVAL(projected) < 0.0_dp .OR. MAXVAL(projected) > 2.0_dp .OR. &
          ABS(count - 2.0_dp) > 5.0e-14_dp .OR. ABS(residual) > 5.0e-14_dp) &
         ERROR STOP "test_rks_waterfill failed"
      PRINT *, "test_rks_waterfill: OK"
   END SUBROUTINE test_rks_waterfill

! **************************************************************************************************
!> \brief Check clipping to [0,1] for one UKS spin channel.
! **************************************************************************************************
   SUBROUTINE test_uks_waterfill()
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: count, residual
      REAL(KIND=dp), DIMENSION(2)                        :: weights
      REAL(KIND=dp), DIMENSION(4, 2)                     :: projected, raw

      raw(:, 1) = [-0.20_dp, 0.15_dp, 0.80_dp, 1.20_dp]
      raw(:, 2) = [0.05_dp, 0.35_dp, 1.05_dp, 1.40_dp]
      weights = [0.25_dp, 0.75_dp]
      CALL project_restart_occupations(raw, weights, 1.0_dp, 1.0_dp, projected, accepted, residual)
      count = weights(1)*SUM(projected(:, 1)) + weights(2)*SUM(projected(:, 2))
      IF (.NOT. accepted .OR. MINVAL(projected) < 0.0_dp .OR. MAXVAL(projected) > 1.0_dp .OR. &
          ABS(count - 1.0_dp) > 5.0e-14_dp .OR. ABS(residual) > 5.0e-14_dp) &
         ERROR STOP "test_uks_waterfill failed"
      PRINT *, "test_uks_waterfill: OK"
   END SUBROUTINE test_uks_waterfill

! **************************************************************************************************
!> \brief Check deterministic weighted conservation for a nonuniform shifted-mesh star.
! **************************************************************************************************
   SUBROUTINE test_shifted_weight_waterfill()
      INTEGER                                            :: i
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: count, residual
      REAL(KIND=dp), DIMENSION(2, 4)                     :: projected, raw
      REAL(KIND=dp), DIMENSION(4)                        :: weights

      raw(:, 1) = [-0.10_dp, 0.25_dp]
      raw(:, 2) = [0.45_dp, 0.85_dp]
      raw(:, 3) = [1.15_dp, 1.55_dp]
      raw(:, 4) = [1.85_dp, 2.10_dp]
      weights = [0.10_dp, 0.20_dp, 0.30_dp, 0.40_dp]
      CALL project_restart_occupations(raw, weights, 1.25_dp, 2.0_dp, projected, accepted, residual)
      count = SUM([(weights(i)*SUM(projected(:, i)), i=1, 4)])
      IF (.NOT. accepted .OR. ABS(count - 1.25_dp) > 5.0e-14_dp .OR. &
          ABS(residual) > 5.0e-14_dp) ERROR STOP "test_shifted_weight_waterfill failed"
      PRINT *, "test_shifted_weight_waterfill: OK"
   END SUBROUTINE test_shifted_weight_waterfill

! **************************************************************************************************
!> \brief Check full nonorthogonal spectral reconstruction and its metric eigenvalue bounds.
! **************************************************************************************************
   SUBROUTINE test_metric_projection()
      COMPLEX(KIND=dp), DIMENSION(2, 2)                  :: density, overlap, projected
      INTEGER                                            :: failure_code
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: residual
      REAL(KIND=dp), DIMENSION(1)                        :: weights
      REAL(KIND=dp), DIMENSION(2)                        :: metric_occupations, raw_occupations, &
                                                            target_occupations
      REAL(KIND=dp), DIMENSION(2, 1)                     :: projected_box, raw_box

      overlap(1, :) = [CMPLX(1.0_dp, 0.0_dp, dp), CMPLX(0.20_dp, 0.10_dp, dp)]
      overlap(2, :) = [CONJG(overlap(1, 2)), CMPLX(1.20_dp, 0.0_dp, dp)]
      density(1, :) = [CMPLX(1.80_dp, 0.0_dp, dp), CMPLX(0.00_dp, 0.40_dp, dp)]
      density(2, :) = [CONJG(density(1, 2)), CMPLX(0.40_dp, 0.0_dp, dp)]
      CALL analyze_metric_density_dense(density, overlap, raw_occupations, accepted, failure_code)
      IF (.NOT. accepted) ERROR STOP "test_metric_projection analysis failed"
      raw_box(:, 1) = raw_occupations
      weights = [1.0_dp]
      CALL project_restart_occupations(raw_box, weights, 2.0_dp, 2.0_dp, &
                                       projected_box, accepted, residual)
      IF (.NOT. accepted) ERROR STOP "test_metric_projection waterfill failed"
      target_occupations = projected_box(:, 1)
      CALL project_metric_density_dense(density, overlap, target_occupations, 2.0_dp, projected, &
                                        metric_occupations, accepted, failure_code)
      IF (.NOT. accepted .OR. MINVAL(metric_occupations) < -1.0e-12_dp .OR. &
          MAXVAL(metric_occupations) > 2.0_dp + 1.0e-12_dp .OR. &
          ABS(SUM(metric_occupations) - 2.0_dp) > 1.0e-12_dp .OR. &
          MAXVAL(ABS(projected - CONJG(TRANSPOSE(projected)))) > 1.0e-12_dp) &
         ERROR STOP "test_metric_projection reconstruction failed"
      PRINT *, "test_metric_projection: OK"
   END SUBROUTINE test_metric_projection

! **************************************************************************************************
!> \brief Check fail-closed handling of a singular/ill-conditioned overlap metric.
! **************************************************************************************************
   SUBROUTINE test_ill_conditioned_overlap()
      COMPLEX(KIND=dp), DIMENSION(2, 2)                  :: density, overlap
      INTEGER                                            :: failure_code
      LOGICAL                                            :: accepted
      REAL(KIND=dp), DIMENSION(2)                        :: occupations

      density = CMPLX(0.0_dp, 0.0_dp, dp)
      overlap = CMPLX(0.0_dp, 0.0_dp, dp)
      density(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(2, 2) = CMPLX(1.0e-14_dp, 0.0_dp, dp)
      CALL analyze_metric_density_dense(density, overlap, occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_overlap_ill_conditioned) &
         ERROR STOP "test_ill_conditioned_overlap failed"
      PRINT *, "test_ill_conditioned_overlap: OK"
   END SUBROUTINE test_ill_conditioned_overlap

! **************************************************************************************************
!> \brief Check fail-closed handling of non-finite density and overlap matrices.
! **************************************************************************************************
   SUBROUTINE test_nonfinite_inputs()
      COMPLEX(KIND=dp), DIMENSION(2, 2)                  :: density, overlap
      INTEGER                                            :: failure_code
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: nan_value
      REAL(KIND=dp), DIMENSION(2)                        :: occupations

      nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
      density = CMPLX(0.0_dp, 0.0_dp, dp)
      overlap = CMPLX(0.0_dp, 0.0_dp, dp)
      density(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(1, 1) = CMPLX(nan_value, 0.0_dp, dp)
      CALL analyze_metric_density_dense(density, overlap, occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_density_nonfinite) &
         ERROR STOP "test_nonfinite_inputs density failed"

      density(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(2, 2) = CMPLX(nan_value, 0.0_dp, dp)
      CALL analyze_metric_density_dense(density, overlap, occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_overlap_nonfinite) &
         ERROR STOP "test_nonfinite_inputs overlap failed"
      PRINT *, "test_nonfinite_inputs: OK"
   END SUBROUTINE test_nonfinite_inputs

! **************************************************************************************************
!> \brief Check fail-closed handling of non-Hermitian density and overlap matrices.
! **************************************************************************************************
   SUBROUTINE test_nonhermitian_inputs()
      COMPLEX(KIND=dp), DIMENSION(2, 2)                  :: density, overlap
      INTEGER                                            :: failure_code
      LOGICAL                                            :: accepted
      REAL(KIND=dp), DIMENSION(2)                        :: occupations

      density = CMPLX(0.0_dp, 0.0_dp, dp)
      overlap = CMPLX(0.0_dp, 0.0_dp, dp)
      density(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(1, 2) = CMPLX(0.2_dp, 0.1_dp, dp)
      CALL analyze_metric_density_dense(density, overlap, occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_density_nonhermitian) &
         ERROR STOP "test_nonhermitian_inputs density failed"

      density(1, 2) = CMPLX(0.0_dp, 0.0_dp, dp)
      overlap(1, 2) = CMPLX(0.2_dp, 0.1_dp, dp)
      CALL analyze_metric_density_dense(density, overlap, occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_overlap_nonhermitian) &
         ERROR STOP "test_nonhermitian_inputs overlap failed"
      PRINT *, "test_nonhermitian_inputs: OK"
   END SUBROUTINE test_nonhermitian_inputs

! **************************************************************************************************
!> \brief Check rejection when the requested electron number exceeds box capacity.
! **************************************************************************************************
   SUBROUTINE test_infeasible_electron_target()
      LOGICAL                                            :: accepted
      REAL(KIND=dp)                                      :: residual
      REAL(KIND=dp), DIMENSION(2)                        :: weights
      REAL(KIND=dp), DIMENSION(2, 2)                     :: projected, raw

      raw = 0.5_dp
      weights = [0.25_dp, 0.75_dp]
      CALL project_restart_occupations(raw, weights, 2.5_dp, 1.0_dp, projected, accepted, residual)
      IF (accepted .OR. ANY(projected /= 0.0_dp)) &
         ERROR STOP "test_infeasible_electron_target failed"
      PRINT *, "test_infeasible_electron_target: OK"
   END SUBROUTINE test_infeasible_electron_target

! **************************************************************************************************
!> \brief Check rejection of prescribed occupations outside the spin box.
! **************************************************************************************************
   SUBROUTINE test_target_occupation_bounds()
      COMPLEX(KIND=dp), DIMENSION(2, 2)                  :: density, overlap, projected
      INTEGER                                            :: failure_code
      LOGICAL                                            :: accepted
      REAL(KIND=dp), DIMENSION(2)                        :: metric_occupations, target_occupations

      density = CMPLX(0.0_dp, 0.0_dp, dp)
      overlap = CMPLX(0.0_dp, 0.0_dp, dp)
      density(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      density(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(1, 1) = CMPLX(1.0_dp, 0.0_dp, dp)
      overlap(2, 2) = CMPLX(1.0_dp, 0.0_dp, dp)
      target_occupations = [-0.1_dp, 1.0_dp]
      CALL project_metric_density_dense(density, overlap, target_occupations, 2.0_dp, projected, &
                                        metric_occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_occupation_bounds) &
         ERROR STOP "test_target_occupation_bounds lower failed"

      target_occupations = [0.5_dp, 2.1_dp]
      CALL project_metric_density_dense(density, overlap, target_occupations, 2.0_dp, projected, &
                                        metric_occupations, accepted, failure_code)
      IF (accepted .OR. failure_code /= metric_projection_occupation_bounds) &
         ERROR STOP "test_target_occupation_bounds upper failed"
      PRINT *, "test_target_occupation_bounds: OK"
   END SUBROUTINE test_target_occupation_bounds

END PROGRAM kpoint_restart_transfer_TEST
! vim: set ts=3 sw=3 tw=132 :
