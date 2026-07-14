!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief CP2K-side tblite-compatible SCC Broyden mixer.
! **************************************************************************************************
MODULE tblite_scc_mixer
#if defined(__TBLITE)
   USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_finite
   USE mctc_env, ONLY: error_type, &
                       fatal_error, &
                       wp
   USE tblite_lapack, ONLY: getrf, &
                            getrs
#endif

   IMPLICIT NONE

   PRIVATE

#if defined(__TBLITE)
   PUBLIC :: cp2k_tblite_broyden_mixer_type, new_cp2k_tblite_mixer

   TYPE :: cp2k_tblite_broyden_mixer_type
      INTEGER                                            :: idif = 0
      INTEGER                                            :: iget = 0
      INTEGER                                            :: iset = 0
      INTEGER                                            :: iter = 0
      INTEGER                                            :: memory = 0
      INTEGER                                            :: ndim = 0
      REAL(wp)                                           :: damp = 0.0_wp
      REAL(wp)                                           :: max_weight = 0.0_wp
      REAL(wp)                                           :: min_weight = 0.0_wp
      REAL(wp)                                           :: omega0 = 0.0_wp
      REAL(wp)                                           :: weight_factor = 0.0_wp
      LOGICAL                                            :: layout_error = .FALSE.
      REAL(wp), ALLOCATABLE, DIMENSION(:, :)             :: a, df, u
      REAL(wp), ALLOCATABLE, DIMENSION(:)                :: dq, dqlast, omega, q_in, qlast_in
   CONTAINS
      PROCEDURE                                          :: diff_1d => cp2k_tblite_mixer_diff_1d
      PROCEDURE                                          :: diff_2d => cp2k_tblite_mixer_diff_2d
      PROCEDURE                                          :: diff_3d => cp2k_tblite_mixer_diff_3d
      GENERIC                                            :: diff => diff_1d, diff_2d, diff_3d
      PROCEDURE                                          :: get_1d => cp2k_tblite_mixer_get_1d
      PROCEDURE                                          :: get_2d => cp2k_tblite_mixer_get_2d
      PROCEDURE                                          :: get_3d => cp2k_tblite_mixer_get_3d
      GENERIC                                            :: get => get_1d, get_2d, get_3d
      PROCEDURE                                          :: get_error => cp2k_tblite_mixer_get_error
      PROCEDURE                                          :: next => cp2k_tblite_mixer_next
      PROCEDURE                                          :: set_1d => cp2k_tblite_mixer_set_1d
      PROCEDURE                                          :: set_2d => cp2k_tblite_mixer_set_2d
      PROCEDURE                                          :: set_3d => cp2k_tblite_mixer_set_3d
      GENERIC                                            :: set => set_1d, set_2d, set_3d
   END TYPE cp2k_tblite_broyden_mixer_type
#endif

CONTAINS

#if defined(__TBLITE)
! **************************************************************************************************
!> \brief Create a CP2K-side tblite-compatible Broyden mixer.
!> \param mixer mixer instance
!> \param memory Broyden history length
!> \param ndim number of SCC variables
!> \param damping first-step damping
!> \param omega0 Broyden regularization weight
!> \param min_weight minimum dynamic Broyden weight
!> \param max_weight maximum dynamic Broyden weight
!> \param weight_factor residual-to-weight scaling factor
! **************************************************************************************************
   SUBROUTINE new_cp2k_tblite_mixer(mixer, memory, ndim, damping, omega0, min_weight, max_weight, &
                                    weight_factor)
      TYPE(cp2k_tblite_broyden_mixer_type), &
         ALLOCATABLE, INTENT(OUT)                        :: mixer
      INTEGER, INTENT(IN)                                :: memory, ndim
      REAL(wp), INTENT(IN)                               :: damping, omega0, min_weight, max_weight, &
                                                            weight_factor

      TYPE(cp2k_tblite_broyden_mixer_type), ALLOCATABLE  :: broyden

      ALLOCATE (broyden)
      broyden%ndim = ndim
      broyden%memory = memory
      broyden%damp = damping
      broyden%omega0 = omega0
      broyden%min_weight = min_weight
      broyden%max_weight = max_weight
      broyden%weight_factor = weight_factor
      ALLOCATE (broyden%a(memory, memory), SOURCE=0.0_wp)
      ALLOCATE (broyden%df(ndim, memory), SOURCE=0.0_wp)
      ALLOCATE (broyden%u(ndim, memory), SOURCE=0.0_wp)
      ALLOCATE (broyden%dq(ndim), SOURCE=0.0_wp)
      ALLOCATE (broyden%dqlast(ndim), SOURCE=0.0_wp)
      ALLOCATE (broyden%omega(memory), SOURCE=0.0_wp)
      ALLOCATE (broyden%q_in(ndim), SOURCE=0.0_wp)
      ALLOCATE (broyden%qlast_in(ndim), SOURCE=0.0_wp)
      CALL MOVE_ALLOC(broyden, mixer)

   END SUBROUTINE new_cp2k_tblite_mixer

! **************************************************************************************************
!> \brief Set input SCC variables.
!> \param self mixer
!> \param qvec SCC-variable vector
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_set_1d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:), INTENT(IN)                  :: qvec

      IF (self%layout_error) RETURN
      IF (self%iset > self%ndim .OR. SIZE(qvec) > self%ndim - self%iset) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%q_in(self%iset + 1:self%iset + SIZE(qvec)) = qvec
      self%iset = self%iset + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_set_1d

! **************************************************************************************************
!> \brief Set a rank-two input SCC-variable array.
!> \param self mixer
!> \param qvec SCC-variable array
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_set_2d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :), INTENT(IN)               :: qvec

      IF (self%layout_error) RETURN
      IF (self%iset > self%ndim .OR. SIZE(qvec) > self%ndim - self%iset) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%q_in(self%iset + 1:self%iset + SIZE(qvec)) = RESHAPE(qvec, [SIZE(qvec)])
      self%iset = self%iset + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_set_2d

! **************************************************************************************************
!> \brief Set a rank-three input SCC-variable array.
!> \param self mixer
!> \param qvec SCC-variable array
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_set_3d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :, :), INTENT(IN)            :: qvec

      IF (self%layout_error) RETURN
      IF (self%iset > self%ndim .OR. SIZE(qvec) > self%ndim - self%iset) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%q_in(self%iset + 1:self%iset + SIZE(qvec)) = RESHAPE(qvec, [SIZE(qvec)])
      self%iset = self%iset + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_set_3d

! **************************************************************************************************
!> \brief Set SCC residual.
!> \param self mixer
!> \param qvec output SCC-variable vector before mixing
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_diff_1d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:), INTENT(IN)                  :: qvec

      IF (self%layout_error) RETURN
      IF (self%idif > self%ndim .OR. SIZE(qvec) > self%ndim - self%idif) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%dq(self%idif + 1:self%idif + SIZE(qvec)) = &
         qvec - self%q_in(self%idif + 1:self%idif + SIZE(qvec))
      self%idif = self%idif + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_diff_1d

! **************************************************************************************************
!> \brief Set a rank-two SCC residual.
!> \param self mixer
!> \param qvec output SCC-variable array before mixing
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_diff_2d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :), INTENT(IN)               :: qvec

      IF (self%layout_error) RETURN
      IF (self%idif > self%ndim .OR. SIZE(qvec) > self%ndim - self%idif) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%dq(self%idif + 1:self%idif + SIZE(qvec)) = RESHAPE(qvec, [SIZE(qvec)]) - &
                                                      self%q_in(self%idif + 1:self%idif + SIZE(qvec))
      self%idif = self%idif + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_diff_2d

! **************************************************************************************************
!> \brief Set a rank-three SCC residual.
!> \param self mixer
!> \param qvec output SCC-variable array before mixing
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_diff_3d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :, :), INTENT(IN)            :: qvec

      IF (self%layout_error) RETURN
      IF (self%idif > self%ndim .OR. SIZE(qvec) > self%ndim - self%idif) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite(qvec))) THEN
         self%layout_error = .TRUE.
         RETURN
      END IF
      self%dq(self%idif + 1:self%idif + SIZE(qvec)) = RESHAPE(qvec, [SIZE(qvec)]) - &
                                                      self%q_in(self%idif + 1:self%idif + SIZE(qvec))
      self%idif = self%idif + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_diff_3d

! **************************************************************************************************
!> \brief Apply the next Broyden update.
!> \param self mixer
!> \param error error handling
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_next(self, error)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      TYPE(error_type), ALLOCATABLE, INTENT(OUT)          :: error

      CHARACTER(LEN=192)                                 :: message
      INTEGER                                            :: info

      IF (self%layout_error) THEN
         CALL fatal_error(error, "Invalid or non-finite tblite SCC mixer set/diff/get input")
         RETURN
      END IF
      IF (self%idif /= self%ndim) THEN
         WRITE (message, "(A,I0,A,I0)") "Incomplete tblite SCC mixer residual: got ", &
            self%idif, " values, expected ", self%ndim
         CALL fatal_error(error, TRIM(message))
         RETURN
      END IF
      IF (self%iter == 0) THEN
         IF (self%iset /= self%ndim) THEN
            WRITE (message, "(A,I0,A,I0)") "Incomplete initial tblite SCC mixer input: got ", &
               self%iset, " values, expected ", self%ndim
            CALL fatal_error(error, TRIM(message))
            RETURN
         END IF
      ELSE
         ! qs_charge_mixing keeps q_in inside the mixer after its initial set, whereas the native
         ! tblite path replaces all components before every update.  Reject partial replacement.
         IF (self%iset /= 0 .AND. self%iset /= self%ndim) THEN
            WRITE (message, "(A,I0,A,I0)") "Partial tblite SCC mixer input: got ", &
               self%iset, " values, expected either 0 or ", self%ndim
            CALL fatal_error(error, TRIM(message))
            RETURN
         END IF
         IF (self%iget /= self%ndim) THEN
            WRITE (message, "(A,I0,A,I0)") "Incomplete tblite SCC mixer output read: got ", &
               self%iget, " values, expected ", self%ndim
            CALL fatal_error(error, TRIM(message))
            RETURN
         END IF
      END IF
      IF (.NOT. ALL(ieee_is_finite(self%q_in)) .OR. &
          .NOT. ALL(ieee_is_finite(self%dq))) THEN
         CALL fatal_error(error, "Non-finite tblite SCC mixer vector before Broyden update")
         RETURN
      END IF

      self%iset = 0
      self%idif = 0
      self%iget = 0
      self%iter = self%iter + 1
      CALL cp2k_tblite_broyden(self%ndim, self%q_in, self%qlast_in, self%dq, self%dqlast, &
                               self%iter, self%memory, self%damp, self%omega0, &
                               self%min_weight, self%max_weight, self%weight_factor, &
                               self%omega, self%df, self%u, self%a, info)
      SELECT CASE (info)
      CASE (0)
         IF (.NOT. ALL(ieee_is_finite(self%q_in))) THEN
            CALL fatal_error(error, "Non-finite tblite SCC mixer vector after Broyden update")
         END IF
      CASE (-1)
         CALL fatal_error(error, "Invalid dimensions or parameters in tblite Broyden mixer")
      CASE (-2)
         CALL fatal_error(error, "Non-finite input or history in tblite Broyden mixer")
      CASE (-3)
         CALL fatal_error(error, "Non-finite intermediate or output in tblite Broyden mixer")
      CASE DEFAULT
         CALL fatal_error(error, "Broyden linear solve failed to obtain next iteration")
      END SELECT

   END SUBROUTINE cp2k_tblite_mixer_next

! **************************************************************************************************
!> \brief Get mixed SCC variables.
!> \param self mixer
!> \param qvec mixed SCC-variable vector
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_get_1d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:), INTENT(OUT)                 :: qvec

      IF (self%layout_error) THEN
         qvec = 0.0_wp
         RETURN
      END IF
      IF (self%iget > self%ndim .OR. SIZE(qvec) > self%ndim - self%iget) THEN
         self%layout_error = .TRUE.
         qvec = 0.0_wp
         RETURN
      END IF
      qvec(:) = self%q_in(self%iget + 1:self%iget + SIZE(qvec))
      IF (.NOT. ALL(ieee_is_finite(qvec))) self%layout_error = .TRUE.
      self%iget = self%iget + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_get_1d

! **************************************************************************************************
!> \brief Get a rank-two mixed SCC-variable array.
!> \param self mixer
!> \param qvec mixed SCC-variable array
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_get_2d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :), INTENT(OUT)              :: qvec

      IF (self%layout_error) THEN
         qvec = 0.0_wp
         RETURN
      END IF
      IF (self%iget > self%ndim .OR. SIZE(qvec) > self%ndim - self%iget) THEN
         self%layout_error = .TRUE.
         qvec = 0.0_wp
         RETURN
      END IF
      qvec = RESHAPE(self%q_in(self%iget + 1:self%iget + SIZE(qvec)), SHAPE(qvec))
      IF (.NOT. ALL(ieee_is_finite(qvec))) self%layout_error = .TRUE.
      self%iget = self%iget + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_get_2d

! **************************************************************************************************
!> \brief Get a rank-three mixed SCC-variable array.
!> \param self mixer
!> \param qvec mixed SCC-variable array
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_mixer_get_3d(self, qvec)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(INOUT) :: self
      REAL(wp), DIMENSION(:, :, :), INTENT(OUT)           :: qvec

      IF (self%layout_error) THEN
         qvec = 0.0_wp
         RETURN
      END IF
      IF (self%iget > self%ndim .OR. SIZE(qvec) > self%ndim - self%iget) THEN
         self%layout_error = .TRUE.
         qvec = 0.0_wp
         RETURN
      END IF
      qvec = RESHAPE(self%q_in(self%iget + 1:self%iget + SIZE(qvec)), SHAPE(qvec))
      IF (.NOT. ALL(ieee_is_finite(qvec))) self%layout_error = .TRUE.
      self%iget = self%iget + SIZE(qvec)

   END SUBROUTINE cp2k_tblite_mixer_get_3d

! **************************************************************************************************
!> \brief Get RMS SCC residual.
!> \param self mixer
!> \return residual
! **************************************************************************************************
   PURE FUNCTION cp2k_tblite_mixer_get_error(self) RESULT(error)
      CLASS(cp2k_tblite_broyden_mixer_type), INTENT(IN)  :: self
      REAL(wp)                                           :: error

      IF (self%layout_error .OR. .NOT. ALL(ieee_is_finite(self%dq))) THEN
         ! Keep the convergence metric finite; next() reports the detailed fatal mixer error.
         error = HUGE(1.0_wp)
      ELSE
         error = SQRT(SUM(self%dq**2)/SIZE(self%dq))
         IF (.NOT. ieee_is_finite(error)) error = HUGE(1.0_wp)
      END IF

   END FUNCTION cp2k_tblite_mixer_get_error

! **************************************************************************************************
!> \brief Modified Broyden update matching tblite's default algorithm, with explicit constants.
!> \param n ...
!> \param q ...
!> \param qlast ...
!> \param dq ...
!> \param dqlast ...
!> \param iter ...
!> \param memory ...
!> \param alpha ...
!> \param omega0 ...
!> \param minw ...
!> \param maxw ...
!> \param wfac ...
!> \param omega ...
!> \param df ...
!> \param u ...
!> \param a ...
!> \param info ...
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_broyden(n, q, qlast, dq, dqlast, iter, memory, alpha, omega0, minw, &
                                  maxw, wfac, omega, df, u, a, info)
      INTEGER, INTENT(IN)                                :: n
      REAL(wp), DIMENSION(n), INTENT(INOUT)              :: q, qlast
      REAL(wp), DIMENSION(n), INTENT(IN)                 :: dq
      REAL(wp), DIMENSION(n), INTENT(INOUT)              :: dqlast
      INTEGER, INTENT(IN)                                :: iter, memory
      REAL(wp), INTENT(IN)                               :: alpha, omega0, minw, maxw, wfac
      REAL(wp), DIMENSION(memory), INTENT(INOUT)         :: omega
      REAL(wp), DIMENSION(n, memory), INTENT(INOUT)      :: df, u
      REAL(wp), DIMENSION(memory, memory), INTENT(INOUT) :: a
      INTEGER, INTENT(OUT)                               :: info

      INTEGER                                            :: current_memory, i, it1, j, sloti, slotj, start_index
      REAL(wp)                                           :: inv, norm_dq, norm_dqlast, norm_dr, norm_dr_tol
      REAL(wp), ALLOCATABLE, DIMENSION(:, :)             :: beta, c

      info = 0
      IF (n < 1 .OR. memory < 1 .OR. iter < 1 .OR. maxw <= 0.0_wp .OR. &
          minw < 0.0_wp .OR. minw > maxw .OR. omega0 <= 0.0_wp) THEN
         info = -1
         RETURN
      END IF
      IF (.NOT. ALL(ieee_is_finite([alpha, omega0, minw, maxw, wfac])) .OR. &
          .NOT. ALL(ieee_is_finite(q)) .OR. .NOT. ALL(ieee_is_finite(dq))) THEN
         info = -2
         RETURN
      END IF

      IF (iter == 1) THEN
         dqlast(:) = dq
         qlast(:) = q
         q(:) = q + alpha*dq
         IF (.NOT. ALL(ieee_is_finite(q))) info = -3
         RETURN
      END IF

      IF (.NOT. ALL(ieee_is_finite(qlast)) .OR. &
          .NOT. ALL(ieee_is_finite(dqlast))) THEN
         info = -2
         RETURN
      END IF

      ! Store the new secant in a cyclic buffer, but expose the active history to LAPACK as a
      ! compact matrix.  The latter is essential once the physical slots wrap around memory.
      it1 = MOD(iter - 2, memory) + 1
      current_memory = MIN(iter - 1, memory)
      start_index = iter - 1 - current_memory
      ALLOCATE (beta(current_memory, current_memory), SOURCE=0.0_wp)
      ALLOCATE (c(current_memory, 1), SOURCE=0.0_wp)

      norm_dq = SQRT(DOT_PRODUCT(dq, dq))
      norm_dqlast = SQRT(DOT_PRODUCT(dqlast, dqlast))
      df(:, it1) = dq - dqlast
      norm_dr = SQRT(DOT_PRODUCT(df(:, it1), df(:, it1)))
      IF (.NOT. ALL(ieee_is_finite([norm_dq, norm_dqlast, norm_dr]))) THEN
         info = -3
         RETURN
      END IF
      norm_dr_tol = EPSILON(1.0_wp)*MAX(1.0_wp, norm_dq, norm_dqlast)

      IF (norm_dr <= norm_dr_tol) THEN
         ! A degenerate secant carries no Broyden information.  Give its slot zero weight instead
         ! of forming (q-qlast)/norm_dr; older non-degenerate history remains usable.
         omega(it1) = 0.0_wp
         df(:, it1) = 0.0_wp
         u(:, it1) = 0.0_wp
      ELSE
         omega(it1) = norm_dq
         IF (omega(it1) > (wfac/maxw)) THEN
            omega(it1) = wfac/omega(it1)
         ELSE
            omega(it1) = maxw
         END IF
         omega(it1) = MIN(MAX(omega(it1), minw), maxw)

         inv = 1.0_wp/norm_dr
         df(:, it1) = inv*df(:, it1)
         u(:, it1) = alpha*df(:, it1) + inv*(q - qlast)
      END IF

      IF (.NOT. ieee_is_finite(omega(it1)) .OR. &
          .NOT. ALL(ieee_is_finite(df(:, it1))) .OR. &
          .NOT. ALL(ieee_is_finite(u(:, it1)))) THEN
         info = -3
         RETURN
      END IF

      DO i = 1, current_memory
         sloti = MOD(start_index + i - 1, memory) + 1
         a(sloti, it1) = DOT_PRODUCT(df(:, sloti), df(:, it1))
         a(it1, sloti) = a(sloti, it1)
      END DO

      dqlast(:) = dq
      qlast(:) = q

      DO i = 1, current_memory
         sloti = MOD(start_index + i - 1, memory) + 1
         c(i, 1) = omega(sloti)*DOT_PRODUCT(df(:, sloti), dq)
         DO j = 1, current_memory
            slotj = MOD(start_index + j - 1, memory) + 1
            beta(i, j) = omega(sloti)*omega(slotj)*a(sloti, slotj)
         END DO
         beta(i, i) = beta(i, i) + omega0*omega0
      END DO

      IF (.NOT. ALL(ieee_is_finite(beta)) .OR. .NOT. ALL(ieee_is_finite(c))) THEN
         info = -3
         RETURN
      END IF

      CALL cp2k_tblite_lineq(beta, c, info)
      IF (info /= 0) RETURN
      IF (.NOT. ALL(ieee_is_finite(c))) THEN
         info = -3
         RETURN
      END IF

      q(:) = q + alpha*dq

      DO i = 1, current_memory
         sloti = MOD(start_index + i - 1, memory) + 1
         q(:) = q - omega(sloti)*c(i, 1)*u(:, sloti)
      END DO
      IF (.NOT. ALL(ieee_is_finite(q))) info = -3

   END SUBROUTINE cp2k_tblite_broyden

! **************************************************************************************************
!> \brief Solve the small Broyden linear system.
!> \param a ...
!> \param c ...
!> \param info ...
! **************************************************************************************************
   SUBROUTINE cp2k_tblite_lineq(a, c, info)
      REAL(wp), DIMENSION(:, :), INTENT(INOUT)           :: a, c
      INTEGER, INTENT(OUT)                               :: info

      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: ipiv

      ALLOCATE (ipiv(SIZE(a, 1)))
      CALL getrf(a, ipiv, info)
      IF (info == 0) CALL getrs(a, c, ipiv, info, trans="t")

   END SUBROUTINE cp2k_tblite_lineq
#endif

END MODULE tblite_scc_mixer
