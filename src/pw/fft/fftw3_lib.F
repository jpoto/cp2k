!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
MODULE fftw3_lib
   USE ISO_C_BINDING, ONLY: C_ASSOCIATED, &
                            C_CHAR, &
                            C_DOUBLE, &
                            C_DOUBLE_COMPLEX, &
                            C_INT, &
                            C_PTR
#if defined(__FFTW3)
   USE ISO_C_BINDING, ONLY: &
      C_FLOAT, &
      C_FLOAT_COMPLEX, &
      C_FUNPTR, &
      C_INT32_T, &
      C_INTPTR_T, &
      C_LOC, &
      C_NULL_CHAR, &
      C_SIZE_T, C_F_POINTER
   USE mathconstants, ONLY: z_zero
#endif
   USE cp_files, ONLY: get_unit_number
   USE fft_kinds, ONLY: dp
   USE fft_plan, ONLY: fft_plan_type

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

#include "../../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: fftw3_do_init, fftw3_do_cleanup, fftw3_get_lengths

#if defined(__FFTW3)
#include "fftw3.f03"
#endif

CONTAINS

#if defined(__FFTW3)
! **************************************************************************************************
!> \brief A workaround that allows us to compile with -Werror=unused-parameter
! **************************************************************************************************
   SUBROUTINE dummy_routine_to_call_mark_used()
      MARK_USED(FFTW_R2HC)
      MARK_USED(FFTW_HC2R)
      MARK_USED(FFTW_DHT)
      MARK_USED(FFTW_REDFT00)
      MARK_USED(FFTW_REDFT01)
      MARK_USED(FFTW_REDFT10)
      MARK_USED(FFTW_REDFT11)
      MARK_USED(FFTW_RODFT00)
      MARK_USED(FFTW_RODFT01)
      MARK_USED(FFTW_RODFT10)
      MARK_USED(FFTW_RODFT11)
      MARK_USED(FFTW_FORWARD)
      MARK_USED(FFTW_BACKWARD)
      MARK_USED(FFTW_MEASURE)
      MARK_USED(FFTW_DESTROY_INPUT)
      MARK_USED(FFTW_UNALIGNED)
      MARK_USED(FFTW_CONSERVE_MEMORY)
      MARK_USED(FFTW_EXHAUSTIVE)
      MARK_USED(FFTW_PRESERVE_INPUT)
      MARK_USED(FFTW_PATIENT)
      MARK_USED(FFTW_ESTIMATE)
      MARK_USED(FFTW_WISDOM_ONLY)
      MARK_USED(FFTW_ESTIMATE_PATIENT)
      MARK_USED(FFTW_BELIEVE_PCOST)
      MARK_USED(FFTW_NO_DFT_R2HC)
      MARK_USED(FFTW_NO_NONTHREADED)
      MARK_USED(FFTW_NO_BUFFERING)
      MARK_USED(FFTW_NO_INDIRECT_OP)
      MARK_USED(FFTW_ALLOW_LARGE_GENERIC)
      MARK_USED(FFTW_NO_RANK_SPLITS)
      MARK_USED(FFTW_NO_VRANK_SPLITS)
      MARK_USED(FFTW_NO_VRECURSE)
      MARK_USED(FFTW_NO_SIMD)
      MARK_USED(FFTW_NO_SLOW)
      MARK_USED(FFTW_NO_FIXED_RADIX_LARGE_N)
      MARK_USED(FFTW_ALLOW_PRUNING)
   END SUBROUTINE dummy_routine_to_call_mark_used
#endif

! **************************************************************************************************
!> \brief ...
!> \param wisdom_file ...
!> \param ionode ...
! **************************************************************************************************
   SUBROUTINE fftw3_do_cleanup(wisdom_file, ionode)

      CHARACTER(LEN=*), INTENT(IN)             :: wisdom_file
      LOGICAL                                  :: ionode

#if defined(__FFTW3)
      CHARACTER(LEN=1, KIND=C_CHAR), DIMENSION(:), ALLOCATABLE :: wisdom_file_name_c
      INTEGER                                  :: file_name_length, i, iunit, istat
      INTEGER(KIND=C_INT)                      :: isuccess
      ! Write out FFTW3 wisdom to file (if we can)
      ! only the ionode updates the wisdom
      IF (ionode) THEN
         iunit = get_unit_number()
         ! Check whether the file can be opened in the necessary manner
         OPEN (UNIT=iunit, FILE=wisdom_file, STATUS="UNKNOWN", FORM="FORMATTED", ACTION="WRITE", IOSTAT=istat)
         IF (istat == 0) THEN
            CLOSE (iunit)
            file_name_length = LEN_TRIM(wisdom_file)
            ALLOCATE (wisdom_file_name_c(file_name_length + 1))
            DO i = 1, file_name_length
               wisdom_file_name_c(i) = wisdom_file(i:i)
            END DO
            wisdom_file_name_c(file_name_length + 1) = C_NULL_CHAR
            isuccess = fftw_export_wisdom_to_filename(wisdom_file_name_c)
            IF (isuccess == 0) &
               CALL cp_warn(__LOCATION__, "Error exporting wisdom to file "//TRIM(wisdom_file)//". "// &
                            "Wisdom was not exported.")
         END IF
      END IF

      CALL fftw_cleanup()
#else
      MARK_USED(wisdom_file)
      MARK_USED(ionode)
#endif

   END SUBROUTINE fftw3_do_cleanup

! **************************************************************************************************
!> \brief ...
!> \param wisdom_file ...
! **************************************************************************************************
   SUBROUTINE fftw3_do_init(wisdom_file)

      CHARACTER(LEN=*), INTENT(IN)             :: wisdom_file

#if defined(__FFTW3)
      CHARACTER(LEN=1, KIND=C_CHAR), DIMENSION(:), ALLOCATABLE :: wisdom_file_name_c
      INTEGER                                  :: file_name_length, i, istat, iunit
      INTEGER(KIND=C_INT)                      :: isuccess
      LOGICAL :: file_exists

      isuccess = fftw_init_threads()
      IF (isuccess == 0) &
         CPABORT("Error initializing FFTW with threads")

      ! Read FFTW wisdom (if available)
      ! all nodes are opening the file here...
      INQUIRE (FILE=wisdom_file, exist=file_exists)
      IF (file_exists) THEN
         iunit = get_unit_number()
         file_name_length = LEN_TRIM(wisdom_file)
         ! Check whether the file can be opened in the necessary manner
         OPEN (UNIT=iunit, FILE=wisdom_file, STATUS="OLD", FORM="FORMATTED", POSITION="REWIND", &
               ACTION="READ", IOSTAT=istat)
         IF (istat == 0) THEN
            CLOSE (iunit)
            file_name_length = LEN_TRIM(wisdom_file)
            ALLOCATE (wisdom_file_name_c(file_name_length + 1))
            DO i = 1, file_name_length
               wisdom_file_name_c(i) = wisdom_file(i:i)
            END DO
            wisdom_file_name_c(file_name_length + 1) = C_NULL_CHAR
            isuccess = fftw_import_wisdom_from_filename(wisdom_file_name_c)
            IF (isuccess == 0) &
               CALL cp_warn(__LOCATION__, "Error importing wisdom from file "//TRIM(wisdom_file)//". "// &
                            "Maybe the file was created with a different configuration than CP2K is run with. "// &
                            "CP2K continues without importing wisdom.")
         END IF
      END IF
#else
      MARK_USED(wisdom_file)
#endif

   END SUBROUTINE fftw3_do_init

! **************************************************************************************************
!> \brief ...
!> \param DATA ...
!> \param max_length ...
!> \par History
!>      JGH 23-Jan-2006 : initial version
!>      Adapted for new interface
!>      IAB 09-Jan-2009 : Modified to cache plans in fft_plan_type
!>                        (c) The Numerical Algorithms Group (NAG) Ltd, 2009 on behalf of the HECToR project
!>      IAB 09-Oct-2009 : Added OpenMP directives to 1D FFT, and planning routines
!>                        (c) The Numerical Algorithms Group (NAG) Ltd, 2009 on behalf of the HECToR project
!>      IAB 11-Sep-2012 : OpenMP parallel 3D FFT (Ruyman Reyes, PRACE)
!> \author JGH
! **************************************************************************************************
   SUBROUTINE fftw3_get_lengths(DATA, max_length)

      INTEGER, DIMENSION(*)                              :: DATA
      INTEGER, INTENT(INOUT)                             :: max_length

      INTEGER                                            :: h, i, j, k, m, maxn, maxn_elevens, &
                                                            maxn_fives, maxn_sevens, &
                                                            maxn_thirteens, maxn_threes, &
                                                            maxn_twos, ndata, nmax, number
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: dlocal, idx

!------------------------------------------------------------------------------
! compute ndata
!! FFTW can do arbitrary(?) lengths, maybe you want to limit them to some
!!    powers of small prime numbers though...

      maxn_twos = 15
      maxn_threes = 3
      maxn_fives = 2
      maxn_sevens = 1
      maxn_elevens = 1
      maxn_thirteens = 0
      maxn = 37748736

      ndata = 0
      DO h = 0, maxn_twos
         nmax = HUGE(0)/2**h
         DO i = 0, maxn_threes
            DO j = 0, maxn_fives
               DO k = 0, maxn_sevens
                  DO m = 0, maxn_elevens
                     number = (3**i)*(5**j)*(7**k)*(11**m)

                     IF (number > nmax) CYCLE

                     number = number*2**h
                     IF (number >= maxn) CYCLE

                     ndata = ndata + 1
                  END DO
               END DO
            END DO
         END DO
      END DO

      ALLOCATE (dlocal(ndata), idx(ndata))

      ndata = 0
      dlocal(:) = 0
      DO h = 0, maxn_twos
         nmax = HUGE(0)/2**h
         DO i = 0, maxn_threes
            DO j = 0, maxn_fives
               DO k = 0, maxn_sevens
                  DO m = 0, maxn_elevens
                     number = (3**i)*(5**j)*(7**k)*(11**m)

                     IF (number > nmax) CYCLE

                     number = number*2**h
                     IF (number >= maxn) CYCLE

                     ndata = ndata + 1
                     dlocal(ndata) = number
                  END DO
               END DO
            END DO
         END DO
      END DO

      CALL sortint(dlocal, ndata, idx)
      ndata = MIN(ndata, max_length)
      DATA(1:ndata) = dlocal(1:ndata)
      max_length = ndata

      DEALLOCATE (dlocal, idx)

   END SUBROUTINE fftw3_get_lengths

! **************************************************************************************************
!> \brief ...
!> \param iarr ...
!> \param n ...
!> \param index ...
! **************************************************************************************************
   SUBROUTINE sortint(iarr, n, index)

      INTEGER, INTENT(IN)                                :: n
      INTEGER, INTENT(INOUT)                             :: iarr(1:n)
      INTEGER, INTENT(OUT)                               :: INDEX(1:n)

      INTEGER, PARAMETER                                 :: m = 7, nstack = 50

      INTEGER                                            :: a, i, ib, ir, istack(1:nstack), itemp, &
                                                            j, jstack, k, l, temp

!------------------------------------------------------------------------------

      DO i = 1, n
         INDEX(i) = i
      END DO
      jstack = 0
      l = 1
      ir = n
      DO WHILE (.TRUE.)
      IF (ir - l < m) THEN
         DO j = l + 1, ir
            a = iarr(j)
            ib = INDEX(j)
            DO i = j - 1, 0, -1
               IF (i == 0) EXIT
               IF (iarr(i) <= a) EXIT
               iarr(i + 1) = iarr(i)
               INDEX(i + 1) = INDEX(i)
            END DO
            iarr(i + 1) = a
            INDEX(i + 1) = ib
         END DO
         IF (jstack == 0) RETURN
         ir = istack(jstack)
         l = istack(jstack - 1)
         jstack = jstack - 2
      ELSE
         k = (l + ir)/2
         temp = iarr(k)
         iarr(k) = iarr(l + 1)
         iarr(l + 1) = temp
         itemp = INDEX(k)
         INDEX(k) = INDEX(l + 1)
         INDEX(l + 1) = itemp
         IF (iarr(l + 1) > iarr(ir)) THEN
            temp = iarr(l + 1)
            iarr(l + 1) = iarr(ir)
            iarr(ir) = temp
            itemp = INDEX(l + 1)
            INDEX(l + 1) = INDEX(ir)
            INDEX(ir) = itemp
         END IF
         IF (iarr(l) > iarr(ir)) THEN
            temp = iarr(l)
            iarr(l) = iarr(ir)
            iarr(ir) = temp
            itemp = INDEX(l)
            INDEX(l) = INDEX(ir)
            INDEX(ir) = itemp
         END IF
         IF (iarr(l + 1) > iarr(l)) THEN
            temp = iarr(l + 1)
            iarr(l + 1) = iarr(l)
            iarr(l) = temp
            itemp = INDEX(l + 1)
            INDEX(l + 1) = INDEX(l)
            INDEX(l) = itemp
         END IF
         i = l + 1
         j = ir
         a = iarr(l)
         ib = INDEX(l)
         DO WHILE (.TRUE.)
            i = i + 1
            DO WHILE (iarr(i) < a)
               i = i + 1
            END DO
            j = j - 1
            DO WHILE (iarr(j) > a)
               j = j - 1
            END DO
            IF (j < i) EXIT
            temp = iarr(i)
            iarr(i) = iarr(j)
            iarr(j) = temp
            itemp = INDEX(i)
            INDEX(i) = INDEX(j)
            INDEX(j) = itemp
         END DO
         iarr(l) = iarr(j)
         iarr(j) = a
         INDEX(l) = INDEX(j)
         INDEX(j) = ib
         jstack = jstack + 2
         IF (jstack > nstack) CPABORT("Nstack too small in sortint")
         IF (ir - i + 1 >= j - l) THEN
            istack(jstack) = ir
            istack(jstack - 1) = i
            ir = j - 1
         ELSE
            istack(jstack) = j - 1
            istack(jstack - 1) = l
            l = i
         END IF
      END IF

      END DO

   END SUBROUTINE sortint
   END MODULE fftw3_lib
