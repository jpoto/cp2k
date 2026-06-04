!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief This module defines the grid data type and some basic operations on it
!> \note
!>      pw_grid_create : set the defaults
!>      pw_grid_release : release all memory connected to type
!>      pw_grid_setup  : main routine to set up a grid
!>           input: cell (the box for the grid)
!>                  pw_grid (the grid; pw_grid%grid_span has to be set)
!>                  cutoff (optional, if not given pw_grid%bounds has to be set)
!>                  pe_group (optional, if not given we have a local grid)
!>
!>                  if no cutoff or a negative cutoff is given, all g-vectors
!>                  in the box are included (no spherical cutoff)
!>
!>                  for a distributed setup the array in para rs_dims has to
!>                  be initialized
!>           output: pw_grid
!>
!>      pw_grid_change : updates g-vectors after a change of the box
!> \par History
!>      JGH (20-12-2000) : Adapted for parallel use
!>      JGH (07-02-2001) : Added constructor and destructor routines
!>      JGH (21-02-2003) : Generalized reference grid concept
!>      JGH (19-11-2007) : Refactoring and modularization
!>      JGH (21-12-2007) : pw_grid_setup refactoring
!> \author apsi
!>      CJM
! **************************************************************************************************
MODULE pw_grids
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE fft_api,                         ONLY: fft_create_grid, &
                                              fft_create_grid_from_reference, &
                                              fft_free_grid, &
                                              fft_grid_get_comm, &
                                              fft_grid_get_proc2local_rs, &
                                              fft_grid_get_proc2local_ms, &
                                              fft_grid_get_proc2local_gs, &
                                              fft_grid_get_number_of_gs_points, &
                                              fft_grid_get_gs_index_to_vector, &
                                              fft_grid_get_gs_index_to_gsquared, &
                                              fft_grid_set_hmat
   USE mathconstants,                   ONLY: twopi
   USE mathlib,                         ONLY: det_3x3,&
                                              inv_3x3
   USE message_passing,                 ONLY: mp_comm_self,&
                                              mp_comm_type,&
                                              mp_dims_create
   USE pw_grid_info,                    ONLY: pw_find_cutoff,&
                                              pw_grid_bounds_from_n,&
                                              pw_grid_init_setup
   USE pw_grid_types,                   ONLY: FULLSPACE,&
                                              HALFSPACE,&
                                              PW_MODE_DISTRIBUTED,&
                                              PW_MODE_LOCAL,&
                                              map_pn,&
                                              pw_grid_type
   USE util,                            ONLY: get_limit,&
                                              sort
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: pw_grid_create, pw_grid_retain, pw_grid_release
   PUBLIC :: get_pw_grid_info, pw_grid_compare
   PUBLIC :: pw_grid_change

   INTEGER :: grid_tag = 0
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pw_grids'

   ! Distribution in g-space can be
   INTEGER, PARAMETER, PUBLIC               :: do_pw_grid_blocked_false = 0, &
                                               do_pw_grid_blocked_true = 1, &
                                               do_pw_grid_blocked_free = 2

   INTERFACE pw_grid_create
      MODULE PROCEDURE pw_grid_create_local
      MODULE PROCEDURE pw_grid_create_extended
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Initialize a PW grid with bounds only (used by some routines)
!> \param pw_grid ...
!> \param bounds ...
!> \par History
!>      JGH (21-Feb-2003) : initialize pw_grid%reference
!> \author JGH (7-Feb-2001) & fawzi
! **************************************************************************************************
   SUBROUTINE pw_grid_create_local(pw_grid, bounds)

      TYPE(pw_grid_type), POINTER                        :: pw_grid
      INTEGER, DIMENSION(2, 3), INTENT(IN)               :: bounds

      INTEGER, DIMENSION(2)                              :: rs_dims

      CPASSERT(.NOT. ASSOCIATED(pw_grid))
      ALLOCATE (pw_grid)
      pw_grid%bounds = bounds
      pw_grid%bounds_local = bounds
      pw_grid%npts = bounds(2, :) - bounds(1, :) + 1
      pw_grid%npts_local = pw_grid%npts
      pw_grid%ngpts = PRODUCT(INT(pw_grid%npts, KIND=int_8))
      pw_grid%ngpts_cut = pw_grid%ngpts
      pw_grid%ngpts_local = PRODUCT(pw_grid%npts)
      pw_grid%ngpts_cut_local = pw_grid%ngpts_local
      pw_grid%grid_span = FULLSPACE
      pw_grid%para%mode = PW_MODE_LOCAL
      pw_grid%reference = 0
      pw_grid%ref_count = 1
      NULLIFY (pw_grid%g)
      NULLIFY (pw_grid%gsq)
      NULLIFY (pw_grid%g_hatmap)
      NULLIFY (pw_grid%gidx)
      NULLIFY (pw_grid%grays)

      ! assign a unique tag to this grid
      grid_tag = grid_tag + 1
      pw_grid%id_nr = grid_tag

      ! parallel info
      rs_dims = 1
      CALL pw_grid%para%group%create(mp_comm_self, 2, rs_dims)
      IF (pw_grid%para%group%num_pe > 1) THEN
         pw_grid%para%mode = PW_MODE_DISTRIBUTED
      ELSE
         pw_grid%para%mode = PW_MODE_LOCAL
      END IF

   END SUBROUTINE pw_grid_create_local

! **************************************************************************************************
!> \brief Check if two pw_grids are equal
!> \param grida ...
!> \param gridb ...
!> \return ...
!> \par History
!>      none
!> \author JGH (14-Feb-2001)
! **************************************************************************************************
   FUNCTION pw_grid_compare(grida, gridb) RESULT(equal)

      TYPE(pw_grid_type), INTENT(IN)                     :: grida, gridb
      LOGICAL                                            :: equal

!------------------------------------------------------------------------------

      IF (grida%id_nr == gridb%id_nr) THEN
         equal = .TRUE.
      ELSE
         ! for the moment all grids with different identifiers are considered as not equal
         ! later we can get this more relaxed
         equal = .FALSE.
      END IF

   END FUNCTION pw_grid_compare

! **************************************************************************************************
!> \brief Access to information stored in the pw_grid_type
!> \param pw_grid ...
!> \param id_nr ...
!> \param mode ...
!> \param vol ...
!> \param dvol ...
!> \param npts ...
!> \param ngpts ...
!> \param ngpts_cut ...
!> \param dr ...
!> \param cutoff ...
!> \param orthorhombic ...
!> \param gvectors ...
!> \param gsquare ...
!> \par History
!>      none
!> \author JGH (17-Nov-2007)
! **************************************************************************************************
   SUBROUTINE get_pw_grid_info(pw_grid, id_nr, mode, vol, dvol, npts, ngpts, &
                               ngpts_cut, dr, cutoff, orthorhombic, gvectors, gsquare)

      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid
      INTEGER, INTENT(OUT), OPTIONAL                     :: id_nr, mode
      REAL(dp), INTENT(OUT), OPTIONAL                    :: vol, dvol
      INTEGER, DIMENSION(3), INTENT(OUT), OPTIONAL       :: npts
      INTEGER(int_8), INTENT(OUT), OPTIONAL              :: ngpts, ngpts_cut
      REAL(dp), DIMENSION(3), INTENT(OUT), OPTIONAL      :: dr
      REAL(dp), INTENT(OUT), OPTIONAL                    :: cutoff
      LOGICAL, INTENT(OUT), OPTIONAL                     :: orthorhombic
      REAL(dp), DIMENSION(:, :), OPTIONAL, POINTER       :: gvectors
      REAL(dp), DIMENSION(:), OPTIONAL, POINTER          :: gsquare

      CPASSERT(pw_grid%ref_count > 0)

      IF (PRESENT(id_nr)) id_nr = pw_grid%id_nr
      IF (PRESENT(mode)) mode = pw_grid%para%mode
      IF (PRESENT(vol)) vol = pw_grid%vol
      IF (PRESENT(dvol)) dvol = pw_grid%dvol
      IF (PRESENT(npts)) npts(1:3) = pw_grid%npts(1:3)
      IF (PRESENT(ngpts)) ngpts = pw_grid%ngpts
      IF (PRESENT(ngpts_cut)) ngpts_cut = pw_grid%ngpts_cut
      IF (PRESENT(dr)) dr = pw_grid%dr
      IF (PRESENT(cutoff)) cutoff = pw_grid%cutoff
      IF (PRESENT(orthorhombic)) orthorhombic = pw_grid%orthorhombic
      IF (PRESENT(gvectors)) gvectors => pw_grid%g
      IF (PRESENT(gsquare)) gsquare => pw_grid%gsq

   END SUBROUTINE get_pw_grid_info

! **************************************************************************************************
!> \brief Set some information stored in the pw_grid_type
!> \param pw_grid ...
!> \param grid_span ...
!> \param npts ...
!> \param bounds ...
!> \param cutoff ...
!> \param spherical ...
!> \par History
!>      none
!> \author JGH (19-Nov-2007)
! **************************************************************************************************
   SUBROUTINE set_pw_grid_info(pw_grid, grid_span, npts, bounds, cutoff, spherical)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, INTENT(in), OPTIONAL                      :: grid_span
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL        :: npts
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: cutoff
      LOGICAL, INTENT(IN), OPTIONAL                      :: spherical

      CPASSERT(pw_grid%ref_count > 0)

      IF (PRESENT(grid_span)) THEN
         pw_grid%grid_span = grid_span
      END IF
      IF (PRESENT(bounds) .AND. PRESENT(npts)) THEN
         pw_grid%bounds = bounds
         pw_grid%npts = npts
         CPASSERT(ALL(npts == bounds(2, :) - bounds(1, :) + 1))
      ELSE IF (PRESENT(bounds)) THEN
         pw_grid%bounds = bounds
         pw_grid%npts = bounds(2, :) - bounds(1, :) + 1
      ELSE IF (PRESENT(npts)) THEN
         pw_grid%npts = npts
         pw_grid%bounds = pw_grid_bounds_from_n(npts)
      END IF
      IF (PRESENT(cutoff)) THEN
         pw_grid%cutoff = cutoff
         IF (PRESENT(spherical)) THEN
            pw_grid%spherical = spherical
         ELSE
            pw_grid%spherical = .FALSE.
         END IF
      END IF

   END SUBROUTINE set_pw_grid_info

! **************************************************************************************************
!> \brief sets up a pw_grid
!> \param pw_grid ...
!> \param mp_comm ...
!> \param cell_hmat ...
!> \param grid_span ...
!> \param cutoff ...
!> \param bounds ...
!> \param bounds_local ...
!> \param npts ...
!> \param spherical ...
!> \param odd ...
!> \param fft_compatible_grid ...
!> \param ncommensurate ...
!> \param icommensurate ...
!> \param blocked ...
!> \param ref_grid ...
!> \param rs_dims ...
!> \param iounit ...
!> \author JGH (21-Dec-2007)
!> \note
!>      this is the function that should be used in the future
! **************************************************************************************************
   SUBROUTINE pw_grid_create_extended(pw_grid, mp_comm, cell_hmat, grid_span, cutoff, bounds, bounds_local, npts, &
                                      spherical, odd, fft_compatible_grid, ncommensurate, icommensurate, blocked, ref_grid, &
                                      rs_dims, iounit)

      TYPE(pw_grid_type), POINTER                        :: pw_grid
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat

      CLASS(mp_comm_type), INTENT(IN)                    :: mp_comm
      INTEGER, INTENT(in), OPTIONAL                      :: grid_span
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: cutoff
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds, bounds_local
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL        :: npts
      LOGICAL, INTENT(in), OPTIONAL                      :: spherical, odd, fft_compatible_grid
      INTEGER, INTENT(in), OPTIONAL                      :: ncommensurate, icommensurate, blocked
      TYPE(pw_grid_type), INTENT(IN), OPTIONAL           :: ref_grid
      INTEGER, DIMENSION(2), INTENT(in), OPTIONAL        :: rs_dims
      INTEGER, INTENT(in), OPTIONAL                      :: iounit

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_setup'

      INTEGER                                            :: handle, my_icommensurate, &
                                                            my_ncommensurate
      INTEGER, DIMENSION(3)                              :: n
      LOGICAL                                            :: my_fft_compatible_grid, my_odd, my_spherical
      REAL(KIND=dp)                                      :: cell_deth, my_cutoff
      REAL(KIND=dp), DIMENSION(3, 3)                     :: cell_h_inv

      CALL timeset(routineN, handle)

      CPASSERT(.NOT. ASSOCIATED(pw_grid))
      ALLOCATE (pw_grid)
      pw_grid%bounds = 0
      pw_grid%cutoff = 0.0_dp
      pw_grid%grid_span = FULLSPACE
      pw_grid%para%mode = PW_MODE_LOCAL
      pw_grid%reference = 0
      pw_grid%ref_count = 1
      NULLIFY (pw_grid%g)
      NULLIFY (pw_grid%gsq)
      NULLIFY (pw_grid%g_hatmap)
      NULLIFY (pw_grid%gidx)
      NULLIFY (pw_grid%grays)

      ! assign a unique tag to this grid
      grid_tag = grid_tag + 1
      pw_grid%id_nr = grid_tag

      ! parallel info
      IF (mp_comm%num_pe > 1) THEN
         pw_grid%para%mode = PW_MODE_DISTRIBUTED
      ELSE
         pw_grid%para%mode = PW_MODE_LOCAL
      END IF

      cell_deth = ABS(det_3x3(cell_hmat))
      IF (cell_deth < 1.0E-10_dp) THEN
         CALL cp_abort(__LOCATION__, &
                       "An invalid set of cell vectors was specified. "// &
                       "The determinant det(h) is too small")
      END IF
      cell_h_inv = inv_3x3(cell_hmat)

      IF (PRESENT(grid_span)) THEN
         CALL set_pw_grid_info(pw_grid, grid_span=grid_span)
      END IF

      IF (PRESENT(spherical)) THEN
         my_spherical = spherical
      ELSE
         my_spherical = .FALSE.
      END IF

      IF (PRESENT(odd)) THEN
         my_odd = odd
      ELSE
         my_odd = .FALSE.
      END IF

      IF (PRESENT(fft_compatible_grid)) THEN
         my_fft_compatible_grid = fft_compatible_grid
         IF (my_fft_compatible_grid .AND. PRESENT(bounds_local)) &
         CPABORT("FFT_compatible grids need to determine their bounds themselves!")
      ELSE
         my_fft_compatible_grid = .FALSE.
      END IF

      IF (PRESENT(ncommensurate)) THEN
         my_ncommensurate = ncommensurate
         IF (PRESENT(icommensurate)) THEN
            my_icommensurate = icommensurate
         ELSE
            my_icommensurate = MIN(1, ncommensurate)
         END IF
      ELSE
         my_ncommensurate = 0
         my_icommensurate = 1
      END IF

      IF (PRESENT(bounds)) THEN
         IF (PRESENT(cutoff)) THEN
            CALL set_pw_grid_info(pw_grid, bounds=bounds, cutoff=cutoff, &
                                  spherical=my_spherical)
         ELSE
            n = bounds(2, :) - bounds(1, :) + 1
            my_cutoff = pw_find_cutoff(n, cell_h_inv)
            my_cutoff = 0.5_dp*my_cutoff*my_cutoff
            CALL set_pw_grid_info(pw_grid, bounds=bounds, cutoff=my_cutoff, &
                                  spherical=my_spherical)
         END IF
      ELSE IF (PRESENT(npts)) THEN
         n = npts
         IF (PRESENT(cutoff)) THEN
            my_cutoff = cutoff
         ELSE
            my_cutoff = pw_find_cutoff(npts, cell_h_inv)
            my_cutoff = 0.5_dp*my_cutoff*my_cutoff
         END IF
         IF (my_fft_compatible_grid) THEN
            n = pw_grid_init_setup(cell_hmat, cutoff=my_cutoff, &
                                   spherical=my_spherical, odd=my_odd, fft_compatible_grid=my_fft_compatible_grid, &
                                   ncommensurate=my_ncommensurate, icommensurate=my_icommensurate, &
                                   ref_grid=ref_grid, n_orig=n)
         END IF
         CALL set_pw_grid_info(pw_grid, npts=n, cutoff=my_cutoff, &
                               spherical=my_spherical)
      ELSE IF (PRESENT(cutoff)) THEN
         n = pw_grid_init_setup(cell_hmat, cutoff=cutoff, &
                                spherical=my_spherical, odd=my_odd, fft_compatible_grid=my_fft_compatible_grid, &
                                ncommensurate=my_ncommensurate, icommensurate=my_icommensurate, &
                                ref_grid=ref_grid)
         CALL set_pw_grid_info(pw_grid, npts=n, cutoff=cutoff, &
                               spherical=my_spherical)
      ELSE
         CPABORT("BOUNDS, NPTS or CUTOFF have to be specified")
      END IF
      WRITE(*,*) "Bounds", pw_grid%bounds

      IF (PRESENT(ref_grid)) THEN
         CALL fft_create_grid_from_reference(pw_grid%fft_grid, pw_grid%npts, cutoff, bounds_local, ref_grid%fft_grid)
      ELSE
         ! TODO: pass h matrix instead of h_inv instead of unscaled h_inv (ERROR!!!)
         CALL fft_create_grid(pw_grid%fft_grid, pw_grid%npts, mp_comm, cell_h_inv, &
                              pw_grid%grid_span == HALFSPACE, pw_grid%cutoff, rs_dims, bounds_local)
      END IF
      CALL fft_grid_get_comm(pw_grid%fft_grid, pw_grid%para%group)

      CALL pw_grid_setup_internal(cell_hmat, cell_h_inv, cell_deth, pw_grid, bounds_local=bounds_local, &
                                  blocked=blocked, ref_grid=ref_grid, iounit=iounit)

      CALL timestop(handle)

   END SUBROUTINE pw_grid_create_extended
! **************************************************************************************************
!> \brief sets up a pw_grid, needs valid bounds as input, it is up to you to
!>      make sure of it using pw_grid_bounds_from_n
!> \param cell_hmat ...
!> \param cell_h_inv ...
!> \param cell_deth ...
!> \param pw_grid ...
!> \param bounds_local ...
!> \param blocked ...
!> \param ref_grid ...
!> \param iounit ...
!> \par History
!>      JGH (20-Dec-2000) : Adapted for parallel use
!>      JGH (28-Feb-2001) : New optional argument fft_compatible_grid
!>      JGH (21-Mar-2001) : Reference grid code
!>      JGH (21-Mar-2001) : New optional argument symm_usage
!>      JGH (22-Mar-2001) : Simplify group assignment (mp_comm_dup)
!>      JGH (21-May-2002) : Remove orthorhombic keyword (code is fast enough)
!>      JGH (19-Feb-2003) : Negative cutoff can be used for non-spheric grids
!>      Joost VandeVondele (Feb-2004) : optionally generate pw grids that are commensurate in rs
!>      JGH (18-Dec-2007) : Refactoring
!> \author fawzi
!> \note
!>      this is the function that should be used in the future
! **************************************************************************************************
   SUBROUTINE pw_grid_setup_internal(cell_hmat, cell_h_inv, cell_deth, pw_grid, bounds_local, &
                                     blocked, ref_grid, iounit)
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat, cell_h_inv
      REAL(KIND=dp), INTENT(IN)                          :: cell_deth
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds_local
      INTEGER, INTENT(in), OPTIONAL                      :: blocked
      TYPE(pw_grid_type), INTENT(in), OPTIONAL           :: ref_grid
      INTEGER, INTENT(in), OPTIONAL                      :: iounit

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_setup_internal'

      INTEGER                                            :: handle, n(3)
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: yz_mask
      REAL(KIND=dp)                                      :: ecut

!------------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      CPASSERT(pw_grid%ref_count > 0)

      ! set pointer to possible reference grid
      IF (PRESENT(ref_grid)) THEN
         pw_grid%reference = ref_grid%id_nr
      END IF

      IF (pw_grid%spherical) THEN
         ecut = pw_grid%cutoff
      ELSE
         ecut = 1.e10_dp
      END IF

      n(:) = pw_grid%npts(:)

      ! Find the number of grid points
      ! yz_mask counts the number of g-vectors orthogonal to the yz plane
      ! the indices in yz_mask are from -n/2 .. n/2 shifted by n/2 + 1
      ! these are not mapped indices !
      ALLOCATE (yz_mask(n(2), n(3)))
      CALL pw_grid_count(cell_h_inv, pw_grid, pw_grid%para%group, ecut, yz_mask)

      ! Check if reference grid is compatible
      IF (PRESENT(ref_grid)) THEN
         CPASSERT(pw_grid%para%mode == ref_grid%para%mode)
         CPASSERT(pw_grid%grid_span == ref_grid%grid_span)
         CPASSERT(pw_grid%spherical .EQV. ref_grid%spherical)
      END IF

      ! Distribute grid
      CALL pw_grid_distribute(pw_grid, yz_mask, bounds_local=bounds_local, ref_grid=ref_grid, blocked=blocked)

      ! Allocate the grid fields
      CALL pw_grid_allocate(pw_grid, pw_grid%ngpts_cut_local, &
                            pw_grid%bounds)

      ! Fill in the grid structure
      CALL pw_grid_assign(cell_h_inv, pw_grid, ecut)

      ! Sort g vector wrt length (only local for each processor)
      CALL pw_grid_sort(pw_grid, ref_grid)

      CALL pw_grid_remap(pw_grid, yz_mask)

      DEALLOCATE (yz_mask)

      CALL cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)
      !
      ! Output: All the information of this grid type
      !

      IF (PRESENT(iounit)) THEN
         CALL pw_grid_print(pw_grid, iounit)
      END IF

      CALL timestop(handle)

   END SUBROUTINE pw_grid_setup_internal

! **************************************************************************************************
!> \brief Helper routine used by pw_grid_setup_internal and pw_grid_change
!> \param cell_hmat ...
!> \param cell_h_inv ...
!> \param cell_deth ...
!> \param pw_grid ...
!> \par History moved common code into new subroutine
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat, cell_h_inv
      REAL(KIND=dp), INTENT(IN)                          :: cell_deth
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      pw_grid%vol = ABS(cell_deth)
      pw_grid%dvol = pw_grid%vol/REAL(pw_grid%ngpts, KIND=dp)
      pw_grid%dr(1) = SQRT(SUM(cell_hmat(:, 1)**2)) &
                      /REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dr(2) = SQRT(SUM(cell_hmat(:, 2)**2)) &
                      /REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dr(3) = SQRT(SUM(cell_hmat(:, 3)**2)) &
                      /REAL(pw_grid%npts(3), KIND=dp)
      pw_grid%dh(:, 1) = cell_hmat(:, 1)/REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dh(:, 2) = cell_hmat(:, 2)/REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dh(:, 3) = cell_hmat(:, 3)/REAL(pw_grid%npts(3), KIND=dp)
      pw_grid%dh_inv(1, :) = cell_h_inv(1, :)*REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dh_inv(2, :) = cell_h_inv(2, :)*REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dh_inv(3, :) = cell_h_inv(3, :)*REAL(pw_grid%npts(3), KIND=dp)

      CALL fft_grid_set_hmat(pw_grid%fft_grid, cell_hmat)

      IF ((cell_hmat(1, 2) == 0.0_dp) .AND. (cell_hmat(1, 3) == 0.0_dp) .AND. &
          (cell_hmat(2, 1) == 0.0_dp) .AND. (cell_hmat(2, 3) == 0.0_dp) .AND. &
          (cell_hmat(3, 1) == 0.0_dp) .AND. (cell_hmat(3, 2) == 0.0_dp)) THEN
         pw_grid%orthorhombic = .TRUE.
      ELSE
         pw_grid%orthorhombic = .FALSE.
      END IF
   END SUBROUTINE cell2grid

! **************************************************************************************************
!> \brief Output of information on pw_grid
!> \param pw_grid ...
!> \param info ...
!> \author JGH[18-05-2007] from earlier versions
! **************************************************************************************************
   SUBROUTINE pw_grid_print(pw_grid, info)

      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid
      INTEGER, INTENT(IN)                                :: info

      INTEGER                                            :: i
      INTEGER(KIND=int_8)                                :: n(3)
      REAL(KIND=dp)                                      :: rv(3, 3)

!------------------------------------------------------------------------------
!
! Output: All the information of this grid type
!

      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
         IF (info > 0) THEN
            WRITE (info, '(/,A,T71,I10)') &
               " PW_GRID| Information for grid number ", pw_grid%id_nr
            IF (pw_grid%reference > 0) THEN
               WRITE (info, '(A,T71,I10)') &
                  " PW_GRID| Number of the reference grid ", pw_grid%reference
            END IF
            WRITE (info, '(" PW_GRID| Cutoff [a.u.]",T71,f10.1)') pw_grid%cutoff
            IF (pw_grid%spherical) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", "YES"
               WRITE (info, '(A,T71,I10)') " PW_GRID| Grid points within cutoff", &
                  pw_grid%ngpts_cut
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", " NO"
            END IF
            DO i = 1, 3
               WRITE (info, '(A,I3,T30,2I8,T62,A,T71,I10)') " PW_GRID|   Bounds ", &
                  i, pw_grid%bounds(1, I), pw_grid%bounds(2, I), &
                  "Points:", pw_grid%npts(I)
            END DO
            WRITE (info, '(A,G12.4,T50,A,T67,F14.4)') &
               " PW_GRID| Volume element (a.u.^3)", &
               pw_grid%dvol, " Volume (a.u.^3) :", pw_grid%vol
            IF (pw_grid%grid_span == HALFSPACE) THEN
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "HALFSPACE"
            ELSE
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "FULLSPACE"
            END IF
         END IF
      ELSE

         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%sum(n(1:2))
         n(3) = SUM(pw_grid%para%nyzray)
         rv(:, 1) = REAL(n, KIND=dp)/REAL(pw_grid%para%group%num_pe, KIND=dp)
         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%max(n(1:2))
         n(3) = MAXVAL(pw_grid%para%nyzray)
         rv(:, 2) = REAL(n, KIND=dp)
         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%min(n(1:2))
         n(3) = MINVAL(pw_grid%para%nyzray)
         rv(:, 3) = REAL(n, KIND=dp)

         IF (info > 0) THEN
            WRITE (info, '(/,A,T71,I10)') &
               " PW_GRID| Information for grid number ", pw_grid%id_nr
            IF (pw_grid%reference > 0) THEN
               WRITE (info, '(A,T71,I10)') &
                  " PW_GRID| Number of the reference grid ", pw_grid%reference
            END IF
            WRITE (info, '(A,T60,I10,A)') &
               " PW_GRID| Grid distributed over ", pw_grid%para%group%num_pe, &
               " processors"
            WRITE (info, '(A,T71,2I5)') &
               " PW_GRID| Real space group dimensions ", pw_grid%para%group%num_pe_cart
            IF (pw_grid%para%blocked) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| the grid is blocked: ", "YES"
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| the grid is blocked: ", " NO"
            END IF
            WRITE (info, '(" PW_GRID| Cutoff [a.u.]",T71,f10.1)') pw_grid%cutoff
            IF (pw_grid%spherical) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", "YES"
               WRITE (info, '(A,T71,I10)') " PW_GRID| Grid points within cutoff", &
                  pw_grid%ngpts_cut
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", " NO"
            END IF
            DO i = 1, 3
               WRITE (info, '(A,I3,T30,2I8,T62,A,T71,I10)') " PW_GRID|   Bounds ", &
                  i, pw_grid%bounds(1, I), pw_grid%bounds(2, I), &
                  "Points:", pw_grid%npts(I)
            END DO
            WRITE (info, '(A,G12.4,T50,A,T67,F14.4)') &
               " PW_GRID| Volume element (a.u.^3)", &
               pw_grid%dvol, " Volume (a.u.^3) :", pw_grid%vol
            IF (pw_grid%grid_span == HALFSPACE) THEN
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "HALFSPACE"
            ELSE
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "FULLSPACE"
            END IF
            WRITE (info, '(A,T48,A)') " PW_GRID|   Distribution", &
               "  Average         Max         Min"
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   G-Vectors", &
               rv(1, 1), NINT(rv(1, 2)), NINT(rv(1, 3))
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   G-Rays   ", &
               rv(3, 1), NINT(rv(3, 2)), NINT(rv(3, 3))
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   Real Space Points", &
               rv(2, 1), NINT(rv(2, 2)), NINT(rv(2, 3))
         END IF ! group head
      END IF ! local

   END SUBROUTINE pw_grid_print

! **************************************************************************************************
!> \brief Distribute grids in real and Fourier Space to the processors in group
!> \param pw_grid ...
!> \param yz_mask ...
!> \param bounds_local ...
!> \param ref_grid ...
!> \param blocked ...
!> \par History
!>      JGH (01-Mar-2001) optional reference grid
!>      JGH (22-May-2002) bug fix for pre_tag and HALFSPACE grids
!>      JGH (09-Sep-2003) reduce scaling for distribution
!> \author JGH (22-12-2000)
! **************************************************************************************************
   SUBROUTINE pw_grid_distribute(pw_grid, yz_mask, bounds_local, ref_grid, blocked)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      INTEGER, DIMENSION(:, :), INTENT(INOUT)            :: yz_mask
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds_local
      TYPE(pw_grid_type), INTENT(IN), OPTIONAL           :: ref_grid
      INTEGER, INTENT(IN), OPTIONAL                      :: blocked

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_distribute'

      INTEGER                                            :: blocked_local, coor(2), gmax, handle, i, &
                                                            i1, i2, ip, ipl, ipp, itmp, j, l, lby, &
                                                            lbz, lo(2), m, n, np, ns, nx, ny, nz, &
                                                            rsd(2), min_bounds(3)
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: pemap
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: yz_index
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: axis_dist_all
      INTEGER, DIMENSION(2)                              :: my_rs_dims
      INTEGER, DIMENSION(2, 3)                           :: axis_dist
      LOGICAL                                            :: blocking

!------------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)

      pw_grid%ngpts = PRODUCT(INT(pw_grid%npts, KIND=int_8))

      IF (PRESENT(blocked)) THEN
         blocked_local = blocked
      ELSE
         blocked_local = do_pw_grid_blocked_free
      END IF

      pw_grid%para%blocked = .FALSE.

      ALLOCATE(pw_grid%para%bo(2, 3, 0:pw_grid%para%group%num_pe-1, 3))
      min_bounds = pw_grid%bounds(1, :)
      IF (PRESENT(bounds_local)) THEN
         CPASSERT(bounds_local(1, 3) == 1)
         min_bounds = bounds_local(1, :)
         CALL pw_grid%para%group%min(min_bounds)
      END IF
      CALL fft_grid_get_proc2local_rs(pw_grid%fft_grid, pw_grid%para%bo(:, :, :, 1), min_bounds)
      min_bounds(2) = 1
      CALL fft_grid_get_proc2local_ms(pw_grid%fft_grid, pw_grid%para%bo(:, :, :, 2), min_bounds)
      CALL fft_grid_get_proc2local_gs(pw_grid%fft_grid, pw_grid%para%bo(:, :, :, 3))
      pw_grid%bounds_local(:, :) = pw_grid%para%bo(:, :, pw_grid%para%group%mepos, 1)
      IF (pw_grid%para%group%mepos==0) WRITE(*,*) "local bounds", pw_grid%para%bo

      pw_grid%ngpts_cut_local = fft_grid_get_number_of_gs_points(pw_grid%fft_grid)

      ALLOCATE(pw_grid%g_hat(3, pw_grid%ngpts_cut_local))
      ALLOCATE(pw_grid%gsq(pw_grid%ngpts_cut_local))

      CALL fft_grid_get_gs_index_to_vector(pw_grid%fft_grid, pw_grid%g_hat)
      CALL fft_grid_get_gs_index_to_gsquared(pw_grid%fft_grid, pw_grid%gsq)

      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN

         pw_grid%para%ray_distribution = .FALSE.

         pw_grid%npts_local = pw_grid%npts
         CPASSERT(pw_grid%ngpts_cut < HUGE(pw_grid%ngpts_cut_local))
         pw_grid%ngpts_cut_local = INT(pw_grid%ngpts_cut)
         CPASSERT(pw_grid%ngpts < HUGE(pw_grid%ngpts_local))
         pw_grid%ngpts_local = INT(pw_grid%ngpts)
         my_rs_dims = 1

      ELSE

         !..find the real space distribution
         nx = pw_grid%npts(1)
         ny = pw_grid%npts(2)
         nz = pw_grid%npts(3)
         np = pw_grid%para%group%num_pe

         ! The user can specify 2 strictly positive indices => specific layout
         !                      1 strictly positive index   => the other is fixed by the number of CPUs
         !                      0 strictly positive indices => fully free distribution
         ! if fully free or the user request can not be fulfilled, we optimize heuristically ourselves by:
         !                      1) nx>np -> taking a plane distribution (/np,1/)
         !                      2) nx<np -> taking the most square distribution
         ! if blocking is free:
         !                      1) blocked=.FALSE. for plane distributions
         !                      2) blocked=.TRUE.  for non-plane distributions

         !..fetch the communicator of the FFT grid

         ! now fix the blocking if we have a choice
         SELECT CASE (blocked_local)
         CASE (do_pw_grid_blocked_false)
            blocking = .FALSE.
         CASE (do_pw_grid_blocked_true)
            blocking = .TRUE.
         CASE (do_pw_grid_blocked_free)
            IF (ALL(my_rs_dims == [np, 1])) THEN
               blocking = .FALSE.
            ELSE
               blocking = .TRUE.
            END IF
         CASE DEFAULT
            CPABORT("")
         END SELECT

         pw_grid%npts_local(:) = pw_grid%bounds_local(2, :) &
                                 - pw_grid%bounds_local(1, :) + 1

         !..the third distribution is needed for the second step in the FFT
         rsd = pw_grid%para%group%num_pe_cart
         !..find the g space distribution
         pw_grid%ngpts_cut_local = 0

         ALLOCATE (pw_grid%para%nyzray(0:np - 1))

         ALLOCATE (pw_grid%para%yzq(ny, nz))

         IF (pw_grid%spherical .OR. pw_grid%grid_span == HALFSPACE &
             .OR. .NOT. blocking) THEN

            pw_grid%para%ray_distribution = .TRUE.

            pw_grid%para%yzq = -1
            IF (PRESENT(ref_grid)) THEN
               ! tag all vectors from the reference grid
               CALL pre_tag(pw_grid, yz_mask, ref_grid)
            END IF

            ! Round Robin distribution
            ! Processors 0 .. NP-1, NP-1 .. 0  get the largest remaining batch
            ! of g vectors in turn

            i1 = SIZE(yz_mask, 1)
            i2 = SIZE(yz_mask, 2)
            ALLOCATE (yz_index(2, i1*i2))
            CALL order_mask(yz_mask, yz_index)
            DO i = 1, i1*i2
               lo(1) = yz_index(1, i)
               lo(2) = yz_index(2, i)
               IF (lo(1)*lo(2) == 0) CYCLE
               gmax = yz_mask(lo(1), lo(2))
               IF (gmax == 0) CYCLE
               yz_mask(lo(1), lo(2)) = 0
               ip = MOD(i - 1, 2*np)
               IF (ip > np - 1) ip = 2*np - ip - 1
               IF (ip == pw_grid%para%group%mepos) THEN
                  pw_grid%ngpts_cut_local = pw_grid%ngpts_cut_local + gmax
               END IF
               pw_grid%para%yzq(lo(1), lo(2)) = ip
               IF (pw_grid%grid_span == HALFSPACE) THEN
                  m = -lo(1) - 2*lby + 2
                  n = -lo(2) - 2*lbz + 2
                  pw_grid%para%yzq(m, n) = ip
                  yz_mask(m, n) = 0
               END IF
            END DO

            DEALLOCATE (yz_index)

            ! Count the total number of rays on each processor
            pw_grid%para%nyzray = 0
            DO i = 1, nz
               DO j = 1, ny
                  ip = pw_grid%para%yzq(j, i)
                  IF (ip >= 0) pw_grid%para%nyzray(ip) = &
                     pw_grid%para%nyzray(ip) + 1
               END DO
            END DO

            ! Allocate mapping array (y:z, nray, nproc)
            ns = MAXVAL(pw_grid%para%nyzray(0:np - 1))
            ALLOCATE (pw_grid%para%yzp(2, ns, 0:np - 1))

            ! Fill mapping array, recalculate nyzray for convenience
            pw_grid%para%nyzray = 0
            DO i = 1, nz
               DO j = 1, ny
                  ip = pw_grid%para%yzq(j, i)
                  IF (ip >= 0) THEN
                     pw_grid%para%nyzray(ip) = &
                        pw_grid%para%nyzray(ip) + 1
                     ns = pw_grid%para%nyzray(ip)
                     pw_grid%para%yzp(1, ns, ip) = j
                     pw_grid%para%yzp(2, ns, ip) = i
                     IF (ip == pw_grid%para%group%mepos) THEN
                        pw_grid%para%yzq(j, i) = ns
                     ELSE
                        pw_grid%para%yzq(j, i) = -1
                     END IF
                  ELSE
                     pw_grid%para%yzq(j, i) = -2
                  END IF
               END DO
            END DO

            pw_grid%ngpts_local = PRODUCT(pw_grid%npts_local)

         ELSE
            !
            !  block distribution of g vectors, we do not have a spherical cutoff
            !

            pw_grid%para%blocked = .TRUE.
            pw_grid%para%ray_distribution = .FALSE.

            DO ip = 0, np - 1
               m = pw_grid%para%bo(2, 2, ip, 3) - &
                   pw_grid%para%bo(1, 2, ip, 3) + 1
               n = pw_grid%para%bo(2, 3, ip, 3) - &
                   pw_grid%para%bo(1, 3, ip, 3) + 1
               pw_grid%para%nyzray(ip) = n*m
            END DO

            ipl = pw_grid%para%group%mepos
            l = pw_grid%para%bo(2, 1, ipl, 3) - &
                pw_grid%para%bo(1, 1, ipl, 3) + 1
            m = pw_grid%para%bo(2, 2, ipl, 3) - &
                pw_grid%para%bo(1, 2, ipl, 3) + 1
            n = pw_grid%para%bo(2, 3, ipl, 3) - &
                pw_grid%para%bo(1, 3, ipl, 3) + 1
            pw_grid%ngpts_cut_local = l*m*n
            pw_grid%ngpts_local = pw_grid%ngpts_cut_local

            pw_grid%para%yzq = 0
            ny = pw_grid%para%bo(2, 2, ipl, 3) - &
                 pw_grid%para%bo(1, 2, ipl, 3) + 1
            DO n = pw_grid%para%bo(1, 3, ipl, 3), &
               pw_grid%para%bo(2, 3, ipl, 3)
               i = n - pw_grid%para%bo(1, 3, ipl, 3)
               DO m = pw_grid%para%bo(1, 2, ipl, 3), &
                  pw_grid%para%bo(2, 2, ipl, 3)
                  j = m - pw_grid%para%bo(1, 2, ipl, 3) + 1
                  pw_grid%para%yzq(m, n) = j + i*ny
               END DO
            END DO

            ! Allocate mapping array (y:z, nray, nproc)
            ns = MAXVAL(pw_grid%para%nyzray(0:np - 1))
            ALLOCATE (pw_grid%para%yzp(2, ns, 0:np - 1))
            pw_grid%para%yzp = 0

            ALLOCATE (pemap(0:np - 1))
            pemap = 0
            pemap(pw_grid%para%group%mepos) = pw_grid%para%group%mepos
            CALL pw_grid%para%group%sum(pemap)

            DO ip = 0, np - 1
               ipp = pemap(ip)
               ns = 0
               DO n = pw_grid%para%bo(1, 3, ipp, 3), &
                  pw_grid%para%bo(2, 3, ipp, 3)
                  i = n - pw_grid%bounds(1, 3) + 1
                  DO m = pw_grid%para%bo(1, 2, ipp, 3), &
                     pw_grid%para%bo(2, 2, ipp, 3)
                     j = m - pw_grid%bounds(1, 2) + 1
                     ns = ns + 1
                     pw_grid%para%yzp(1, ns, ip) = j
                     pw_grid%para%yzp(2, ns, ip) = i
                  END DO
               END DO
               CPASSERT(ns == pw_grid%para%nyzray(ip))
            END DO

            DEALLOCATE (pemap)

         END IF

      END IF

      ! pos_of_x(i) tells on which cpu pw%array(i,:,:) is located
      ! should be computable in principle, without the need for communication
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         ALLOCATE (pw_grid%para%pos_of_x(pw_grid%bounds(1, 1):pw_grid%bounds(2, 1)))
         pw_grid%para%pos_of_x = 0
         pw_grid%para%pos_of_x(pw_grid%bounds_local(1, 1):pw_grid%bounds_local(2, 1)) = pw_grid%para%group%mepos
         CALL pw_grid%para%group%sum(pw_grid%para%pos_of_x)
      ELSE
         ! this should not be needed
         ALLOCATE (pw_grid%para%pos_of_x(pw_grid%bounds(1, 1):pw_grid%bounds(2, 1)))
         pw_grid%para%pos_of_x = 0
      END IF

      CALL timestop(handle)

   END SUBROUTINE pw_grid_distribute

! **************************************************************************************************
!> \brief ...
!> \param pw_grid ...
!> \param yz_mask ...
!> \param ref_grid ...
!> \par History
!>      - Fix mapping bug for pw_grid eqv to ref_grid (21.11.2019, MK)
! **************************************************************************************************
   SUBROUTINE pre_tag(pw_grid, yz_mask, ref_grid)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, DIMENSION(:, :), INTENT(INOUT)            :: yz_mask
      TYPE(pw_grid_type), INTENT(IN)                     :: ref_grid

      INTEGER                                            :: gmax, ig, ip, lby, lbz, my, mz, ny, nz, &
                                                            uby, ubz, y, yp, z, zp

      ny = ref_grid%npts(2)
      nz = ref_grid%npts(3)
      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)
      uby = pw_grid%bounds(2, 2)
      ubz = pw_grid%bounds(2, 3)
      my = SIZE(yz_mask, 1)
      mz = SIZE(yz_mask, 2)

      ! loop over all processors and all g vectors yz lines on this processor
      DO ip = 0, ref_grid%para%group%num_pe - 1
         DO ig = 1, ref_grid%para%nyzray(ip)
            ! go from mapped coordinates to original coordinates
            ! 1, 2, ..., n-1, n -> 0, 1, ..., (n/2)-1, -(n/2), -(n/2)+1, ..., -2, -1
            y = ref_grid%para%yzp(1, ig, ip) - 1
            IF (y >= ny/2) y = y - ny
            z = ref_grid%para%yzp(2, ig, ip) - 1
            IF (z >= nz/2) z = z - nz
            ! check if this is inside the realm of the new grid
            IF (y < lby .OR. y > uby .OR. z < lbz .OR. z > ubz) CYCLE
            ! go to shifted coordinates
            y = y - lby + 1
            z = z - lbz + 1
            ! this tag is outside the cutoff range of the new grid
            IF (pw_grid%grid_span == HALFSPACE) THEN
               yp = -y - 2*lby + 2
               zp = -z - 2*lbz + 2
               ! if the reference grid is larger than the mirror point may be
               ! outside the new grid even if the original point is inside
               IF (yp < 1 .OR. yp > my .OR. zp < 1 .OR. zp > mz) CYCLE
               gmax = MAX(yz_mask(y, z), yz_mask(yp, zp))
               IF (gmax == 0) CYCLE
               yz_mask(y, z) = 0
               yz_mask(yp, zp) = 0
               pw_grid%para%yzq(y, z) = ip
               pw_grid%para%yzq(yp, zp) = ip
            ELSE
               gmax = yz_mask(y, z)
               IF (gmax == 0) CYCLE
               yz_mask(y, z) = 0
               pw_grid%para%yzq(y, z) = ip
            END IF
            IF (ip == pw_grid%para%group%mepos) THEN
               pw_grid%ngpts_cut_local = pw_grid%ngpts_cut_local + gmax
            END IF
         END DO
      END DO

   END SUBROUTINE pre_tag

! **************************************************************************************************
!> \brief ...
!> \param yz_mask ...
!> \param yz_index ...
! **************************************************************************************************
   PURE SUBROUTINE order_mask(yz_mask, yz_index)

      INTEGER, DIMENSION(:, :), INTENT(IN)               :: yz_mask
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: yz_index

      INTEGER                                            :: i1, i2, ic, icount, ii, im, jc, jj

!NB load balance
!------------------------------------------------------------------------------
!NB spiral out from origin, so that even if overall grid is full and
!NB block distributed, spherical cutoff still leads to good load
!NB balance in cp_ddapc_apply_CD

      i1 = SIZE(yz_mask, 1)
      i2 = SIZE(yz_mask, 2)
      yz_index = 0

      icount = 1
      ic = i1/2
      jc = i2/2
      ii = ic
      jj = jc
      IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
         IF (yz_mask(ii, jj) /= 0) THEN
            yz_index(1, icount) = ii
            yz_index(2, icount) = jj
            icount = icount + 1
         END IF
      END IF
      DO im = 1, MAX(ic + 1, jc + 1)
         ii = ic - im
         DO jj = jc - im, jc + im
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         ii = ic + im
         DO jj = jc - im, jc + im
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         jj = jc - im
         DO ii = ic - im + 1, ic + im - 1
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         jj = jc + im
         DO ii = ic - im + 1, ic + im - 1
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
      END DO

   END SUBROUTINE order_mask
! **************************************************************************************************
!> \brief compute the length of g vectors
!> \param h_inv ...
!> \param length_x ...
!> \param length_y ...
!> \param length_z ...
!> \param length ...
!> \param l ...
!> \param m ...
!> \param n ...
! **************************************************************************************************
   PURE SUBROUTINE pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: h_inv
      REAL(KIND=dp), INTENT(OUT)                         :: length_x, length_y, length_z, length
      INTEGER, INTENT(IN)                                :: l, m, n

      length_x &
         = REAL(l, dp)*h_inv(1, 1) &
           + REAL(m, dp)*h_inv(2, 1) &
           + REAL(n, dp)*h_inv(3, 1)
      length_y &
         = REAL(l, dp)*h_inv(1, 2) &
           + REAL(m, dp)*h_inv(2, 2) &
           + REAL(n, dp)*h_inv(3, 2)
      length_z &
         = REAL(l, dp)*h_inv(1, 3) &
           + REAL(m, dp)*h_inv(2, 3) &
           + REAL(n, dp)*h_inv(3, 3)

      ! enforce strict zero-ness in this case (compiler optimization)
      IF (l == 0 .AND. m == 0 .AND. n == 0) THEN
         length_x = 0
         length_y = 0
         length_z = 0
      END IF

      length_x = length_x*twopi
      length_y = length_y*twopi
      length_z = length_z*twopi

      length = length_x**2 + length_y**2 + length_z**2

   END SUBROUTINE pw_vec_length

! **************************************************************************************************
!> \brief Count total number of g vectors
!> \param h_inv ...
!> \param pw_grid ...
!> \param mp_comm ...
!> \param cutoff ...
!> \param yz_mask ...
!> \par History
!>      JGH (22-12-2000) : Adapted for parallel use
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_count(h_inv, pw_grid, mp_comm, cutoff, yz_mask)

      REAL(KIND=dp), DIMENSION(3, 3)                     :: h_inv
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CLASS(mp_comm_type), INTENT(IN) :: mp_comm
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: yz_mask

      INTEGER                                            :: l, m, mm, n, n_upperlimit, nlim(2), nn
      INTEGER(KIND=int_8)                                :: gpt
      REAL(KIND=dp)                                      :: length, length_x, length_y, length_z

      ASSOCIATE (bounds => pw_grid%bounds)

         IF (pw_grid%grid_span == HALFSPACE) THEN
            n_upperlimit = 0
         ELSE IF (pw_grid%grid_span == FULLSPACE) THEN
            n_upperlimit = bounds(2, 3)
         ELSE
            CPABORT("No type set for the grid")
         END IF

         ! finds valid g-points within grid
         gpt = 0
         IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
            nlim(1) = bounds(1, 3)
            nlim(2) = n_upperlimit
         ELSE IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
            n = n_upperlimit - bounds(1, 3) + 1
            nlim = get_limit(n, mp_comm%num_pe, mp_comm%mepos)
            nlim = nlim + bounds(1, 3) - 1
         ELSE
            CPABORT("para % mode not specified")
         END IF

         yz_mask = 0
         DO n = nlim(1), nlim(2)
            nn = n - bounds(1, 3) + 1
            DO m = bounds(1, 2), bounds(2, 2)
               mm = m - bounds(1, 2) + 1
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (pw_grid%grid_span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) THEN
                     gpt = gpt + 1
                     yz_mask(mm, nn) = yz_mask(mm, nn) + 1
                  END IF

               END DO
            END DO
         END DO
      END ASSOCIATE

      ! number of g-vectors for grid
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         CALL mp_comm%sum(gpt)
         CALL mp_comm%sum(yz_mask)
      END IF
      pw_grid%ngpts_cut = gpt

   END SUBROUTINE pw_grid_count

! **************************************************************************************************
!> \brief Setup maps from 1d to 3d space
!> \param h_inv ...
!> \param pw_grid ...
!> \param cutoff ...
!> \par History
!>      JGH (29-12-2000) : Adapted for parallel use
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_assign(h_inv, pw_grid, cutoff)

      REAL(KIND=dp), DIMENSION(3, 3)                     :: h_inv
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      REAL(KIND=dp), INTENT(IN)                          :: cutoff

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_assign'

      INTEGER                                            :: gpt, handle, i, ip, l, lby, lbz, ll, m, &
                                                            mm, n, n_upperlimit, nn
      INTEGER(KIND=int_8)                                :: gpt_global
      INTEGER, DIMENSION(2, 3)                           :: bol, bounds
      REAL(KIND=dp)                                      :: length, length_x, length_y, length_z

      CALL timeset(routineN, handle)

      bounds = pw_grid%bounds
      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)

      IF (pw_grid%grid_span == HALFSPACE) THEN
         n_upperlimit = 0
      ELSE IF (pw_grid%grid_span == FULLSPACE) THEN
         n_upperlimit = bounds(2, 3)
      ELSE
         CPABORT("No type set for the grid")
      END IF

      ! finds valid g-points within grid
      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
         gpt = 0
         DO n = bounds(1, 3), n_upperlimit
            DO m = bounds(1, 2), bounds(2, 2)
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (pw_grid%grid_span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) THEN
                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z
                  END IF

               END DO
            END DO
         END DO

      ELSE

         IF (pw_grid%para%ray_distribution) THEN

            gpt = 0
            ip = pw_grid%para%group%mepos
            DO i = 1, pw_grid%para%nyzray(ip)
               n = pw_grid%para%yzp(2, i, ip) + lbz - 1
               m = pw_grid%para%yzp(1, i, ip) + lby - 1
               IF (n > n_upperlimit) CYCLE
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (pw_grid%grid_span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) THEN
                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z
                  END IF

               END DO
            END DO

         ELSE

            bol = pw_grid%para%bo(:, :, pw_grid%para%group%mepos, 3)
            gpt = 0
            DO n = bounds(1, 3), bounds(2, 3)
               IF (n < 0) THEN
                  nn = n + pw_grid%npts(3) + 1
               ELSE
                  nn = n + 1
               END IF
               IF (nn < bol(1, 3) .OR. nn > bol(2, 3)) CYCLE
               DO m = bounds(1, 2), bounds(2, 2)
                  IF (m < 0) THEN
                     mm = m + pw_grid%npts(2) + 1
                  ELSE
                     mm = m + 1
                  END IF
                  IF (mm < bol(1, 2) .OR. mm > bol(2, 2)) CYCLE
                  DO l = bounds(1, 1), bounds(2, 1)
                     IF (l < 0) THEN
                        ll = l + pw_grid%npts(1) + 1
                     ELSE
                        ll = l + 1
                     END IF
                     IF (ll < bol(1, 1) .OR. ll > bol(2, 1)) CYCLE

                     CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z

                  END DO
               END DO
            END DO

         END IF

      END IF

      ! Check the number of g-vectors for grid
      CPASSERT(pw_grid%ngpts_cut_local == gpt)
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         gpt_global = gpt
         CALL pw_grid%para%group%sum(gpt_global)
         CPASSERT(pw_grid%ngpts_cut == gpt_global)
      END IF

      pw_grid%have_g0 = .FALSE.
      pw_grid%first_gne0 = 1
      DO gpt = 1, pw_grid%ngpts_cut_local
         IF (ALL(pw_grid%g_hat(:, gpt) == 0)) THEN
            pw_grid%have_g0 = .TRUE.
            pw_grid%first_gne0 = 2
            EXIT
         END IF
      END DO

      CALL pw_grid_set_maps(pw_grid%grid_span, pw_grid%g_hat, &
                            pw_grid%mapl, pw_grid%mapm, pw_grid%mapn, pw_grid%npts)

      CALL timestop(handle)

   END SUBROUTINE pw_grid_assign

! **************************************************************************************************
!> \brief Setup maps from 1d to 3d space
!> \param grid_span ...
!> \param g_hat ...
!> \param mapl ...
!> \param mapm ...
!> \param mapn ...
!> \param npts ...
!> \par History
!>      JGH (21-12-2000) : Size of g_hat locally determined
!> \author apsi
!>      Christopher Mundy
!> \note
!>      Maps are to full 3D space (not distributed)
! **************************************************************************************************
   SUBROUTINE pw_grid_set_maps(grid_span, g_hat, mapl, mapm, mapn, npts)

      INTEGER, INTENT(IN)                                :: grid_span
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: g_hat
      TYPE(map_pn), INTENT(INOUT)                        :: mapl, mapm, mapn
      INTEGER, DIMENSION(:), INTENT(IN)                  :: npts

      INTEGER                                            :: gpt, l, m, n, ng

      ng = SIZE(g_hat, 2)

      DO gpt = 1, ng
         l = g_hat(1, gpt)
         m = g_hat(2, gpt)
         n = g_hat(3, gpt)
         IF (l < 0) THEN
            mapl%pos(l) = l + npts(1)
         ELSE
            mapl%pos(l) = l
         END IF
         IF (m < 0) THEN
            mapm%pos(m) = m + npts(2)
         ELSE
            mapm%pos(m) = m
         END IF
         IF (n < 0) THEN
            mapn%pos(n) = n + npts(3)
         ELSE
            mapn%pos(n) = n
         END IF

         ! Generating the maps to the full 3-d space

         IF (grid_span == HALFSPACE) THEN

            IF (l <= 0) THEN
               mapl%neg(l) = -l
            ELSE
               mapl%neg(l) = npts(1) - l
            END IF
            IF (m <= 0) THEN
               mapm%neg(m) = -m
            ELSE
               mapm%neg(m) = npts(2) - m
            END IF
            IF (n <= 0) THEN
               mapn%neg(n) = -n
            ELSE
               mapn%neg(n) = npts(3) - n
            END IF

         END IF

      END DO

   END SUBROUTINE pw_grid_set_maps

! **************************************************************************************************
!> \brief Allocate all (Pointer) Arrays in pw_grid
!> \param pw_grid ...
!> \param ng ...
!> \param bounds ...
!> \par History
!>      JGH (20-12-2000) : Added status variable
!>                         Bounds of arrays now from calling routine, this
!>                         makes it independent from parallel setup
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_allocate(pw_grid, ng, bounds)

      ! Argument
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, INTENT(IN)                                :: ng
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: bounds

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_allocate'

      INTEGER                                            :: handle, nmaps

      CALL timeset(routineN, handle)

      ALLOCATE (pw_grid%g(3, ng))

      nmaps = 1
      IF (pw_grid%grid_span == HALFSPACE) nmaps = 2
      ALLOCATE (pw_grid%g_hatmap(1, 1))

      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         ALLOCATE (pw_grid%grays(pw_grid%npts(1), &
                                 pw_grid%para%nyzray(pw_grid%para%group%mepos)))
      END IF

      ALLOCATE (pw_grid%mapl%pos(bounds(1, 1):bounds(2, 1)))
      ALLOCATE (pw_grid%mapl%neg(bounds(1, 1):bounds(2, 1)))
      ALLOCATE (pw_grid%mapm%pos(bounds(1, 2):bounds(2, 2)))
      ALLOCATE (pw_grid%mapm%neg(bounds(1, 2):bounds(2, 2)))
      ALLOCATE (pw_grid%mapn%pos(bounds(1, 3):bounds(2, 3)))
      ALLOCATE (pw_grid%mapn%neg(bounds(1, 3):bounds(2, 3)))

      CALL timestop(handle)

   END SUBROUTINE pw_grid_allocate

! **************************************************************************************************
!> \brief Recalculate the g-vectors after a change of the box
!> \param cell_hmat ...
!> \param pw_grid ...
!> \par History
!>      JGH (20-12-2000) : get local grid size from definition of g.
!>                         Assume that gsq is allocated.
!>                         Local routine, no information on distribution of
!>                         PW required.
!>      JGH (8-Mar-2001) : also update information on volume and grid spaceing
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_change(cell_hmat, pw_grid)
      ! Argument
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      INTEGER                                            :: gpt
      REAL(KIND=dp)                                      :: cell_deth, l, m, n
      REAL(KIND=dp), DIMENSION(3, 3)                     :: cell_h_inv
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: g

      cell_deth = ABS(det_3x3(cell_hmat))
      IF (cell_deth < 1.0E-10_dp) THEN
         CALL cp_abort(__LOCATION__, &
                       "An invalid set of cell vectors was specified. "// &
                       "The determinant det(h) is too small")
      END IF
      cell_h_inv = inv_3x3(cell_hmat)

      g => pw_grid%g

      CALL cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)

      DO gpt = 1, SIZE(g, 2)

         l = twopi*REAL(pw_grid%g_hat(1, gpt), KIND=dp)
         m = twopi*REAL(pw_grid%g_hat(2, gpt), KIND=dp)
         n = twopi*REAL(pw_grid%g_hat(3, gpt), KIND=dp)

         g(1, gpt) = l*cell_h_inv(1, 1) + m*cell_h_inv(2, 1) + n*cell_h_inv(3, 1)
         g(2, gpt) = l*cell_h_inv(1, 2) + m*cell_h_inv(2, 2) + n*cell_h_inv(3, 2)
         g(3, gpt) = l*cell_h_inv(1, 3) + m*cell_h_inv(2, 3) + n*cell_h_inv(3, 3)

         pw_grid%gsq(gpt) = g(1, gpt)*g(1, gpt) &
                            + g(2, gpt)*g(2, gpt) &
                            + g(3, gpt)*g(3, gpt)

      END DO

   END SUBROUTINE pw_grid_change

! **************************************************************************************************
!> \brief retains the given pw grid
!> \param pw_grid the pw grid to retain
!> \par History
!>      04.2003 created [fawzi]
!> \author fawzi
!> \note
!>      see doc/ReferenceCounting.html
! **************************************************************************************************
   SUBROUTINE pw_grid_retain(pw_grid)
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CPASSERT(pw_grid%ref_count > 0)
      pw_grid%ref_count = pw_grid%ref_count + 1
   END SUBROUTINE pw_grid_retain

! **************************************************************************************************
!> \brief releases the given pw grid
!> \param pw_grid the pw grid to release
!> \par History
!>      04.2003 created [fawzi]
!> \author fawzi
!> \note
!>      see doc/ReferenceCounting.html
! **************************************************************************************************
   SUBROUTINE pw_grid_release(pw_grid)

      TYPE(pw_grid_type), POINTER                        :: pw_grid

      IF (ASSOCIATED(pw_grid)) THEN
         CPASSERT(pw_grid%ref_count > 0)
         pw_grid%ref_count = pw_grid%ref_count - 1
         IF (pw_grid%ref_count == 0) THEN
            IF (ASSOCIATED(pw_grid%gidx)) THEN
               DEALLOCATE (pw_grid%gidx)
            END IF
            IF (ASSOCIATED(pw_grid%g)) THEN
               DEALLOCATE (pw_grid%g)
            END IF
            IF (ASSOCIATED(pw_grid%gsq)) THEN
               DEALLOCATE (pw_grid%gsq)
            END IF
            IF (ALLOCATED(pw_grid%g_hat)) THEN
               DEALLOCATE (pw_grid%g_hat)
            END IF
            IF (ASSOCIATED(pw_grid%g_hatmap)) THEN
               DEALLOCATE (pw_grid%g_hatmap)
            END IF
            IF (ASSOCIATED(pw_grid%grays)) THEN
               DEALLOCATE (pw_grid%grays)
            END IF
            IF (ALLOCATED(pw_grid%mapl%pos)) THEN
               DEALLOCATE (pw_grid%mapl%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapm%pos)) THEN
               DEALLOCATE (pw_grid%mapm%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapn%pos)) THEN
               DEALLOCATE (pw_grid%mapn%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapl%neg)) THEN
               DEALLOCATE (pw_grid%mapl%neg)
            END IF
            IF (ALLOCATED(pw_grid%mapm%neg)) THEN
               DEALLOCATE (pw_grid%mapm%neg)
            END IF
            IF (ALLOCATED(pw_grid%mapn%neg)) THEN
               DEALLOCATE (pw_grid%mapn%neg)
            END IF
            IF (ALLOCATED(pw_grid%para%bo)) THEN
               DEALLOCATE (pw_grid%para%bo)
            END IF
            IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
               IF (ALLOCATED(pw_grid%para%yzp)) THEN
                  DEALLOCATE (pw_grid%para%yzp)
               END IF
               IF (ALLOCATED(pw_grid%para%yzq)) THEN
                  DEALLOCATE (pw_grid%para%yzq)
               END IF
               IF (ALLOCATED(pw_grid%para%nyzray)) THEN
                  DEALLOCATE (pw_grid%para%nyzray)
               END IF
            END IF
            ! also release groups
            CALL pw_grid%para%group%free()
            IF (ALLOCATED(pw_grid%para%pos_of_x)) THEN
               DEALLOCATE (pw_grid%para%pos_of_x)
            END IF
            CALL fft_free_grid(pw_grid%fft_grid)

            IF (ASSOCIATED(pw_grid)) THEN
               DEALLOCATE (pw_grid)
            END IF
         END IF
      END IF
      NULLIFY (pw_grid)
   END SUBROUTINE pw_grid_release

END MODULE pw_grids
