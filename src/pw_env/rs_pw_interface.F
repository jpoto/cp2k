!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Transfers densities from PW to RS grids and potentials from PW to RS
!> \par History
!>      - Copied from qs_coolocate_Density and qs_integrate_potenntial
!> \author JGH (04.2014)
! **************************************************************************************************
MODULE rs_pw_interface
   USE cp_log_handling,                 ONLY: cp_to_string
   USE cp_spline_utils,                 ONLY: pw_interp,&
                                              pw_prolongate_s3,&
                                              pw_restrict_s3,&
                                              spline3_pbc_interp
   USE gaussian_gridlevels,             ONLY: gridlevel_info_type
   USE input_section_types,             ONLY: section_vals_val_get
   USE iso_c_binding,                   ONLY: C_ASSOCIATED,&
                                              C_LOC,&
                                              C_NULL_PTR,&
                                              C_SIZE_T
   USE kinds,                           ONLY: dp
   USE offload_api,                     ONLY: offload_buffer_set_device_resident,&
                                              offload_device_alloc,&
                                              offload_device_d2d,&
                                              offload_device_d2h,&
                                              offload_device_free,&
                                              offload_get_buffer_device_pointer
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_type
   USE pw_methods,                      ONLY: pw_axpy,&
                                              pw_copy,&
                                              pw_transfer,&
                                              pw_zero
   USE pw_offload,                      ONLY: pw_offload_fused_enabled
   USE pw_pool_types,                   ONLY: pw_pool_p_type,&
                                              pw_pools_create_pws,&
                                              pw_pools_give_back_pws
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
   USE realspace_grid_types,            ONLY: realspace_grid_desc_p_type,&
                                              realspace_grid_type,&
                                              transfer_pw2rs,&
                                              transfer_rs2pw
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'rs_pw_interface'
! *** Public subroutines ***

   PUBLIC :: density_rs2pw, &
             potential_pw2rs

CONTAINS

! **************************************************************************************************
!> \brief given partial densities on the realspace multigrids,
!>      computes the full density on the plane wave grids, both in real and
!>      gspace
!> \param pw_env ...
!> \param rs_rho ...
!> \param rho ...
!> \param rho_gspace ...
!> \param device ...
!> \note
!>      should contain all communication in the collocation of the density
!>      in the case of replicated grids
! **************************************************************************************************
   SUBROUTINE density_rs2pw(pw_env, rs_rho, rho, rho_gspace, device)

      TYPE(pw_env_type), INTENT(IN)                      :: pw_env
      TYPE(realspace_grid_type), DIMENSION(:), &
         INTENT(IN)                                      :: rs_rho
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: rho
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_gspace
      LOGICAL, INTENT(IN), OPTIONAL                      :: device

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'density_rs2pw'

      INTEGER                                            :: handle, igrid_level, interp_kind
      INTEGER(KIND=C_SIZE_T)                             :: nbytes
      LOGICAL                                            :: use_device
      TYPE(gridlevel_info_type), POINTER                 :: gridlevel_info
      TYPE(pw_c1d_gs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_gspace
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools
      TYPE(pw_r3d_rs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_rspace
      TYPE(realspace_grid_desc_p_type), DIMENSION(:), &
         POINTER                                         :: rs_descs

      CALL timeset(routineN, handle)
      use_device = .FALSE.
      IF (PRESENT(device)) use_device = device
      NULLIFY (gridlevel_info, rs_descs, pw_pools)
      CALL pw_env_get(pw_env, rs_descs=rs_descs, pw_pools=pw_pools)

      gridlevel_info => pw_env%gridlevel_info

      CALL section_vals_val_get(pw_env%interp_section, "KIND", i_val=interp_kind)

      CALL pw_pools_create_pws(pw_pools, mgrid_rspace)

      CALL pw_pools_create_pws(pw_pools, mgrid_gspace)

      IF (gridlevel_info%ngrid_levels == 1) THEN
         CALL transfer_rs2pw(rs_rho(1), rho, device=use_device)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
         ! Grid -> device producer: the collocate result already lives in the
         ! realspace grid's device buffer. Keep it on the device (device->device)
         ! so the following FFT can skip its host->device upload. Only valid when
         ! the rs->pw mapping is a plain copy (single grid, non-distributed,
         ! no border, equal sizes); otherwise the host path is used.
         IF (.NOT. use_device .AND. .NOT. rs_rho(1)%desc%distributed .AND. &
             .NOT. rs_rho(1)%desc%parallel .AND. &
             rs_rho(1)%desc%border == 0 .AND. SIZE(rho%array) == rs_rho(1)%ngpts_local .AND. &
             pw_offload_fused_enabled()) THEN
            nbytes = INT(SIZE(rho%array), KIND=C_SIZE_T)*8_C_SIZE_T
            IF (C_ASSOCIATED(offload_get_buffer_device_pointer(rs_rho(1)%buffer))) THEN
               IF (C_ASSOCIATED(rho%device_data) .AND. rho%device_capacity < nbytes) THEN
                  CALL offload_device_free(rho%device_data)
                  rho%device_data = C_NULL_PTR
                  rho%device_capacity = 0_C_SIZE_T
                  rho%device_valid = .FALSE.
               END IF
               IF (.NOT. C_ASSOCIATED(rho%device_data)) THEN
                  rho%device_data = offload_device_alloc(nbytes)
                  rho%device_capacity = nbytes
               END IF
               CALL offload_device_d2d(rho%device_data, offload_get_buffer_device_pointer(rs_rho(1)%buffer), nbytes)
               rho%device_valid = .TRUE.
            END IF
         END IF
#endif
         CALL pw_transfer(rho, rho_gspace)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
         ! If the FFT produced rho_gspace on the device, bring it back for host consumers.
         IF (C_ASSOCIATED(rho_gspace%device_data) .AND. .NOT. rho_gspace%host_valid) THEN
            CALL offload_device_d2h(C_LOC(rho_gspace%array(1)), rho_gspace%device_data, &
                                    INT(SIZE(rho_gspace%array), KIND=C_SIZE_T)*16_C_SIZE_T)
            rho_gspace%host_valid = .TRUE.
         END IF
#endif
         IF (rho%pw_grid%spherical) THEN ! rho_gspace = rho
            CALL pw_transfer(rho_gspace, rho)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
            IF (C_ASSOCIATED(rho%device_data) .AND. .NOT. rho%host_valid) THEN
               CALL offload_device_d2h(C_LOC(rho%array(1, 1, 1)), rho%device_data, &
                                       INT(SIZE(rho%array), KIND=C_SIZE_T)*8_C_SIZE_T)
               rho%host_valid = .TRUE.
            END IF
#endif
         END IF
      ELSE
         DO igrid_level = 1, gridlevel_info%ngrid_levels
            CALL transfer_rs2pw(rs_rho(igrid_level), &
                                mgrid_rspace(igrid_level))
         END DO

         ! we want both rho and rho_gspace, the latter for Hartree and co-workers.
         SELECT CASE (interp_kind)
         CASE (pw_interp)
            CALL pw_zero(rho_gspace)
            DO igrid_level = 1, gridlevel_info%ngrid_levels
               CALL pw_transfer(mgrid_rspace(igrid_level), mgrid_gspace(igrid_level))
               CALL pw_axpy(mgrid_gspace(igrid_level), rho_gspace)
            END DO
            CALL pw_transfer(rho_gspace, rho)
         CASE (spline3_pbc_interp)
            DO igrid_level = gridlevel_info%ngrid_levels, 2, -1
               CALL pw_prolongate_s3(mgrid_rspace(igrid_level), &
                                     mgrid_rspace(igrid_level - 1), pw_pools(igrid_level)%pool, &
                                     pw_env%interp_section)
            END DO
            CALL pw_copy(mgrid_rspace(1), rho)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
            ! Multi-grid: keep rho_gspace device-resident for the downstream
            ! Poisson / c1dr3d (M3/M5). Upload rho, run the FFT on the device,
            ! then sync rho_gspace back but leave its device mirror current.
            IF (pw_offload_fused_enabled()) CALL rho%ensure_device()
#endif
            CALL pw_transfer(rho, rho_gspace)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
            IF (pw_offload_fused_enabled()) CALL rho_gspace%ensure_host()
#endif
         CASE default
            CALL cp_abort(__LOCATION__, &
                          "interpolator "// &
                          cp_to_string(interp_kind))
         END SELECT
      END IF

      ! *** give back the pw multi-grids
      CALL pw_pools_give_back_pws(pw_pools, mgrid_gspace)
      CALL pw_pools_give_back_pws(pw_pools, mgrid_rspace)
      CALL timestop(handle)

   END SUBROUTINE density_rs2pw

! **************************************************************************************************
!> \brief transfers a potential from a pw_grid to a vector of
!>      realspace multigrids
!> \param rs_v OUTPUT: the potential on the realspace multigrids
!> \param v_rspace INPUT : the potential on a planewave grid in Rspace
!> \param pw_env ...
!> \param device ...
!> \par History
!>      09.2006 created [Joost VandeVondele]
!> \note
!>      extracted from integrate_v_rspace
!>      should contain all parallel communication of integrate_v_rspace in the
!>      case of replicated grids.
! **************************************************************************************************
   SUBROUTINE potential_pw2rs(rs_v, v_rspace, pw_env, device)

      TYPE(realspace_grid_type), DIMENSION(:), &
         INTENT(IN)                                      :: rs_v
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: v_rspace
      TYPE(pw_env_type), INTENT(IN)                      :: pw_env
      LOGICAL, INTENT(IN), OPTIONAL                      :: device

      CHARACTER(len=*), PARAMETER                        :: routineN = 'potential_pw2rs'

      INTEGER                                            :: auxbas_grid, handle, igrid_level, &
                                                            interp_kind
      LOGICAL                                            :: use_device
      REAL(KIND=dp)                                      :: scale
      TYPE(gridlevel_info_type), POINTER                 :: gridlevel_info
      TYPE(pw_c1d_gs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_gspace
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools
      TYPE(pw_r3d_rs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_rspace

      CALL timeset(routineN, handle)

      ! *** set up of the potential on the multigrids
      CALL pw_env_get(pw_env, pw_pools=pw_pools, gridlevel_info=gridlevel_info, &
                      auxbas_grid=auxbas_grid)

      ! Phase A: only the single-grid case can stay fully device-resident; the
      ! multi-grid spline path interpolates on the host.
      use_device = .FALSE.
      IF (PRESENT(device)) use_device = device
      use_device = use_device .AND. (gridlevel_info%ngrid_levels == 1)
      IF (use_device) THEN
         DO igrid_level = 1, gridlevel_info%ngrid_levels
            CALL offload_buffer_set_device_resident(rs_v(igrid_level)%buffer, .TRUE.)
         END DO
      END IF

      CALL pw_pools_create_pws(pw_pools, mgrid_rspace)

      ! use either realspace or fft techniques to get the potential on the rs multigrids
      CALL section_vals_val_get(pw_env%interp_section, "KIND", i_val=interp_kind)
      SELECT CASE (interp_kind)
      CASE (pw_interp)
         CALL pw_pools_create_pws(pw_pools, mgrid_gspace)
         CALL pw_transfer(v_rspace, mgrid_gspace(auxbas_grid))
         DO igrid_level = 1, gridlevel_info%ngrid_levels
            IF (igrid_level /= auxbas_grid) THEN
               CALL pw_copy(mgrid_gspace(auxbas_grid), mgrid_gspace(igrid_level))
               CALL pw_transfer(mgrid_gspace(igrid_level), mgrid_rspace(igrid_level))
            ELSE
               IF (mgrid_gspace(auxbas_grid)%pw_grid%spherical) THEN
                  CALL pw_transfer(mgrid_gspace(auxbas_grid), mgrid_rspace(auxbas_grid))
               ELSE ! fft forward + backward should be identical
                  CALL pw_copy(v_rspace, mgrid_rspace(auxbas_grid))
               END IF
            END IF
            ! *** Multiply by the grid volume element ratio ***
            IF (igrid_level /= auxbas_grid) THEN
               scale = mgrid_rspace(igrid_level)%pw_grid%dvol/ &
                       mgrid_rspace(auxbas_grid)%pw_grid%dvol
               mgrid_rspace(igrid_level)%array = &
                  scale*mgrid_rspace(igrid_level)%array
            END IF
         END DO
         CALL pw_pools_give_back_pws(pw_pools, mgrid_gspace)
      CASE (spline3_pbc_interp)
         CALL pw_copy(v_rspace, mgrid_rspace(1))
         DO igrid_level = 1, gridlevel_info%ngrid_levels - 1
            CALL pw_zero(mgrid_rspace(igrid_level + 1))
            CALL pw_restrict_s3(mgrid_rspace(igrid_level), &
                                mgrid_rspace(igrid_level + 1), pw_pools(igrid_level + 1)%pool, &
                                pw_env%interp_section)
            ! *** Multiply by the grid volume element ratio
            mgrid_rspace(igrid_level + 1)%array = &
               mgrid_rspace(igrid_level + 1)%array*8._dp
         END DO
      CASE default
         CALL cp_abort(__LOCATION__, &
                       "interpolation not supported "// &
                       cp_to_string(interp_kind))
      END SELECT

      DO igrid_level = 1, gridlevel_info%ngrid_levels
         CALL transfer_pw2rs(rs_v(igrid_level), &
                             mgrid_rspace(igrid_level), device=use_device)
      END DO
      ! *** give back the pw multi-grids
      CALL pw_pools_give_back_pws(pw_pools, mgrid_rspace)

      CALL timestop(handle)

   END SUBROUTINE potential_pw2rs

END MODULE rs_pw_interface
