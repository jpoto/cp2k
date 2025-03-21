!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief qs_environment methods that use many other modules
!> \par History
!>      09.2002 created [fawzi]
!>      - local atom distribution (25.06.2003,MK)
!> \author Fawzi Mohamed
! **************************************************************************************************
MODULE qs_environment_methods
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_types,                      ONLY: cell_type
   USE cp_blacs_env,                    ONLY: cp_blacs_env_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_distribution_type
   USE cp_dbcsr_operations,             ONLY: cp_dbcsr_dist2d_to_dist
   USE distribution_2d_types,           ONLY: distribution_2d_release,&
                                              distribution_2d_type
   USE distribution_methods,            ONLY: distribute_molecules_2d
   USE ewald_environment_types,         ONLY: ewald_environment_type
   USE ewald_pw_methods,                ONLY: ewald_pw_grid_update
   USE ewald_pw_types,                  ONLY: ewald_pw_type
   USE input_constants,                 ONLY: do_ppl_grid
   USE kinds,                           ONLY: dp
   USE message_passing,                 ONLY: mp_para_env_type
   USE molecule_kind_types,             ONLY: molecule_kind_type
   USE molecule_types,                  ONLY: molecule_type
   USE particle_types,                  ONLY: particle_type
   USE pw_env_methods,                  ONLY: pw_env_create,&
                                              pw_env_rebuild
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_release,&
                                              pw_env_type
   USE pw_pool_types,                   ONLY: pw_pool_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
   USE qs_charges_types,                ONLY: qs_charges_create,&
                                              qs_charges_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type,&
                                              set_qs_env
   USE qs_kind_types,                   ONLY: has_nlcc,&
                                              qs_kind_type
   USE qs_ks_types,                     ONLY: get_ks_env,&
                                              qs_ks_env_type,&
                                              set_ks_env
   USE qs_matrix_pools,                 ONLY: mpools_rebuild_fm_pools
   USE qs_outer_scf,                    ONLY: outer_loop_variables_count
   USE qs_rho0_ggrid,                   ONLY: rho0_s_grid_create
   USE qs_rho0_types,                   ONLY: rho0_mpole_type
   USE scf_control_types,               ONLY: scf_control_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_environment_methods'

   PUBLIC :: qs_env_rebuild_pw_env, &
             qs_env_setup, &
             qs_env_time_update
!***
CONTAINS

! **************************************************************************************************
!> \brief initializes various components of the qs_env, that need only
!>      atomic_kind_set, cell, dft_control, scf_control, c(i)%nmo,
!>      c(i)%nao, and particle_set to be initialized.
!>      The previous components of qs_env must be valid.
!>      Initializes pools, charges and pw_env.
!> \param qs_env the qs_env to set up
!> \par History
!>      10.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE qs_env_setup(qs_env)

      TYPE(qs_environment_type), POINTER                 :: qs_env

      CHARACTER(len=*), PARAMETER                        :: routineN = 'qs_env_setup'

      INTEGER                                            :: handle, nhistory, nvariables
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: gradient_history, outer_scf_history, &
                                                            variable_history
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(dbcsr_distribution_type), POINTER             :: dbcsr_dist
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(distribution_2d_type), POINTER                :: distribution_2d
      TYPE(molecule_kind_type), DIMENSION(:), POINTER    :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), POINTER         :: molecule_set
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_charges_type), POINTER                     :: qs_charges
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(scf_control_type), POINTER                    :: scf_control

      CALL timeset(routineN, handle)

      NULLIFY (qs_kind_set, atomic_kind_set, dft_control, scf_control, qs_charges, para_env, &
               distribution_2d, molecule_kind_set, molecule_set, particle_set, cell, &
               ks_env, blacs_env)

      CALL get_qs_env(qs_env=qs_env, &
                      qs_kind_set=qs_kind_set, &
                      atomic_kind_set=atomic_kind_set, &
                      dft_control=dft_control, &
                      molecule_kind_set=molecule_kind_set, &
                      molecule_set=molecule_set, &
                      particle_set=particle_set, &
                      scf_control=scf_control, &
                      para_env=para_env, &
                      blacs_env=blacs_env, &
                      cell=cell, &
                      ks_env=ks_env)

      CPASSERT(ASSOCIATED(qs_kind_set))
      CPASSERT(ASSOCIATED(atomic_kind_set))
      CPASSERT(ASSOCIATED(dft_control))
      CPASSERT(ASSOCIATED(scf_control))
      ! allocate qs_charges
      ALLOCATE (qs_charges)
      CALL qs_charges_create(qs_charges, nspins=dft_control%nspins)
      CALL set_qs_env(qs_env, qs_charges=qs_charges)

      ! outer scf setup
      IF (scf_control%outer_scf%have_scf) THEN
         nvariables = outer_loop_variables_count(scf_control)
         nhistory = scf_control%outer_scf%extrapolation_order
         ALLOCATE (outer_scf_history(nvariables, nhistory))
         ALLOCATE (gradient_history(nvariables, 2))
         gradient_history = 0.0_dp
         ALLOCATE (variable_history(nvariables, 2))
         variable_history = 0.0_dp
         CALL set_qs_env(qs_env, outer_scf_history=outer_scf_history, &
                         gradient_history=gradient_history, &
                         variable_history=variable_history)
         CALL set_qs_env(qs_env, outer_scf_ihistory=0)
      END IF

      ! set up pw_env
      CALL qs_env_rebuild_pw_env(qs_env)

      ! rebuilds fm_pools

      ! XXXX should get rid of the mpools
      IF (ASSOCIATED(qs_env%mos)) THEN
         CALL mpools_rebuild_fm_pools(qs_env%mpools, mos=qs_env%mos, &
                                      blacs_env=blacs_env, para_env=para_env)
      END IF

      ! create 2d distribution

      CALL distribute_molecules_2d(cell=cell, &
                                   atomic_kind_set=atomic_kind_set, &
                                   qs_kind_set=qs_kind_set, &
                                   particle_set=particle_set, &
                                   molecule_kind_set=molecule_kind_set, &
                                   molecule_set=molecule_set, &
                                   distribution_2d=distribution_2d, &
                                   blacs_env=blacs_env, &
                                   force_env_section=qs_env%input)

      ! and use it to create the dbcsr_dist, which should be the sole user of distribution_2d by now.
      ALLOCATE (dbcsr_dist)
      CALL cp_dbcsr_dist2d_to_dist(distribution_2d, dbcsr_dist)
      CALL set_ks_env(ks_env, dbcsr_dist=dbcsr_dist)

      ! also keep distribution_2d in qs_env
      CALL set_ks_env(ks_env, distribution_2d=distribution_2d)
      CALL distribution_2d_release(distribution_2d)

      CALL timestop(handle)

   END SUBROUTINE qs_env_setup

! **************************************************************************************************
!> \brief rebuilds the pw_env in the given qs_env, allocating it if necessary
!> \param qs_env the qs_env whose pw_env has to be rebuilt
!> \par History
!>      10.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE qs_env_rebuild_pw_env(qs_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env

      CHARACTER(len=*), PARAMETER :: routineN = 'qs_env_rebuild_pw_env'

      INTEGER                                            :: handle
      LOGICAL                                            :: nlcc
      TYPE(cell_type), POINTER                           :: cell
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(ewald_environment_type), POINTER              :: ewald_env
      TYPE(ewald_pw_type), POINTER                       :: ewald_pw
      TYPE(pw_c1d_gs_type), POINTER                      :: rho_core, rho_nlcc_g
      TYPE(pw_env_type), POINTER                         :: new_pw_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_r3d_rs_type), POINTER                      :: embed_pot, external_vxc, rho_nlcc, &
                                                            spin_embed_pot, v_hartree_rspace, vee, &
                                                            vppl
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(rho0_mpole_type), POINTER                     :: rho0_mpole

      CALL timeset(routineN, handle)
      ! rebuild pw_env
      NULLIFY (dft_control, cell, ks_env, v_hartree_rspace, auxbas_pw_pool)
      NULLIFY (rho0_mpole)
      NULLIFY (ewald_env, ewald_pw, new_pw_env, external_vxc, rho_core, rho_nlcc, rho_nlcc_g, vee, vppl, &
               embed_pot, spin_embed_pot)

      CALL get_qs_env(qs_env, ks_env=ks_env, pw_env=new_pw_env)
      IF (.NOT. ASSOCIATED(new_pw_env)) THEN
         CALL pw_env_create(new_pw_env)
         CALL set_ks_env(ks_env, pw_env=new_pw_env)
         CALL pw_env_release(new_pw_env)
      END IF

      CALL get_qs_env(qs_env, pw_env=new_pw_env, dft_control=dft_control, &
                      cell=cell)

      IF (ANY(new_pw_env%cell_hmat /= cell%hmat)) THEN
         ! only rebuild if necessary
         new_pw_env%cell_hmat = cell%hmat
         CALL pw_env_rebuild(new_pw_env, qs_env=qs_env)

         ! reallocate rho_core
         CALL get_qs_env(qs_env, pw_env=new_pw_env, rho_core=rho_core)
         CPASSERT(ASSOCIATED(new_pw_env))
         IF (dft_control%qs_control%gapw) THEN
            IF (ASSOCIATED(rho_core)) THEN
               CALL rho_core%release()
               DEALLOCATE (rho_core)
            END IF
            IF (dft_control%qs_control%gapw_control%nopaw_as_gpw) THEN
               ALLOCATE (rho_core)
               CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
               CALL auxbas_pw_pool%create_pw(rho_core)
               CALL set_ks_env(ks_env, rho_core=rho_core)
            END IF
            CALL get_qs_env(qs_env=qs_env, rho0_mpole=rho0_mpole)
            CALL rho0_s_grid_create(new_pw_env, rho0_mpole)
         ELSE IF (dft_control%qs_control%semi_empirical) THEN
            IF (dft_control%qs_control%se_control%do_ewald .OR. &
                dft_control%qs_control%se_control%do_ewald_gks) THEN
               ! rebuild Ewald environment
               CALL get_qs_env(qs_env=qs_env, ewald_env=ewald_env, ewald_pw=ewald_pw)
               CALL ewald_pw_grid_update(ewald_pw, ewald_env, cell%hmat)
            END IF
         ELSE IF (dft_control%qs_control%dftb) THEN
            IF (dft_control%qs_control%dftb_control%do_ewald) THEN
               ! rebuild Ewald environment
               CALL get_qs_env(qs_env=qs_env, ewald_env=ewald_env, ewald_pw=ewald_pw)
               CALL ewald_pw_grid_update(ewald_pw, ewald_env, cell%hmat)
            END IF
         ELSE IF (dft_control%qs_control%xtb) THEN
            IF (dft_control%qs_control%xtb_control%do_ewald) THEN
               ! rebuild Ewald environment
               CALL get_qs_env(qs_env=qs_env, ewald_env=ewald_env, ewald_pw=ewald_pw)
               CALL ewald_pw_grid_update(ewald_pw, ewald_env, cell%hmat)
            END IF
         ELSE
            IF (ASSOCIATED(rho_core)) THEN
               CALL rho_core%release()
               DEALLOCATE (rho_core)
            END IF
            ALLOCATE (rho_core)
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(rho_core)
            CALL set_ks_env(ks_env, rho_core=rho_core)
         END IF

         ! reallocate vppl (realspace grid of local pseudopotential
         IF (dft_control%qs_control%do_ppl_method == do_ppl_grid) THEN
            NULLIFY (vppl)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, vppl=vppl)
            IF (ASSOCIATED(vppl)) THEN
               CALL vppl%release()
            ELSE
               ALLOCATE (vppl)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(vppl)
            CALL set_ks_env(ks_env, vppl=vppl)
         END IF

         ! reallocate rho_nlcc
         CALL get_qs_env(qs_env=qs_env, qs_kind_set=qs_kind_set)
         nlcc = has_nlcc(qs_kind_set)
         IF (nlcc) THEN
            ! the realspace version
            NULLIFY (rho_nlcc)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, rho_nlcc=rho_nlcc)
            IF (ASSOCIATED(rho_nlcc)) THEN
               CALL rho_nlcc%release()
            ELSE
               ALLOCATE (rho_nlcc)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(rho_nlcc)
            CALL set_ks_env(ks_env, rho_nlcc=rho_nlcc)
            ! the g-space version
            NULLIFY (rho_nlcc_g)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, rho_nlcc_g=rho_nlcc_g)
            IF (ASSOCIATED(rho_nlcc_g)) THEN
               CALL rho_nlcc_g%release()
            ELSE
               ALLOCATE (rho_nlcc_g)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(rho_nlcc_g)
            CALL set_ks_env(ks_env, rho_nlcc_g=rho_nlcc_g)
         END IF

         ! reallocate vee: external electrostatic potential
         IF (dft_control%apply_external_potential) THEN
            NULLIFY (vee)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, vee=vee)
            IF (ASSOCIATED(vee)) THEN
               CALL vee%release()
               DEALLOCATE (vee)
            END IF
            ALLOCATE (vee)
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(vee)
            CALL set_ks_env(ks_env, vee=vee)
            dft_control%eval_external_potential = .TRUE.
         END IF

         ! ZMP Reallocate external_vxc: external vxc potential
         IF (dft_control%apply_external_vxc) THEN
            NULLIFY (external_vxc)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, external_vxc=external_vxc)
            IF (ASSOCIATED(external_vxc)) THEN
               CALL external_vxc%release()
            ELSE
               ALLOCATE (external_vxc)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(external_vxc)
            CALL set_qs_env(qs_env, external_vxc=external_vxc)
            dft_control%read_external_vxc = .TRUE.
         END IF

         ! Embedding Reallocate: embed_pot
         IF (dft_control%apply_embed_pot) THEN
            NULLIFY (embed_pot)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, embed_pot=embed_pot)
            IF (ASSOCIATED(embed_pot)) THEN
               CALL embed_pot%release()
            ELSE
               ALLOCATE (embed_pot)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(embed_pot)
            CALL set_qs_env(qs_env, embed_pot=embed_pot)

            NULLIFY (spin_embed_pot)
            CALL get_qs_env(qs_env, pw_env=new_pw_env, spin_embed_pot=spin_embed_pot)
            IF (ASSOCIATED(spin_embed_pot)) THEN
               CALL spin_embed_pot%release()
               DEALLOCATE (spin_embed_pot)
            ELSE
               ALLOCATE (spin_embed_pot)
            END IF
            CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
            CALL auxbas_pw_pool%create_pw(spin_embed_pot)
            CALL set_qs_env(qs_env, spin_embed_pot=spin_embed_pot)
         END IF

         CALL get_ks_env(ks_env, v_hartree_rspace=v_hartree_rspace)
         IF (ASSOCIATED(v_hartree_rspace)) THEN
            CALL v_hartree_rspace%release()
            DEALLOCATE (v_hartree_rspace)
         END IF
         CALL get_qs_env(qs_env, pw_env=new_pw_env)
         CALL pw_env_get(new_pw_env, auxbas_pw_pool=auxbas_pw_pool)
         ALLOCATE (v_hartree_rspace)
         CALL auxbas_pw_pool%create_pw(v_hartree_rspace)
         CALL set_ks_env(ks_env, v_hartree_rspace=v_hartree_rspace)
      END IF

      !update the time in the poisson environment, to update time dependant constraints
      new_pw_env%poisson_env%parameters%dbc_params%time = qs_env%sim_time

      CALL timestop(handle)

   END SUBROUTINE qs_env_rebuild_pw_env

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param time ...
!> \param itimes ...
! **************************************************************************************************
   SUBROUTINE qs_env_time_update(qs_env, time, itimes)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      REAL(KIND=dp), INTENT(IN)                          :: time
      INTEGER, INTENT(IN)                                :: itimes

      TYPE(dft_control_type), POINTER                    :: dft_control

      qs_env%sim_time = time
      qs_env%sim_step = itimes

      CALL get_qs_env(qs_env=qs_env, dft_control=dft_control)

      IF (dft_control%apply_external_potential) THEN
         IF (.NOT. dft_control%expot_control%static) THEN
            dft_control%eval_external_potential = .TRUE.
         END IF
      END IF

   END SUBROUTINE qs_env_time_update

END MODULE qs_environment_methods
