!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines for the real time propagation.
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

MODULE rt_propagation
   USE bibliography,                    ONLY: Andermatt2016,&
                                              cite_reference
   USE cp_control_types,                ONLY: dft_control_type,&
                                              rtp_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_copy,&
                                              dbcsr_p_type
   USE cp_external_control,             ONLY: external_control
   USE cp_fm_types,                     ONLY: cp_fm_set_all,&
                                              cp_fm_to_fm,&
                                              cp_fm_type
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type,&
                                              cp_to_string
   USE cp_output_handling,              ONLY: cp_add_iter_level,&
                                              cp_iterate,&
                                              cp_p_file,&
                                              cp_print_key_generate_filename,&
                                              cp_print_key_should_output,&
                                              cp_print_key_unit_nr,&
                                              cp_rm_iter_level
   USE efield_utils,                    ONLY: calculate_ecore_efield
   USE force_env_methods,               ONLY: force_env_calc_energy_force
   USE force_env_types,                 ONLY: force_env_get,&
                                              force_env_type
   USE global_types,                    ONLY: global_environment_type
   USE hfx_admm_utils,                  ONLY: hfx_admm_init
   USE input_constants,                 ONLY: real_time_propagation,&
                                              use_restart_wfn,&
                                              use_rt_restart,&
                                              use_scf_wfn
   USE input_cp2k_restarts,             ONLY: write_restart
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get,&
                                              section_vals_val_set
   USE kinds,                           ONLY: default_path_length,&
                                              dp
   USE machine,                         ONLY: m_walltime
   USE md_environment_types,            ONLY: md_environment_type
   USE pw_env_types,                    ONLY: pw_env_type
   USE qs_core_hamiltonian,             ONLY: qs_matrix_h_allocate_imag_from_real
   USE qs_energy_init,                  ONLY: qs_energies_init
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_external_potential,           ONLY: external_c_potential,&
                                              external_e_potential
   USE qs_ks_methods,                   ONLY: qs_ks_allocate_basics,&
                                              qs_ks_update_qs_env
   USE qs_ks_types,                     ONLY: qs_ks_did_change,&
                                              qs_ks_env_type,&
                                              set_ks_env
   USE qs_mo_io,                        ONLY: wfn_restart_file_name
   USE qs_mo_types,                     ONLY: get_mo_set,&
                                              init_mo_set,&
                                              mo_set_type
   USE qs_rho_methods,                  ONLY: allocate_rho_ao_imag_from_real
   USE qs_rho_types,                    ONLY: qs_rho_set,&
                                              qs_rho_type
   USE rt_delta_pulse,                  ONLY: apply_delta_pulse
   USE rt_hfx_utils,                    ONLY: rtp_hfx_rebuild
   USE rt_projection_mo_utils,          ONLY: init_mo_projection
   USE rt_propagation_methods,          ONLY: propagation_step
   USE rt_propagation_output,           ONLY: rt_prop_output
   USE rt_propagation_types,            ONLY: get_rtp,&
                                              rt_prop_create,&
                                              rt_prop_type,&
                                              rtp_create_SinvH_imag,&
                                              rtp_history_create
   USE rt_propagation_utils,            ONLY: calc_S_derivs,&
                                              calc_update_rho,&
                                              calc_update_rho_sparse,&
                                              get_restart_wfn,&
                                              warn_section_unused
   USE rt_propagation_velocity_gauge,   ONLY: velocity_gauge_ks_matrix
   USE rt_propagator_init,              ONLY: init_propagators,&
                                              rt_initialize_rho_from_mos
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'rt_propagation'

   PUBLIC :: rt_prop_setup

CONTAINS

! **************************************************************************************************
!> \brief creates rtp_type, gets the initial state, either by reading MO's
!>        from file or calling SCF run
!> \param force_env ...
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

   SUBROUTINE rt_prop_setup(force_env)
      TYPE(force_env_type), POINTER                      :: force_env

      INTEGER                                            :: aspc_order
      LOGICAL                                            :: magnetic, vel_reprs
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(global_environment_type), POINTER             :: globenv
      TYPE(qs_energy_type), POINTER                      :: energy
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(rt_prop_type), POINTER                        :: rtp
      TYPE(rtp_control_type), POINTER                    :: rtp_control
      TYPE(section_vals_type), POINTER                   :: hfx_sections, input, ls_scf_section, &
                                                            md_section, motion_section, &
                                                            print_moments_section, &
                                                            rtp_print_section

      NULLIFY (qs_env, rtp_control, dft_control)

      CALL cite_reference(Andermatt2016)

      CALL force_env_get(force_env=force_env, qs_env=qs_env, globenv=globenv)
      CALL get_qs_env(qs_env, dft_control=dft_control, energy=energy)
      rtp_control => dft_control%rtp_control

      ! Takes care that an initial wavefunction/density is available
      ! Can either be by performing an scf loop or reading a restart
      CALL rt_initial_guess(qs_env, force_env, rtp_control)

      ! Initializes the extrapolation
      NULLIFY (rtp)
      CALL get_qs_env(qs_env=qs_env, rtp=rtp, input=input)
      aspc_order = rtp_control%aspc_order
      CALL rtp_history_create(rtp, aspc_order)

      ! Reads the simulation parameters from the input
      motion_section => section_vals_get_subs_vals(force_env%root_section, "MOTION")
      md_section => section_vals_get_subs_vals(motion_section, "MD")
      hfx_sections => section_vals_get_subs_vals(force_env%root_section, "FORCE_EVAL%DFT%XC%HF")
      print_moments_section => section_vals_get_subs_vals(force_env%root_section, "FORCE_EVAL%DFT%PRINT%MOMENTS")
      CALL section_vals_val_get(md_section, "TIMESTEP", r_val=qs_env%rtp%dt)
      CALL section_vals_val_get(md_section, "STEP_START_VAL", i_val=qs_env%rtp%i_start)
      CALL section_vals_val_get(md_section, "STEPS", i_val=rtp%nsteps)
      CALL section_vals_val_get(md_section, "MAX_STEPS", i_val=rtp%max_steps)

      ls_scf_section => section_vals_get_subs_vals(input, "DFT%LS_SCF")
      CALL section_vals_val_get(ls_scf_section, "EPS_FILTER", r_val=rtp%filter_eps)
      IF (.NOT. qs_env%rtp%linear_scaling) rtp%filter_eps = 0.0_dp
      IF (rtp_control%acc_ref < 1) rtp_control%acc_ref = 1
      rtp%filter_eps_small = rtp%filter_eps/rtp_control%acc_ref
      CALL section_vals_val_get(ls_scf_section, "EPS_LANCZOS", r_val=rtp%lanzcos_threshold)
      CALL section_vals_val_get(ls_scf_section, "MAX_ITER_LANCZOS", i_val=rtp%lanzcos_max_iter)
      CALL section_vals_val_get(ls_scf_section, "SIGN_SQRT_ORDER", i_val=rtp%newton_schulz_order)
      CALL section_vals_get(hfx_sections, explicit=rtp%do_hfx)
      CALL section_vals_val_get(print_moments_section, "MAGNETIC", l_val=magnetic)
      CALL section_vals_val_get(print_moments_section, "VEL_REPRS", l_val=vel_reprs)

      rtp%track_imag_density = (magnetic) .OR. (vel_reprs) .OR. (rtp_control%velocity_gauge) &
                               .OR. (rtp%do_hfx) .OR. (.NOT. rtp_control%fixed_ions)
      rtp%propagate_complex_ks = rtp%do_hfx .OR. rtp_control%velocity_gauge

      ! Marek : In case some print sections that apply so far only to RTBSE are present,
      !         warn the user that the quantities will not be in fact printed out
      rtp_print_section => section_vals_get_subs_vals(force_env%root_section, &
                                                      "FORCE_EVAL%DFT%REAL_TIME_PROPAGATION%PRINT")
      CALL warn_section_unused(rtp_print_section, "POLARIZABILITY", &
                               "POLARIZABILITY printing not implemented for non-RTBSE code.")
      CALL warn_section_unused(rtp_print_section, "MOMENTS", &
                               "MOMENTS not implemented: Use DFT%PRINT%MOMENTS for moments printing with TDDFT.")
      CALL warn_section_unused(rtp_print_section, "MOMENTS_FT", &
                               "MOMENTS_FT printing not implemented for non-RTBSE code.")
      CALL warn_section_unused(rtp_print_section, "DENSITY_MATRIX", &
                               "DENSITY_MATRIX printing not implemented for non-RTBSE code.")

      CALL rt_init_complex_quantities(qs_env, imag_p=rtp%track_imag_density, &
                                      imag_ks=rtp%propagate_complex_ks, imag_h=rtp_control%velocity_gauge)

      ! Hmm, not really like to initialize with the structure of S but I reckon it is
      ! done everywhere like this
      IF (rtp%do_hfx) CALL rtp_hfx_rebuild(qs_env)

      ! Setup the MO projection environment if required
      IF (rtp_control%is_proj_mo) CALL init_mo_projection(qs_env, rtp_control)

      CALL init_propagation_run(qs_env)
      IF (.NOT. rtp_control%fixed_ions) THEN
         !derivativs of the overlap needed for EMD
         CALL calc_S_derivs(qs_env)
         ! a bit hidden, but computes SinvH and SinvB (calc_SinvH for CN,EM and ARNOLDI)
         ! make_etrs_exp in case of ETRS in combination with TAYLOR and PADE
      END IF
      CALL init_propagators(qs_env)
      IF (rtp_control%fixed_ions) THEN
         CALL run_propagation(qs_env, force_env, globenv)
      ELSE
         rtp_control%initial_step = .TRUE.
         CALL force_env_calc_energy_force(force_env, calc_force=.TRUE.)
         rtp_control%initial_step = .FALSE.
         rtp%energy_old = energy%total
      END IF

   END SUBROUTINE rt_prop_setup

! **************************************************************************************************
!> \brief calculates the matrices needed in the first step of EMD/RTP
!> \param qs_env ...
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

   SUBROUTINE init_propagation_run(qs_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env

      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      INTEGER                                            :: i, ispin, re
      INTEGER, DIMENSION(2)                              :: nelectron_spin
      TYPE(cp_fm_type), DIMENSION(:), POINTER            :: mos_new, mos_old
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks, rho_new, rho_old
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(mo_set_type), DIMENSION(:), POINTER           :: mos
      TYPE(rt_prop_type), POINTER                        :: rtp
      TYPE(rtp_control_type), POINTER                    :: rtp_control

      NULLIFY (dft_control, rtp, rtp_control)

      CALL cite_reference(Andermatt2016)

      CALL get_qs_env(qs_env, &
                      rtp=rtp, &
                      dft_control=dft_control)
      rtp_control => dft_control%rtp_control

      IF (rtp_control%initial_wfn == use_scf_wfn) THEN
         IF (rtp_control%apply_delta_pulse .OR. rtp_control%apply_delta_pulse_mag) THEN
            CALL apply_delta_pulse(qs_env, rtp, rtp_control)
         ELSE
            IF (.NOT. rtp%linear_scaling) THEN
               CALL get_rtp(rtp=rtp, mos_old=mos_old)
               CALL get_qs_env(qs_env, mos=mos)
               DO i = 1, SIZE(mos)
                  CALL cp_fm_to_fm(mos(i)%mo_coeff, mos_old(2*i - 1))
                  CALL cp_fm_set_all(mos_old(2*i), zero, zero)
               END DO
            END IF
         END IF
      END IF

      IF (.NOT. rtp%linear_scaling) THEN
         CALL get_rtp(rtp=rtp, mos_old=mos_old, mos_new=mos_new)
         DO i = 1, SIZE(mos_old)
            CALL cp_fm_to_fm(mos_old(i), mos_new(i))
         END DO
         CALL calc_update_rho(qs_env)
      ELSE
         IF (rtp_control%initial_wfn == use_scf_wfn) THEN
            CALL get_qs_env(qs_env, &
                            matrix_ks=matrix_ks, &
                            mos=mos, &
                            nelectron_spin=nelectron_spin)
            IF (ASSOCIATED(mos)) THEN
               !The wavefunction was minimized by an mo based algorith. P is therefore calculated from the mos
               IF (ASSOCIATED(rtp%mos)) THEN
                  IF (ASSOCIATED(rtp%mos%old)) THEN
                     ! Delta kick was applied and the results is in rtp%mos%old
                     CALL rt_initialize_rho_from_mos(rtp, mos, mos_old=rtp%mos%old)
                  ELSE
                     CALL rt_initialize_rho_from_mos(rtp, mos)
                  END IF
               ELSE
                  CALL rt_initialize_rho_from_mos(rtp, mos)
               END IF
            ELSE
               !The wavefunction was minimized using a linear scaling method. The density matrix is therefore taken from the ls_scf_env.
               CALL get_rtp(rtp=rtp, rho_old=rho_old, rho_new=rho_new)
               DO ispin = 1, SIZE(rho_old)/2
                  re = 2*ispin - 1
                  CALL dbcsr_copy(rho_old(re)%matrix, qs_env%ls_scf_env%matrix_p(ispin))
                  CALL dbcsr_copy(rho_new(re)%matrix, qs_env%ls_scf_env%matrix_p(ispin))
               END DO
            END IF
            CALL calc_update_rho_sparse(qs_env)
         END IF
      END IF
      ! Modify KS matrix to include the additional terms in the velocity gauge
      IF (rtp_control%velocity_gauge) THEN
         ! As matrix_h and matrix_h_im are not updated by qs_ks_update_qs_env()
         ! the non-gauge transformed non-local part has to be subtracted here
         CALL velocity_gauge_ks_matrix(qs_env, subtract_nl_term=.TRUE.)
      END IF
      CALL qs_ks_update_qs_env(qs_env, calculate_forces=.FALSE.)

   END SUBROUTINE init_propagation_run

! **************************************************************************************************
!> \brief performs the real RTP run, gets information from MD section
!>        uses MD as iteration level
!> \param qs_env ...
!> \param force_env ...
!> \param globenv ...
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

   SUBROUTINE run_propagation(qs_env, force_env, globenv)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(global_environment_type), POINTER             :: globenv

      CHARACTER(len=*), PARAMETER                        :: routineN = 'run_propagation'

      INTEGER                                            :: aspc_order, handle, i_iter, i_step, &
                                                            max_iter, max_steps, output_unit
      LOGICAL                                            :: should_stop
      REAL(Kind=dp)                                      :: eps_ener, time_iter_start, &
                                                            time_iter_stop, used_time
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(pw_env_type), POINTER                         :: pw_env
      TYPE(qs_energy_type), POINTER                      :: energy
      TYPE(rt_prop_type), POINTER                        :: rtp
      TYPE(rtp_control_type), POINTER                    :: rtp_control
      TYPE(section_vals_type), POINTER                   :: input, rtp_section

      should_stop = .FALSE.
      CALL timeset(routineN, handle)

      CALL cite_reference(Andermatt2016)

      NULLIFY (logger, dft_control, energy, rtp, rtp_control, input, rtp_section)
      logger => cp_get_default_logger()

      CALL get_qs_env(qs_env=qs_env, dft_control=dft_control, rtp=rtp, energy=energy, input=input)

      rtp_control => dft_control%rtp_control
      max_steps = MIN(rtp%nsteps, rtp%max_steps)
      max_iter = rtp_control%max_iter
      eps_ener = rtp_control%eps_ener

      aspc_order = rtp_control%aspc_order

      rtp%energy_old = energy%total
      time_iter_start = m_walltime()
      CALL cp_add_iter_level(logger%iter_info, "MD")
      CALL cp_iterate(logger%iter_info, iter_nr=0)
      IF (rtp%i_start >= max_steps) CALL cp_abort(__LOCATION__, &
                                                  "maximum step number smaller than initial step value")

      rtp_section => section_vals_get_subs_vals(input, "DFT%REAL_TIME_PROPAGATION")
      output_unit = cp_print_key_unit_nr(logger, rtp_section, "PRINT%PROGRAM_RUN_INFO", &
                                         extension=".scfLog")

      DO i_step = rtp%i_start + 1, max_steps
         IF (output_unit > 0) THEN
            WRITE (output_unit, FMT="(/,(T2,A,T40,I6))") &
               "Real time propagation step:", i_step
         END IF
         energy%efield_core = 0.0_dp
         qs_env%sim_time = REAL(i_step, dp)*rtp%dt
         CALL get_qs_env(qs_env, pw_env=pw_env)
         pw_env%poisson_env%parameters%dbc_params%time = qs_env%sim_time
         qs_env%sim_step = i_step
         rtp%istep = i_step - rtp%i_start
         CALL calculate_ecore_efield(qs_env, .FALSE.)
         IF (dft_control%apply_external_potential) THEN
            IF (.NOT. dft_control%expot_control%static) THEN
               dft_control%eval_external_potential = .TRUE.
            END IF
         END IF
         CALL external_c_potential(qs_env, calculate_forces=.FALSE.)
         CALL external_e_potential(qs_env)
         CALL cp_iterate(logger%iter_info, last=(i_step == max_steps), iter_nr=i_step)
         rtp%converged = .FALSE.
         DO i_iter = 1, max_iter
            IF (i_step == rtp%i_start + 1 .AND. i_iter == 2 .AND. rtp_control%hfx_redistribute) &
               CALL qs_ks_did_change(qs_env%ks_env, s_mstruct_changed=.TRUE.)
            rtp%iter = i_iter
            CALL propagation_step(qs_env, rtp, rtp_control)
            CALL qs_ks_update_qs_env(qs_env, calculate_forces=.FALSE.)
            rtp%energy_new = energy%total
            IF (rtp%converged) EXIT
            CALL rt_prop_output(qs_env, real_time_propagation, rtp%delta_iter)
         END DO
         IF (rtp%converged) THEN
            CALL external_control(should_stop, "MD", globenv=globenv)
            IF (should_stop) CALL cp_iterate(logger%iter_info, last=.TRUE., iter_nr=i_step)
            time_iter_stop = m_walltime()
            used_time = time_iter_stop - time_iter_start
            time_iter_start = time_iter_stop
            CALL rt_prop_output(qs_env, real_time_propagation, delta_iter=rtp%delta_iter, used_time=used_time)
            CALL rt_write_input_restart(force_env=force_env, qs_env=qs_env)
            IF (should_stop) EXIT
         ELSE
            EXIT
         END IF
      END DO
      CALL cp_rm_iter_level(logger%iter_info, "MD")

      IF (.NOT. rtp%converged) &
         CALL cp_abort(__LOCATION__, "propagation did not converge, "// &
                       "either increase MAX_ITER or use a smaller TIMESTEP")

      CALL timestop(handle)

   END SUBROUTINE run_propagation

! **************************************************************************************************
!> \brief overwrites some values in the input file such that the .restart
!>        file will contain the appropriate information
!> \param md_env ...
!> \param qs_env ...
!> \param force_env ...
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

   SUBROUTINE rt_write_input_restart(md_env, qs_env, force_env)
      TYPE(md_environment_type), OPTIONAL, POINTER       :: md_env
      TYPE(qs_environment_type), OPTIONAL, POINTER       :: qs_env
      TYPE(force_env_type), POINTER                      :: force_env

      CHARACTER(len=default_path_length)                 :: file_name
      REAL(KIND=dp), DIMENSION(:), POINTER               :: tmp_vals
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(section_vals_type), POINTER                   :: dft_section, efield_section, &
                                                            motion_section, print_key, &
                                                            root_section, rt_section

      CALL get_qs_env(qs_env=qs_env, dft_control=dft_control)
      root_section => force_env%root_section
      motion_section => section_vals_get_subs_vals(root_section, "MOTION")
      dft_section => section_vals_get_subs_vals(root_section, "FORCE_EVAL%DFT")
      rt_section => section_vals_get_subs_vals(root_section, "FORCE_EVAL%DFT%REAL_TIME_PROPAGATION")

      CALL section_vals_val_set(rt_section, "INITIAL_WFN", i_val=use_rt_restart)
      CALL section_vals_val_set(rt_section, "APPLY_DELTA_PULSE", l_val=.FALSE.)
      CALL section_vals_val_set(rt_section, "APPLY_DELTA_PULSE_MAG", l_val=.FALSE.)
      CALL section_vals_val_set(rt_section, "APPLY_WFN_MIX_INIT_RESTART", l_val=.FALSE.)

      logger => cp_get_default_logger()

      ! to continue propagating the TD wavefunction we need to read from the new .rtpwfn
      IF (BTEST(cp_print_key_should_output(logger%iter_info, &
                                           rt_section, "PRINT%RESTART"), cp_p_file)) THEN
         print_key => section_vals_get_subs_vals(rt_section, "PRINT%RESTART")
         file_name = cp_print_key_generate_filename(logger, print_key, &
                                                    extension=".rtpwfn", my_local=.FALSE.)
         CALL section_vals_val_set(dft_section, "WFN_RESTART_FILE_NAME", c_val=TRIM(file_name))
      END IF

      ! coming from RTP
      IF (.NOT. PRESENT(md_env)) THEN
         CALL section_vals_val_set(motion_section, "MD%STEP_START_VAL", i_val=force_env%qs_env%sim_step)
      END IF

      IF (dft_control%apply_vector_potential) THEN
         efield_section => section_vals_get_subs_vals(root_section, "FORCE_EVAL%DFT%EFIELD")
         NULLIFY (tmp_vals)
         ALLOCATE (tmp_vals(3))
         tmp_vals = dft_control%efield_fields(1)%efield%vec_pot_initial
         CALL section_vals_val_set(efield_section, "VEC_POT_INITIAL", &
                                   r_vals_ptr=tmp_vals, &
                                   i_rep_section=1)
      END IF

      CALL write_restart(md_env=md_env, root_section=root_section)

   END SUBROUTINE rt_write_input_restart

! **************************************************************************************************
!> \brief Creates the initial electronic states and allocates the necessary
!>        matrices
!> \param qs_env ...
!> \param force_env ...
!> \param rtp_control ...
!> \author Florian Schiffmann (02.09)
! **************************************************************************************************

   SUBROUTINE rt_initial_guess(qs_env, force_env, rtp_control)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(rtp_control_type), POINTER                    :: rtp_control

      INTEGER                                            :: homo, ispin
      LOGICAL                                            :: energy_consistency
      TYPE(cp_fm_type), POINTER                          :: mo_coeff
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s
      TYPE(dft_control_type), POINTER                    :: dft_control

      NULLIFY (matrix_s, dft_control)
      CALL get_qs_env(qs_env, dft_control=dft_control)
      CPASSERT(ASSOCIATED(qs_env))

      SELECT CASE (rtp_control%initial_wfn)
      CASE (use_scf_wfn)
         qs_env%sim_time = 0.0_dp
         qs_env%sim_step = 0
         energy_consistency = .TRUE.
         !in the linear scaling case we need a correct kohn-sham matrix, which we cannot get with consistent energies
         IF (rtp_control%linear_scaling) energy_consistency = .FALSE.
         CALL force_env_calc_energy_force(force_env, calc_force=.FALSE., &
                                          consistent_energies=energy_consistency)
         qs_env%run_rtp = .TRUE.
         ALLOCATE (qs_env%rtp)
         CALL get_qs_env(qs_env, matrix_s=matrix_s)
         IF (dft_control%do_admm) THEN
            CALL hfx_admm_init(qs_env)
            CALL rt_prop_create(qs_env%rtp, qs_env%mos, qs_env%mpools, dft_control, matrix_s(1)%matrix, &
                                rtp_control%linear_scaling, qs_env%admm_env%mos_aux_fit)
         ELSE
            CALL rt_prop_create(qs_env%rtp, qs_env%mos, qs_env%mpools, dft_control, matrix_s(1)%matrix, &
                                rtp_control%linear_scaling)
         END IF

      CASE (use_restart_wfn, use_rt_restart)
         CALL qs_energies_init(qs_env, .FALSE.)
         IF (.NOT. rtp_control%linear_scaling .OR. rtp_control%initial_wfn == use_restart_wfn) THEN
            DO ispin = 1, SIZE(qs_env%mos)
               CALL get_mo_set(qs_env%mos(ispin), mo_coeff=mo_coeff, homo=homo)
               IF (.NOT. ASSOCIATED(mo_coeff)) THEN
                  CALL init_mo_set(qs_env%mos(ispin), &
                                   qs_env%mpools%ao_mo_fm_pools(ispin)%pool, &
                                   name="qs_env%mo"//TRIM(ADJUSTL(cp_to_string(ispin))))
               END IF
            END DO
            IF (dft_control%do_admm) CALL hfx_admm_init(qs_env)
         END IF
         ALLOCATE (qs_env%rtp)
         CALL get_qs_env(qs_env, matrix_s=matrix_s)
         CALL rt_prop_create(qs_env%rtp, qs_env%mos, qs_env%mpools, dft_control, matrix_s(1)%matrix, &
                             rtp_control%linear_scaling, qs_env%admm_env%mos_aux_fit)
         CALL get_restart_wfn(qs_env)
         CPASSERT(ASSOCIATED(qs_env))

         qs_env%run_rtp = .TRUE.
      END SELECT

   END SUBROUTINE rt_initial_guess

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param imag_p ...
!> \param imag_ks ...
!> \param imag_h ...
! **************************************************************************************************
   SUBROUTINE rt_init_complex_quantities(qs_env, imag_p, imag_ks, imag_h)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      LOGICAL, INTENT(in)                                :: imag_p, imag_ks, imag_h

      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(rt_prop_type), POINTER                        :: rtp

      NULLIFY (ks_env, rho, dft_control)

      CALL get_qs_env(qs_env, &
                      dft_control=dft_control, &
                      ks_env=ks_env, &
                      rho=rho, &
                      rtp=rtp)

      ! rho
      CALL qs_rho_set(rho, complex_rho_ao=imag_p)
      IF (imag_p) CALL allocate_rho_ao_imag_from_real(rho, qs_env)

      ! ks
      CALL set_ks_env(ks_env, complex_ks=imag_ks)
      IF (imag_ks) THEN
         CALL qs_ks_allocate_basics(qs_env, is_complex=imag_ks)
         IF (.NOT. dft_control%rtp_control%fixed_ions) &
            CALL rtp_create_SinvH_imag(rtp, dft_control%nspins)
      END IF

      ! h
      IF (imag_h) CALL qs_matrix_h_allocate_imag_from_real(qs_env)

   END SUBROUTINE rt_init_complex_quantities

END MODULE rt_propagation
