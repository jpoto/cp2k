!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
MODULE optimize_input
   USE cell_types,                      ONLY: parse_cell_line
   USE cp2k_info,                       ONLY: write_restart_header
   USE cp_external_control,             ONLY: external_control
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_io_unit,&
                                              cp_logger_type
   USE cp_output_handling,              ONLY: cp_add_iter_level,&
                                              cp_iterate,&
                                              cp_print_key_finished_output,&
                                              cp_print_key_unit_nr,&
                                              cp_rm_iter_level
   USE cp_parser_methods,               ONLY: parser_read_line
   USE cp_parser_types,                 ONLY: cp_parser_type,&
                                              parser_create,&
                                              parser_release
   USE environment,                     ONLY: cp2k_get_walltime
   USE f77_interface,                   ONLY: calc_force,&
                                              create_force_env,&
                                              destroy_force_env,&
                                              set_cell
   USE input_constants,                 ONLY: opt_force_matching
   USE input_section_types,             ONLY: section_type,&
                                              section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get,&
                                              section_vals_val_set,&
                                              section_vals_write
   USE kinds,                           ONLY: default_path_length,&
                                              default_string_length,&
                                              dp
   USE machine,                         ONLY: m_flush,&
                                              m_walltime
   USE memory_utilities,                ONLY: reallocate
   USE message_passing,                 ONLY: mp_comm_type,&
                                              mp_para_env_type
   USE parallel_rng_types,              ONLY: UNIFORM,&
                                              rng_stream_type
   USE physcon,                         ONLY: bohr
   USE powell,                          ONLY: opt_state_type,&
                                              powell_optimize
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   PUBLIC ::  run_optimize_input

   TYPE fm_env_type
      CHARACTER(LEN=default_path_length) :: optimize_file_name = ""

      CHARACTER(LEN=default_path_length) :: ref_traj_file_name = ""
      CHARACTER(LEN=default_path_length) :: ref_force_file_name = ""
      CHARACTER(LEN=default_path_length) :: ref_cell_file_name = ""

      INTEGER :: group_size = -1

      REAL(KIND=dp) :: energy_weight = -1.0_dp
      REAL(KIND=dp) :: shift_mm = -1.0_dp
      REAL(KIND=dp) :: shift_qm = -1.0_dp
      LOGICAL       :: shift_average = .FALSE.
      INTEGER :: frame_start = -1, frame_stop = -1, frame_stride = -1, frame_count = -1
   END TYPE

   TYPE variable_type
      CHARACTER(LEN=default_string_length) :: label = ""
      REAL(KIND=dp)                        :: value = -1.0_dp
      LOGICAL                              :: fixed = .FALSE.
   END TYPE

   TYPE oi_env_type
      INTEGER :: method = -1
      INTEGER :: seed = -1
      CHARACTER(LEN=default_path_length) :: project_name = ""
      TYPE(fm_env_type) :: fm_env = fm_env_type()
      TYPE(variable_type), DIMENSION(:), ALLOCATABLE :: variables
      REAL(KIND=dp) :: rhobeg = -1.0_dp, rhoend = -1.0_dp
      INTEGER       :: maxfun = -1
      INTEGER       :: iter_start_val = -1
      REAL(KIND=dp) :: randomize_variables = -1.0_dp
      REAL(KIND=dp) :: start_time = -1.0_dp, target_time = -1.0_dp
   END TYPE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'optimize_input'

CONTAINS

! **************************************************************************************************
!> \brief main entry point for methods aimed at optimizing parameters in a CP2K input file
!> \param input_declaration ...
!> \param root_section ...
!> \param para_env ...
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE run_optimize_input(input_declaration, root_section, para_env)
      TYPE(section_type), POINTER                        :: input_declaration
      TYPE(section_vals_type), POINTER                   :: root_section
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CHARACTER(len=*), PARAMETER :: routineN = 'run_optimize_input'

      INTEGER                                            :: handle, i_var
      REAL(KIND=dp)                                      :: random_number, seed(3, 2)
      TYPE(oi_env_type)                                  :: oi_env
      TYPE(rng_stream_type), ALLOCATABLE                 :: rng_stream

      CALL timeset(routineN, handle)

      oi_env%start_time = m_walltime()

      CALL parse_input(oi_env, root_section)

      ! if we have been asked to randomize the variables, we do this.
      IF (oi_env%randomize_variables .NE. 0.0_dp) THEN
         seed = REAL(oi_env%seed, KIND=dp)
         rng_stream = rng_stream_type("run_optimize_input", distribution_type=UNIFORM, seed=seed)
         DO i_var = 1, SIZE(oi_env%variables, 1)
            IF (.NOT. oi_env%variables(i_var)%fixed) THEN
               ! change with a random percentage the variable
               random_number = rng_stream%next()
               oi_env%variables(i_var)%value = oi_env%variables(i_var)%value* &
                                               (1.0_dp + (2*random_number - 1.0_dp)*oi_env%randomize_variables/100.0_dp)
            END IF
         END DO
      END IF

      ! proceed to actual methods
      SELECT CASE (oi_env%method)
      CASE (opt_force_matching)
         CALL force_matching(oi_env, input_declaration, root_section, para_env)
      CASE DEFAULT
         CPABORT("")
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE run_optimize_input

! **************************************************************************************************
!> \brief optimizes parameters by force/energy matching results against reference values
!> \param oi_env ...
!> \param input_declaration ...
!> \param root_section ...
!> \param para_env ...
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE force_matching(oi_env, input_declaration, root_section, para_env)
      TYPE(oi_env_type)                                  :: oi_env
      TYPE(section_type), POINTER                        :: input_declaration
      TYPE(section_vals_type), POINTER                   :: root_section
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CHARACTER(len=*), PARAMETER                        :: routineN = 'force_matching'

      CHARACTER(len=default_path_length)                 :: input_path, output_path
      CHARACTER(len=default_string_length), &
         ALLOCATABLE, DIMENSION(:, :)                    :: initial_variables
      INTEGER :: color, energies_unit, handle, history_unit, i_atom, i_el, i_frame, i_free_var, &
         i_var, ierr, mepos_master, mepos_minion, n_atom, n_el, n_frames, n_free_var, n_groups, &
         n_var, new_env_id, num_pe_master, output_unit, restart_unit, state
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: free_var_index
      INTEGER, DIMENSION(:), POINTER                     :: group_distribution
      LOGICAL                                            :: should_stop
      REAL(KIND=dp)                                      :: e1, e2, e3, e4, e_pot, energy_weight, &
                                                            re, rf, shift_mm, shift_qm, t1, t2, &
                                                            t3, t4, t5
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: force, free_vars, pos
      REAL(KIND=dp), DIMENSION(:), POINTER               :: energy_traj, energy_traj_read, energy_var
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: cell_traj, cell_traj_read, force_traj, &
                                                            force_traj_read, force_var, pos_traj, &
                                                            pos_traj_read
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(mp_comm_type)                                 :: mpi_comm_master, mpi_comm_minion, &
                                                            mpi_comm_minion_primus
      TYPE(opt_state_type)                               :: ostate
      TYPE(section_vals_type), POINTER                   :: oi_section, variable_section

      CALL timeset(routineN, handle)

      logger => cp_get_default_logger()
      CALL cp_add_iter_level(logger%iter_info, "POWELL_OPT")
      output_unit = cp_logger_get_default_io_unit(logger)

      IF (output_unit > 0) THEN
         WRITE (output_unit, '(T2,A)') 'FORCE_MATCHING| good morning....'
      END IF

      ! do IO of ref traj / frc / cell
      NULLIFY (cell_traj)
      NULLIFY (cell_traj_read, force_traj_read, pos_traj_read, energy_traj_read)
      CALL read_reference_data(oi_env, para_env, force_traj_read, pos_traj_read, energy_traj_read, cell_traj_read)
      n_atom = SIZE(pos_traj_read, 2)

      ! adjust read data with respect to start/stop/stride
      IF (oi_env%fm_env%frame_stop < 0) oi_env%fm_env%frame_stop = SIZE(pos_traj_read, 3)

      IF (oi_env%fm_env%frame_count > 0) THEN
         oi_env%fm_env%frame_stride = (oi_env%fm_env%frame_stop - oi_env%fm_env%frame_start + 1 + &
                                       oi_env%fm_env%frame_count - 1)/(oi_env%fm_env%frame_count)
      END IF
      n_frames = (oi_env%fm_env%frame_stop - oi_env%fm_env%frame_start + oi_env%fm_env%frame_stride)/oi_env%fm_env%frame_stride

      ALLOCATE (force_traj(3, n_atom, n_frames), pos_traj(3, n_atom, n_frames), energy_traj(n_frames))
      IF (ASSOCIATED(cell_traj_read)) ALLOCATE (cell_traj(3, 3, n_frames))

      n_frames = 0
      DO i_frame = oi_env%fm_env%frame_start, oi_env%fm_env%frame_stop, oi_env%fm_env%frame_stride
         n_frames = n_frames + 1
         force_traj(:, :, n_frames) = force_traj_read(:, :, i_frame)
         pos_traj(:, :, n_frames) = pos_traj_read(:, :, i_frame)
         energy_traj(n_frames) = energy_traj_read(i_frame)
         IF (ASSOCIATED(cell_traj)) cell_traj(:, :, n_frames) = cell_traj_read(:, :, i_frame)
      END DO
      DEALLOCATE (force_traj_read, pos_traj_read, energy_traj_read)
      IF (ASSOCIATED(cell_traj_read)) DEALLOCATE (cell_traj_read)

      n_el = 3*n_atom
      ALLOCATE (pos(n_el), force(n_el))
      ALLOCATE (energy_var(n_frames), force_var(3, n_atom, n_frames))

      ! split the para_env in a set of sub_para_envs that will do the force_env communications
      mpi_comm_master = para_env
      num_pe_master = para_env%num_pe
      mepos_master = para_env%mepos
      ALLOCATE (group_distribution(0:num_pe_master - 1))
      IF (oi_env%fm_env%group_size > para_env%num_pe) oi_env%fm_env%group_size = para_env%num_pe

      CALL mpi_comm_minion%from_split(mpi_comm_master, n_groups, group_distribution, subgroup_min_size=oi_env%fm_env%group_size)
      mepos_minion = mpi_comm_minion%mepos
      color = 0
      IF (mepos_minion == 0) color = 1
      CALL mpi_comm_minion_primus%from_split(mpi_comm_master, color)

      ! assign initial variables
      n_var = SIZE(oi_env%variables, 1)
      ALLOCATE (initial_variables(2, n_var))
      n_free_var = 0
      DO i_var = 1, n_var
         initial_variables(1, i_var) = oi_env%variables(i_var)%label
         WRITE (initial_variables(2, i_var), *) oi_env%variables(i_var)%value
         IF (.NOT. oi_env%variables(i_var)%fixed) n_free_var = n_free_var + 1
      END DO
      ALLOCATE (free_vars(n_free_var), free_var_index(n_free_var))
      i_free_var = 0
      DO i_var = 1, n_var
         IF (.NOT. oi_env%variables(i_var)%fixed) THEN
            i_free_var = i_free_var + 1
            free_var_index(i_free_var) = i_var
            free_vars(i_free_var) = oi_env%variables(free_var_index(i_free_var))%value
         END IF
      END DO

      ! create input and output file names.
      input_path = oi_env%fm_env%optimize_file_name
      WRITE (output_path, '(A,I0,A)') TRIM(oi_env%project_name)//"-worker-", group_distribution(mepos_master), ".out"

      ! initialize the powell optimizer
      energy_weight = oi_env%fm_env%energy_weight
      shift_mm = oi_env%fm_env%shift_mm
      shift_qm = oi_env%fm_env%shift_qm

      IF (para_env%is_source()) THEN
         ostate%nf = 0
         ostate%nvar = n_free_var
         ostate%rhoend = oi_env%rhoend
         ostate%rhobeg = oi_env%rhobeg
         ostate%maxfun = oi_env%maxfun
         ostate%iprint = 1
         ostate%unit = output_unit
         ostate%state = 0
      END IF

      IF (output_unit > 0) THEN
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| number of atoms per frame ', n_atom
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| number of frames ', n_frames
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| number of parallel groups ', n_groups
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| number of variables ', n_var
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| number of free variables ', n_free_var
         WRITE (output_unit, '(T2,A,A)') 'FORCE_MATCHING| optimize file name ', TRIM(input_path)
         WRITE (output_unit, '(T2,A,T60,F20.12)') 'FORCE_MATCHING| accuracy', ostate%rhoend
         WRITE (output_unit, '(T2,A,T60,F20.12)') 'FORCE_MATCHING| step size', ostate%rhobeg
         WRITE (output_unit, '(T2,A,T60,I20)') 'FORCE_MATCHING| max function evaluation', ostate%maxfun
         WRITE (output_unit, '(T2,A,T60,L20)') 'FORCE_MATCHING| shift average', oi_env%fm_env%shift_average
         WRITE (output_unit, '(T2,A)') 'FORCE_MATCHING| initial values:'
         DO i_var = 1, n_var
            WRITE (output_unit, '(T2,A,1X,E28.16)') TRIM(oi_env%variables(i_var)%label), oi_env%variables(i_var)%value
         END DO
         WRITE (output_unit, '(T2,A)') 'FORCE_MATCHING| switching to POWELL optimization of the free parameters'
         WRITE (output_unit, '()')
         WRITE (output_unit, '(T2,A20,A20,A11,A11)') 'iteration number', 'function value', 'time', 'time Force'
         CALL m_flush(output_unit)
      END IF

      t1 = m_walltime()

      DO

         ! globalize the state
         IF (para_env%is_source()) state = ostate%state
         CALL para_env%bcast(state)

         ! if required get the energy of this set of params
         IF (state == 2) THEN

            CALL cp_iterate(logger%iter_info, last=.FALSE.)

            ! create a new force env, updating the free vars as needed
            DO i_free_var = 1, n_free_var
               WRITE (initial_variables(2, free_var_index(i_free_var)), *) free_vars(i_free_var)
               oi_env%variables(free_var_index(i_free_var))%value = free_vars(i_free_var)
            END DO

            ierr = 0
            CALL create_force_env(new_env_id=new_env_id, input_declaration=input_declaration, &
                                  input_path=input_path, output_path=output_path, &
                                  mpi_comm=mpi_comm_minion, initial_variables=initial_variables, ierr=ierr)

            ! set to zero initialy, for easier mp_summing
            energy_var = 0.0_dp
            force_var = 0.0_dp

            ! compute energies and forces for all frames, doing the work on a minion sub group based on round robin
            t5 = 0.0_dp
            DO i_frame = group_distribution(mepos_master) + 1, n_frames, n_groups

               ! set new cell if needed
               IF (ASSOCIATED(cell_traj)) THEN
                  CALL set_cell(env_id=new_env_id, new_cell=cell_traj(:, :, i_frame), ierr=ierr)
               END IF

               ! copy pos from ref
               i_el = 0
               DO i_atom = 1, n_atom
                  pos(i_el + 1) = pos_traj(1, i_atom, i_frame)
                  pos(i_el + 2) = pos_traj(2, i_atom, i_frame)
                  pos(i_el + 3) = pos_traj(3, i_atom, i_frame)
                  i_el = i_el + 3
               END DO

               ! evaluate energy/force with new pos
               t3 = m_walltime()
               CALL calc_force(env_id=new_env_id, pos=pos, n_el_pos=n_el, e_pot=e_pot, force=force, n_el_force=n_el, ierr=ierr)
               t4 = m_walltime()
               t5 = t5 + t4 - t3

               ! copy force and energy in place
               energy_var(i_frame) = e_pot
               i_el = 0
               DO i_atom = 1, n_atom
                  force_var(1, i_atom, i_frame) = force(i_el + 1)
                  force_var(2, i_atom, i_frame) = force(i_el + 2)
                  force_var(3, i_atom, i_frame) = force(i_el + 3)
                  i_el = i_el + 3
               END DO

            END DO

            ! clean up force env, get ready for the next round
            CALL destroy_force_env(env_id=new_env_id, ierr=ierr)

            ! get data everywhere on the master group, we could reduce the amount of data by reducing to partial RMSD first
            ! furthermore, we should only do this operation among the masters of the minion group
            IF (mepos_minion == 0) THEN
               CALL mpi_comm_minion_primus%sum(energy_var)
               CALL mpi_comm_minion_primus%sum(force_var)
            END IF

            ! now evaluate the target function to be minimized (only valid on mepos_minion==0)
            IF (para_env%is_source()) THEN
               rf = SQRT(SUM((force_var - force_traj)**2)/(REAL(n_frames, KIND=dp)*REAL(n_atom, KIND=dp)))
               IF (oi_env%fm_env%shift_average) THEN
                  shift_mm = SUM(energy_var)/n_frames
                  shift_qm = SUM(energy_traj)/n_frames
               END IF
               re = SQRT(SUM(((energy_var - shift_mm) - (energy_traj - shift_qm))**2)/n_frames)
               ostate%f = energy_weight*re + rf
               t2 = m_walltime()
               WRITE (output_unit, '(T2,I20,F20.12,F11.3,F11.3)') oi_env%iter_start_val + ostate%nf, ostate%f, t2 - t1, t5
               CALL m_flush(output_unit)
               t1 = m_walltime()
            END IF

            ! the history file with the trajectory of the parameters
            history_unit = cp_print_key_unit_nr(logger, root_section, "OPTIMIZE_INPUT%HISTORY", &
                                                extension=".dat")
            IF (history_unit > 0) THEN
               WRITE (UNIT=history_unit, FMT="(I20,F20.12,1000F20.12)") oi_env%iter_start_val + ostate%nf, ostate%f, free_vars
            END IF
            CALL cp_print_key_finished_output(history_unit, logger, root_section, "OPTIMIZE_INPUT%HISTORY")

            ! the energy profile for all frames
            energies_unit = cp_print_key_unit_nr(logger, root_section, "OPTIMIZE_INPUT%FORCE_MATCHING%COMPARE_ENERGIES", &
                                                 file_position="REWIND", extension=".dat")
            IF (energies_unit > 0) THEN
               WRITE (UNIT=energies_unit, FMT="(A20,A20,A20,A20)") "#frame", "ref", "fit", "diff"
               DO i_frame = 1, n_frames
                  e1 = energy_traj(i_frame) - shift_qm
                  e2 = energy_var(i_frame) - shift_mm
                  WRITE (UNIT=energies_unit, FMT="(I20,F20.12,F20.12,F20.12)") i_frame, e1, e2, e1 - e2
               END DO
            END IF
            CALL cp_print_key_finished_output(energies_unit, logger, root_section, "OPTIMIZE_INPUT%FORCE_MATCHING%COMPARE_ENERGIES")

            ! the force profile for all frames
            energies_unit = cp_print_key_unit_nr(logger, root_section, "OPTIMIZE_INPUT%FORCE_MATCHING%COMPARE_FORCES", &
                                                 file_position="REWIND", extension=".dat")
            IF (energies_unit > 0) THEN
               WRITE (UNIT=energies_unit, FMT="(A20,A20,A20,A20)") "#frame", "normalized diff", "diff", "ref", "ref sum"
               DO i_frame = 1, n_frames
                  e1 = SQRT(SUM((force_var(:, :, i_frame) - force_traj(:, :, i_frame))**2))
                  e2 = SQRT(SUM((force_traj(:, :, i_frame))**2))
                  e3 = SQRT(SUM(SUM(force_traj(:, :, i_frame), DIM=2)**2))
                  e4 = SQRT(SUM(SUM(force_var(:, :, i_frame), DIM=2)**2))
                  WRITE (UNIT=energies_unit, FMT="(I20,F20.12,F20.12,F20.12,2F20.12)") i_frame, e1/e2, e1, e2, e3, e4
               END DO
            END IF
            CALL cp_print_key_finished_output(energies_unit, logger, root_section, "OPTIMIZE_INPUT%FORCE_MATCHING%COMPARE_FORCES")

            ! a restart file with the current values of the parameters
            restart_unit = cp_print_key_unit_nr(logger, root_section, "OPTIMIZE_INPUT%RESTART", extension=".restart", &
                                                file_position="REWIND", do_backup=.TRUE.)
            IF (restart_unit > 0) THEN
               oi_section => section_vals_get_subs_vals(root_section, "OPTIMIZE_INPUT")
               CALL section_vals_val_set(oi_section, "ITER_START_VAL", i_val=oi_env%iter_start_val + ostate%nf)
               variable_section => section_vals_get_subs_vals(oi_section, "VARIABLE")
               DO i_free_var = 1, n_free_var
                  CALL section_vals_val_set(variable_section, "VALUE", i_rep_section=free_var_index(i_free_var), &
                                            r_val=free_vars(i_free_var))
               END DO
               CALL write_restart_header(restart_unit)
               CALL section_vals_write(root_section, unit_nr=restart_unit, hide_root=.TRUE.)
            END IF
            CALL cp_print_key_finished_output(restart_unit, logger, root_section, "OPTIMIZE_INPUT%RESTART")

         END IF

         IF (state == -1) EXIT

         CALL external_control(should_stop, "OPTIMIZE_INPUT", target_time=oi_env%target_time, start_time=oi_env%start_time)

         IF (should_stop) EXIT

         ! do a powell step if needed
         IF (para_env%is_source()) THEN
            CALL powell_optimize(ostate%nvar, free_vars, ostate)
         END IF
         CALL para_env%bcast(free_vars)

      END DO

      ! finally, get the best set of variables
      IF (para_env%is_source()) THEN
         ostate%state = 8
         CALL powell_optimize(ostate%nvar, free_vars, ostate)
      END IF
      CALL para_env%bcast(free_vars)
      DO i_free_var = 1, n_free_var
         WRITE (initial_variables(2, free_var_index(i_free_var)), *) free_vars(i_free_var)
         oi_env%variables(free_var_index(i_free_var))%value = free_vars(i_free_var)
      END DO
      IF (para_env%is_source()) THEN
         WRITE (output_unit, '(T2,A)') ''
         WRITE (output_unit, '(T2,A,T60,F20.12)') 'FORCE_MATCHING| optimal function value found so far:', ostate%fopt
         WRITE (output_unit, '(T2,A)') 'FORCE_MATCHING| optimal variables found so far:'
         DO i_var = 1, n_var
            WRITE (output_unit, '(T2,A,1X,E28.16)') TRIM(oi_env%variables(i_var)%label), oi_env%variables(i_var)%value
         END DO
      END IF

      CALL cp_rm_iter_level(logger%iter_info, "POWELL_OPT")

      ! deallocate for cleanup
      IF (ASSOCIATED(cell_traj)) DEALLOCATE (cell_traj)
      DEALLOCATE (pos, force, force_traj, pos_traj, force_var)
      DEALLOCATE (group_distribution, energy_traj, energy_var)
      CALL mpi_comm_minion%free()
      CALL mpi_comm_minion_primus%free()

      CALL timestop(handle)

   END SUBROUTINE force_matching

! **************************************************************************************************
!> \brief reads the reference data for force matching results, the format of the files needs to be the CP2K xyz trajectory format
!> \param oi_env ...
!> \param para_env ...
!> \param force_traj forces
!> \param pos_traj position
!> \param energy_traj energies, as extracted from the forces file
!> \param cell_traj cell parameters, as extracted from a CP2K cell file
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE read_reference_data(oi_env, para_env, force_traj, pos_traj, energy_traj, cell_traj)
      TYPE(oi_env_type)                                  :: oi_env
      TYPE(mp_para_env_type), POINTER                    :: para_env
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: force_traj, pos_traj
      REAL(KIND=dp), DIMENSION(:), POINTER               :: energy_traj
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: cell_traj

      CHARACTER(len=*), PARAMETER :: routineN = 'read_reference_data'

      CHARACTER(len=default_path_length)                 :: filename
      CHARACTER(len=default_string_length)               :: AA
      INTEGER                                            :: cell_itimes, handle, i, iframe, &
                                                            n_frames, n_frames_current, nread, &
                                                            trj_itimes
      LOGICAL                                            :: at_end, test_ok
      REAL(KIND=dp)                                      :: cell_time, trj_epot, trj_time, vec(3), &
                                                            vol
      TYPE(cp_parser_type)                               :: local_parser

      CALL timeset(routineN, handle)

      ! do IO of ref traj / frc / cell

      ! trajectory
      n_frames = 0
      n_frames_current = 0
      NULLIFY (pos_traj, energy_traj, force_traj)
      filename = oi_env%fm_env%ref_traj_file_name
      IF (filename .EQ. "") &
         CPABORT("The reference trajectory file name is empty")
      CALL parser_create(local_parser, filename, para_env=para_env)
      DO
         CALL parser_read_line(local_parser, 1, at_end=at_end)
         IF (at_end) EXIT
         READ (local_parser%input_line, FMT="(I8)") nread
         n_frames = n_frames + 1

         IF (n_frames > n_frames_current) THEN
            n_frames_current = 5*(n_frames_current + 10)/3
            CALL reallocate(pos_traj, 1, 3, 1, nread, 1, n_frames_current)
         END IF

         ! title line
         CALL parser_read_line(local_parser, 1)

         ! actual coordinates
         DO i = 1, nread
            CALL parser_read_line(local_parser, 1)
            READ (local_parser%input_line(1:LEN_TRIM(local_parser%input_line)), *) AA, vec
            pos_traj(:, i, n_frames) = vec*bohr
         END DO

      END DO
      CALL parser_release(local_parser)

      n_frames_current = n_frames
      CALL reallocate(energy_traj, 1, n_frames_current)
      CALL reallocate(force_traj, 1, 3, 1, nread, 1, n_frames_current)
      CALL reallocate(pos_traj, 1, 3, 1, nread, 1, n_frames_current)

      ! now force reference trajectory
      filename = oi_env%fm_env%ref_force_file_name
      IF (filename .EQ. "") &
         CPABORT("The reference force file name is empty")
      CALL parser_create(local_parser, filename, para_env=para_env)
      DO iframe = 1, n_frames
         CALL parser_read_line(local_parser, 1)
         READ (local_parser%input_line, FMT="(I8)") nread

         ! title line
         test_ok = .FALSE.
         CALL parser_read_line(local_parser, 1)
         READ (local_parser%input_line, FMT="(T6,I8,T23,F12.3,T41,F20.10)", ERR=999) trj_itimes, trj_time, trj_epot
         test_ok = .TRUE.
999      CONTINUE
         IF (.NOT. test_ok) THEN
            CPABORT("Could not parse the title line of the trajectory file")
         END IF
         energy_traj(iframe) = trj_epot

         ! actual forces, in a.u.
         DO i = 1, nread
            CALL parser_read_line(local_parser, 1)
            READ (local_parser%input_line(1:LEN_TRIM(local_parser%input_line)), *) AA, vec
            force_traj(:, i, iframe) = vec
         END DO
      END DO
      CALL parser_release(local_parser)

      ! and cell, which is optional
      NULLIFY (cell_traj)
      filename = oi_env%fm_env%ref_cell_file_name
      IF (filename .NE. "") THEN
         CALL parser_create(local_parser, filename, para_env=para_env)
         ALLOCATE (cell_traj(3, 3, n_frames))
         DO iframe = 1, n_frames
            CALL parser_read_line(local_parser, 1)
            CALL parse_cell_line(local_parser%input_line, cell_itimes, cell_time, cell_traj(:, :, iframe), vol)
         END DO
         CALL parser_release(local_parser)
      END IF

      CALL timestop(handle)

   END SUBROUTINE read_reference_data

! **************************************************************************************************
!> \brief parses the input section, and stores in the optimize input environment
!> \param oi_env optimize input environment
!> \param root_section ...
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE parse_input(oi_env, root_section)
      TYPE(oi_env_type)                                  :: oi_env
      TYPE(section_vals_type), POINTER                   :: root_section

      CHARACTER(len=*), PARAMETER                        :: routineN = 'parse_input'

      INTEGER                                            :: handle, ivar, n_var
      LOGICAL                                            :: explicit
      TYPE(section_vals_type), POINTER                   :: fm_section, oi_section, variable_section

      CALL timeset(routineN, handle)

      CALL section_vals_val_get(root_section, "GLOBAL%PROJECT", c_val=oi_env%project_name)
      CALL section_vals_val_get(root_section, "GLOBAL%SEED", i_val=oi_env%seed)
      CALL cp2k_get_walltime(section=root_section, keyword_name="GLOBAL%WALLTIME", &
                             walltime=oi_env%target_time)

      oi_section => section_vals_get_subs_vals(root_section, "OPTIMIZE_INPUT")
      variable_section => section_vals_get_subs_vals(oi_section, "VARIABLE")

      CALL section_vals_val_get(oi_section, "ACCURACY", r_val=oi_env%rhoend)
      CALL section_vals_val_get(oi_section, "STEP_SIZE", r_val=oi_env%rhobeg)
      CALL section_vals_val_get(oi_section, "MAX_FUN", i_val=oi_env%maxfun)
      CALL section_vals_val_get(oi_section, "ITER_START_VAL", i_val=oi_env%iter_start_val)
      CALL section_vals_val_get(oi_section, "RANDOMIZE_VARIABLES", r_val=oi_env%randomize_variables)

      CALL section_vals_get(variable_section, explicit=explicit, n_repetition=n_var)
      IF (explicit) THEN
         ALLOCATE (oi_env%variables(1:n_var))
         DO ivar = 1, n_var
            CALL section_vals_val_get(variable_section, "VALUE", i_rep_section=ivar, &
                                      r_val=oi_env%variables(ivar)%value)
            CALL section_vals_val_get(variable_section, "FIXED", i_rep_section=ivar, &
                                      l_val=oi_env%variables(ivar)%fixed)
            CALL section_vals_val_get(variable_section, "LABEL", i_rep_section=ivar, &
                                      c_val=oi_env%variables(ivar)%label)
         END DO
      END IF

      CALL section_vals_val_get(oi_section, "METHOD", i_val=oi_env%method)
      SELECT CASE (oi_env%method)
      CASE (opt_force_matching)
         fm_section => section_vals_get_subs_vals(oi_section, "FORCE_MATCHING")
         CALL section_vals_val_get(fm_section, "REF_TRAJ_FILE_NAME", c_val=oi_env%fm_env%ref_traj_file_name)
         CALL section_vals_val_get(fm_section, "REF_FORCE_FILE_NAME", c_val=oi_env%fm_env%ref_force_file_name)
         CALL section_vals_val_get(fm_section, "REF_CELL_FILE_NAME", c_val=oi_env%fm_env%ref_cell_file_name)
         CALL section_vals_val_get(fm_section, "OPTIMIZE_FILE_NAME", c_val=oi_env%fm_env%optimize_file_name)
         CALL section_vals_val_get(fm_section, "FRAME_START", i_val=oi_env%fm_env%frame_start)
         CALL section_vals_val_get(fm_section, "FRAME_STOP", i_val=oi_env%fm_env%frame_stop)
         CALL section_vals_val_get(fm_section, "FRAME_STRIDE", i_val=oi_env%fm_env%frame_stride)
         CALL section_vals_val_get(fm_section, "FRAME_COUNT", i_val=oi_env%fm_env%frame_count)

         CALL section_vals_val_get(fm_section, "GROUP_SIZE", i_val=oi_env%fm_env%group_size)

         CALL section_vals_val_get(fm_section, "ENERGY_WEIGHT", r_val=oi_env%fm_env%energy_weight)
         CALL section_vals_val_get(fm_section, "SHIFT_MM", r_val=oi_env%fm_env%shift_mm)
         CALL section_vals_val_get(fm_section, "SHIFT_QM", r_val=oi_env%fm_env%shift_qm)
         CALL section_vals_val_get(fm_section, "SHIFT_AVERAGE", l_val=oi_env%fm_env%shift_average)
      CASE DEFAULT
         CPABORT("")
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE parse_input

END MODULE optimize_input
