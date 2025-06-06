!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Module to perform a counterpoise correction (BSSE)
!> \par History
!>      6.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
MODULE bsse
   USE atomic_kind_types,               ONLY: get_atomic_kind
   USE cell_types,                      ONLY: cell_type
   USE cp2k_info,                       ONLY: write_restart_header
   USE cp_external_control,             ONLY: external_control
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type
   USE cp_output_handling,              ONLY: cp_add_iter_level,&
                                              cp_iterate,&
                                              cp_print_key_finished_output,&
                                              cp_print_key_unit_nr,&
                                              cp_rm_iter_level
   USE cp_subsys_methods,               ONLY: create_small_subsys
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_release,&
                                              cp_subsys_type
   USE force_env_types,                 ONLY: force_env_get,&
                                              force_env_type
   USE global_types,                    ONLY: global_environment_type
   USE input_constants,                 ONLY: do_qs
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get,&
                                              section_vals_val_set,&
                                              section_vals_write
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE memory_utilities,                ONLY: reallocate
   USE message_passing,                 ONLY: mp_para_env_type
   USE particle_list_types,             ONLY: particle_list_type
   USE qs_energy,                       ONLY: qs_energies
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment,                  ONLY: qs_init
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_env_create,&
                                              qs_env_release,&
                                              qs_environment_type
   USE string_utilities,                ONLY: compress
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'bsse'

   PUBLIC :: do_bsse_calculation

CONTAINS

! **************************************************************************************************
!> \brief Perform an COUNTERPOISE CORRECTION (BSSE)
!>      For a 2-body system the correction scheme can be represented as:
!>
!>      E_{AB}^{2}        = E_{AB}(AB) - E_A(AB) - E_B(AB)  [BSSE-corrected interaction energy]
!>      E_{AB}^{2,uncorr} = E_{AB}(AB) - E_A(A)  - E_B(B)
!>      E_{AB}^{CP}       = E_{AB}(AB) + [ E_A(A) - E_A(AB) ] + [ E_B(B) - E_B(AB) ]
!>                                                          [CP-corrected total energy of AB]
!> \param force_env ...
!> \param globenv ...
!> \par History
!>      06.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE do_bsse_calculation(force_env, globenv)
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(global_environment_type), POINTER             :: globenv

      INTEGER                                            :: i, istart, k, num_of_conf, Num_of_Frag
      INTEGER, DIMENSION(:, :), POINTER                  :: conf
      LOGICAL                                            :: explicit, should_stop
      REAL(KIND=dp), DIMENSION(:), POINTER               :: Em
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(section_vals_type), POINTER                   :: bsse_section, fragment_energies_section, &
                                                            n_frags, root_section

      NULLIFY (bsse_section, n_frags, Em, conf)
      logger => cp_get_default_logger()
      root_section => force_env%root_section
      bsse_section => section_vals_get_subs_vals(force_env%force_env_section, "BSSE")
      n_frags => section_vals_get_subs_vals(bsse_section, "FRAGMENT")
      CALL section_vals_get(n_frags, n_repetition=Num_of_Frag)

      ! Number of configurations
      num_of_conf = 0
      DO k = 1, Num_of_frag
         num_of_conf = num_of_conf + FACT(Num_of_frag)/(FACT(k)*FACT(Num_of_frag - k))
      END DO
      ALLOCATE (conf(num_of_conf, Num_of_frag))
      ALLOCATE (Em(num_of_conf))
      CALL gen_Nbody_conf(Num_of_frag, conf)

      should_stop = .FALSE.
      istart = 0
      fragment_energies_section => section_vals_get_subs_vals(bsse_section, "FRAGMENT_ENERGIES")
      CALL section_vals_get(fragment_energies_section, explicit=explicit)
      IF (explicit) THEN
         CALL section_vals_val_get(fragment_energies_section, "_DEFAULT_KEYWORD_", n_rep_val=istart)
         DO i = 1, istart
            CALL section_vals_val_get(fragment_energies_section, "_DEFAULT_KEYWORD_", r_val=Em(i), &
                                      i_rep_val=i)
         END DO
      END IF
      ! Setup the iteration level for BSSE
      CALL cp_add_iter_level(logger%iter_info, "BSSE")
      CALL cp_iterate(logger%iter_info, last=.FALSE., iter_nr=istart)

      ! Evaluating the energy of the N-body cluster terms
      DO i = istart + 1, num_of_conf
         CALL cp_iterate(logger%iter_info, last=(i == num_of_conf), iter_nr=i)
         CALL eval_bsse_energy(conf(i, :), Em(i), force_env, n_frags, &
                               root_section, globenv, should_stop)
         IF (should_stop) EXIT

         ! If no signal was received in the inner loop let's check also at this stage
         CALL external_control(should_stop, "BSSE", globenv=globenv)
         IF (should_stop) EXIT

         ! Dump Restart info only if the calculation of the energy of a configuration
         ! ended nicely..
         CALL section_vals_val_set(fragment_energies_section, "_DEFAULT_KEYWORD_", r_val=Em(i), &
                                   i_rep_val=i)
         CALL write_bsse_restart(bsse_section, root_section)
      END DO
      IF (.NOT. should_stop) CALL dump_bsse_results(conf, Em, num_of_frag, bsse_section)
      CALL cp_rm_iter_level(logger%iter_info, "BSSE")
      DEALLOCATE (Em)
      DEALLOCATE (conf)

   END SUBROUTINE do_bsse_calculation

! **************************************************************************************************
!> \brief Evaluate the N-body energy contribution to the BSSE evaluation
!> \param conf ...
!> \param Em ...
!> \param force_env ...
!> \param n_frags ...
!> \param root_section ...
!> \param globenv ...
!> \param should_stop ...
!> \par History
!>      07.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE eval_bsse_energy(conf, Em, force_env, n_frags, root_section, &
                               globenv, should_stop)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: conf
      REAL(KIND=dp), INTENT(OUT)                         :: Em
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(section_vals_type), POINTER                   :: n_frags, root_section
      TYPE(global_environment_type), POINTER             :: globenv
      LOGICAL, INTENT(OUT)                               :: should_stop

      INTEGER                                            :: i, j, k, Num_of_sub_conf, Num_of_sub_frag
      INTEGER, DIMENSION(:, :), POINTER                  :: conf_loc
      REAL(KIND=dp)                                      :: my_energy
      REAL(KIND=dp), DIMENSION(:), POINTER               :: Em_loc

      NULLIFY (conf_loc, Em_loc)
      should_stop = .FALSE.
      ! Count the number of subconfiguration to evaluate..
      Num_of_sub_frag = COUNT(conf == 1)
      Num_of_sub_conf = 0
      IF (Num_of_sub_frag == 1) THEN
         CALL eval_bsse_energy_low(force_env, conf, conf, n_frags, root_section, globenv, Em)
      ELSE
         my_energy = 0.0_dp
         DO k = 1, Num_of_sub_frag
            Num_of_sub_conf = Num_of_sub_conf + &
                              FACT(Num_of_sub_frag)/(FACT(k)*FACT(Num_of_sub_frag - k))
         END DO
         ALLOCATE (conf_loc(Num_of_sub_conf, Num_of_sub_frag))
         ALLOCATE (Em_loc(Num_of_sub_conf))
         Em_loc = 0.0_dp
         CALL gen_Nbody_conf(Num_of_sub_frag, conf_loc)
         CALL make_plan_conf(conf, conf_loc)
         DO i = 1, Num_of_sub_conf
            CALL eval_bsse_energy_low(force_env, conf, conf_loc(i, :), n_frags, &
                                      root_section, globenv, Em_loc(i))
            CALL external_control(should_stop, "BSSE", globenv=globenv)
            IF (should_stop) EXIT
         END DO
         ! Energy
         k = COUNT(conf == 1)
         DO i = 1, Num_of_sub_conf
            j = COUNT(conf_loc(i, :) == 1)
            my_energy = my_energy + (-1.0_dp)**(k + j)*Em_loc(i)
         END DO
         Em = my_energy
         DEALLOCATE (Em_loc)
         DEALLOCATE (conf_loc)
      END IF

   END SUBROUTINE eval_bsse_energy

! **************************************************************************************************
!> \brief Evaluate the N-body energy contribution to the BSSE evaluation
!> \param force_env ...
!> \param conf ...
!> \param conf_loc ...
!> \param n_frags ...
!> \param root_section ...
!> \param globenv ...
!> \param energy ...
!> \par History
!>      07.2005 created [tlaino]
!>      2014/09/17 made atom list to be read from repeated occurrence of LIST [LTong]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE eval_bsse_energy_low(force_env, conf, conf_loc, n_frags, &
                                   root_section, globenv, energy)
      TYPE(force_env_type), POINTER                      :: force_env
      INTEGER, DIMENSION(:), INTENT(IN)                  :: conf, conf_loc
      TYPE(section_vals_type), POINTER                   :: n_frags, root_section
      TYPE(global_environment_type), POINTER             :: globenv
      REAL(KIND=dp), INTENT(OUT)                         :: energy

      CHARACTER(LEN=default_string_length)               :: name
      CHARACTER(len=default_string_length), &
         DIMENSION(:), POINTER                           :: atom_type
      INTEGER                                            :: i, ir, isize, j, k, method_name_id, &
                                                            my_targ, n_rep, num_of_frag, old_size, &
                                                            present_charge, present_multpl
      INTEGER, DIMENSION(:), POINTER                     :: atom_index, atom_list, my_conf, tmplist
      TYPE(cell_type), POINTER                           :: cell
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_list_type), POINTER                  :: particles
      TYPE(section_vals_type), POINTER                   :: bsse_section, dft_section, &
                                                            force_env_section, subsys_section

      CALL section_vals_get(n_frags, n_repetition=num_of_frag)
      CPASSERT(SIZE(conf) == num_of_frag)
      NULLIFY (subsys, particles, para_env, cell, atom_index, atom_type, tmplist, &
               force_env_section)
      CALL force_env_get(force_env, force_env_section=force_env_section)
      CALL section_vals_val_get(force_env_section, "METHOD", i_val=method_name_id)
      bsse_section => section_vals_get_subs_vals(force_env_section, "BSSE")
      subsys_section => section_vals_get_subs_vals(force_env_section, "SUBSYS")
      dft_section => section_vals_get_subs_vals(force_env_section, "DFT")

      ALLOCATE (my_conf(SIZE(conf)))
      my_conf = conf
      CALL force_env_get(force_env=force_env, subsys=subsys, para_env=para_env, &
                         cell=cell)
      CALL cp_subsys_get(subsys, particles=particles)
      isize = 0
      ALLOCATE (atom_index(isize))
      DO i = 1, num_of_frag
         IF (conf(i) == 1) THEN
            !
            ! Get the list of atoms creating the present fragment
            !
            old_size = isize
            CALL section_vals_val_get(n_frags, "LIST", i_rep_section=i, n_rep_val=n_rep)
            IF (n_rep /= 0) THEN
               DO ir = 1, n_rep
                  CALL section_vals_val_get(n_frags, "LIST", i_rep_section=i, i_rep_val=ir, i_vals=tmplist)
                  CALL reallocate(atom_index, 1, isize + SIZE(tmplist))
                  atom_index(isize + 1:isize + SIZE(tmplist)) = tmplist
                  isize = SIZE(atom_index)
               END DO
            END IF
            my_conf(i) = isize - old_size
            CPASSERT(conf(i) /= 0)
         END IF
      END DO
      CALL conf_info_setup(present_charge, present_multpl, conf, conf_loc, bsse_section, &
                           dft_section)
      !
      ! Get names and modify the ghost ones
      !
      ALLOCATE (atom_type(isize))
      DO j = 1, isize
         my_targ = atom_index(j)
         DO k = 1, SIZE(particles%els)
            CALL get_atomic_kind(particles%els(k)%atomic_kind, atom_list=atom_list, name=name)
            IF (ANY(atom_list == my_targ)) EXIT
         END DO
         atom_type(j) = name
      END DO
      DO i = 1, SIZE(conf_loc)
         IF (my_conf(i) /= 0 .AND. conf_loc(i) == 0) THEN
            DO j = SUM(my_conf(1:i - 1)) + 1, SUM(my_conf(1:i))
               atom_type(j) = TRIM(atom_type(j))//"_ghost"
            END DO
         END IF
      END DO
      CALL dump_bsse_info(atom_index, atom_type, conf, conf_loc, bsse_section, &
                          present_charge, present_multpl)
      !
      ! Let's start setting up environments and calculations
      !
      energy = 0.0_dp
      IF (method_name_id == do_qs) THEN
         BLOCK
            TYPE(qs_environment_type), POINTER :: qs_env
            TYPE(qs_energy_type), POINTER                      :: qs_energy
            TYPE(cp_subsys_type), POINTER                      :: subsys_loc
            NULLIFY (subsys_loc)
            CALL create_small_subsys(subsys_loc, big_subsys=subsys, &
                                     small_para_env=para_env, small_cell=cell, sub_atom_index=atom_index, &
                                     sub_atom_kind_name=atom_type, para_env=para_env, &
                                     force_env_section=force_env_section, subsys_section=subsys_section)

            ALLOCATE (qs_env)
            CALL qs_env_create(qs_env, globenv)
            CALL qs_init(qs_env, para_env, root_section, globenv=globenv, cp_subsys=subsys_loc, &
                         force_env_section=force_env_section, subsys_section=subsys_section, &
                         use_motion_section=.FALSE.)
            CALL cp_subsys_release(subsys_loc)

            !
            ! Evaluate Energy
            !
            CALL qs_energies(qs_env)
            CALL get_qs_env(qs_env, energy=qs_energy)
            energy = qs_energy%total
            CALL qs_env_release(qs_env)
            DEALLOCATE (qs_env)
         END BLOCK
      ELSE
         CPABORT("")
      END IF
      DEALLOCATE (atom_index)
      DEALLOCATE (atom_type)
      DEALLOCATE (my_conf)

   END SUBROUTINE eval_bsse_energy_low

! **************************************************************************************************
!> \brief Dumps bsse information (configuration fragment)
!> \param atom_index ...
!> \param atom_type ...
!> \param conf ...
!> \param conf_loc ...
!> \param bsse_section ...
!> \param present_charge ...
!> \param present_multpl ...
!> \par History
!>      07.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE dump_bsse_info(atom_index, atom_type, conf, conf_loc, bsse_section, &
                             present_charge, present_multpl)
      INTEGER, DIMENSION(:), POINTER                     :: atom_index
      CHARACTER(len=default_string_length), &
         DIMENSION(:), POINTER                           :: atom_type
      INTEGER, DIMENSION(:), INTENT(IN)                  :: conf, conf_loc
      TYPE(section_vals_type), POINTER                   :: bsse_section
      INTEGER, INTENT(IN)                                :: present_charge, present_multpl

      INTEGER                                            :: i, istat, iw
      CHARACTER(len=20*SIZE(conf))                       :: conf_loc_s, conf_s
      TYPE(cp_logger_type), POINTER                      :: logger

      NULLIFY (logger)
      logger => cp_get_default_logger()
      iw = cp_print_key_unit_nr(logger, bsse_section, "PRINT%PROGRAM_RUN_INFO", &
                                extension=".log")
      IF (iw > 0) THEN
         WRITE (conf_s, fmt="(1000I0)", iostat=istat) conf; 
         IF (istat .NE. 0) conf_s = "exceeded"
         CALL compress(conf_s, full=.TRUE.)
         WRITE (conf_loc_s, fmt="(1000I0)", iostat=istat) conf_loc; 
         IF (istat .NE. 0) conf_loc_s = "exceeded"
         CALL compress(conf_loc_s, full=.TRUE.)

         WRITE (UNIT=iw, FMT="(/,T2,A)") REPEAT("-", 79)
         WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T5,A,T30,A,T55,A,T80,A)") &
            "-", "BSSE CALCULATION", "FRAGMENT CONF: "//TRIM(conf_s), "FRAGMENT SUBCONF: "//TRIM(conf_loc_s), "-"
         WRITE (UNIT=iw, FMT="(T2,A,T30,A,I6,T55,A,I6,T80,A)") "-", "CHARGE =", present_charge, "MULTIPLICITY =", &
            present_multpl, "-"
         WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T20,A,T60,A,T80,A)") "-", "ATOM INDEX", "ATOM NAME", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T20,A,T60,A,T80,A)") "-", "----------", "---------", "-"
         DO i = 1, SIZE(atom_index)
            WRITE (UNIT=iw, FMT="(T2,A,T20,I6,T61,A,T80,A)") "-", atom_index(i), TRIM(atom_type(i)), "-"
         END DO
         WRITE (UNIT=iw, FMT="(T2,A)") REPEAT("-", 79)

         CALL cp_print_key_finished_output(iw, logger, bsse_section, &
                                           "PRINT%PROGRAM_RUN_INFO")

      END IF
   END SUBROUTINE dump_bsse_info

! **************************************************************************************************
!> \brief Read modified parameters for configurations
!> \param present_charge ...
!> \param present_multpl ...
!> \param conf ...
!> \param conf_loc ...
!> \param bsse_section ...
!> \param dft_section ...
!> \par History
!>      09.2007 created [tlaino]
!> \author Teodoro Laino - University of Zurich
! **************************************************************************************************
   SUBROUTINE conf_info_setup(present_charge, present_multpl, conf, conf_loc, &
                              bsse_section, dft_section)
      INTEGER, INTENT(OUT)                               :: present_charge, present_multpl
      INTEGER, DIMENSION(:), INTENT(IN)                  :: conf, conf_loc
      TYPE(section_vals_type), POINTER                   :: bsse_section, dft_section

      INTEGER                                            :: i, nconf
      INTEGER, DIMENSION(:), POINTER                     :: glb_conf, sub_conf
      LOGICAL                                            :: explicit
      TYPE(section_vals_type), POINTER                   :: configurations

      present_charge = 0
      present_multpl = 0
      NULLIFY (configurations, glb_conf, sub_conf)
      ! Loop over all configurations to pick up the right one
      configurations => section_vals_get_subs_vals(bsse_section, "CONFIGURATION")
      CALL section_vals_get(configurations, explicit=explicit, n_repetition=nconf)
      IF (explicit) THEN
         DO i = 1, nconf
            CALL section_vals_val_get(configurations, "GLB_CONF", i_rep_section=i, i_vals=glb_conf)
            IF (SIZE(glb_conf) /= SIZE(conf)) &
               CALL cp_abort(__LOCATION__, &
                             "GLB_CONF requires a binary description of the configuration. Number of integer "// &
                             "different from the number of fragments defined!")
            CALL section_vals_val_get(configurations, "SUB_CONF", i_rep_section=i, i_vals=sub_conf)
            IF (SIZE(sub_conf) /= SIZE(conf)) &
               CALL cp_abort(__LOCATION__, &
                             "SUB_CONF requires a binary description of the configuration. Number of integer "// &
                             "different from the number of fragments defined!")
            IF (ALL(conf == glb_conf) .AND. ALL(conf_loc == sub_conf)) THEN
               CALL section_vals_val_get(configurations, "CHARGE", i_rep_section=i, &
                                         i_val=present_charge)
               CALL section_vals_val_get(configurations, "MULTIPLICITY", i_rep_section=i, &
                                         i_val=present_multpl)
            END IF
         END DO
      END IF
      ! Setup parameter for this configuration
      CALL section_vals_val_set(dft_section, "CHARGE", i_val=present_charge)
      CALL section_vals_val_set(dft_section, "MULTIPLICITY", i_val=present_multpl)
   END SUBROUTINE conf_info_setup

! **************************************************************************************************
!> \brief Dumps results
!> \param conf ...
!> \param Em ...
!> \param num_of_frag ...
!> \param bsse_section ...
!> \par History
!>      09.2007 created [tlaino]
!> \author Teodoro Laino - University of Zurich
! **************************************************************************************************
   SUBROUTINE dump_bsse_results(conf, Em, num_of_frag, bsse_section)
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: conf
      REAL(KIND=dp), DIMENSION(:), POINTER               :: Em
      INTEGER, INTENT(IN)                                :: num_of_frag
      TYPE(section_vals_type), POINTER                   :: bsse_section

      INTEGER                                            :: i, iw
      TYPE(cp_logger_type), POINTER                      :: logger

      NULLIFY (logger)
      logger => cp_get_default_logger()
      iw = cp_print_key_unit_nr(logger, bsse_section, "PRINT%PROGRAM_RUN_INFO", &
                                extension=".log")

      IF (iw > 0) THEN
         WRITE (UNIT=iw, FMT="(/,T2,A)") REPEAT("-", 79)
         WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T36,A,T80,A)") &
            "-", "BSSE RESULTS", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=iw, FMT="(T2,A,T20,A,F16.6,T80,A)") "-", "CP-corrected Total energy:", SUM(Em), "-"
         WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
         DO i = 1, SIZE(conf, 1)
            IF (i .GT. 1) THEN
               IF (SUM(conf(i - 1, :)) == 1 .AND. SUM(conf(i, :)) /= 1) THEN
                  WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
               END IF
            END IF
            WRITE (UNIT=iw, FMT="(T2,A,T24,I3,A,F16.6,T80,A)") "-", SUM(conf(i, :)), "-body contribution:", Em(i), "-"
         END DO
         WRITE (UNIT=iw, FMT="(T2,A,T20,A,F16.6,T80,A)") "-", "BSSE-free interaction energy:", SUM(Em(Num_of_frag + 1:)), "-"
         WRITE (UNIT=iw, FMT="(T2,A)") REPEAT("-", 79)
      END IF

      CALL cp_print_key_finished_output(iw, logger, bsse_section, &
                                        "PRINT%PROGRAM_RUN_INFO")

   END SUBROUTINE dump_bsse_results

! **************************************************************************************************
!> \brief generate the N-body configuration for the N-body BSSE evaluation
!> \param Num_of_frag ...
!> \param conf ...
!> \par History
!>      07.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE gen_Nbody_conf(Num_of_frag, conf)
      INTEGER, INTENT(IN)                                :: Num_of_frag
      INTEGER, DIMENSION(:, :), POINTER                  :: conf

      INTEGER                                            :: k, my_ind

      my_ind = 0
      !
      ! Set up the N-body configurations
      !
      conf = 0
      DO k = 1, Num_of_frag
         CALL build_Nbody_conf(1, Num_of_frag, conf, k, my_ind)
      END DO
   END SUBROUTINE gen_Nbody_conf

! **************************************************************************************************
!> \brief ...
!> \param ldown ...
!> \param lup ...
!> \param conf ...
!> \param k ...
!> \param my_ind ...
! **************************************************************************************************
   RECURSIVE SUBROUTINE build_Nbody_conf(ldown, lup, conf, k, my_ind)
      INTEGER, INTENT(IN)                                :: ldown, lup
      INTEGER, DIMENSION(:, :), POINTER                  :: conf
      INTEGER, INTENT(IN)                                :: k
      INTEGER, INTENT(INOUT)                             :: my_ind

      INTEGER                                            :: i, kloc, my_ind0

      kloc = k - 1
      my_ind0 = my_ind
      IF (kloc /= 0) THEN
         DO i = ldown, lup
            CALL build_Nbody_conf(i + 1, lup, conf, kloc, my_ind)
            conf(my_ind0 + 1:my_ind, i) = 1
            my_ind0 = my_ind
         END DO
      ELSE
         DO i = ldown, lup
            my_ind = my_ind + 1
            conf(my_ind, i) = 1
         END DO
      END IF
   END SUBROUTINE build_Nbody_conf

! **************************************************************************************************
!> \brief ...
!> \param num ...
!> \return ...
! **************************************************************************************************
   RECURSIVE FUNCTION FACT(num) RESULT(my_fact)
      INTEGER, INTENT(IN)                                :: num
      INTEGER                                            :: my_fact

      IF (num <= 1) THEN
         my_fact = 1
      ELSE
         my_fact = num*FACT(num - 1)
      END IF
   END FUNCTION FACT

! **************************************************************************************************
!> \brief ...
!> \param main_conf ...
!> \param conf ...
! **************************************************************************************************
   SUBROUTINE make_plan_conf(main_conf, conf)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: main_conf
      INTEGER, DIMENSION(:, :), POINTER                  :: conf

      INTEGER                                            :: i, ind
      INTEGER, DIMENSION(:, :), POINTER                  :: tmp_conf

      ALLOCATE (tmp_conf(SIZE(conf, 1), SIZE(main_conf)))
      tmp_conf = 0
      ind = 0
      DO i = 1, SIZE(main_conf)
         IF (main_conf(i) /= 0) THEN
            ind = ind + 1
            tmp_conf(:, i) = conf(:, ind)
         END IF
      END DO
      DEALLOCATE (conf)
      ALLOCATE (conf(SIZE(tmp_conf, 1), SIZE(tmp_conf, 2)))
      conf = tmp_conf
      DEALLOCATE (tmp_conf)

   END SUBROUTINE make_plan_conf

! **************************************************************************************************
!> \brief Writes restart for BSSE calculations
!> \param bsse_section ...
!> \param root_section ...
!> \par History
!>      01.2008 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE write_bsse_restart(bsse_section, root_section)

      TYPE(section_vals_type), POINTER                   :: bsse_section, root_section

      INTEGER                                            :: ires
      TYPE(cp_logger_type), POINTER                      :: logger

      logger => cp_get_default_logger()
      ires = cp_print_key_unit_nr(logger, bsse_section, "PRINT%RESTART", &
                                  extension=".restart", do_backup=.FALSE., file_position="REWIND")

      IF (ires > 0) THEN
         CALL write_restart_header(ires)
         CALL section_vals_write(root_section, unit_nr=ires, hide_root=.TRUE.)
      END IF

      CALL cp_print_key_finished_output(ires, logger, bsse_section, &
                                        "PRINT%RESTART")

   END SUBROUTINE write_bsse_restart

END MODULE bsse
