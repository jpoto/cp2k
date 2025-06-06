!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Setup of regions with different temperature
!> \par History
!>   - Added support for Langevin regions (2014/01/08, LT)
!>   - Added print subroutine for langevin regions (2014/02/04, LT)
!>   - Changed print_thermal_regions to print_thermal_regions_temperature
!>     (2014/02/04, LT)
!> \author MI
! **************************************************************************************************
MODULE thermal_region_utils

   USE bibliography,                    ONLY: Kantorovich2008,&
                                              Kantorovich2008a,&
                                              cite_reference
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type,&
                                              cp_to_string
   USE cp_output_handling,              ONLY: cp_p_file,&
                                              cp_print_key_finished_output,&
                                              cp_print_key_should_output,&
                                              cp_print_key_unit_nr
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_type
   USE force_env_types,                 ONLY: force_env_get,&
                                              force_env_type
   USE input_constants,                 ONLY: langevin_ensemble,&
                                              npt_f_ensemble,&
                                              npt_i_ensemble,&
                                              npt_ia_ensemble,&
                                              nvt_ensemble
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE memory_utilities,                ONLY: reallocate
   USE particle_list_types,             ONLY: particle_list_type
   USE physcon,                         ONLY: femtoseconds,&
                                              kelvin
   USE simpar_types,                    ONLY: simpar_type
   USE string_utilities,                ONLY: integer_to_string
   USE thermal_region_types,            ONLY: allocate_thermal_regions,&
                                              release_thermal_regions,&
                                              thermal_region_type,&
                                              thermal_regions_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: create_thermal_regions, &
             print_thermal_regions_temperature, &
             print_thermal_regions_langevin

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'thermal_region_utils'

CONTAINS

! **************************************************************************************************
!> \brief create thermal_regions
!> \param thermal_regions ...
!> \param md_section ...
!> \param simpar ...
!> \param force_env ...
!> \par History
!>   - Added support for Langevin regions (2014/01/08, LT)
!> \author
! **************************************************************************************************
   SUBROUTINE create_thermal_regions(thermal_regions, md_section, simpar, force_env)
      TYPE(thermal_regions_type), POINTER                :: thermal_regions
      TYPE(section_vals_type), POINTER                   :: md_section
      TYPE(simpar_type), POINTER                         :: simpar
      TYPE(force_env_type), POINTER                      :: force_env

      CHARACTER(LEN=default_string_length)               :: my_region
      INTEGER                                            :: i, il, ipart, ireg, nlist, nregions
      INTEGER, DIMENSION(:), POINTER                     :: tmplist
      LOGICAL                                            :: apply_thermostat, do_langevin, &
                                                            do_langevin_default, do_read_ngr, &
                                                            explicit
      REAL(KIND=dp)                                      :: temp, temp_tol
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(particle_list_type), POINTER                  :: particles
      TYPE(section_vals_type), POINTER                   :: region_sections, thermal_region_section
      TYPE(thermal_region_type), POINTER                 :: t_region

      NULLIFY (region_sections, t_region, thermal_region_section, particles, subsys, tmplist)
      ALLOCATE (thermal_regions)
      CALL allocate_thermal_regions(thermal_regions)
      thermal_region_section => section_vals_get_subs_vals(md_section, "THERMAL_REGION")
      CALL section_vals_get(thermal_region_section, explicit=explicit)
      IF (explicit) THEN
         apply_thermostat = (simpar%ensemble == nvt_ensemble) .OR. &
                            (simpar%ensemble == npt_f_ensemble) .OR. &
                            (simpar%ensemble == npt_ia_ensemble) .OR. &
                            (simpar%ensemble == npt_i_ensemble)
         IF (apply_thermostat) THEN
            CALL cp_warn(__LOCATION__, &
                         "With the chosen ensemble the temperature is "// &
                         "controlled by thermostats. The definition of different thermal "// &
                         "regions might result inconsistent with the presence of thermostats.")
         END IF
         IF (simpar%temp_tol > 0.0_dp) THEN
            CALL cp_warn(__LOCATION__, &
                         "Control of the global temperature by rescaling of the velocity "// &
                         "is not consistent with the presence of different thermal regions. "// &
                         "The temperature of different regions is rescaled separatedly.")
         END IF
         CALL section_vals_val_get(thermal_region_section, "FORCE_RESCALING", &
                                   l_val=thermal_regions%force_rescaling)
         region_sections => section_vals_get_subs_vals(thermal_region_section, &
                                                       "DEFINE_REGION")
         CALL section_vals_get(region_sections, n_repetition=nregions)
         IF (nregions > 0) THEN
            thermal_regions%nregions = nregions
            thermal_regions%section => thermal_region_section
            ALLOCATE (thermal_regions%thermal_region(nregions))
            CALL force_env_get(force_env, subsys=subsys)
            CALL cp_subsys_get(subsys, particles=particles)
            IF (simpar%ensemble == langevin_ensemble) THEN
               CALL cite_reference(Kantorovich2008)
               CALL cite_reference(Kantorovich2008a)
               CALL section_vals_val_get(thermal_region_section, "DO_LANGEVIN_DEFAULT", &
                                         l_val=do_langevin_default)
               ALLOCATE (thermal_regions%do_langevin(particles%n_els))
               thermal_regions%do_langevin = do_langevin_default
            END IF
            DO ireg = 1, nregions
               NULLIFY (t_region)
               t_region => thermal_regions%thermal_region(ireg)
               t_region%region_index = ireg
               CALL section_vals_val_get(region_sections, "LIST", &
                                         i_rep_section=ireg, n_rep_val=nlist)
               NULLIFY (t_region%part_index)
               t_region%npart = 0
               IF (simpar%ensemble == langevin_ensemble) THEN
                  CALL section_vals_val_get(region_sections, "DO_LANGEVIN", &
                                            i_rep_section=ireg, l_val=do_langevin)
               END IF
               DO il = 1, nlist
                  CALL section_vals_val_get(region_sections, "LIST", i_rep_section=ireg, &
                                            i_rep_val=il, i_vals=tmplist)
                  CALL reallocate(t_region%part_index, 1, t_region%npart + SIZE(tmplist))
                  DO i = 1, SIZE(tmplist)
                     ipart = tmplist(i)
                     CPASSERT(((ipart > 0) .AND. (ipart <= particles%n_els)))
                     t_region%npart = t_region%npart + 1
                     t_region%part_index(t_region%npart) = ipart
                     particles%els(ipart)%t_region_index = ireg
                     IF (simpar%ensemble == langevin_ensemble) THEN
                        thermal_regions%do_langevin(ipart) = do_langevin
                     END IF
                  END DO
               END DO
               CALL section_vals_val_get(region_sections, "TEMPERATURE", i_rep_section=ireg, &
                                         r_val=temp)
               t_region%temp_expected = temp
               CALL section_vals_val_get(region_sections, "TEMP_TOL", i_rep_section=ireg, &
                                         r_val=temp_tol)
               t_region%temp_tol = temp_tol
               CALL section_vals_val_get(region_sections, "NOISY_GAMMA_REGION", i_rep_section=ireg, explicit=do_read_ngr)
               IF (do_read_ngr) THEN
                  CALL section_vals_val_get(region_sections, "NOISY_GAMMA_REGION", i_rep_section=ireg, &
                                            r_val=t_region%noisy_gamma_region)
                  IF (simpar%ensemble == langevin_ensemble) THEN
                     IF (.NOT. do_langevin) THEN
                        CALL integer_to_string(ireg, my_region)
                        CALL cp_warn(__LOCATION__, &
                                     "You provided NOISY_GAMMA_REGION but atoms in thermal region "//TRIM(my_region)// &
                                     " will not undergo Langevin MD. "// &
                                     "NOISY_GAMMA_REGION will be ignored and its value discarded!")
                     END IF
                  ELSE
                     CALL cp_warn(__LOCATION__, &
                                  "You provided NOISY_GAMMA_REGION but the Langevin Ensamble is not selected "// &
                                  "NOISY_GAMMA_REGION will be ignored and its value discarded!")
                  END IF
               ELSE
                  t_region%noisy_gamma_region = simpar%noisy_gamma
               END IF
            END DO
            simpar%do_thermal_region = .TRUE.
         ELSE
            CALL release_thermal_regions(thermal_regions)
            DEALLOCATE (thermal_regions)
            simpar%do_thermal_region = .FALSE.
         END IF
      ELSE
         CALL release_thermal_regions(thermal_regions)
         DEALLOCATE (thermal_regions)
         simpar%do_thermal_region = .FALSE.
      END IF

   END SUBROUTINE create_thermal_regions

! **************************************************************************************************
!> \brief print_thermal_regions_temperature
!> \param thermal_regions : thermal regions type contains information
!>                          about the regions
!> \param itimes          : iteration number of the time step
!> \param time            : simulation time of the time step
!> \param pos             : file position
!> \param act             : file action
!> \par History
!>   - added doxygen header and changed subroutine name from
!>     print_thermal_regions to print_thermal_regions_temperature
!>     (2014/02/04, LT)
!> \author
! **************************************************************************************************
   SUBROUTINE print_thermal_regions_temperature(thermal_regions, itimes, time, pos, act)
      TYPE(thermal_regions_type), POINTER                :: thermal_regions
      INTEGER, INTENT(IN)                                :: itimes
      REAL(KIND=dp), INTENT(IN)                          :: time
      CHARACTER(LEN=default_string_length)               :: pos, act

      CHARACTER(LEN=default_string_length)               :: fmd
      INTEGER                                            :: ireg, nregions, unit
      LOGICAL                                            :: new_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: temp
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(section_vals_type), POINTER                   :: print_key

      NULLIFY (logger)
      logger => cp_get_default_logger()

      IF (ASSOCIATED(thermal_regions)) THEN
         print_key => section_vals_get_subs_vals(thermal_regions%section, "PRINT%TEMPERATURE")
         IF (BTEST(cp_print_key_should_output(logger%iter_info, print_key), cp_p_file)) THEN
            unit = cp_print_key_unit_nr(logger, thermal_regions%section, "PRINT%TEMPERATURE", &
                                        extension=".tregion", file_position=pos, &
                                        file_action=act, is_new_file=new_file)
            IF (unit > 0) THEN
               IF (new_file) THEN
                  WRITE (unit, '(A)') "# Temperature per Region"
                  WRITE (unit, '("#",3X,A,2X,A,13X,A)') "Step Nr.", "Time[fs]", "Temp.[K] ...."
               END IF
               nregions = thermal_regions%nregions
               ALLOCATE (temp(0:nregions))
               temp = 0.0_dp
               temp(0) = thermal_regions%temp_reg0
               DO ireg = 1, nregions
                  temp(ireg) = thermal_regions%thermal_region(ireg)%temperature
               END DO
               fmd = "(I10,F20.3,"//TRIM(ADJUSTL(cp_to_string(nregions + 1)))//"F20.6)"
               fmd = TRIM(fmd)
               WRITE (UNIT=unit, FMT=fmd) itimes, time, temp(0:nregions)
               DEALLOCATE (temp)
            END IF
            CALL cp_print_key_finished_output(unit, logger, thermal_regions%section, "PRINT%TEMPERATURE")
         END IF
      END IF
   END SUBROUTINE print_thermal_regions_temperature

! **************************************************************************************************
!> \brief print out information regarding to langevin regions defined in
!>        thermal_regions section
!> \param thermal_regions : thermal regions type containing the relevant
!>                          langevin regions information
!> \param simpar          : wrapper for simulation parameters
!> \param pos             : file position
!> \param act             : file action
!> \par History
!>   - created (2014/02/02, LT)
!> \author Lianheng Tong [LT] (tonglianheng@gmail.com)
! **************************************************************************************************
   SUBROUTINE print_thermal_regions_langevin(thermal_regions, simpar, pos, act)
      TYPE(thermal_regions_type), POINTER                :: thermal_regions
      TYPE(simpar_type), POINTER                         :: simpar
      CHARACTER(LEN=default_string_length)               :: pos, act

      INTEGER                                            :: ipart, ipart_reg, ireg, natoms, &
                                                            print_unit
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: region_id
      LOGICAL                                            :: new_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: noisy_gamma_region, temperature
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(section_vals_type), POINTER                   :: print_key

      NULLIFY (logger)
      logger => cp_get_default_logger()

      IF (ASSOCIATED(thermal_regions)) THEN
         IF (ASSOCIATED(thermal_regions%do_langevin)) THEN
            print_key => section_vals_get_subs_vals(thermal_regions%section, &
                                                    "PRINT%LANGEVIN_REGIONS")
            IF (BTEST(cp_print_key_should_output(logger%iter_info, print_key), &
                      cp_p_file)) THEN
               print_unit = cp_print_key_unit_nr(logger, thermal_regions%section, &
                                                 "PRINT%LANGEVIN_REGIONS", &
                                                 extension=".lgv_regions", &
                                                 file_position=pos, file_action=act, &
                                                 is_new_file=new_file)
               IF (print_unit > 0) THEN
                  IF (new_file) THEN
                     WRITE (print_unit, '(A)') "# Atoms Undergoing Langevin MD"
                     WRITE (print_unit, '(A,3X,A,3X,A,3X,A,3X,A,3X,A)') &
                        "#", "Atom_ID", "Region_ID", "Langevin(L)/NVE(N)", "Expected_T[K]", "[NoisyGamma]"
                  END IF
                  natoms = SIZE(thermal_regions%do_langevin)
                  ALLOCATE (temperature(natoms))
                  ALLOCATE (region_id(natoms))
                  ALLOCATE (noisy_gamma_region(natoms))
                  temperature(:) = simpar%temp_ext
                  region_id(:) = 0
                  noisy_gamma_region(:) = simpar%noisy_gamma
                  DO ireg = 1, thermal_regions%nregions
                     DO ipart_reg = 1, thermal_regions%thermal_region(ireg)%npart
                        ipart = thermal_regions%thermal_region(ireg)%part_index(ipart_reg)
                        temperature(ipart) = thermal_regions%thermal_region(ireg)%temp_expected
                        region_id(ipart) = thermal_regions%thermal_region(ireg)%region_index
                        noisy_gamma_region(ipart) = thermal_regions%thermal_region(ireg)%noisy_gamma_region
                     END DO
                  END DO
                  DO ipart = 1, natoms
                     WRITE (print_unit, '(1X,I10,2X)', advance='no') ipart
                     WRITE (print_unit, '(I10,3X)', advance='no') region_id(ipart)
                     IF (thermal_regions%do_langevin(ipart)) THEN
                        WRITE (print_unit, '(A,3X)', advance='no') "L"
                        IF (noisy_gamma_region(ipart) > 0._dp) THEN
                           WRITE (print_unit, '(10X,F20.3,3X,ES9.3)') temperature(ipart)*kelvin, &
                              noisy_gamma_region(ipart)/femtoseconds
                        ELSE
                           WRITE (print_unit, '(10X,F20.3)') temperature(ipart)*kelvin
                        END IF
                     ELSE
                        WRITE (print_unit, '(A,3X)', advance='no') "N"
                        WRITE (print_unit, '(18X,A)') "--"
                     END IF
                  END DO
                  DEALLOCATE (region_id)
                  DEALLOCATE (temperature)
                  DEALLOCATE (noisy_gamma_region)
               END IF
               CALL cp_print_key_finished_output(print_unit, logger, thermal_regions%section, &
                                                 "PRINT%LANGEVIN_REGIONS")
            END IF
         END IF
      END IF
   END SUBROUTINE print_thermal_regions_langevin

END MODULE thermal_region_utils
