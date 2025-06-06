!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief The environment for the empirical interatomic potential methods.
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
MODULE eip_environment_types
   USE atomic_kind_list_types,          ONLY: atomic_kind_list_create,&
                                              atomic_kind_list_release,&
                                              atomic_kind_list_type
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_types,                      ONLY: cell_release,&
                                              cell_retain,&
                                              cell_type
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_release,&
                                              cp_subsys_set,&
                                              cp_subsys_type
   USE distribution_1d_types,           ONLY: distribution_1d_type
   USE input_section_types,             ONLY: section_vals_release,&
                                              section_vals_type
   USE kinds,                           ONLY: dp
   USE molecule_kind_list_types,        ONLY: molecule_kind_list_create,&
                                              molecule_kind_list_release,&
                                              molecule_kind_list_type
   USE molecule_kind_types,             ONLY: molecule_kind_type
   USE molecule_list_types,             ONLY: molecule_list_create,&
                                              molecule_list_release,&
                                              molecule_list_type
   USE molecule_types,                  ONLY: molecule_type
   USE particle_list_types,             ONLY: particle_list_create,&
                                              particle_list_release,&
                                              particle_list_type
   USE particle_types,                  ONLY: particle_type
   USE virial_types,                    ONLY: virial_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'eip_environment_types'

   ! *** Public data types ***
   PUBLIC :: eip_environment_type

   ! *** Public subroutines ***
   PUBLIC :: eip_env_release, &
             eip_env_set, &
             eip_env_get, &
             eip_env_create

! **************************************************************************************************
!> \brief The empirical interatomic potential environment
!> \param eip_model Specifies which EIP model is in use.
!> \param eip_kinetic_energy The EIP kinetic energy
!> \param eip_potential_energy The EIP potential energy
!> \param eip_energy The total eip energy
!> \param eip_energy_var Variance of the energy/atom
!> \param eip_forces The final eip forces [eV/A]
!> \param coord_avg The average coordination number
!> \param coord_var The variance of the coordination number
!> \param count Counts how often the empirical interatomic potential function
!>               is called. Don't ask why this is a real!
!> \param subsystem the particles, molecules,... of this environment
!> \param eip_input Pointer to the EIP input section
!> \param force_env_input Pointer to the force_env input section
!> \param cell The simulation cell
!> \param cell_ref The reference simulation cell
!> \param use_ref_cell Logical which indicates if reference
!>                      simulation cell is used
!> \param virial Dummy virial pointer
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
   TYPE eip_environment_type
      INTEGER                                          :: eip_model = 0
      REAL(KIND=dp)                                    :: eip_energy = 0.0_dp, &
                                                          eip_kinetic_energy = 0.0_dp, &
                                                          eip_potential_energy = 0.0_dp, &
                                                          eip_energy_var = 0.0_dp
      REAL(KIND=dp), DIMENSION(:, :), POINTER           :: eip_forces => Null()
      REAL(KIND=dp)                                    :: coord_avg = 0.0_dp, &
                                                          coord_var = 0.0_dp, &
                                                          count = 0.0_dp
      TYPE(cp_subsys_type), POINTER                    :: subsys => Null()
      TYPE(section_vals_type), POINTER                 :: eip_input => Null(), &
                                                          force_env_input => Null()
      TYPE(cell_type), POINTER                         :: cell_ref => Null()
      LOGICAL                                          :: use_ref_cell = .FALSE.
   END TYPE eip_environment_type

CONTAINS

! **************************************************************************************************
!> \brief Releases the given eip environment (see doc/ReferenceCounting.html)
!> \param eip_env The eip environment to release
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
   SUBROUTINE eip_env_release(eip_env)

      TYPE(eip_environment_type), INTENT(INOUT)          :: eip_env

      IF (ASSOCIATED(eip_env%eip_forces)) THEN
         DEALLOCATE (eip_env%eip_forces)
      END IF
      IF (ASSOCIATED(eip_env%subsys)) THEN
         CALL cp_subsys_release(eip_env%subsys)
      END IF
      IF (ASSOCIATED(eip_env%subsys)) THEN
         CALL cp_subsys_release(eip_env%subsys)
      END IF
      !IF (ASSOCIATED(eip_env%eip_input)) THEN
      !  CALL section_vals_release(eip_env%eip_input)
      !END IF
      !IF (ASSOCIATED(eip_env%force_env_input)) THEN
      !  CALL section_vals_release(eip_env%force_env_input)
      !END IF
      IF (ASSOCIATED(eip_env%cell_ref)) THEN
         CALL cell_release(eip_env%cell_ref)
      END IF
   END SUBROUTINE eip_env_release

! **************************************************************************************************
!> \brief Returns various attributes of the eip environment
!> \param eip_env The enquired eip environment
!> \param eip_model Specifies which EIP model is in use.
!> \param eip_energy The total eip energy
!> \param eip_energy_var Variance of the energy/atom
!> \param eip_forces The final eip forces [eV/A]
!> \param coord_avg The average coordination number
!> \param coord_var The variance of the coordination number
!> \param count Counts how often the empirical interatomic potential function
!>               is called. Don't ask why this is a real!
!> \param subsys the particles, molecules,... of this environment
!> \param atomic_kind_set The set of all atomic kinds involved
!> \param particle_set The set of all particles
!> \param local_particles All particles on this particular node
!> \param molecule_kind_set The set of all different molecule kinds involved
!> \param molecule_set The set of all molecules
!> \param local_molecules All molecules on this particular node
!> \param eip_input the pointer to the EIP input section
!> \param force_env_input Pointer to the force_env input section
!> \param cell The simulation cell
!> \param cell_ref The reference simulation cell
!> \param use_ref_cell Logical which indicates if reference
!>                      simulation cell is used
!> \param eip_kinetic_energy The EIP kinetic energy
!> \param eip_potential_energy The EIP potential energy
!> \param virial Dummy virial pointer
!>
!>      For possible missing arguments see the attributes of
!>      eip_environment_type
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
   SUBROUTINE eip_env_get(eip_env, eip_model, eip_energy, eip_energy_var, &
                          eip_forces, coord_avg, coord_var, count, subsys, &
                          atomic_kind_set, particle_set, local_particles, &
                          molecule_kind_set, molecule_set, local_molecules, &
                          eip_input, force_env_input, cell, cell_ref, &
                          use_ref_cell, eip_kinetic_energy, eip_potential_energy, &
                          virial)

      TYPE(eip_environment_type), INTENT(IN)             :: eip_env
      INTEGER, INTENT(OUT), OPTIONAL                     :: eip_model
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: eip_energy, eip_energy_var
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: eip_forces
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: coord_avg, coord_var, count
      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys
      TYPE(atomic_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: atomic_kind_set
      TYPE(particle_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: particle_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_particles
      TYPE(molecule_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_molecules
      TYPE(section_vals_type), OPTIONAL, POINTER         :: eip_input, force_env_input
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell, cell_ref
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_ref_cell
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: eip_kinetic_energy, eip_potential_energy
      TYPE(virial_type), OPTIONAL, POINTER               :: virial

      TYPE(atomic_kind_list_type), POINTER               :: atomic_kinds
      TYPE(molecule_kind_list_type), POINTER             :: molecule_kinds
      TYPE(molecule_list_type), POINTER                  :: molecules
      TYPE(particle_list_type), POINTER                  :: particles

!   ------------------------------------------------------------------------

      NULLIFY (atomic_kinds, particles, molecules, molecule_kinds)

      IF (PRESENT(eip_model)) eip_model = eip_env%eip_model
      IF (PRESENT(eip_kinetic_energy)) THEN
         eip_kinetic_energy = eip_env%eip_kinetic_energy
      END IF
      IF (PRESENT(eip_potential_energy)) THEN
         eip_potential_energy = eip_env%eip_potential_energy
      END IF
      IF (PRESENT(eip_energy)) eip_energy = eip_env%eip_energy
      IF (PRESENT(eip_energy_var)) eip_energy_var = eip_env%eip_energy_var
      IF (PRESENT(eip_forces)) eip_forces = eip_env%eip_forces
      IF (PRESENT(coord_avg)) coord_avg = eip_env%coord_avg
      IF (PRESENT(coord_var)) coord_var = eip_env%coord_var
      IF (PRESENT(count)) count = eip_env%count
      IF (PRESENT(subsys)) subsys => eip_env%subsys
      CALL cp_subsys_get(eip_env%subsys, &
                         atomic_kinds=atomic_kinds, &
                         particles=particles, &
                         molecule_kinds=molecule_kinds, &
                         molecules=molecules, &
                         local_molecules=local_molecules, &
                         local_particles=local_particles, &
                         virial=virial, &
                         cell=cell)
      IF (PRESENT(atomic_kind_set)) atomic_kind_set => atomic_kinds%els
      IF (PRESENT(particle_set)) particle_set => particles%els
      IF (PRESENT(molecule_kind_set)) molecule_kind_set => molecule_kinds%els
      IF (PRESENT(molecule_set)) molecule_set => molecules%els

      IF (PRESENT(eip_input)) eip_input => eip_env%eip_input
      IF (PRESENT(force_env_input)) force_env_input => eip_env%force_env_input
      IF (PRESENT(cell_ref)) cell_ref => eip_env%cell_ref
      IF (PRESENT(use_ref_cell)) use_ref_cell = eip_env%use_ref_cell

   END SUBROUTINE eip_env_get

! **************************************************************************************************
!> \brief Sets various attributes of the eip environment
!> \param eip_env The enquired eip environment
!> \param eip_model Specifies which EIP model is in use
!> \param eip_energy The total eip energy
!> \param eip_energy_var Variance of the energy/atom
!> \param eip_forces The final eip forces [eV/A]
!> \param coord_avg The average coordination number
!> \param coord_var The variance of the coordination number
!> \param count Counts how often the empirical interatomic potential function
!>               is called. Don't ask why this is a real!
!> \param subsys the particles, molecules,... of this environment
!> \param atomic_kind_set The set of all atomic kinds involved
!> \param particle_set The set of all particles
!> \param local_particles All particles on this particular node
!> \param molecule_kind_set The set of all different molecule kinds involved
!> \param molecule_set The set of all molecules
!> \param local_molecules All molecules on this particular node
!> \param eip_input the pointer to the EIP input section
!> \param force_env_input Pointer to the force_env input section
!> \param cell_ref The reference simulation cell
!> \param use_ref_cell Logical which indicates if reference
!>                      simulation cell is used
!> \param eip_kinetic_energy The EIP kinetic energy
!> \param eip_potential_energy The EIP potential energy
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
!> \note
!>   For possible missing arguments see the attributes of eip_environment_type
! **************************************************************************************************
   SUBROUTINE eip_env_set(eip_env, eip_model, eip_energy, eip_energy_var, &
                          eip_forces, coord_avg, coord_var, count, subsys, &
                          atomic_kind_set, particle_set, local_particles, &
                          molecule_kind_set, molecule_set, local_molecules, &
                          eip_input, force_env_input, cell_ref, &
                          use_ref_cell, eip_kinetic_energy, eip_potential_energy)

      TYPE(eip_environment_type), INTENT(INOUT)          :: eip_env
      INTEGER, INTENT(IN), OPTIONAL                      :: eip_model
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: eip_energy, eip_energy_var
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: eip_forces
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: coord_avg, coord_var, count
      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys
      TYPE(atomic_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: atomic_kind_set
      TYPE(particle_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: particle_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_particles
      TYPE(molecule_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_molecules
      TYPE(section_vals_type), OPTIONAL, POINTER         :: eip_input, force_env_input
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell_ref
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_ref_cell
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: eip_kinetic_energy, eip_potential_energy

      TYPE(atomic_kind_list_type), POINTER               :: atomic_kinds
      TYPE(molecule_kind_list_type), POINTER             :: molecule_kinds
      TYPE(molecule_list_type), POINTER                  :: molecules
      TYPE(particle_list_type), POINTER                  :: particles

      IF (PRESENT(eip_model)) eip_env%eip_model = eip_model
      IF (PRESENT(eip_kinetic_energy)) THEN
         eip_env%eip_kinetic_energy = eip_kinetic_energy
      END IF
      IF (PRESENT(eip_potential_energy)) THEN
         eip_env%eip_potential_energy = eip_potential_energy
      END IF
      IF (PRESENT(eip_energy)) eip_env%eip_energy = eip_energy
      IF (PRESENT(eip_energy_var)) eip_env%eip_energy_var = eip_energy_var
      IF (PRESENT(eip_forces)) eip_env%eip_forces = eip_forces
      IF (PRESENT(coord_avg)) eip_env%coord_avg = coord_avg
      IF (PRESENT(coord_var)) eip_env%coord_var = coord_var
      IF (PRESENT(count)) eip_env%count = count
      IF (PRESENT(subsys)) THEN
         IF (ASSOCIATED(eip_env%subsys)) THEN
         IF (.NOT. ASSOCIATED(eip_env%subsys, subsys)) THEN
            CALL cp_subsys_release(eip_env%subsys)
         END IF
         END IF
         eip_env%subsys => subsys
      END IF
      IF (PRESENT(atomic_kind_set)) THEN
         CALL atomic_kind_list_create(atomic_kinds, els_ptr=atomic_kind_set)
         CALL cp_subsys_set(eip_env%subsys, atomic_kinds=atomic_kinds)
         CALL atomic_kind_list_release(atomic_kinds)
      END IF
      IF (PRESENT(particle_set)) THEN
         CALL particle_list_create(particles, els_ptr=particle_set)
         CALL cp_subsys_set(eip_env%subsys, particles=particles)
         CALL particle_list_release(particles)
      END IF
      IF (PRESENT(molecule_kind_set)) THEN
         CALL molecule_kind_list_create(molecule_kinds, els_ptr=molecule_kind_set)
         CALL cp_subsys_set(eip_env%subsys, molecule_kinds=molecule_kinds)
         CALL molecule_kind_list_release(molecule_kinds)
      END IF
      IF (PRESENT(molecule_set)) THEN
         CALL molecule_list_create(molecules, els_ptr=molecule_set)
         CALL cp_subsys_set(eip_env%subsys, molecules=molecules)
         CALL molecule_list_release(molecules)
      END IF
      IF (PRESENT(local_particles)) THEN
         CALL cp_subsys_set(eip_env%subsys, local_particles=local_particles)
      END IF
      IF (PRESENT(local_molecules)) THEN
         CALL cp_subsys_set(eip_env%subsys, local_molecules=local_molecules)
      END IF

      IF (PRESENT(eip_input)) eip_env%eip_input => eip_input
      IF (PRESENT(force_env_input)) THEN
         eip_env%force_env_input => force_env_input
      END IF
      IF (PRESENT(cell_ref)) THEN
         CALL cell_retain(cell_ref)
         CALL cell_release(eip_env%cell_ref)
         eip_env%cell_ref => cell_ref
      END IF
      IF (PRESENT(use_ref_cell)) eip_env%use_ref_cell = use_ref_cell
   END SUBROUTINE eip_env_set

! **************************************************************************************************
!> \brief Reinitializes the eip environment
!> \param eip_env The eip environment to be reinitialized
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
   SUBROUTINE eip_env_clear(eip_env)

      TYPE(eip_environment_type), INTENT(INOUT)          :: eip_env

      eip_env%eip_model = 0
      eip_env%eip_kinetic_energy = 0.0_dp
      eip_env%eip_potential_energy = 0.0_dp
      eip_env%eip_energy = 0.0_dp
      eip_env%eip_energy_var = 0.0_dp
      eip_env%coord_avg = 0.0_dp
      eip_env%coord_var = 0.0_dp
      eip_env%count = 0.0_dp
      IF (ASSOCIATED(eip_env%eip_forces)) THEN
         eip_env%eip_forces(:, :) = 0.0_dp
      END IF
      IF (ASSOCIATED(eip_env%subsys)) THEN
         CALL cp_subsys_release(eip_env%subsys)
      END IF
      IF (ASSOCIATED(eip_env%eip_input)) THEN
         CALL section_vals_release(eip_env%eip_input)
      END IF
      IF (ASSOCIATED(eip_env%force_env_input)) THEN
         CALL section_vals_release(eip_env%force_env_input)
      END IF
      IF (ASSOCIATED(eip_env%cell_ref)) THEN
         CALL cell_release(eip_env%cell_ref)
      END IF
   END SUBROUTINE eip_env_clear

! **************************************************************************************************
!> \brief Creates the eip environment
!> \param eip_env The eip environment to be created
!> \par History
!>      03.2006 initial create [tdk]
!> \author Thomas D. Kuehne (tkuehne@phys.chem.ethz.ch)
! **************************************************************************************************
   SUBROUTINE eip_env_create(eip_env)

      TYPE(eip_environment_type), INTENT(OUT)            :: eip_env

      NULLIFY (eip_env%eip_forces)
      NULLIFY (eip_env%subsys)
      NULLIFY (eip_env%eip_input)
      NULLIFY (eip_env%force_env_input)
      NULLIFY (eip_env%cell_ref)

      eip_env%use_ref_cell = .FALSE.
      CALL eip_env_clear(eip_env)
   END SUBROUTINE eip_env_create

END MODULE eip_environment_types
