!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2024 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Compute partial charges according to EEQ model
!> Reference: S. Grimme, M. Müller, and A. Hansen, 
!>            J. Chem. Phys., vol. 158, no. 12, p. 124111, (2023)
!>            DOI: 10.1063/5.0137838.
!> \author JVP (09.2024)
! *************************************************************************************************
MODULE eeq_charge_model

   USE atomic_kind_types,               ONLY: get_atomic_kind
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE particle_types,                  ONLY: particle_type
   USE kinds,                           ONLY: dp
   USE mctc_io_structure,               ONLY: new, structure_type
   USE multicharge,                     ONLY: mchrg_model_type, new_eeq2019_model, &
                                              get_covalent_rad, get_lattice_points, &
                                              get_coordination_number
   USE cell_types,                      ONLY: cell_type, get_cell

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'multicharge'

! *** Public data types ***

   PUBLIC :: eeq_charges

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param para_env ...
!> \param calculate_forces ...
! **************************************************************************************************
   SUBROUTINE eeq_charges(qs_env, charges, enshift_type, cn_cutoff)

      TYPE(qs_environment_type), POINTER                   :: qs_env
      REAL(KIND=dp), DIMENSION(:), intent(out), contiguous :: charges  !atomic partial charges
      REAL(KIND=dp)                                        :: enshift_type
      REAL(KIND=dp)                                        :: cn_cutoff

      CHARACTER(LEN=*), PARAMETER :: routineN = 'eeq_charges'

      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      INTEGER                                            :: natom, iatom, ikind
      INTEGER                                            :: handle
      INTEGER, DIMENSION(3)                              :: periodic
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: el_num
      REAL(KIND=dp), ALLOCATABLE                         :: xyz(:,:)
      REAL(KIND=dp), ALLOCATABLE                         :: cn(:)
      REAL(KIND=dp), ALLOCATABLE                         :: rcov(:), trans(:, :)

      
      LOGICAL, DIMENSION(3)                              :: lperiod
      type(structure_type)                               :: mol
      type(mchrg_model_type)                             :: model
      type(cell_type), POINTER                           :: cell

      CALL timeset(routineN, handle)

      NULLIFY (particle_set)
      CALL get_qs_env(qs_env=qs_env, particle_set=particle_set, cell=cell)
      natom = SIZE(particle_set)

      !get information about particles
      ALLOCATE (xyz(3, natom))
      ALLOCATE (el_num(natom))
      DO iatom = 1, natom
         xyz(:, iatom) = particle_set(iatom)%r(:)
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
         el_num(iatom) = ikind
      END DO

      !get information about cell / lattice
      CALL get_cell(cell=cell, periodic=periodic)
      lperiod(1) = periodic(1) == 1
      lperiod(2) = periodic(2) == 1
      lperiod(3) = periodic(3) == 1

      !prepare for the call to the dispersion function
      CALL new(mol, el_num, xyz, lattice=cell%hmat, periodic=lperiod)

      !create electronegativity equilibration model 
      call new_eeq2019_model(mol, model)
      call get_lattice_points(mol%periodic, mol%lattice, cn_cutoff, trans)
      
      rcov = get_covalent_rad(mol%num)
      call get_coordination_number(mol, trans, cn_cutoff, rcov, cn)

      call model%solve(mol, cn, qvec=charges)

      MARK_USED(enshift_type)

      CALL timestop(handle)

   END SUBROUTINE eeq_charges

END MODULE eeq_charge_model