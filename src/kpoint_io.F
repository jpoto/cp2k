!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Restart file for k point calculations
!> \par History
!> \author JGH (30.09.2015)
! **************************************************************************************************
MODULE kpoint_io

   USE atomic_kind_types,               ONLY: get_atomic_kind
   USE basis_set_types,                 ONLY: get_gto_basis_set,&
                                              gto_basis_set_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_copy,&
                                              dbcsr_get_info,&
                                              dbcsr_p_type
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm,&
                                              copy_fm_to_dbcsr
   USE cp_files,                        ONLY: close_file,&
                                              open_file
   USE cp_fm_types,                     ONLY: cp_fm_read_unformatted,&
                                              cp_fm_to_fm,&
                                              cp_fm_type,&
                                              cp_fm_write_unformatted
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type,&
                                              cp_to_string
   USE cp_output_handling,              ONLY: cp_p_file,&
                                              cp_print_key_finished_output,&
                                              cp_print_key_generate_filename,&
                                              cp_print_key_should_output,&
                                              cp_print_key_unit_nr
   USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                              section_vals_type
   USE kinds,                           ONLY: default_path_length,&
                                              default_string_length
   USE kpoint_methods,                  ONLY: kpoint_density_matrices,&
                                              kpoint_density_transform,&
                                              kpoint_initialize_mo_set,&
                                              kpoint_initialize_mos
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_type
   USE message_passing,                 ONLY: mp_para_env_type
   USE orbital_pointers,                ONLY: nso
   USE particle_types,                  ONLY: particle_type
   USE qs_density_matrices,             ONLY: calculate_density_matrix
   USE qs_dftb_types,                   ONLY: qs_dftb_atom_type
   USE qs_dftb_utils,                   ONLY: get_dftb_atom_param
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
   USE qs_mo_io,                        ONLY: read_mo_set_from_restart,&
                                              wfn_restart_file_name
   USE qs_mo_methods,                   ONLY: make_basis_simple
   USE qs_mo_types,                     ONLY: get_mo_set,&
                                              mo_set_type
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE qs_scf_types,                    ONLY: qs_scf_env_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'kpoint_io'

   INTEGER, PARAMETER                   :: version = 1

   PUBLIC :: read_kpoints_restart, &
             write_kpoints_restart
   PUBLIC :: get_cell, write_kpoints_file_header

! **************************************************************************************************

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param denmat ...
!> \param kpoints ...
!> \param scf_env ...
!> \param dft_section ...
!> \param particle_set ...
!> \param qs_kind_set ...
! **************************************************************************************************
   SUBROUTINE write_kpoints_restart(denmat, kpoints, scf_env, dft_section, &
                                    particle_set, qs_kind_set)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(qs_scf_env_type), POINTER                     :: scf_env
      TYPE(section_vals_type), POINTER                   :: dft_section
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CHARACTER(LEN=*), PARAMETER :: routineN = 'write_kpoints_restart'
      CHARACTER(LEN=30), DIMENSION(2), PARAMETER :: &
         keys = ["SCF%PRINT%RESTART_HISTORY", "SCF%PRINT%RESTART        "]

      INTEGER                                            :: handle, ic, ikey, ires, ispin, nimages, &
                                                            nspin
      INTEGER, DIMENSION(3)                              :: cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      TYPE(cp_fm_type), POINTER                          :: fmat
      TYPE(cp_logger_type), POINTER                      :: logger

      CALL timeset(routineN, handle)
      logger => cp_get_default_logger()

      IF (BTEST(cp_print_key_should_output(logger%iter_info, &
                                           dft_section, keys(1)), cp_p_file) .OR. &
          BTEST(cp_print_key_should_output(logger%iter_info, &
                                           dft_section, keys(2)), cp_p_file)) THEN

         DO ikey = 1, SIZE(keys)

            IF (BTEST(cp_print_key_should_output(logger%iter_info, &
                                                 dft_section, keys(ikey)), cp_p_file)) THEN

               ires = cp_print_key_unit_nr(logger, dft_section, keys(ikey), &
                                           extension=".kp", file_status="REPLACE", file_action="WRITE", &
                                           do_backup=.TRUE., file_form="UNFORMATTED")

               CALL write_kpoints_file_header(qs_kind_set, particle_set, ires)

               nspin = SIZE(denmat, 1)
               nimages = SIZE(denmat, 2)
               NULLIFY (cell_to_index)
               IF (nimages > 1) THEN
                  CALL get_kpoint_info(kpoint=kpoints, cell_to_index=cell_to_index)
               END IF
               CPASSERT(ASSOCIATED(scf_env%scf_work1))
               fmat => scf_env%scf_work1(1)

               DO ispin = 1, nspin
                  IF (ires > 0) WRITE (ires) ispin, nspin, nimages
                  DO ic = 1, nimages
                     IF (nimages > 1) THEN
                        cell = get_cell(ic, cell_to_index)
                     ELSE
                        cell = 0
                     END IF
                     IF (ires > 0) WRITE (ires) ic, cell
                     CALL copy_dbcsr_to_fm(denmat(ispin, ic)%matrix, fmat)
                     CALL cp_fm_write_unformatted(fmat, ires)
                  END DO
               END DO

               CALL cp_print_key_finished_output(ires, logger, dft_section, TRIM(keys(ikey)))

            END IF

         END DO

      END IF

      CALL timestop(handle)

   END SUBROUTINE write_kpoints_restart

! **************************************************************************************************
!> \brief ...
!> \param ic ...
!> \param cell_to_index ...
!> \return ...
! **************************************************************************************************
   FUNCTION get_cell(ic, cell_to_index) RESULT(cell)
      INTEGER, INTENT(IN)                                :: ic
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      INTEGER, DIMENSION(3)                              :: cell

      INTEGER                                            :: i1, i2, i3

      cell = 0
      ALLCELL: DO i3 = LBOUND(cell_to_index, 3), UBOUND(cell_to_index, 3)
         DO i2 = LBOUND(cell_to_index, 2), UBOUND(cell_to_index, 2)
            DO i1 = LBOUND(cell_to_index, 1), UBOUND(cell_to_index, 1)
               IF (cell_to_index(i1, i2, i3) == ic) THEN
                  cell(1) = i1
                  cell(2) = i2
                  cell(3) = i3
                  EXIT ALLCELL
               END IF
            END DO
         END DO
      END DO ALLCELL

   END FUNCTION get_cell

! **************************************************************************************************
!> \brief ...
!> \param qs_kind_set ...
!> \param particle_set ...
!> \param ires ...
!> \param basis_type ...
! **************************************************************************************************
   SUBROUTINE write_kpoints_file_header(qs_kind_set, particle_set, ires, basis_type)

      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      INTEGER                                            :: ires
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: basis_type

      CHARACTER(LEN=*), PARAMETER :: routineN = 'write_kpoints_file_header'

      CHARACTER(LEN=default_string_length)               :: basis_type_local
      INTEGER                                            :: handle, iatom, ikind, iset, ishell, &
                                                            lmax, lshell, nao, natom, nset, &
                                                            nset_max, nsgf, nshell_max
      INTEGER, DIMENSION(:), POINTER                     :: nset_info, nshell
      INTEGER, DIMENSION(:, :), POINTER                  :: l, nshell_info
      INTEGER, DIMENSION(:, :, :), POINTER               :: nso_info
      TYPE(gto_basis_set_type), POINTER                  :: orb_basis_set
      TYPE(qs_dftb_atom_type), POINTER                   :: dftb_parameter

      CALL timeset(routineN, handle)

      IF (ires > 0) THEN

         IF (PRESENT(basis_type)) THEN
            basis_type_local = basis_type
         ELSE
            basis_type_local = "ORB"
         END IF

         ! create some info about the basis set first
         natom = SIZE(particle_set, 1)
         nset_max = 0
         nshell_max = 0

         DO iatom = 1, natom
            NULLIFY (orb_basis_set, dftb_parameter)
            CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
            CALL get_qs_kind(qs_kind_set(ikind), basis_type=basis_type_local, &
                             basis_set=orb_basis_set, dftb_parameter=dftb_parameter)
            IF (ASSOCIATED(orb_basis_set)) THEN
               CALL get_gto_basis_set(gto_basis_set=orb_basis_set, &
                                      nset=nset, &
                                      nshell=nshell, &
                                      l=l)
               nset_max = MAX(nset_max, nset)
               DO iset = 1, nset
                  nshell_max = MAX(nshell_max, nshell(iset))
               END DO
            ELSE IF (ASSOCIATED(dftb_parameter)) THEN
               CALL get_dftb_atom_param(dftb_parameter, lmax=lmax)
               nset_max = MAX(nset_max, 1)
               nshell_max = MAX(nshell_max, lmax + 1)
            ELSE
               CPABORT("Unknown basis type.")
            END IF
         END DO

         ALLOCATE (nso_info(nshell_max, nset_max, natom))
         nso_info(:, :, :) = 0

         ALLOCATE (nshell_info(nset_max, natom))
         nshell_info(:, :) = 0

         ALLOCATE (nset_info(natom))
         nset_info(:) = 0

         nao = 0
         DO iatom = 1, natom
            NULLIFY (orb_basis_set, dftb_parameter)
            CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
            CALL get_qs_kind(qs_kind_set(ikind), basis_type=basis_type_local, &
                             basis_set=orb_basis_set, dftb_parameter=dftb_parameter)
            IF (ASSOCIATED(orb_basis_set)) THEN
               CALL get_gto_basis_set(gto_basis_set=orb_basis_set, nsgf=nsgf, nset=nset, nshell=nshell, l=l)
               nao = nao + nsgf
               nset_info(iatom) = nset
               DO iset = 1, nset
                  nshell_info(iset, iatom) = nshell(iset)
                  DO ishell = 1, nshell(iset)
                     lshell = l(ishell, iset)
                     nso_info(ishell, iset, iatom) = nso(lshell)
                  END DO
               END DO
            ELSE IF (ASSOCIATED(dftb_parameter)) THEN
               CALL get_dftb_atom_param(dftb_parameter, lmax=lmax)
               nset_info(iatom) = 1
               nshell_info(1, iatom) = lmax + 1
               DO ishell = 1, lmax + 1
                  lshell = ishell - 1
                  nso_info(ishell, 1, iatom) = nso(lshell)
               END DO
               nao = nao + (lmax + 1)**2
            ELSE
               CPABORT("Unknown basis set type.")
            END IF
         END DO

         WRITE (ires) version
         WRITE (ires) natom, nao, nset_max, nshell_max
         WRITE (ires) nset_info
         WRITE (ires) nshell_info
         WRITE (ires) nso_info

         DEALLOCATE (nset_info, nshell_info, nso_info)
      END IF

      CALL timestop(handle)

   END SUBROUTINE write_kpoints_file_header

! **************************************************************************************************
!> \brief ...
!> \param denmat ...
!> \param kpoints ...
!> \param fmwork ...
!> \param natom ...
!> \param para_env ...
!> \param id_nr ...
!> \param dft_section ...
!> \param natom_mismatch ...
!> \param use_wfn ...
!> \param qs_kind_set ...
!> \param particle_set ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE read_kpoints_restart(denmat, kpoints, fmwork, natom, &
                                   para_env, id_nr, dft_section, natom_mismatch, &
                                   use_wfn, qs_kind_set, particle_set, qs_env)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(cp_fm_type), DIMENSION(:), INTENT(INOUT)      :: fmwork
      INTEGER, INTENT(IN)                                :: natom
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(IN)                                :: id_nr
      TYPE(section_vals_type), POINTER                   :: dft_section
      LOGICAL, INTENT(OUT)                               :: natom_mismatch
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_wfn
      TYPE(qs_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: qs_kind_set
      TYPE(particle_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: particle_set
      TYPE(qs_environment_type), OPTIONAL, POINTER       :: qs_env

      CHARACTER(LEN=*), PARAMETER :: routineN = 'read_kpoints_restart'

      CHARACTER(LEN=default_path_length)                 :: file_name
      INTEGER                                            :: handle, ic, ispin, ncol, nspin, &
                                                            restart_unit
      LOGICAL                                            :: do_wfn_expand, exist
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(mo_set_type), DIMENSION(:), POINTER           :: mo_set
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      TYPE(qs_scf_env_type), POINTER                     :: scf_env
      TYPE(section_vals_type), POINTER                   :: print_key

      CALL timeset(routineN, handle)
      logger => cp_get_default_logger()

      do_wfn_expand = .FALSE.
      IF (PRESENT(use_wfn)) do_wfn_expand = use_wfn

      IF (do_wfn_expand) THEN
         IF (.NOT. PRESENT(qs_kind_set) .OR. .NOT. PRESENT(particle_set)) THEN
            CPABORT("qs_kind_set and particle_set are required when use_wfn=.TRUE.")
         END IF
         IF (PRESENT(qs_env)) THEN
            CALL get_qs_env(qs_env, scf_env=scf_env, sab_orb=sab_orb, &
                            dft_control=dft_control)
         ELSE
            CALL get_qs_env(qs_env)
            CALL get_qs_env(qs_env, scf_env=scf_env, sab_orb=sab_orb, &
                            dft_control=dft_control)
         END IF

         print_key => section_vals_get_subs_vals(dft_section, "SCF%PRINT%RESTART")
         file_name = cp_print_key_generate_filename(logger, print_key, &
                                                    extension=".wfn", my_local=.FALSE.)
         INQUIRE (FILE=file_name, exist=exist)
         IF (exist) THEN

            ALLOCATE (mo_set(2))

            CALL read_mo_set_from_restart(mo_set, qs_kind_set, particle_set, para_env, &
                                          id_nr=id_nr, multiplicity=dft_control%multiplicity, &
                                          dft_section=dft_section, &
                                          natom_mismatch=natom_mismatch)

            nspin = SIZE(mo_set)
            DO ispin = 1, nspin
               CALL get_mo_set(mo_set(ispin), nmo=ncol)
               IF (ncol > 0) CALL make_basis_simple(mo_set(ispin)%mo_coeff, ncol)
               CALL calculate_density_matrix(mo_set(ispin), denmat(ispin, 1)%matrix)
            END DO

            CALL kpoint_initialize_mos(kpoints, mo_set)
            CALL kpoint_initialize_mo_set(kpoints)

            DO ispin = 1, nspin
               DO ic = 1, SIZE(kpoints%kp_range)
                  CALL cp_fm_to_fm(mo_set(ispin)%mo_coeff, &
                                   kpoints%kp_env(ic)%kpoint_env%mos(1, ispin)%mo_coeff)
                  IF (.NOT. kpoints%use_real_wfn) THEN
                     CALL cp_fm_to_fm(mo_set(ispin)%mo_coeff, &
                                      kpoints%kp_env(ic)%kpoint_env%mos(2, ispin)%mo_coeff)
                  END IF
               END DO
            END DO

            CALL kpoint_density_matrices(kpoints, energy_weighted=.FALSE.)

            CPASSERT(ASSOCIATED(scf_env%p_mix_new))
            CALL kpoint_density_transform(kpoints, scf_env%p_mix_new, .FALSE., &
                                          denmat(1, 1)%matrix, sab_orb, fmwork)

            DO ispin = 1, nspin
               DO ic = 1, SIZE(scf_env%p_mix_new, 2)
                  CALL dbcsr_copy(denmat(ispin, ic)%matrix, scf_env%p_mix_new(ispin, ic)%matrix)
               END DO
            END DO

            DEALLOCATE (mo_set)
            CALL timestop(handle)
            RETURN
         END IF
      END IF

      restart_unit = -1

      IF (para_env%is_source()) THEN

         CALL wfn_restart_file_name(file_name, exist, dft_section, logger, kp=.TRUE.)
         IF (id_nr /= 0) THEN
            ! Is it one of the backup files?
            file_name = TRIM(file_name)//".bak-"//ADJUSTL(cp_to_string(id_nr))
         END IF

         CALL open_file(file_name=file_name, &
                        file_action="READ", &
                        file_form="UNFORMATTED", &
                        file_status="OLD", &
                        unit_number=restart_unit)

      END IF

      CALL read_kpoints_restart_low(denmat, kpoints, fmwork(1), para_env, &
                                    natom, restart_unit, natom_mismatch)

      ! only the io_node returns natom_mismatch, must broadcast it
      CALL para_env%bcast(natom_mismatch)

      ! Close restart file
      IF (para_env%is_source()) CALL close_file(unit_number=restart_unit)

      CALL timestop(handle)

   END SUBROUTINE read_kpoints_restart

! **************************************************************************************************
!> \brief Reading the mos from apreviously defined restart file
!> \param denmat ...
!> \param kpoints ...
!> \param fmat ...
!> \param para_env ...
!> \param natom ...
!> \param rst_unit ...
!> \param natom_mismatch ...
!> \par History
!>      12.2007 created [MI]
!> \author MI
! **************************************************************************************************
   SUBROUTINE read_kpoints_restart_low(denmat, kpoints, fmat, para_env, &
                                       natom, rst_unit, natom_mismatch)

      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(cp_fm_type), INTENT(INOUT)                    :: fmat
      TYPE(mp_para_env_type), POINTER                    :: para_env
      INTEGER, INTENT(IN)                                :: natom, rst_unit
      LOGICAL, INTENT(OUT)                               :: natom_mismatch

      INTEGER                                            :: ic, image, ispin, nao, nao_read, &
                                                            natom_read, ni, nimages, nset_max, &
                                                            nshell_max, nspin, nspin_read, &
                                                            version_read
      INTEGER, DIMENSION(3)                              :: cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      LOGICAL                                            :: natom_match

      CPASSERT(ASSOCIATED(denmat))
      CALL dbcsr_get_info(denmat(1, 1)%matrix, nfullrows_total=nao)

      nspin = SIZE(denmat, 1)
      nimages = SIZE(denmat, 2)
      NULLIFY (cell_to_index)
      IF (nimages > 1) THEN
         CALL get_kpoint_info(kpoint=kpoints, cell_to_index=cell_to_index)
      END IF

      IF (para_env%is_source()) THEN
         READ (rst_unit) version_read
         IF (version_read /= version) THEN
            CALL cp_abort(__LOCATION__, &
                          " READ RESTART : Version of restart file not supported")
         END IF
         READ (rst_unit) natom_read, nao_read, nset_max, nshell_max
         natom_match = (natom_read == natom)
         IF (natom_match) THEN
            IF (nao_read /= nao) THEN
               CALL cp_abort(__LOCATION__, &
                             " READ RESTART : Different number of basis functions")
            END IF
            READ (rst_unit)
            READ (rst_unit)
            READ (rst_unit)
         END IF
      END IF

      ! make natom_match and natom_mismatch uniform across all nodes
      CALL para_env%bcast(natom_match)
      natom_mismatch = .NOT. natom_match
      ! handle natom_match false
      IF (natom_mismatch) THEN
         CALL cp_warn(__LOCATION__, &
                      " READ RESTART : WARNING : DIFFERENT natom, returning ")
      ELSE

         DO
            ispin = 0
            IF (para_env%is_source()) THEN
               READ (rst_unit) ispin, nspin_read, ni
            END IF
            CALL para_env%bcast(ispin)
            CALL para_env%bcast(nspin_read)
            CALL para_env%bcast(ni)
            IF (nspin_read /= nspin) THEN
               CALL cp_abort(__LOCATION__, &
                             " READ RESTART : Different spin polarisation not supported")
            END IF
            DO ic = 1, ni
               IF (para_env%is_source()) THEN
                  READ (rst_unit) image, cell
               END IF
               CALL para_env%bcast(image)
               CALL para_env%bcast(cell)
               !
               CALL cp_fm_read_unformatted(fmat, rst_unit)
               !
               IF (nimages > 1) THEN
                  image = get_index(cell, cell_to_index)
               ELSE IF (SUM(ABS(cell(:))) == 0) THEN
                  image = 1
               ELSE
                  image = 0
               END IF
               IF (image >= 1 .AND. image <= nimages) THEN
                  CALL copy_fm_to_dbcsr(fmat, denmat(ispin, image)%matrix)
               END IF
            END DO
            IF (ispin == nspin_read) EXIT
         END DO

      END IF

   END SUBROUTINE read_kpoints_restart_low

! **************************************************************************************************
!> \brief ...
!> \param cell ...
!> \param cell_to_index ...
!> \return ...
! **************************************************************************************************
   FUNCTION get_index(cell, cell_to_index) RESULT(index)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      INTEGER                                            :: index

      IF (CELL(1) >= LBOUND(cell_to_index, 1) .AND. CELL(1) <= UBOUND(cell_to_index, 1) .AND. &
          CELL(2) >= LBOUND(cell_to_index, 2) .AND. CELL(2) <= UBOUND(cell_to_index, 2) .AND. &
          CELL(3) >= LBOUND(cell_to_index, 3) .AND. CELL(3) <= UBOUND(cell_to_index, 3)) THEN

         index = cell_to_index(cell(1), cell(2), cell(3))

      ELSE

         index = 0

      END IF

   END FUNCTION get_index

END MODULE kpoint_io
