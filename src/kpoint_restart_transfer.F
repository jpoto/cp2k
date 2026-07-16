!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Versioned, metadata-validated density-matrix restarts for regular k-point meshes.
!>
!> The payload is a compact list of canonical Born--von Karman Fourier coefficients. In contrast to
!> the legacy k-point restart, it is independent of the neighbour-list image ordering.  The reader
!> is transactional: it constructs temporary P(k) and P(R) matrices and commits them only after all
!> compatibility and physical checks have passed.
! **************************************************************************************************
MODULE kpoint_restart_transfer

   USE atomic_kind_types,               ONLY: get_atomic_kind
   USE basis_set_types,                 ONLY: get_gto_basis_set,&
                                              gto_basis_set_type
   USE cell_types,                      ONLY: cell_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_copy,&
                                              dbcsr_create,&
                                              dbcsr_deallocate_matrix,&
                                              dbcsr_get_info,&
                                              dbcsr_p_type,&
                                              dbcsr_set
   USE cp_dbcsr_contrib,                ONLY: dbcsr_dot
   USE cp_fm_basic_linalg,              ONLY: cp_fm_scale
   USE cp_fm_struct,                    ONLY: cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_release,&
                                              cp_fm_set_submatrix,&
                                              cp_fm_to_fm,&
                                              cp_fm_type
   USE ieee_arithmetic,                 ONLY: ieee_is_finite
   USE input_constants,                 ONLY: kpoint_restart_exact,&
                                              kpoint_restart_legacy_compatible,&
                                              kpoint_restart_validated_bvk_transfer
   USE iso_fortran_env,                 ONLY: iostat_end
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE kpoint_k_r_trafo_simple,         ONLY: replicate_rs_matrices
   USE kpoint_methods,                  ONLY: calibrate_symmetry_phases,&
                                              kpoint_density_transform
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_env_type,&
                                              kpoint_sym_type,&
                                              kpoint_type
   USE mathconstants,                   ONLY: twopi
   USE message_passing,                 ONLY: mp_para_env_type
   USE orbital_pointers,                ONLY: nso
   USE particle_types,                  ONLY: particle_type
   USE qs_dftb_types,                   ONLY: qs_dftb_atom_type
   USE qs_dftb_utils,                   ONLY: get_dftb_atom_param
   USE qs_diis,                         ONLY: qs_diis_b_clear,&
                                              qs_diis_b_clear_kp
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE qs_scf_types,                    ONLY: qs_scf_env_type
   USE scf_control_types,               ONLY: scf_control_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'kpoint_restart_transfer'
   CHARACTER(len=32), PARAMETER         :: restart_magic = "CP2K_KPOINT_RESTART_V2"
   INTEGER, PARAMETER, PUBLIC           :: kpoint_restart_v2_version = 2
   REAL(KIND=dp), PARAMETER             :: metadata_tol = 1.0e-10_dp
   REAL(KIND=dp), PARAMETER             :: hermiticity_tol = 1.0e-10_dp
   REAL(KIND=dp), PARAMETER             :: electron_tol = 1.0e-10_dp
   REAL(KIND=dp), PARAMETER             :: electron_renormalization_tol = 1.0e-6_dp
   REAL(KIND=dp), PARAMETER             :: payload_zero_tol = 1.0e-14_dp
   REAL(KIND=dp), PARAMETER             :: alias_consistency_tol = 1.0e-10_dp

   TYPE phase_snapshot_type
      INTEGER, ALLOCATABLE, DIMENSION(:) :: mode, gauge
   END TYPE phase_snapshot_type

   PUBLIC :: read_kpoint_restart_v2, &
             write_kpoint_restart_v2
   PUBLIC :: canonical_bvk_cell

CONTAINS

! **************************************************************************************************
!> \brief Fold an image cell into the centred BvK Fourier domain and return its twist phase.
!> \param cell input image cell
!> \param nk grid dimensions
!> \param shift k-point shift
!> \param gamma_centered gamma-centred even meshes
!> \param folded canonical image cell
!> \param phase real and imaginary parts of the folding phase
! **************************************************************************************************
   PURE SUBROUTINE canonical_bvk_cell(cell, nk, shift, gamma_centered, folded, phase)
      INTEGER, DIMENSION(3), INTENT(IN)                  :: cell, nk
      REAL(KIND=dp), DIMENSION(3), INTENT(IN)            :: shift
      LOGICAL, INTENT(IN)                                :: gamma_centered
      INTEGER, DIMENSION(3), INTENT(OUT)                 :: folded
      COMPLEX(KIND=dp), INTENT(OUT)                      :: phase

      COMPLEX(KIND=dp)                                   :: twist
      INTEGER                                            :: idim, lower, quotient
      REAL(KIND=dp)                                      :: kfirst

      phase = CMPLX(1.0_dp, 0.0_dp, KIND=dp)
      DO idim = 1, 3
         lower = -nk(idim)/2
         folded(idim) = MODULO(cell(idim) - lower, nk(idim)) + lower
         quotient = (cell(idim) - folded(idim))/nk(idim)
         IF (gamma_centered .AND. MOD(nk(idim), 2) == 0) THEN
            kfirst = REAL(2 - nk(idim), dp)/(2.0_dp*REAL(nk(idim), dp)) + shift(idim)
         ELSE
            kfirst = REAL(1 - nk(idim), dp)/(2.0_dp*REAL(nk(idim), dp)) + shift(idim)
         END IF
         twist = CMPLX(COS(twopi*REAL(nk(idim), dp)*kfirst), &
                       SIN(twopi*REAL(nk(idim), dp)*kfirst), KIND=dp)
         phase = phase*twist**quotient
      END DO
   END SUBROUTINE canonical_bvk_cell

! **************************************************************************************************
!> \brief Write a version-2 canonical BvK density restart to an already opened unit.
!> \param ires ...
!> \param denmat ...
!> \param kpoints ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param cell ...
!> \param dft_control ...
!> \param scf_control ...
!> \param nelectron_spin ...
! **************************************************************************************************
   SUBROUTINE write_kpoint_restart_v2(ires, denmat, kpoints, particle_set, qs_kind_set, cell, &
                                      dft_control, scf_control, nelectron_spin)
      INTEGER, INTENT(IN)                                :: ires
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(scf_control_type), POINTER                    :: scf_control
      INTEGER, DIMENSION(2), INTENT(IN)                  :: nelectron_spin

      CHARACTER(LEN=*), PARAMETER :: routineN = 'write_kpoint_restart_v2'

      CHARACTER(LEN=default_string_length)               :: kp_scheme
      COMPLEX(KIND=dp)                                   :: phase
      INTEGER :: handle, i, iatom, ic, ikind, irecord, ispin, j, nao, natom, nimages, nrecords, &
         nset_max, nshell_max, nspin, smear_method, symmetry_backend, symmetry_reduction_method
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: image_to_record, kind_index, nset_info
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: cells, full_index_to_cell, nshell_info
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: nso_info
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :, :)        :: canonical_count
      INTEGER, DIMENSION(3)                              :: folded, lower_cell, nkp_grid, upper_cell
      INTEGER, DIMENSION(:, :), POINTER                  :: index_to_cell
      INTEGER, DIMENSION(:, :, :), POINTER               :: full_cell_to_index
      LOGICAL                                            :: full_grid, gamma_centered, &
                                                            inversion_symmetry_only, smear, &
                                                            symmetry
      REAL(KIND=dp) :: alias_residual, alias_scale, checksum_limit, checksum_linear, &
         checksum_square, electronic_temperature, eps_geo, payload_max_abs, smearing_width, &
         value_i, value_r
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: positions
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :, :)  :: canonical_i, canonical_r, raw
      REAL(KIND=dp), DIMENSION(3)                        :: kp_shift

      CALL timeset(routineN, handle)

      CPASSERT(ASSOCIATED(denmat))
      CPASSERT(ASSOCIATED(kpoints))
      CPASSERT(ASSOCIATED(cell))
      CPASSERT(ASSOCIATED(dft_control))
      CPASSERT(ASSOCIATED(scf_control))

      nspin = SIZE(denmat, 1)
      nimages = SIZE(denmat, 2)
      natom = SIZE(particle_set)
      CALL dbcsr_get_info(denmat(1, 1)%matrix, nfullrows_total=nao)
      CALL get_kpoint_info(kpoints, kp_scheme=kp_scheme, nkp_grid=nkp_grid, kp_shift=kp_shift, &
                           gamma_centered=gamma_centered, index_to_cell=index_to_cell, &
                           symmetry=symmetry, full_grid=full_grid, &
                           inversion_symmetry_only=inversion_symmetry_only, eps_geo=eps_geo, &
                           symmetry_backend=symmetry_backend, &
                           symmetry_reduction_method=symmetry_reduction_method)
      CPASSERT(ALL(nkp_grid > 0))
      CPASSERT(SIZE(index_to_cell, 2) == nimages)
      IF (TRIM(kp_scheme) /= "MONKHORST-PACK" .AND. TRIM(kp_scheme) /= "MACDONALD") &
         CPABORT("VALIDATED_BVK_TRANSFER requires a MONKHORST-PACK or MACDONALD regular mesh")

      CALL collect_basis_metadata(qs_kind_set, particle_set, "ORB", nao, nset_max, nshell_max, &
                                  nset_info, nshell_info, nso_info)

      ! The compact DBCSR representation may store an (atom i, atom j, R) block only once and
      ! recover its transpose at -R.  The ordinary cell_to_index map can omit that mirror image
      ! (notably with k-point symmetry reduction), whereas replicate_rs_matrices needs a target
      ! slot for it.  Build a temporary union of every stored image and its inverse; this changes
      ! neither the live neighbour list nor the production density matrices.
      lower_cell = MIN(MINVAL(index_to_cell, DIM=2), -MAXVAL(index_to_cell, DIM=2))
      upper_cell = MAX(MAXVAL(index_to_cell, DIM=2), -MINVAL(index_to_cell, DIM=2))
      ALLOCATE (full_cell_to_index(lower_cell(1):upper_cell(1), &
                                   lower_cell(2):upper_cell(2), &
                                   lower_cell(3):upper_cell(3)))
      full_cell_to_index = 0
      ALLOCATE (full_index_to_cell(3, 2*nimages))
      nimages = 0
      DO ic = 1, SIZE(index_to_cell, 2)
         IF (full_cell_to_index(index_to_cell(1, ic), index_to_cell(2, ic), &
                                index_to_cell(3, ic)) == 0) THEN
            nimages = nimages + 1
            full_cell_to_index(index_to_cell(1, ic), index_to_cell(2, ic), &
                               index_to_cell(3, ic)) = nimages
            full_index_to_cell(:, nimages) = index_to_cell(:, ic)
         END IF
         IF (full_cell_to_index(-index_to_cell(1, ic), -index_to_cell(2, ic), &
                                -index_to_cell(3, ic)) == 0) THEN
            nimages = nimages + 1
            full_cell_to_index(-index_to_cell(1, ic), -index_to_cell(2, ic), &
                               -index_to_cell(3, ic)) = nimages
            full_index_to_cell(:, nimages) = -index_to_cell(:, ic)
         END IF
      END DO

      ALLOCATE (raw(nspin, nao, nao, nimages))
      raw = 0.0_dp
      CALL replicate_rs_matrices(denmat, kpoints, raw, full_cell_to_index)

      ALLOCATE (cells(3, nimages), image_to_record(nimages))
      cells = 0
      image_to_record = 0
      nrecords = 0
      DO ic = 1, nimages
         CALL canonical_bvk_cell(full_index_to_cell(:, ic), nkp_grid, kp_shift, gamma_centered, folded, phase)
         irecord = 0
         DO j = 1, nrecords
            IF (ALL(cells(:, j) == folded)) THEN
               irecord = j
               EXIT
            END IF
         END DO
         IF (irecord == 0) THEN
            nrecords = nrecords + 1
            cells(:, nrecords) = folded
            irecord = nrecords
         END IF
         image_to_record(ic) = irecord
      END DO

      ALLOCATE (canonical_r(nspin, nao, nao, nrecords), canonical_i(nspin, nao, nao, nrecords), &
                canonical_count(nspin, nao, nao, nrecords))
      canonical_r = 0.0_dp
      canonical_i = 0.0_dp
      canonical_count = 0
      alias_residual = 0.0_dp
      alias_scale = 0.0_dp
      DO ic = 1, nimages
         CALL canonical_bvk_cell(full_index_to_cell(:, ic), nkp_grid, kp_shift, gamma_centered, folded, phase)
         irecord = image_to_record(ic)
         DO ispin = 1, nspin
            DO j = 1, nao
               DO i = 1, nao
                  IF (raw(ispin, i, j, ic) == 0.0_dp) CYCLE
                  value_r = REAL(phase, dp)*raw(ispin, i, j, ic)
                  value_i = AIMAG(phase)*raw(ispin, i, j, ic)
                  IF (canonical_count(ispin, i, j, irecord) > 0) THEN
                     alias_residual = MAX(alias_residual, &
                                          ABS(value_r - canonical_r(ispin, i, j, irecord)/ &
                                              REAL(canonical_count(ispin, i, j, irecord), dp)), &
                                          ABS(value_i - canonical_i(ispin, i, j, irecord)/ &
                                              REAL(canonical_count(ispin, i, j, irecord), dp)))
                  END IF
                  alias_scale = MAX(alias_scale, ABS(value_r), ABS(value_i))
                  canonical_r(ispin, i, j, irecord) = &
                     canonical_r(ispin, i, j, irecord) + value_r
                  canonical_i(ispin, i, j, irecord) = &
                     canonical_i(ispin, i, j, irecord) + value_i
                  canonical_count(ispin, i, j, irecord) = &
                     canonical_count(ispin, i, j, irecord) + 1
               END DO
            END DO
         END DO
      END DO
      DEALLOCATE (raw, image_to_record, full_index_to_cell, full_cell_to_index)
      IF (alias_residual > alias_consistency_tol*MAX(1.0_dp, alias_scale)) &
         CPABORT("Inconsistent modulo-grid aliases in k-point restart density")
      WHERE (canonical_count > 0)
         canonical_r = canonical_r/REAL(canonical_count, dp)
         canonical_i = canonical_i/REAL(canonical_count, dp)
      END WHERE
      DEALLOCATE (canonical_count)

      IF (.NOT. ALL(ieee_is_finite(canonical_r)) .OR. &
          .NOT. ALL(ieee_is_finite(canonical_i))) &
         CPABORT("Refusing to write a non-finite k-point density restart")
      payload_max_abs = MAX(MAXVAL(ABS(canonical_r)), MAXVAL(ABS(canonical_i)))
      checksum_limit = SQRT(HUGE(1.0_dp)/(REAL(SIZE(canonical_r), dp) + &
                                          REAL(SIZE(canonical_i), dp)))
      IF (payload_max_abs > checksum_limit) &
         CPABORT("Refusing to write a k-point restart whose checksum would overflow")
      checksum_linear = SUM(canonical_r) + SUM(canonical_i)
      checksum_square = SUM(canonical_r*canonical_r) + SUM(canonical_i*canonical_i)
      IF (.NOT. ieee_is_finite(checksum_linear) .OR. &
          .NOT. ieee_is_finite(checksum_square)) &
         CPABORT("Refusing to write a k-point restart with an overflowing checksum")

      ALLOCATE (positions(3, natom), kind_index(natom))
      DO iatom = 1, natom
         positions(:, iatom) = particle_set(iatom)%r
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
         kind_index(iatom) = ikind
      END DO

      smear = .FALSE.
      smear_method = -1
      electronic_temperature = -1.0_dp
      smearing_width = -1.0_dp
      IF (ASSOCIATED(scf_control%smear)) THEN
         smear = scf_control%smear%do_smear
         smear_method = scf_control%smear%method
         electronic_temperature = scf_control%smear%electronic_temperature
         smearing_width = scf_control%smear%smearing_width
      END IF

      IF (ires > 0) THEN
         WRITE (ires) kpoint_restart_v2_version
         WRITE (ires) natom, nao, nset_max, nshell_max
         WRITE (ires) nset_info
         WRITE (ires) nshell_info
         WRITE (ires) nso_info
         CALL write_basis_fingerprint(ires, qs_kind_set, particle_set)
         WRITE (ires) restart_magic
         WRITE (ires) kp_scheme
         WRITE (ires) nkp_grid, kp_shift, gamma_centered
         WRITE (ires) symmetry, full_grid, inversion_symmetry_only, symmetry_backend, &
            symmetry_reduction_method, eps_geo
         WRITE (ires) cell%perd
         WRITE (ires) cell%hmat
         WRITE (ires) positions
         WRITE (ires) kind_index
         WRITE (ires) dft_control%charge, dft_control%multiplicity, nspin, nelectron_spin
         WRITE (ires) smear, smear_method, electronic_temperature, smearing_width
         WRITE (ires) nrecords, checksum_linear, checksum_square
         DO ispin = 1, nspin
            DO irecord = 1, nrecords
               WRITE (ires) ispin, irecord, cells(:, irecord)
               DO j = 1, nao
                  WRITE (ires) canonical_r(ispin, :, j, irecord)
               END DO
               DO j = 1, nao
                  WRITE (ires) canonical_i(ispin, :, j, irecord)
               END DO
            END DO
         END DO
      END IF

      DEALLOCATE (canonical_r, canonical_i, cells, positions, kind_index)
      DEALLOCATE (nset_info, nshell_info, nso_info)
      CALL timestop(handle)
   END SUBROUTINE write_kpoint_restart_v2

! **************************************************************************************************
!> \brief Collect the exact atom-resolved orbital topology stored in a restart header.
!> \param qs_kind_set ...
!> \param particle_set ...
!> \param basis_type ...
!> \param nao ...
!> \param nset_max ...
!> \param nshell_max ...
!> \param nset_info ...
!> \param nshell_info ...
!> \param nso_info ...
! **************************************************************************************************
   SUBROUTINE collect_basis_metadata(qs_kind_set, particle_set, basis_type, nao, nset_max, nshell_max, &
                                     nset_info, nshell_info, nso_info)
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type
      INTEGER, INTENT(OUT)                               :: nao, nset_max, nshell_max
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT)    :: nset_info
      INTEGER, ALLOCATABLE, DIMENSION(:, :), INTENT(OUT) :: nshell_info
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :), &
         INTENT(OUT)                                     :: nso_info

      INTEGER                                            :: iatom, ikind, iset, ishell, lmax, &
                                                            lshell, natom, nset, nsgf
      INTEGER, DIMENSION(:), POINTER                     :: nshell
      INTEGER, DIMENSION(:, :), POINTER                  :: l
      TYPE(gto_basis_set_type), POINTER                  :: orb_basis_set
      TYPE(qs_dftb_atom_type), POINTER                   :: dftb_parameter

      natom = SIZE(particle_set)
      nset_max = 0
      nshell_max = 0
      DO iatom = 1, natom
         NULLIFY (orb_basis_set, dftb_parameter)
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
         CALL get_qs_kind(qs_kind_set(ikind), basis_type=basis_type, basis_set=orb_basis_set, &
                          dftb_parameter=dftb_parameter)
         IF (ASSOCIATED(orb_basis_set)) THEN
            CALL get_gto_basis_set(orb_basis_set, nset=nset, nshell=nshell, l=l)
            nset_max = MAX(nset_max, nset)
            DO iset = 1, nset
               nshell_max = MAX(nshell_max, nshell(iset))
            END DO
         ELSEIF (ASSOCIATED(dftb_parameter)) THEN
            CALL get_dftb_atom_param(dftb_parameter, lmax=lmax)
            nset_max = MAX(nset_max, 1)
            nshell_max = MAX(nshell_max, lmax + 1)
         ELSE
            CPABORT("Unknown basis type in k-point restart metadata")
         END IF
      END DO

      ALLOCATE (nset_info(natom), nshell_info(nset_max, natom), &
                nso_info(nshell_max, nset_max, natom))
      nset_info = 0
      nshell_info = 0
      nso_info = 0
      nao = 0
      DO iatom = 1, natom
         NULLIFY (orb_basis_set, dftb_parameter)
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=ikind)
         CALL get_qs_kind(qs_kind_set(ikind), basis_type=basis_type, basis_set=orb_basis_set, &
                          dftb_parameter=dftb_parameter)
         IF (ASSOCIATED(orb_basis_set)) THEN
            CALL get_gto_basis_set(orb_basis_set, nsgf=nsgf, nset=nset, nshell=nshell, l=l)
            nao = nao + nsgf
            nset_info(iatom) = nset
            DO iset = 1, nset
               nshell_info(iset, iatom) = nshell(iset)
               DO ishell = 1, nshell(iset)
                  lshell = l(ishell, iset)
                  nso_info(ishell, iset, iatom) = nso(lshell)
               END DO
            END DO
         ELSE
            CALL get_dftb_atom_param(dftb_parameter, lmax=lmax)
            nset_info(iatom) = 1
            nshell_info(1, iatom) = lmax + 1
            DO ishell = 1, lmax + 1
               nso_info(ishell, 1, iatom) = nso(ishell - 1)
            END DO
            nao = nao + (lmax + 1)**2
         END IF
      END DO
   END SUBROUTINE collect_basis_metadata

! **************************************************************************************************
!> \brief Serialize every species field and every component of the in-memory orbital basis.
!>        This is intentionally a full fingerprint rather than a name or dimensional checksum:
!>        two bases with equal shell topology but different exponents or contractions must not
!>        share a density-matrix restart.
!> \param ires ...
!> \param qs_kind_set ...
!> \param particle_set ...
! **************************************************************************************************
   SUBROUTINE write_basis_fingerprint(ires, qs_kind_set, particle_set)
      INTEGER, INTENT(IN)                                :: ires
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set

      CHARACTER(LEN=12), DIMENSION(:), POINTER           :: cgf_symbol
      CHARACTER(LEN=2)                                   :: element_symbol, qs_element_symbol
      CHARACTER(LEN=6), DIMENSION(:), POINTER            :: sgf_symbol
      CHARACTER(LEN=default_string_length)               :: aliases, atomic_name, basis_name, &
                                                            dftb_name, dftb_type, qs_name
      INTEGER                                            :: basis_kind, iatom, ikind, lmax_dftb, &
                                                            natorb_dftb, ncgf, norm_type, nset, &
                                                            nsgf, z, z_dftb
      INTEGER, DIMENSION(:), POINTER                     :: lmax, lmin, lx, ly, lz, m, ncgf_set, &
                                                            npgf, nsgf_set, nshell
      INTEGER, DIMENSION(:, :), POINTER                  :: first_cgf, first_sgf, l, last_cgf, &
                                                            last_sgf, n
      LOGICAL                                            :: defined, floating, ghost, monovalent
      REAL(KIND=dp)                                      :: cutoff, di, dudq, energy, kind_radius, &
                                                            mass, qs_zeff, rcdisp, &
                                                            short_kind_radius, xi, zeff_dftb
      REAL(KIND=dp), DIMENSION(0:3)                      :: eta, occupation, skself
      REAL(KIND=dp), DIMENSION(:), POINTER               :: norm_cgf, set_radius
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: ccon, cphi, pgf_radius, scon, sphi, zet
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: gcc
      TYPE(gto_basis_set_type), POINTER                  :: orb_basis_set
      TYPE(qs_dftb_atom_type), POINTER                   :: dftb_parameter

      DO iatom = 1, SIZE(particle_set)
         NULLIFY (orb_basis_set, dftb_parameter, cgf_symbol, sgf_symbol, norm_cgf, set_radius, &
                  lmax, lmin, lx, ly, lz, m, ncgf_set, npgf, nsgf_set, nshell, first_cgf, &
                  first_sgf, l, last_cgf, last_sgf, n, ccon, cphi, pgf_radius, scon, sphi, zet, gcc)
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, element_symbol=element_symbol, &
                              name=atomic_name, mass=mass, kind_number=ikind, z=z)
         CALL get_qs_kind(qs_kind_set(ikind), basis_type="ORB", basis_set=orb_basis_set, &
                          dftb_parameter=dftb_parameter, name=qs_name, &
                          element_symbol=qs_element_symbol, zeff=qs_zeff, ghost=ghost, &
                          monovalent=monovalent, floating=floating)
         WRITE (ires) iatom, z, element_symbol, atomic_name, mass, qs_name, qs_element_symbol, &
            qs_zeff, ghost, monovalent, floating

         IF (ASSOCIATED(orb_basis_set)) THEN
            basis_kind = 1
            WRITE (ires) basis_kind
            CALL get_gto_basis_set(orb_basis_set, name=basis_name, aliases=aliases, &
                                   norm_type=norm_type, kind_radius=kind_radius, &
                                   short_kind_radius=short_kind_radius, ncgf=ncgf, nset=nset, nsgf=nsgf, &
                                   cgf_symbol=cgf_symbol, sgf_symbol=sgf_symbol, norm_cgf=norm_cgf, &
                                   set_radius=set_radius, lmax=lmax, lmin=lmin, lx=lx, ly=ly, lz=lz, &
                                   m=m, ncgf_set=ncgf_set, npgf=npgf, nsgf_set=nsgf_set, &
                                   nshell=nshell, cphi=cphi, pgf_radius=pgf_radius, sphi=sphi, &
                                   scon=scon, zet=zet, first_cgf=first_cgf, first_sgf=first_sgf, &
                                   l=l, last_cgf=last_cgf, last_sgf=last_sgf, n=n, gcc=gcc, ccon=ccon)
            WRITE (ires) basis_name, aliases, norm_type, kind_radius, short_kind_radius, &
               ncgf, nset, nsgf
            CALL write_character_1d(ires, cgf_symbol)
            CALL write_character_1d(ires, sgf_symbol)
            CALL write_real_1d(ires, norm_cgf)
            CALL write_real_1d(ires, set_radius)
            CALL write_integer_1d(ires, lmax)
            CALL write_integer_1d(ires, lmin)
            CALL write_integer_1d(ires, lx)
            CALL write_integer_1d(ires, ly)
            CALL write_integer_1d(ires, lz)
            CALL write_integer_1d(ires, m)
            CALL write_integer_1d(ires, ncgf_set)
            CALL write_integer_1d(ires, npgf)
            CALL write_integer_1d(ires, nsgf_set)
            CALL write_integer_1d(ires, nshell)
            CALL write_real_2d(ires, cphi)
            CALL write_real_2d(ires, pgf_radius)
            CALL write_real_2d(ires, sphi)
            CALL write_real_2d(ires, scon)
            CALL write_real_2d(ires, zet)
            CALL write_real_2d(ires, ccon)
            CALL write_integer_2d(ires, first_cgf)
            CALL write_integer_2d(ires, first_sgf)
            CALL write_integer_2d(ires, l)
            CALL write_integer_2d(ires, last_cgf)
            CALL write_integer_2d(ires, last_sgf)
            CALL write_integer_2d(ires, n)
            CALL write_real_3d(ires, gcc)
         ELSEIF (ASSOCIATED(dftb_parameter)) THEN
            basis_kind = 2
            WRITE (ires) basis_kind
            CALL get_dftb_atom_param(dftb_parameter, name=dftb_name, typ=dftb_type, &
                                     defined=defined, z=z_dftb, zeff=zeff_dftb, &
                                     natorb=natorb_dftb, lmax=lmax_dftb, &
                                     skself=skself, occupation=occupation, eta=eta, energy=energy, &
                                     cutoff=cutoff, xi=xi, di=di, rcdisp=rcdisp, dudq=dudq)
            WRITE (ires) dftb_name, dftb_type, defined, z_dftb, zeff_dftb, natorb_dftb, lmax_dftb, &
               skself, occupation, eta, energy, cutoff, xi, di, rcdisp, dudq
         ELSE
            CPABORT("Unknown orbital basis in k-point restart fingerprint")
         END IF
      END DO
   END SUBROUTINE write_basis_fingerprint

! **************************************************************************************************
!> \brief Read a complete basis/species fingerprint and compare it with the live calculation.
!> \param rst_unit ...
!> \param qs_kind_set ...
!> \param particle_set ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE validate_basis_fingerprint(rst_unit, qs_kind_set, particle_set, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      LOGICAL, INTENT(OUT)                               :: matches
      INTEGER, INTENT(OUT)                               :: ios

      CHARACTER(LEN=12), DIMENSION(:), POINTER           :: cgf_symbol
      CHARACTER(LEN=2)                                   :: element_symbol, element_symbol_file, &
                                                            qs_element_symbol, &
                                                            qs_element_symbol_file
      CHARACTER(LEN=6), DIMENSION(:), POINTER            :: sgf_symbol
      CHARACTER(LEN=default_string_length) :: aliases, aliases_file, atomic_name, &
         atomic_name_file, basis_name, basis_name_file, dftb_name, dftb_name_file, dftb_type, &
         dftb_type_file, qs_name, qs_name_file
      INTEGER :: basis_kind, basis_kind_file, iatom, iatom_file, ikind, lmax_dftb, lmax_dftb_file, &
         natorb_dftb, natorb_dftb_file, ncgf, ncgf_file, norm_type, norm_type_file, nset, &
         nset_file, nsgf, nsgf_file, z, z_dftb, z_dftb_file, z_file
      INTEGER, DIMENSION(:), POINTER                     :: lmax, lmin, lx, ly, lz, m, ncgf_set, &
                                                            npgf, nsgf_set, nshell
      INTEGER, DIMENSION(:, :), POINTER                  :: first_cgf, first_sgf, l, last_cgf, &
                                                            last_sgf, n
      LOGICAL                                            :: defined, defined_file, floating, &
                                                            floating_file, ghost, ghost_file, &
                                                            monovalent, monovalent_file
      REAL(KIND=dp) :: cutoff, cutoff_file, di, di_file, dudq, dudq_file, energy, energy_file, &
         kind_radius, kind_radius_file, mass, mass_file, qs_zeff, qs_zeff_file, rcdisp, &
         rcdisp_file, short_kind_radius, short_kind_radius_file, xi, xi_file, zeff_dftb, &
         zeff_dftb_file
      REAL(KIND=dp), DIMENSION(0:3)                      :: eta, eta_file, occupation, &
                                                            occupation_file, skself, skself_file
      REAL(KIND=dp), DIMENSION(:), POINTER               :: norm_cgf, set_radius
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: ccon, cphi, pgf_radius, scon, sphi, zet
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: gcc
      TYPE(gto_basis_set_type), POINTER                  :: orb_basis_set
      TYPE(qs_dftb_atom_type), POINTER                   :: dftb_parameter

      matches = .TRUE.
      ios = 0
      DO iatom = 1, SIZE(particle_set)
         NULLIFY (orb_basis_set, dftb_parameter, cgf_symbol, sgf_symbol, norm_cgf, set_radius, &
                  lmax, lmin, lx, ly, lz, m, ncgf_set, npgf, nsgf_set, nshell, first_cgf, &
                  first_sgf, l, last_cgf, last_sgf, n, ccon, cphi, pgf_radius, scon, sphi, zet, gcc)
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, element_symbol=element_symbol, &
                              name=atomic_name, mass=mass, kind_number=ikind, z=z)
         CALL get_qs_kind(qs_kind_set(ikind), basis_type="ORB", basis_set=orb_basis_set, &
                          dftb_parameter=dftb_parameter, name=qs_name, &
                          element_symbol=qs_element_symbol, zeff=qs_zeff, ghost=ghost, &
                          monovalent=monovalent, floating=floating)
         READ (rst_unit, IOSTAT=ios) iatom_file, z_file, element_symbol_file, atomic_name_file, &
            mass_file, qs_name_file, qs_element_symbol_file, qs_zeff_file, ghost_file, &
            monovalent_file, floating_file
         IF (ios /= 0) RETURN
         matches = matches .AND. iatom_file == iatom .AND. z_file == z .AND. &
                   element_symbol_file == element_symbol .AND. atomic_name_file == atomic_name .AND. &
                   mass_file == mass .AND. qs_name_file == qs_name .AND. &
                   qs_element_symbol_file == qs_element_symbol .AND. qs_zeff_file == qs_zeff .AND. &
                   (ghost_file .EQV. ghost) .AND. (monovalent_file .EQV. monovalent) .AND. &
                   (floating_file .EQV. floating)

         basis_kind = 0
         IF (ASSOCIATED(orb_basis_set)) THEN
            basis_kind = 1
         ELSEIF (ASSOCIATED(dftb_parameter)) THEN
            basis_kind = 2
         END IF
         READ (rst_unit, IOSTAT=ios) basis_kind_file
         IF (ios /= 0) RETURN
         matches = matches .AND. basis_kind_file == basis_kind

         SELECT CASE (basis_kind_file)
         CASE (1)
            basis_name = ""
            aliases = ""
            norm_type = -1
            kind_radius = 0.0_dp
            short_kind_radius = 0.0_dp
            ncgf = -1
            nset = -1
            nsgf = -1
            IF (ASSOCIATED(orb_basis_set)) THEN
               CALL get_gto_basis_set(orb_basis_set, name=basis_name, aliases=aliases, &
                                      norm_type=norm_type, kind_radius=kind_radius, &
                                      short_kind_radius=short_kind_radius, ncgf=ncgf, nset=nset, nsgf=nsgf, &
                                      cgf_symbol=cgf_symbol, sgf_symbol=sgf_symbol, norm_cgf=norm_cgf, &
                                      set_radius=set_radius, lmax=lmax, lmin=lmin, lx=lx, ly=ly, lz=lz, &
                                      m=m, ncgf_set=ncgf_set, npgf=npgf, nsgf_set=nsgf_set, &
                                      nshell=nshell, cphi=cphi, pgf_radius=pgf_radius, sphi=sphi, &
                                      scon=scon, zet=zet, first_cgf=first_cgf, first_sgf=first_sgf, &
                                      l=l, last_cgf=last_cgf, last_sgf=last_sgf, n=n, gcc=gcc, ccon=ccon)
            END IF
            READ (rst_unit, IOSTAT=ios) basis_name_file, aliases_file, norm_type_file, &
               kind_radius_file, short_kind_radius_file, ncgf_file, nset_file, nsgf_file
            IF (ios /= 0) RETURN
            matches = matches .AND. basis_name_file == basis_name .AND. aliases_file == aliases .AND. &
                      norm_type_file == norm_type .AND. kind_radius_file == kind_radius .AND. &
                      short_kind_radius_file == short_kind_radius .AND. ncgf_file == ncgf .AND. &
                      nset_file == nset .AND. nsgf_file == nsgf
            CALL compare_character_1d(rst_unit, cgf_symbol, matches, ios)
            CALL compare_character_1d(rst_unit, sgf_symbol, matches, ios)
            CALL compare_real_1d(rst_unit, norm_cgf, matches, ios)
            CALL compare_real_1d(rst_unit, set_radius, matches, ios)
            CALL compare_integer_1d(rst_unit, lmax, matches, ios)
            CALL compare_integer_1d(rst_unit, lmin, matches, ios)
            CALL compare_integer_1d(rst_unit, lx, matches, ios)
            CALL compare_integer_1d(rst_unit, ly, matches, ios)
            CALL compare_integer_1d(rst_unit, lz, matches, ios)
            CALL compare_integer_1d(rst_unit, m, matches, ios)
            CALL compare_integer_1d(rst_unit, ncgf_set, matches, ios)
            CALL compare_integer_1d(rst_unit, npgf, matches, ios)
            CALL compare_integer_1d(rst_unit, nsgf_set, matches, ios)
            CALL compare_integer_1d(rst_unit, nshell, matches, ios)
            CALL compare_real_2d(rst_unit, cphi, matches, ios)
            CALL compare_real_2d(rst_unit, pgf_radius, matches, ios)
            CALL compare_real_2d(rst_unit, sphi, matches, ios)
            CALL compare_real_2d(rst_unit, scon, matches, ios)
            CALL compare_real_2d(rst_unit, zet, matches, ios)
            CALL compare_real_2d(rst_unit, ccon, matches, ios)
            CALL compare_integer_2d(rst_unit, first_cgf, matches, ios)
            CALL compare_integer_2d(rst_unit, first_sgf, matches, ios)
            CALL compare_integer_2d(rst_unit, l, matches, ios)
            CALL compare_integer_2d(rst_unit, last_cgf, matches, ios)
            CALL compare_integer_2d(rst_unit, last_sgf, matches, ios)
            CALL compare_integer_2d(rst_unit, n, matches, ios)
            CALL compare_real_3d(rst_unit, gcc, matches, ios)
            IF (ios /= 0) RETURN
         CASE (2)
            dftb_name = ""
            dftb_type = ""
            defined = .FALSE.
            z_dftb = -1
            zeff_dftb = -1.0_dp
            natorb_dftb = -1
            lmax_dftb = -1
            skself = -1.0_dp
            occupation = -1.0_dp
            eta = -1.0_dp
            energy = -1.0_dp
            cutoff = -1.0_dp
            xi = -1.0_dp
            di = -1.0_dp
            rcdisp = -1.0_dp
            dudq = -1.0_dp
            IF (ASSOCIATED(dftb_parameter)) &
               CALL get_dftb_atom_param(dftb_parameter, name=dftb_name, typ=dftb_type, &
                                        defined=defined, z=z_dftb, zeff=zeff_dftb, &
                                        natorb=natorb_dftb, lmax=lmax_dftb, &
                                        skself=skself, occupation=occupation, eta=eta, energy=energy, &
                                        cutoff=cutoff, xi=xi, di=di, rcdisp=rcdisp, dudq=dudq)
            READ (rst_unit, IOSTAT=ios) dftb_name_file, dftb_type_file, defined_file, z_dftb_file, &
               zeff_dftb_file, natorb_dftb_file, lmax_dftb_file, skself_file, occupation_file, &
               eta_file, energy_file, &
               cutoff_file, xi_file, di_file, rcdisp_file, dudq_file
            IF (ios /= 0) RETURN
            matches = matches .AND. dftb_name_file == dftb_name .AND. &
                      dftb_type_file == dftb_type .AND. (defined_file .EQV. defined) .AND. &
                      z_dftb_file == z_dftb .AND. zeff_dftb_file == zeff_dftb .AND. &
                      natorb_dftb_file == natorb_dftb .AND. lmax_dftb_file == lmax_dftb .AND. &
                      ALL(skself_file == skself) .AND. &
                      ALL(occupation_file == occupation) .AND. ALL(eta_file == eta) .AND. &
                      energy_file == energy .AND. cutoff_file == cutoff .AND. xi_file == xi .AND. &
                      di_file == di .AND. rcdisp_file == rcdisp .AND. dudq_file == dudq
         CASE DEFAULT
            ios = -1
            RETURN
         END SELECT
      END DO
   END SUBROUTINE validate_basis_fingerprint

! **************************************************************************************************
!> \brief Write a pointer array together with association and shape metadata.
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_character_1d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN), &
         POINTER                                         :: array

      INTEGER                                            :: n1
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         n1 = SIZE(array, 1)
         WRITE (ires) n1
         WRITE (ires) array
      END IF
   END SUBROUTINE write_character_1d

! **************************************************************************************************
!> \brief ...
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_integer_1d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      INTEGER, DIMENSION(:), INTENT(IN), POINTER         :: array

      INTEGER                                            :: n1
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         n1 = SIZE(array, 1)
         WRITE (ires) n1
         WRITE (ires) array
      END IF
   END SUBROUTINE write_integer_1d

! **************************************************************************************************
!> \brief ...
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_integer_2d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      INTEGER, DIMENSION(:, :), INTENT(IN), POINTER      :: array

      INTEGER, DIMENSION(2)                              :: dims
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         dims = SHAPE(array)
         WRITE (ires) dims
         WRITE (ires) array
      END IF
   END SUBROUTINE write_integer_2d

! **************************************************************************************************
!> \brief ...
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_real_1d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      REAL(KIND=dp), DIMENSION(:), INTENT(IN), POINTER   :: array

      INTEGER                                            :: n1
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         n1 = SIZE(array, 1)
         WRITE (ires) n1
         WRITE (ires) array
      END IF
   END SUBROUTINE write_real_1d

! **************************************************************************************************
!> \brief ...
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_real_2d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN), &
         POINTER                                         :: array

      INTEGER, DIMENSION(2)                              :: dims
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         dims = SHAPE(array)
         WRITE (ires) dims
         WRITE (ires) array
      END IF
   END SUBROUTINE write_real_2d

! **************************************************************************************************
!> \brief ...
!> \param ires ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE write_real_3d(ires, array)
      INTEGER, INTENT(IN)                                :: ires
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN), &
         POINTER                                         :: array

      INTEGER, DIMENSION(3)                              :: dims
      LOGICAL                                            :: present_array

      present_array = ASSOCIATED(array)
      WRITE (ires) present_array
      IF (present_array) THEN
         dims = SHAPE(array)
         WRITE (ires) dims
         WRITE (ires) array
      END IF
   END SUBROUTINE write_real_3d

! **************************************************************************************************
!> \brief Read an array fingerprint, consume it completely, and compare with the live pointer.
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_character_1d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN), &
         POINTER                                         :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      CHARACTER(LEN=:), ALLOCATABLE, DIMENSION(:)        :: file_array
      INTEGER                                            :: n1
      LOGICAL                                            :: present_file

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) n1
      IF (ios /= 0) RETURN
      IF (n1 < 0) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (SIZE(array, 1) /= n1) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (CHARACTER(LEN=LEN(array)) :: file_array(n1))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_character_1d

! **************************************************************************************************
!> \brief ...
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_integer_1d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      INTEGER, DIMENSION(:), INTENT(IN), POINTER         :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      INTEGER                                            :: n1
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: file_array
      LOGICAL                                            :: present_file

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) n1
      IF (ios /= 0) RETURN
      IF (n1 < 0) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (SIZE(array, 1) /= n1) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (file_array(n1))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_integer_1d

! **************************************************************************************************
!> \brief ...
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_integer_2d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      INTEGER, DIMENSION(:, :), INTENT(IN), POINTER      :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: file_array
      INTEGER, DIMENSION(2)                              :: dims
      LOGICAL                                            :: present_file

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) dims
      IF (ios /= 0) RETURN
      IF (ANY(dims < 0)) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (ANY(SHAPE(array) /= dims)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (file_array(dims(1), dims(2)))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_integer_2d

! **************************************************************************************************
!> \brief ...
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_real_1d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      REAL(KIND=dp), DIMENSION(:), INTENT(IN), POINTER   :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      INTEGER                                            :: n1
      LOGICAL                                            :: present_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: file_array

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) n1
      IF (ios /= 0) RETURN
      IF (n1 < 0) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (SIZE(array, 1) /= n1) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (file_array(n1))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_real_1d

! **************************************************************************************************
!> \brief ...
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_real_2d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN), &
         POINTER                                         :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      INTEGER, DIMENSION(2)                              :: dims
      LOGICAL                                            :: present_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: file_array

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) dims
      IF (ios /= 0) RETURN
      IF (ANY(dims < 0)) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (ANY(SHAPE(array) /= dims)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (file_array(dims(1), dims(2)))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_real_2d

! **************************************************************************************************
!> \brief ...
!> \param rst_unit ...
!> \param array ...
!> \param matches ...
!> \param ios ...
! **************************************************************************************************
   SUBROUTINE compare_real_3d(rst_unit, array, matches, ios)
      INTEGER, INTENT(IN)                                :: rst_unit
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN), &
         POINTER                                         :: array
      LOGICAL, INTENT(INOUT)                             :: matches
      INTEGER, INTENT(INOUT)                             :: ios

      INTEGER, DIMENSION(3)                              :: dims
      LOGICAL                                            :: present_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: file_array

      IF (ios /= 0) RETURN
      READ (rst_unit, IOSTAT=ios) present_file
      IF (ios /= 0) RETURN
      matches = matches .AND. (present_file .EQV. ASSOCIATED(array))
      IF (.NOT. present_file) RETURN
      READ (rst_unit, IOSTAT=ios) dims
      IF (ios /= 0) RETURN
      IF (ANY(dims < 0)) THEN
         ios = -1
         RETURN
      END IF
      IF (.NOT. ASSOCIATED(array)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      ELSEIF (ANY(SHAPE(array) /= dims)) THEN
         matches = .FALSE.
         READ (rst_unit, IOSTAT=ios)
         RETURN
      END IF
      ALLOCATE (file_array(dims(1), dims(2), dims(3)))
      READ (rst_unit, IOSTAT=ios) file_array
      IF (ios /= 0) RETURN
      IF (ANY(file_array /= array)) matches = .FALSE.
   END SUBROUTINE compare_real_3d

! **************************************************************************************************
!> \brief Read and validate a version-2 k-point restart from an already opened unit.
!> \param rst_unit ...
!> \param denmat ...
!> \param kpoints ...
!> \param fmwork ...
!> \param para_env ...
!> \param particle_set ...
!> \param qs_kind_set ...
!> \param cell ...
!> \param dft_control ...
!> \param scf_control ...
!> \param nelectron_spin ...
!> \param overlap_rs ...
!> \param scf_env ...
!> \param restart_mode ...
!> \param accepted true only after temporary P(k) and P(R) have passed all gates and were committed
!> \param reason single diagnostic explaining a rejected transfer
! **************************************************************************************************
   SUBROUTINE read_kpoint_restart_v2(rst_unit, denmat, kpoints, fmwork, para_env, particle_set, &
                                     qs_kind_set, cell, dft_control, scf_control, nelectron_spin, &
                                     overlap_rs, scf_env, restart_mode, accepted, reason)
      INTEGER, INTENT(IN)                                :: rst_unit
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(cp_fm_type), DIMENSION(:), INTENT(INOUT)      :: fmwork
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(scf_control_type), POINTER                    :: scf_control
      INTEGER, DIMENSION(2), INTENT(IN)                  :: nelectron_spin
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: overlap_rs
      TYPE(qs_scf_env_type), POINTER                     :: scf_env
      INTEGER, INTENT(IN)                                :: restart_mode
      LOGICAL, INTENT(OUT)                               :: accepted
      CHARACTER(LEN=*), INTENT(OUT)                      :: reason

      CHARACTER(LEN=*), PARAMETER :: routineN = 'read_kpoint_restart_v2'

      CHARACTER(LEN=32)                                  :: magic_file
      CHARACTER(LEN=default_string_length)               :: kp_scheme_file, kp_scheme_target
      INTEGER :: charge_file, handle, iatom, ios, irecord, ispin, j, multiplicity_file, &
         nao_current, nao_file, natom_current, natom_file, nrecords, nset_max_current, &
         nset_max_file, nshell_max_current, nshell_max_file, nspin_file, record_file, &
         smear_method_file, spin_file, symmetry_backend_file, symmetry_backend_target, &
         symmetry_reduction_method_file, symmetry_reduction_method_target, version_file
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: kind_current, kind_file, nset_current, &
                                                            nset_file
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: cells, nshell_current, nshell_file
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: nso_current, nso_file
      INTEGER, DIMENSION(2)                              :: nelectron_file
      INTEGER, DIMENSION(3)                              :: cell_file, lower_bvk, nkp_grid_file, &
                                                            nkp_grid_target, periodic_file, &
                                                            upper_bvk
      LOGICAL :: fingerprint_ok, full_grid_file, full_grid_target, gamma_file, gamma_target, &
         inversion_symmetry_only_file, inversion_symmetry_only_target, io_ok, smear_file, &
         symmetry_file, symmetry_target
      REAL(KIND=dp) :: checksum_limit, checksum_linear, checksum_linear_file, checksum_square, &
         checksum_square_file, electronic_temperature_file, eps_geo_file, eps_geo_target, &
         payload_max_abs, smearing_width_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: positions_current, positions_file
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :, :)  :: payload_i, payload_r
      REAL(KIND=dp), DIMENSION(3)                        :: kp_shift_file, kp_shift_target
      REAL(KIND=dp), DIMENSION(3, 3)                     :: hmat_file

      CALL timeset(routineN, handle)
      accepted = .FALSE.
      reason = ""
      io_ok = .TRUE.
      version_file = -1
      natom_file = -1
      nao_file = -1
      nset_max_file = -1
      nshell_max_file = -1
      natom_current = SIZE(particle_set)
      CALL collect_basis_metadata(qs_kind_set, particle_set, "ORB", nao_current, &
                                  nset_max_current, nshell_max_current, nset_current, &
                                  nshell_current, nso_current)

      IF (para_env%is_source()) THEN
         READ (rst_unit, IOSTAT=ios) version_file
         io_ok = ios == 0
         IF (.NOT. io_ok) reason = "truncated version-2 header"
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(version_file)
      IF (.NOT. io_ok) GOTO 900
      IF (version_file /= kpoint_restart_v2_version) THEN
         reason = "not a version-2 k-point restart"
         GOTO 900
      END IF

      IF (para_env%is_source()) THEN
         READ (rst_unit, IOSTAT=ios) natom_file, nao_file, nset_max_file, nshell_max_file
         io_ok = ios == 0
         IF (.NOT. io_ok) reason = "truncated basis header"
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(natom_file)
      CALL para_env%bcast(nao_file)
      CALL para_env%bcast(nset_max_file)
      CALL para_env%bcast(nshell_max_file)
      IF (.NOT. io_ok) GOTO 900
      IF (natom_file /= natom_current .OR. nao_file /= nao_current .OR. &
          nset_max_file /= nset_max_current .OR. nshell_max_file /= nshell_max_current) THEN
         reason = "atom count or orbital topology dimensions differ"
         GOTO 900
      END IF

      ALLOCATE (nset_file(natom_file), nshell_file(nset_max_file, natom_file), &
                nso_file(nshell_max_file, nset_max_file, natom_file))
      nset_file = 0
      nshell_file = 0
      nso_file = 0
      IF (para_env%is_source()) THEN
         READ (rst_unit, IOSTAT=ios) nset_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) nshell_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) nso_file
         io_ok = ios == 0
         IF (.NOT. io_ok) reason = "truncated orbital topology"
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(nset_file)
      CALL para_env%bcast(nshell_file)
      CALL para_env%bcast(nso_file)
      IF (.NOT. io_ok) GOTO 900
      IF (ANY(nset_file /= nset_current) .OR. ANY(nshell_file /= nshell_current) .OR. &
          ANY(nso_file /= nso_current)) THEN
         reason = "atom-resolved orbital topology differs"
         GOTO 900
      END IF

      fingerprint_ok = .TRUE.
      IF (para_env%is_source()) THEN
         CALL validate_basis_fingerprint(rst_unit, qs_kind_set, particle_set, fingerprint_ok, ios)
         io_ok = ios == 0
         IF (.NOT. io_ok) THEN
            reason = "truncated basis/species fingerprint"
         ELSEIF (.NOT. fingerprint_ok) THEN
            reason = "complete basis/species fingerprint differs"
         END IF
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(fingerprint_ok)
      IF (.NOT. io_ok .OR. .NOT. fingerprint_ok) GOTO 900

      ALLOCATE (positions_file(3, natom_file), positions_current(3, natom_file), &
                kind_file(natom_file), kind_current(natom_file))
      positions_file = 0.0_dp
      kind_file = -1
      DO iatom = 1, natom_file
         positions_current(:, iatom) = particle_set(iatom)%r
         CALL get_atomic_kind(particle_set(iatom)%atomic_kind, kind_number=kind_current(iatom))
      END DO

      magic_file = ""
      kp_scheme_file = ""
      nkp_grid_file = -1
      kp_shift_file = 0.0_dp
      gamma_file = .FALSE.
      symmetry_file = .FALSE.
      full_grid_file = .FALSE.
      inversion_symmetry_only_file = .FALSE.
      symmetry_backend_file = -1
      symmetry_reduction_method_file = -1
      eps_geo_file = -1.0_dp
      periodic_file = -1
      hmat_file = 0.0_dp
      charge_file = 0
      multiplicity_file = 0
      nspin_file = 0
      nelectron_file = 0
      smear_file = .FALSE.
      smear_method_file = -1
      electronic_temperature_file = -1.0_dp
      smearing_width_file = -1.0_dp
      nrecords = -1
      checksum_linear_file = 0.0_dp
      checksum_square_file = 0.0_dp

      IF (para_env%is_source()) THEN
         READ (rst_unit, IOSTAT=ios) magic_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) kp_scheme_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) nkp_grid_file, kp_shift_file, gamma_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) symmetry_file, full_grid_file, &
            inversion_symmetry_only_file, symmetry_backend_file, &
            symmetry_reduction_method_file, eps_geo_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) periodic_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) hmat_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) positions_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) kind_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) charge_file, multiplicity_file, &
            nspin_file, nelectron_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) smear_file, smear_method_file, &
            electronic_temperature_file, smearing_width_file
         IF (ios == 0) READ (rst_unit, IOSTAT=ios) nrecords, checksum_linear_file, &
            checksum_square_file
         io_ok = ios == 0
         IF (.NOT. io_ok) reason = "truncated version-2 metadata"
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(magic_file)
      CALL para_env%bcast(kp_scheme_file)
      CALL para_env%bcast(nkp_grid_file)
      CALL para_env%bcast(kp_shift_file)
      CALL para_env%bcast(gamma_file)
      CALL para_env%bcast(symmetry_file)
      CALL para_env%bcast(full_grid_file)
      CALL para_env%bcast(inversion_symmetry_only_file)
      CALL para_env%bcast(symmetry_backend_file)
      CALL para_env%bcast(symmetry_reduction_method_file)
      CALL para_env%bcast(eps_geo_file)
      CALL para_env%bcast(periodic_file)
      CALL para_env%bcast(hmat_file)
      CALL para_env%bcast(positions_file)
      CALL para_env%bcast(kind_file)
      CALL para_env%bcast(charge_file)
      CALL para_env%bcast(multiplicity_file)
      CALL para_env%bcast(nspin_file)
      CALL para_env%bcast(nelectron_file)
      CALL para_env%bcast(smear_file)
      CALL para_env%bcast(smear_method_file)
      CALL para_env%bcast(electronic_temperature_file)
      CALL para_env%bcast(smearing_width_file)
      CALL para_env%bcast(nrecords)
      CALL para_env%bcast(checksum_linear_file)
      CALL para_env%bcast(checksum_square_file)
      IF (.NOT. io_ok) GOTO 900

      IF (.NOT. ALL(ieee_is_finite(kp_shift_file)) .OR. &
          .NOT. ieee_is_finite(eps_geo_file) .OR. &
          .NOT. ALL(ieee_is_finite(hmat_file)) .OR. &
          .NOT. ALL(ieee_is_finite(positions_file)) .OR. &
          .NOT. ieee_is_finite(electronic_temperature_file) .OR. &
          .NOT. ieee_is_finite(smearing_width_file) .OR. &
          .NOT. ieee_is_finite(checksum_linear_file) .OR. &
          .NOT. ieee_is_finite(checksum_square_file)) THEN
         reason = "version-2 metadata contains non-finite values"
         GOTO 900
      END IF

      IF (magic_file /= restart_magic) THEN
         reason = "invalid version-2 restart marker"
         GOTO 900
      END IF
      IF (ANY(nkp_grid_file <= 0) .OR. nrecords <= 0) THEN
         reason = "invalid number of canonical BvK records"
         GOTO 900
      END IF
      IF (nrecords > PRODUCT(nkp_grid_file)) THEN
         reason = "invalid number of canonical BvK records"
         GOTO 900
      END IF

      CALL get_kpoint_info(kpoints, kp_scheme=kp_scheme_target, nkp_grid=nkp_grid_target, &
                           kp_shift=kp_shift_target, gamma_centered=gamma_target, &
                           symmetry=symmetry_target, full_grid=full_grid_target, &
                           inversion_symmetry_only=inversion_symmetry_only_target, &
                           eps_geo=eps_geo_target, symmetry_backend=symmetry_backend_target, &
                           symmetry_reduction_method=symmetry_reduction_method_target)
      IF (TRIM(kp_scheme_file) /= TRIM(kp_scheme_target) .OR. &
          (TRIM(kp_scheme_file) /= "MONKHORST-PACK" .AND. TRIM(kp_scheme_file) /= "MACDONALD")) THEN
         reason = "source and target are not the same supported regular k-point scheme"
         GOTO 900
      END IF
      IF (ANY(ABS(kp_shift_file - kp_shift_target) > metadata_tol) .OR. &
          (gamma_file .NEQV. gamma_target)) THEN
         reason = "k-point shift or gamma-centering differs"
         GOTO 900
      END IF
      IF ((symmetry_file .NEQV. symmetry_target) .OR. &
          (full_grid_file .NEQV. full_grid_target) .OR. &
          (inversion_symmetry_only_file .NEQV. inversion_symmetry_only_target) .OR. &
          symmetry_backend_file /= symmetry_backend_target .OR. &
          symmetry_reduction_method_file /= symmetry_reduction_method_target .OR. &
          ABS(eps_geo_file - eps_geo_target) > metadata_tol) THEN
         reason = "k-point symmetry configuration differs"
         GOTO 900
      END IF
      IF (ANY(periodic_file /= cell%perd)) THEN
         reason = "periodicity mask differs"
         GOTO 900
      END IF
      DO iatom = 1, 3
         IF (cell%perd(iatom) == 0 .AND. &
             (nkp_grid_file(iatom) /= 1 .OR. nkp_grid_target(iatom) /= 1)) THEN
            reason = "an open direction has more than one k point"
            GOTO 900
         END IF
      END DO
      SELECT CASE (restart_mode)
      CASE (kpoint_restart_legacy_compatible)
         IF (ANY(nkp_grid_file /= nkp_grid_target)) THEN
            reason = "COMPATIBLE mode requires an identical k-point mesh"
            GOTO 900
         END IF
      CASE (kpoint_restart_exact)
         IF (ANY(nkp_grid_file /= nkp_grid_target)) THEN
            reason = "EXACT mode requires an identical k-point mesh"
            GOTO 900
         END IF
      CASE (kpoint_restart_validated_bvk_transfer)
         IF (ANY(nkp_grid_target < nkp_grid_file)) THEN
            reason = "stage-1 BvK transfer supports same meshes and refinement only"
            GOTO 900
         END IF
      CASE DEFAULT
         reason = "unknown KPOINT_RESTART_MODE"
         GOTO 900
      END SELECT

      IF (MAXVAL(ABS(hmat_file - cell%hmat)) > metadata_tol*MAX(1.0_dp, MAXVAL(ABS(cell%hmat))) .OR. &
          MAXVAL(ABS(positions_file - positions_current)) > &
          metadata_tol*MAX(1.0_dp, MAXVAL(ABS(positions_current)))) THEN
         reason = "stage-1 transfer requires identical cell and coordinates"
         GOTO 900
      END IF
      IF (ANY(kind_file /= kind_current)) THEN
         reason = "atomic kind ordering differs"
         GOTO 900
      END IF
      IF (charge_file /= dft_control%charge .OR. multiplicity_file /= dft_control%multiplicity .OR. &
          nspin_file /= SIZE(denmat, 1) .OR. ANY(nelectron_file /= nelectron_spin)) THEN
         reason = "charge, multiplicity, spin, or electron counts differ"
         GOTO 900
      END IF
      IF (.NOT. compatible_smearing(scf_control, smear_file, smear_method_file, &
                                    electronic_temperature_file, smearing_width_file)) THEN
         reason = "smearing metadata differs"
         GOTO 900
      END IF

      ALLOCATE (cells(3, nrecords), &
                payload_r(nspin_file, nao_file, nao_file, nrecords), &
                payload_i(nspin_file, nao_file, nao_file, nrecords))
      payload_r = 0.0_dp
      payload_i = 0.0_dp
      cells = 0
      IF (para_env%is_source()) THEN
         ios = 0
         DO ispin = 1, nspin_file
            DO irecord = 1, nrecords
               READ (rst_unit, IOSTAT=ios) spin_file, record_file, cell_file
               IF (ios /= 0) EXIT
               IF (spin_file /= ispin .OR. record_file /= irecord) THEN
                  ios = -1
                  EXIT
               END IF
               IF (ispin == 1) THEN
                  cells(:, irecord) = cell_file
               ELSEIF (ANY(cells(:, irecord) /= cell_file)) THEN
                  ios = -1
                  EXIT
               END IF
               DO j = 1, nao_file
                  READ (rst_unit, IOSTAT=ios) payload_r(ispin, :, j, irecord)
                  IF (ios /= 0) EXIT
               END DO
               IF (ios /= 0) EXIT
               DO j = 1, nao_file
                  READ (rst_unit, IOSTAT=ios) payload_i(ispin, :, j, irecord)
                  IF (ios /= 0) EXIT
               END DO
               IF (ios /= 0) EXIT
            END DO
            IF (ios /= 0) EXIT
         END DO
         IF (ios == 0) THEN
            ! A version-2 restart is a closed transaction.  Reject appended records instead of
            ! silently accepting a valid prefix of a corrupted or concatenated file.
            READ (rst_unit, IOSTAT=ios)
            io_ok = ios == iostat_end
            IF (.NOT. io_ok) reason = "trailing data after version-2 payload"
         ELSE
            io_ok = .FALSE.
            reason = "truncated or misordered version-2 payload"
         END IF
      END IF
      CALL para_env%bcast(io_ok)
      CALL para_env%bcast(cells)
      DO ispin = 1, nspin_file
         DO irecord = 1, nrecords
            CALL para_env%bcast(payload_r(ispin, :, :, irecord))
            CALL para_env%bcast(payload_i(ispin, :, :, irecord))
         END DO
      END DO
      IF (.NOT. io_ok) GOTO 900

      lower_bvk = -nkp_grid_file/2
      upper_bvk = lower_bvk + nkp_grid_file - 1
      DO irecord = 1, nrecords
         IF (ANY(cells(:, irecord) < lower_bvk) .OR. ANY(cells(:, irecord) > upper_bvk)) THEN
            reason = "payload contains a noncanonical BvK cell"
            GOTO 900
         END IF
         DO j = 1, irecord - 1
            IF (ALL(cells(:, irecord) == cells(:, j))) THEN
               reason = "payload contains duplicate canonical BvK cells"
               GOTO 900
            END IF
         END DO
      END DO

      IF (.NOT. ALL(ieee_is_finite(payload_r)) .OR. .NOT. ALL(ieee_is_finite(payload_i))) THEN
         reason = "payload contains non-finite values"
         GOTO 900
      END IF
      payload_max_abs = MAX(MAXVAL(ABS(payload_r)), MAXVAL(ABS(payload_i)))
      checksum_limit = SQRT(HUGE(1.0_dp)/(REAL(SIZE(payload_r), dp) + &
                                          REAL(SIZE(payload_i), dp)))
      IF (payload_max_abs > checksum_limit) THEN
         reason = "payload magnitude would overflow its checksum"
         GOTO 900
      END IF
      checksum_linear = SUM(payload_r) + SUM(payload_i)
      checksum_square = SUM(payload_r*payload_r) + SUM(payload_i*payload_i)
      IF (.NOT. ieee_is_finite(checksum_linear) .OR. &
          .NOT. ieee_is_finite(checksum_square)) THEN
         reason = "payload checksum overflowed"
         GOTO 900
      END IF
      IF (ABS(checksum_linear - checksum_linear_file) > &
          metadata_tol*MAX(1.0_dp, ABS(checksum_linear_file)) .OR. &
          ABS(checksum_square - checksum_square_file) > &
          metadata_tol*MAX(1.0_dp, ABS(checksum_square_file))) THEN
         reason = "payload checksum mismatch"
         GOTO 900
      END IF

      CALL apply_bvk_transfer(denmat, kpoints, fmwork, para_env, overlap_rs, scf_env, &
                              nkp_grid_file, cells, payload_r, payload_i, nelectron_spin, &
                              accepted, reason)

900   CONTINUE
      IF (.NOT. accepted) CALL para_env%bcast(reason)
      IF (ALLOCATED(payload_r)) DEALLOCATE (payload_r)
      IF (ALLOCATED(payload_i)) DEALLOCATE (payload_i)
      IF (ALLOCATED(cells)) DEALLOCATE (cells)
      IF (ALLOCATED(positions_file)) DEALLOCATE (positions_file)
      IF (ALLOCATED(positions_current)) DEALLOCATE (positions_current)
      IF (ALLOCATED(kind_file)) DEALLOCATE (kind_file)
      IF (ALLOCATED(kind_current)) DEALLOCATE (kind_current)
      IF (ALLOCATED(nset_file)) DEALLOCATE (nset_file)
      IF (ALLOCATED(nshell_file)) DEALLOCATE (nshell_file)
      IF (ALLOCATED(nso_file)) DEALLOCATE (nso_file)
      DEALLOCATE (nset_current, nshell_current, nso_current)
      CALL timestop(handle)
   END SUBROUTINE read_kpoint_restart_v2

! **************************************************************************************************
!> \brief Compare restart and current smearing controls.
!> \param scf_control ...
!> \param smear_file ...
!> \param method_file ...
!> \param temperature_file ...
!> \param width_file ...
!> \return ...
! **************************************************************************************************
   LOGICAL FUNCTION compatible_smearing(scf_control, smear_file, method_file, temperature_file, width_file)
      TYPE(scf_control_type), POINTER                    :: scf_control
      LOGICAL, INTENT(IN)                                :: smear_file
      INTEGER, INTENT(IN)                                :: method_file
      REAL(KIND=dp), INTENT(IN)                          :: temperature_file, width_file

      compatible_smearing = .NOT. smear_file
      IF (.NOT. ASSOCIATED(scf_control%smear)) RETURN
      compatible_smearing = smear_file .EQV. scf_control%smear%do_smear
      IF (.NOT. compatible_smearing .OR. .NOT. smear_file) RETURN
      compatible_smearing = method_file == scf_control%smear%method .AND. &
                            ABS(temperature_file - scf_control%smear%electronic_temperature) <= metadata_tol .AND. &
                            ABS(width_file - scf_control%smear%smearing_width) <= metadata_tol
   END FUNCTION compatible_smearing

! **************************************************************************************************
!> \brief Construct temporary target P(k), rebuild P(R), validate, and commit atomically.
!> \param denmat ...
!> \param kpoints ...
!> \param fmwork ...
!> \param para_env ...
!> \param overlap_rs ...
!> \param scf_env ...
!> \param nkp_grid_source ...
!> \param cells ...
!> \param payload_r ...
!> \param payload_i ...
!> \param nelectron_spin ...
!> \param accepted ...
!> \param reason ...
! **************************************************************************************************
   SUBROUTINE apply_bvk_transfer(denmat, kpoints, fmwork, para_env, overlap_rs, scf_env, &
                                 nkp_grid_source, cells, payload_r, payload_i, nelectron_spin, &
                                 accepted, reason)
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(cp_fm_type), DIMENSION(:), INTENT(INOUT)      :: fmwork
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: overlap_rs
      TYPE(qs_scf_env_type), POINTER                     :: scf_env
      INTEGER, DIMENSION(3), INTENT(IN)                  :: nkp_grid_source
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: cells
      REAL(KIND=dp), DIMENSION(:, :, :, :), INTENT(IN)   :: payload_r, payload_i
      INTEGER, DIMENSION(2), INTENT(IN)                  :: nelectron_spin
      LOGICAL, INTENT(OUT)                               :: accepted
      CHARACTER(LEN=*), INTENT(OUT)                      :: reason

      CHARACTER(LEN=*), PARAMETER :: routineN = 'apply_bvk_transfer'

      COMPLEX(KIND=dp)                                   :: phase
      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :)     :: dense
      INTEGER                                            :: dense_nonfinite, handle, ic, idim, ikp, &
                                                            irecord, ispin, kplocal, nao, nc, &
                                                            nimages, nrecords, nspin
      INTEGER, DIMENSION(3)                              :: nkp_grid_target
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      LOGICAL                                            :: need_renormalization, nyquist_present, &
                                                            phase_ok, phase_snapshot_taken
      REAL(KIND=dp) :: electron_error, expected_electrons, herm_error, herm_error_global, &
         imag_error, imag_error_global, matrix_norm, normalization_factor, nyquist_norm, &
         spin_trace, trace_term
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: dense_i, dense_r
      TYPE(cp_fm_struct_type), POINTER                   :: matrix_struct
      TYPE(cp_fm_type), ALLOCATABLE, DIMENSION(:, :, :)  :: pmat_tmp
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: denmat_tmp
      TYPE(kpoint_env_type), POINTER                     :: kp
      TYPE(kpoint_sym_type), POINTER                     :: kpsym
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      TYPE(phase_snapshot_type), ALLOCATABLE, &
         DIMENSION(:)                                    :: phase_snapshot

      CALL timeset(routineN, handle)
      accepted = .FALSE.
      reason = ""
      NULLIFY (cell_to_index, denmat_tmp, kp, kpsym, matrix_struct, sab_nl)
      phase_snapshot_taken = .FALSE.

      nspin = SIZE(denmat, 1)
      nimages = SIZE(denmat, 2)
      nao = SIZE(payload_r, 2)
      nrecords = SIZE(cells, 2)
      CALL get_kpoint_info(kpoints, nkp_grid=nkp_grid_target, sab_nl=sab_nl, &
                           cell_to_index=cell_to_index)

      ! Snapshot all mutable phase selectors, calibrate through the recoverable API, and retain
      ! the result only if the complete density transaction commits.  Thus a failed calibration
      ! or a later physical gate leaves the live symmetry environment byte-for-byte unchanged.
      IF (.NOT. kpoints%full_grid) THEN
         ALLOCATE (phase_snapshot(kpoints%nkp))
         DO ikp = 1, kpoints%nkp
            kpsym => kpoints%kp_sym(ikp)%kpoint_sym
            IF (.NOT. ASSOCIATED(kpsym)) THEN
               reason = "target k-point symmetry metadata is unavailable"
               GOTO 800
            END IF
            IF (.NOT. ASSOCIATED(kpsym%phase_mode) .OR. &
                .NOT. ASSOCIATED(kpsym%phase_gauge)) THEN
               reason = "target symmetry phases are unavailable"
               GOTO 800
            END IF
            ALLOCATE (phase_snapshot(ikp)%mode(SIZE(kpsym%phase_mode)), &
                      phase_snapshot(ikp)%gauge(SIZE(kpsym%phase_gauge)))
            phase_snapshot(ikp)%mode(:) = kpsym%phase_mode(:)
            phase_snapshot(ikp)%gauge(:) = kpsym%phase_gauge(:)
         END DO
         phase_snapshot_taken = .TRUE.
         CALL calibrate_symmetry_phases(kpoints, overlap_rs, overlap_rs(1, 1)%matrix, sab_nl, &
                                        cell_to_index, success=phase_ok, error_message=reason)
         IF (.NOT. phase_ok) GOTO 800
      END IF

      ! An even source mesh has a single aliased Nyquist coefficient. Splitting it into the two
      ! independent modes of a refined mesh is deliberately deferred to stage 2. A zero plane is
      ! harmless and can be refined without special treatment.
      DO idim = 1, 3
         IF (MOD(nkp_grid_source(idim), 2) == 0 .AND. &
             nkp_grid_target(idim) > nkp_grid_source(idim)) THEN
            nyquist_norm = 0.0_dp
            nyquist_present = .FALSE.
            DO irecord = 1, nrecords
               IF (cells(idim, irecord) == -nkp_grid_source(idim)/2) THEN
                  nyquist_present = .TRUE.
                  nyquist_norm = MAX(nyquist_norm, MAXVAL(ABS(payload_r(:, :, :, irecord))))
                  nyquist_norm = MAX(nyquist_norm, MAXVAL(ABS(payload_i(:, :, :, irecord))))
               END IF
            END DO
            IF (nyquist_present .AND. nyquist_norm > payload_zero_tol) THEN
               reason = "refinement of a nonzero even-mesh Nyquist plane is a stage-2 feature"
               GOTO 800
            END IF
         END IF
      END DO

      kplocal = SIZE(kpoints%kp_env)
      kp => kpoints%kp_env(1)%kpoint_env
      CPASSERT(ASSOCIATED(kp%pmat))
      nc = SIZE(kp%pmat, 1)
      IF (nc < 1 .OR. nc > 2 .OR. SIZE(kp%pmat, 2) /= nspin) THEN
         reason = "target P(k) storage is incompatible with the restart"
         GOTO 800
      END IF

      ALLOCATE (pmat_tmp(kplocal, nc, nspin))
      DO ikp = 1, kplocal
         kp => kpoints%kp_env(ikp)%kpoint_env
         DO ispin = 1, nspin
            DO ic = 1, nc
               CALL cp_fm_get_info(kp%pmat(ic, ispin), matrix_struct=matrix_struct)
               CALL cp_fm_create(pmat_tmp(ikp, ic, ispin), matrix_struct, set_zero=.TRUE.)
            END DO
         END DO
      END DO

      ALLOCATE (dense(nao, nao), dense_r(nao, nao), dense_i(nao, nao))
      herm_error = 0.0_dp
      imag_error = 0.0_dp
      dense_nonfinite = 0
      DO ikp = 1, kplocal
         kp => kpoints%kp_env(ikp)%kpoint_env
         DO ispin = 1, nspin
            dense = CMPLX(0.0_dp, 0.0_dp, KIND=dp)
            DO irecord = 1, nrecords
               phase = CMPLX(COS(twopi*DOT_PRODUCT(kp%xkp, REAL(cells(:, irecord), dp))), &
                             SIN(twopi*DOT_PRODUCT(kp%xkp, REAL(cells(:, irecord), dp))), KIND=dp)
               DO ic = 1, nao
                  dense(:, ic) = dense(:, ic) + phase*CMPLX(payload_r(ispin, :, ic, irecord), &
                                                            payload_i(ispin, :, ic, irecord), KIND=dp)
               END DO
            END DO
            IF (.NOT. ALL(ieee_is_finite(REAL(dense, dp))) .OR. &
                .NOT. ALL(ieee_is_finite(AIMAG(dense)))) THEN
               dense_nonfinite = 1
               CYCLE
            END IF
            matrix_norm = MAX(1.0_dp, MAXVAL(ABS(dense)))
            herm_error = MAX(herm_error, MAXVAL(ABS(dense - CONJG(TRANSPOSE(dense))))/matrix_norm)
            dense(:, :) = 0.5_dp*(dense + CONJG(TRANSPOSE(dense)))
            dense_r(:, :) = REAL(dense, dp)
            dense_i(:, :) = AIMAG(dense)
            CALL cp_fm_set_submatrix(pmat_tmp(ikp, 1, ispin), dense_r)
            IF (nc == 2) THEN
               CALL cp_fm_set_submatrix(pmat_tmp(ikp, 2, ispin), dense_i)
            ELSE
               imag_error = MAX(imag_error, MAXVAL(ABS(dense_i))/matrix_norm)
            END IF
         END DO
      END DO
      herm_error_global = herm_error
      imag_error_global = imag_error
      CALL para_env%max(dense_nonfinite)
      CALL para_env%max(herm_error_global)
      CALL para_env%max(imag_error_global)
      DEALLOCATE (dense, dense_r, dense_i)
      IF (dense_nonfinite /= 0) THEN
         reason = "target P(k) contains non-finite values"
         GOTO 800
      END IF
      IF (herm_error_global > hermiticity_tol) THEN
         WRITE (reason, '(A,ES12.4)') "target P(k) is not Hermitian; relative residual=", herm_error_global
         GOTO 800
      END IF
      IF (imag_error_global > hermiticity_tol) THEN
         WRITE (reason, '(A,ES12.4)') "real-only target has a nonzero imaginary density; residual=", imag_error_global
         GOTO 800
      END IF

      ALLOCATE (denmat_tmp(nspin, nimages))
      DO ispin = 1, nspin
         DO ic = 1, nimages
            ALLOCATE (denmat_tmp(ispin, ic)%matrix)
            CALL dbcsr_create(denmat_tmp(ispin, ic)%matrix, template=denmat(ispin, ic)%matrix)
            CALL dbcsr_copy(denmat_tmp(ispin, ic)%matrix, denmat(ispin, ic)%matrix)
            CALL dbcsr_set(denmat_tmp(ispin, ic)%matrix, 0.0_dp)
         END DO
      END DO

      CALL kpoint_density_transform(kpoints, denmat_tmp, .FALSE., &
                                    tempmat=overlap_rs(1, 1)%matrix, sab_nl=sab_nl, &
                                    fmwork=fmwork, pmat_ext=pmat_tmp)
      need_renormalization = .FALSE.
      DO ispin = 1, nspin
         spin_trace = 0.0_dp
         DO ic = 1, nimages
            trace_term = 0.0_dp
            CALL dbcsr_dot(overlap_rs(1, ic)%matrix, denmat_tmp(ispin, ic)%matrix, trace_term)
            spin_trace = spin_trace + trace_term
         END DO
         IF (nspin == 1) THEN
            expected_electrons = REAL(SUM(nelectron_spin), dp)
         ELSE
            expected_electrons = REAL(nelectron_spin(ispin), dp)
         END IF
         electron_error = ABS(spin_trace - expected_electrons)
         IF (.NOT. ieee_is_finite(spin_trace) .OR. electron_error > electron_renormalization_tol) THEN
            WRITE (reason, '(A,I0,A,ES12.4,A,ES12.4)') &
               "spin-resolved electron-count gate failed for spin ", ispin, ": Tr[PS]=", &
               spin_trace, ", error=", electron_error
            GOTO 800
         END IF
         IF (electron_error > electron_tol) THEN
            IF (ABS(spin_trace) <= TINY(1.0_dp)) THEN
               reason = "cannot normalize a zero-trace restart density"
               GOTO 800
            END IF
            normalization_factor = expected_electrons/spin_trace
            DO ikp = 1, kplocal
               DO ic = 1, nc
                  CALL cp_fm_scale(normalization_factor, pmat_tmp(ikp, ic, ispin))
               END DO
            END DO
            need_renormalization = .TRUE.
         END IF
      END DO

      IF (need_renormalization) THEN
         CALL kpoint_density_transform(kpoints, denmat_tmp, .FALSE., &
                                       tempmat=overlap_rs(1, 1)%matrix, sab_nl=sab_nl, &
                                       fmwork=fmwork, pmat_ext=pmat_tmp)
         DO ispin = 1, nspin
            spin_trace = 0.0_dp
            DO ic = 1, nimages
               trace_term = 0.0_dp
               CALL dbcsr_dot(overlap_rs(1, ic)%matrix, denmat_tmp(ispin, ic)%matrix, trace_term)
               spin_trace = spin_trace + trace_term
            END DO
            IF (nspin == 1) THEN
               expected_electrons = REAL(SUM(nelectron_spin), dp)
            ELSE
               expected_electrons = REAL(nelectron_spin(ispin), dp)
            END IF
            electron_error = ABS(spin_trace - expected_electrons)
            IF (.NOT. ieee_is_finite(spin_trace) .OR. electron_error > electron_tol) THEN
               WRITE (reason, '(A,I0,A,ES12.4,A,ES12.4)') &
                  "normalized spin electron-count gate failed for spin ", ispin, ": Tr[PS]=", &
                  spin_trace, ", error=", electron_error
               GOTO 800
            END IF
         END DO
      END IF

      ! Commit only after all validations. Both representations are replaced together.
      DO ikp = 1, kplocal
         kp => kpoints%kp_env(ikp)%kpoint_env
         DO ispin = 1, nspin
            DO ic = 1, nc
               CALL cp_fm_to_fm(pmat_tmp(ikp, ic, ispin), kp%pmat(ic, ispin))
            END DO
         END DO
      END DO
      DO ispin = 1, nspin
         DO ic = 1, nimages
            CALL dbcsr_copy(denmat(ispin, ic)%matrix, denmat_tmp(ispin, ic)%matrix)
         END DO
      END DO
      IF (ASSOCIATED(kpoints%scf_diis_buffer)) CALL qs_diis_b_clear_kp(kpoints%scf_diis_buffer)
      IF (ASSOCIATED(scf_env%scf_diis_buffer)) CALL qs_diis_b_clear(scf_env%scf_diis_buffer)
      accepted = .TRUE.

800   CONTINUE
      IF (phase_snapshot_taken .AND. .NOT. accepted) THEN
         DO ikp = 1, kpoints%nkp
            kpsym => kpoints%kp_sym(ikp)%kpoint_sym
            kpsym%phase_mode = phase_snapshot(ikp)%mode
            kpsym%phase_gauge = phase_snapshot(ikp)%gauge
         END DO
      END IF
      IF (ALLOCATED(pmat_tmp)) THEN
         DO ikp = 1, SIZE(pmat_tmp, 1)
            DO ic = 1, SIZE(pmat_tmp, 2)
               DO ispin = 1, SIZE(pmat_tmp, 3)
                  CALL cp_fm_release(pmat_tmp(ikp, ic, ispin))
               END DO
            END DO
         END DO
         DEALLOCATE (pmat_tmp)
      END IF
      IF (ASSOCIATED(denmat_tmp)) THEN
         DO ispin = 1, SIZE(denmat_tmp, 1)
            DO ic = 1, SIZE(denmat_tmp, 2)
               IF (ASSOCIATED(denmat_tmp(ispin, ic)%matrix)) &
                  CALL dbcsr_deallocate_matrix(denmat_tmp(ispin, ic)%matrix)
            END DO
         END DO
         DEALLOCATE (denmat_tmp)
      END IF
      CALL timestop(handle)
   END SUBROUTINE apply_bvk_transfer

END MODULE kpoint_restart_transfer
