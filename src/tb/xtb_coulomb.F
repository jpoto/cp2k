!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Calculation of Coulomb contributions in xTB
!> \author JGH
! **************************************************************************************************
MODULE xtb_coulomb
   USE ai_contraction,                  ONLY: block_add,&
                                              contraction
   USE ai_overlap,                      ONLY: overlap_ab
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE atprop_types,                    ONLY: atprop_array_init,&
                                              atprop_type
   USE basis_set_types,                 ONLY: gto_basis_set_p_type,&
                                              gto_basis_set_type
   USE cell_types,                      ONLY: cell_type,&
                                              get_cell,&
                                              pbc
   USE cp_control_types,                ONLY: dft_control_type,&
                                              xtb_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_add,&
                                              dbcsr_get_block_p,&
                                              dbcsr_iterator_blocks_left,&
                                              dbcsr_iterator_next_block,&
                                              dbcsr_iterator_start,&
                                              dbcsr_iterator_stop,&
                                              dbcsr_iterator_type,&
                                              dbcsr_p_type
   USE distribution_1d_types,           ONLY: distribution_1d_type
   USE ewald_environment_types,         ONLY: ewald_env_get,&
                                              ewald_environment_type
   USE ewald_methods_tb,                ONLY: tb_ewald_overlap,&
                                              tb_spme_evaluate
   USE ewald_pw_types,                  ONLY: ewald_pw_type
   USE kinds,                           ONLY: dp
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_type
   USE mathconstants,                   ONLY: oorootpi,&
                                              pi
   USE message_passing,                 ONLY: mp_para_env_type
   USE orbital_pointers,                ONLY: ncoset
   USE particle_types,                  ONLY: particle_type
   USE pw_poisson_types,                ONLY: do_ewald_ewald,&
                                              do_ewald_none,&
                                              do_ewald_pme,&
                                              do_ewald_spme
   USE qmmm_tb_coulomb,                 ONLY: build_tb_coulomb_qmqm
   USE qs_dftb3_methods,                ONLY: build_dftb3_diagonal
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_force_types,                  ONLY: qs_force_type
   USE qs_integral_utils,               ONLY: basis_set_list_setup,&
                                              get_memory_usage
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
   USE qs_neighbor_list_types,          ONLY: get_iterator_info,&
                                              neighbor_list_iterate,&
                                              neighbor_list_iterator_create,&
                                              neighbor_list_iterator_p_type,&
                                              neighbor_list_iterator_release,&
                                              neighbor_list_set_p_type
   USE qs_rho_types,                    ONLY: qs_rho_get,&
                                              qs_rho_type
   USE sap_kind_types,                  ONLY: clist_type,&
                                              release_sap_int,&
                                              sap_int_type
   USE virial_methods,                  ONLY: virial_pair_force
   USE virial_types,                    ONLY: virial_type
   USE xtb_spinpol,                     ONLY: build_xtb_spinpol
   USE xtb_types,                       ONLY: get_xtb_atom_param,&
                                              xtb_atom_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xtb_coulomb'

   PUBLIC :: build_xtb_coulomb, gamma_rab_sr, dgamma_rab_sr, xtb_dsint_list

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param ks_matrix ...
!> \param rho ...
!> \param charges ...
!> \param mcharge ...
!> \param energy ...
!> \param calculate_forces ...
!> \param just_energy ...
! **************************************************************************************************
   SUBROUTINE build_xtb_coulomb(qs_env, ks_matrix, rho, charges, mcharge, energy, &
                                calculate_forces, just_energy)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: ks_matrix
      TYPE(qs_rho_type), POINTER                         :: rho
      REAL(dp), DIMENSION(:, :), INTENT(in)              :: charges
      REAL(dp), DIMENSION(:), INTENT(in)                 :: mcharge
      TYPE(qs_energy_type), POINTER                      :: energy
      LOGICAL, INTENT(in)                                :: calculate_forces, just_energy

      CHARACTER(len=*), PARAMETER                        :: routineN = 'build_xtb_coulomb'

      INTEGER :: atom_i, atom_j, ewald_type, handle, i, ia, iac, iatom, ic, icol, ikind, img, &
         irow, is, j, jatom, jkind, la, lb, lmaxa, lmaxb, natom, natorb_a, natorb_b, ni, nimg, nj, &
         nkind, nmat, za, zb
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: atom_of_kind, kind_of
      INTEGER, DIMENSION(25)                             :: laoa, laob
      INTEGER, DIMENSION(3)                              :: cellind, periodic
      INTEGER, DIMENSION(5)                              :: occ
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      LOGICAL                                            :: defined, do_ewald, do_gamma_stress, &
                                                            found, use_virial
      REAL(KIND=dp)                                      :: alpha, deth, dr, ecsr, etaa, etab, f1, &
                                                            f2, fi, gmij, kg, rcut, rcuta, rcutb, &
                                                            zeff
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: xgamma, zeffk
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: gammab, gcij, gmcharge
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: gchrg
      REAL(KIND=dp), DIMENSION(25)                       :: gcint
      REAL(KIND=dp), DIMENSION(3)                        :: fij, rij
      REAL(KIND=dp), DIMENSION(5)                        :: kappaa, kappab
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: dsblock, ksblock, pblock, sblock
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dsint
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(atprop_type), POINTER                         :: atprop
      TYPE(cell_type), POINTER                           :: cell
      TYPE(dbcsr_iterator_type)                          :: iter
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_p, matrix_s
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(distribution_1d_type), POINTER                :: local_particles
      TYPE(ewald_environment_type), POINTER              :: ewald_env
      TYPE(ewald_pw_type), POINTER                       :: ewald_pw
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: nl_iterator
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: n_list
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_force_type), DIMENSION(:), POINTER         :: force
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(sap_int_type), DIMENSION(:), POINTER          :: sap_int
      TYPE(virial_type), POINTER                         :: virial
      TYPE(xtb_atom_type), POINTER                       :: xtb_atom_a, xtb_atom_b, xtb_kind
      TYPE(xtb_control_type), POINTER                    :: xtb_control

      CALL timeset(routineN, handle)

      NULLIFY (matrix_p, matrix_s, virial, atprop, dft_control)

      CALL get_qs_env(qs_env, &
                      qs_kind_set=qs_kind_set, &
                      particle_set=particle_set, &
                      cell=cell, &
                      virial=virial, &
                      atprop=atprop, &
                      dft_control=dft_control)

      xtb_control => dft_control%qs_control%xtb_control

      use_virial = .FALSE.
      IF (calculate_forces) THEN
         use_virial = virial%pv_availability .AND. (.NOT. virial%pv_numer)
      END IF

      do_gamma_stress = .FALSE.
      IF (.NOT. just_energy .AND. use_virial) THEN
         IF (dft_control%nimages == 1) do_gamma_stress = .TRUE.
      END IF

      IF (atprop%energy) THEN
         CALL get_qs_env(qs_env=qs_env, particle_set=particle_set)
         natom = SIZE(particle_set)
         CALL atprop_array_init(atprop%atecoul, natom)
      END IF

      IF (calculate_forces) THEN
         nmat = 4
      ELSE
         nmat = 1
      END IF

      CALL get_qs_env(qs_env, nkind=nkind, natom=natom)
      ALLOCATE (gchrg(natom, 5, nmat))
      gchrg = 0._dp
      ALLOCATE (gmcharge(natom, nmat))
      gmcharge = 0._dp

      ! short range contribution (gamma)
      CALL get_cell(cell=cell, periodic=periodic)
      IF (xtb_control%klopman_ohno_ewald .AND. ANY(periodic == 1)) THEN
         IF (ABS(xtb_control%kg - 2.0_dp) > 1.e-12_dp) &
            CPABORT("XTB/KLOPMAN_OHNO_EWALD requires KG=2")
         IF (use_virial) &
            CPABORT("XTB/KLOPMAN_OHNO_EWALD does not provide an analytical stress tensor")
         CALL build_klopman_ohno_ewald_coulomb(qs_env, charges, gchrg, calculate_forces)
      ELSE
      ! loop over all atom pairs (sab_xtbe)
      kg = xtb_control%kg
      NULLIFY (n_list)
      CALL get_qs_env(qs_env=qs_env, sab_xtbe=n_list)
      IF (.NOT. ASSOCIATED(n_list)) THEN
         CPABORT("sab_xtbe neighbor list is not associated in build_xtb_coulomb")
      END IF
      CALL neighbor_list_iterator_create(nl_iterator, n_list)
      DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
         CALL get_iterator_info(nl_iterator, ikind=ikind, jkind=jkind, &
                                iatom=iatom, jatom=jatom, r=rij, cell=cellind)
         CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
         CALL get_xtb_atom_param(xtb_atom_a, defined=defined, natorb=natorb_a)
         IF (.NOT. defined .OR. natorb_a < 1) CYCLE
         CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
         CALL get_xtb_atom_param(xtb_atom_b, defined=defined, natorb=natorb_b)
         IF (.NOT. defined .OR. natorb_b < 1) CYCLE
         ! atomic parameters
         CALL get_xtb_atom_param(xtb_atom_a, eta=etaa, lmax=lmaxa, kappa=kappaa, rcut=rcuta)
         CALL get_xtb_atom_param(xtb_atom_b, eta=etab, lmax=lmaxb, kappa=kappab, rcut=rcutb)
         ! gamma matrix
         ni = lmaxa + 1
         nj = lmaxb + 1
         ALLOCATE (gammab(ni, nj))
         rcut = rcuta + rcutb
         dr = SQRT(SUM(rij(:)**2))
         CALL gamma_rab_sr(gammab, dr, ni, kappaa, etaa, nj, kappab, etab, kg, rcut)
         gchrg(iatom, 1:ni, 1) = gchrg(iatom, 1:ni, 1) + MATMUL(gammab, charges(jatom, 1:nj))
         IF (iatom /= jatom) THEN
            gchrg(jatom, 1:nj, 1) = gchrg(jatom, 1:nj, 1) + MATMUL(charges(iatom, 1:ni), gammab)
         END IF
         IF (calculate_forces) THEN
            IF (dr > 1.e-6_dp) THEN
               CALL dgamma_rab_sr(gammab, dr, ni, kappaa, etaa, nj, kappab, etab, kg, rcut)
               DO i = 1, 3
                  gchrg(iatom, 1:ni, i + 1) = gchrg(iatom, 1:ni, i + 1) &
                                              + MATMUL(gammab, charges(jatom, 1:nj))*rij(i)/dr
                  IF (iatom /= jatom) THEN
                     gchrg(jatom, 1:nj, i + 1) = gchrg(jatom, 1:nj, i + 1) &
                                                 - MATMUL(charges(iatom, 1:ni), gammab)*rij(i)/dr
                  END IF
               END DO
               IF (use_virial) THEN
                  gcint(1:ni) = MATMUL(gammab, charges(jatom, 1:nj))
                  DO i = 1, 3
                     fij(i) = -SUM(charges(iatom, 1:ni)*gcint(1:ni))*rij(i)/dr
                  END DO
                  fi = 1.0_dp
                  IF (iatom == jatom) fi = 0.5_dp
                  CALL virial_pair_force(virial%pv_virial, fi, fij, rij)
               END IF
            END IF
         END IF
         DEALLOCATE (gammab)
      END DO
      CALL neighbor_list_iterator_release(nl_iterator)
      END IF

      ! 1/R contribution

      IF (xtb_control%coulomb_lr) THEN
         do_ewald = xtb_control%do_ewald
         IF (do_ewald) THEN
            ! Ewald sum
            NULLIFY (ewald_env, ewald_pw)
            CALL get_qs_env(qs_env=qs_env, &
                            ewald_env=ewald_env, ewald_pw=ewald_pw)
            CALL get_cell(cell=cell, periodic=periodic, deth=deth)
            CALL ewald_env_get(ewald_env, alpha=alpha, ewald_type=ewald_type)
            CALL get_qs_env(qs_env=qs_env, sab_tbe=n_list)
            CALL tb_ewald_overlap(gmcharge, mcharge, alpha, n_list, virial, use_virial)
            SELECT CASE (ewald_type)
            CASE DEFAULT
               CPABORT("Invalid Ewald type")
            CASE (do_ewald_none)
               CPABORT("Not allowed with xTB/DFTB")
            CASE (do_ewald_ewald)
               CPABORT("Standard Ewald not implemented in xTB/DFTB")
            CASE (do_ewald_pme)
               CPABORT("PME not implemented in xTB/DFTB")
            CASE (do_ewald_spme)
               CALL tb_spme_evaluate(ewald_env, ewald_pw, particle_set, cell, &
                                     gmcharge, mcharge, calculate_forces, virial, use_virial)
            END SELECT
         ELSE
            ! direct sum
            CALL get_qs_env(qs_env=qs_env, &
                            local_particles=local_particles)
            DO ikind = 1, SIZE(local_particles%n_el)
               DO ia = 1, local_particles%n_el(ikind)
                  iatom = local_particles%list(ikind)%array(ia)
                  DO jatom = 1, iatom - 1
                     rij = particle_set(iatom)%r - particle_set(jatom)%r
                     rij = pbc(rij, cell)
                     dr = SQRT(SUM(rij(:)**2))
                     IF (dr > 1.e-6_dp) THEN
                        gmcharge(iatom, 1) = gmcharge(iatom, 1) + mcharge(jatom)/dr
                        gmcharge(jatom, 1) = gmcharge(jatom, 1) + mcharge(iatom)/dr
                        DO i = 2, nmat
                           gmcharge(iatom, i) = gmcharge(iatom, i) + rij(i - 1)*mcharge(jatom)/dr**3
                           gmcharge(jatom, i) = gmcharge(jatom, i) - rij(i - 1)*mcharge(iatom)/dr**3
                        END DO
                        IF (use_virial) THEN
                           DO i = 1, 3
                              fij(i) = mcharge(iatom)*mcharge(jatom)*rij(i)/dr**3
                           END DO
                           CALL virial_pair_force(virial%pv_virial, 1.0_dp, fij, rij)
                        END IF
                     END IF
                  END DO
               END DO
            END DO
         END IF
      END IF

      ! global sum of gamma*p arrays
      CALL get_qs_env(qs_env=qs_env, &
                      atomic_kind_set=atomic_kind_set, &
                      force=force, para_env=para_env)
      CALL para_env%sum(gmcharge(:, 1))
      CALL para_env%sum(gchrg(:, :, 1))

      IF (xtb_control%coulomb_lr) THEN
         IF (do_ewald) THEN
            ! add self charge interaction and background charge contribution
            gmcharge(:, 1) = gmcharge(:, 1) - 2._dp*alpha*oorootpi*mcharge(:)
            IF (ANY(periodic(:) == 1)) THEN
               gmcharge(:, 1) = gmcharge(:, 1) - pi/alpha**2/deth
            END IF
         END IF
      END IF

      ! energy
      CALL get_atomic_kind_set(atomic_kind_set=atomic_kind_set, &
                               kind_of=kind_of, &
                               atom_of_kind=atom_of_kind)
      ecsr = 0.0_dp
      DO iatom = 1, natom
         ikind = kind_of(iatom)
         CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_kind)
         CALL get_xtb_atom_param(xtb_kind, lmax=ni)
         ni = ni + 1
         ecsr = ecsr + SUM(charges(iatom, 1:ni)*gchrg(iatom, 1:ni, 1))
      END DO

      energy%hartree = energy%hartree + 0.5_dp*ecsr
      energy%hartree = energy%hartree + 0.5_dp*SUM(mcharge(:)*gmcharge(:, 1))

      IF (atprop%energy) THEN
         CALL get_qs_env(qs_env=qs_env, local_particles=local_particles)
         DO ikind = 1, SIZE(local_particles%n_el)
            CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_kind)
            CALL get_xtb_atom_param(xtb_kind, lmax=ni, occupation=occ)
            ni = ni + 1
            zeff = SUM(REAL(occ, KIND=dp))
            DO ia = 1, local_particles%n_el(ikind)
               iatom = local_particles%list(ikind)%array(ia)
               atprop%atecoul(iatom) = atprop%atecoul(iatom) + &
                                       0.5_dp*SUM(REAL(occ(1:ni), KIND=dp)*gchrg(iatom, 1:ni, 1))
               atprop%atecoul(iatom) = atprop%atecoul(iatom) + &
                                       0.5_dp*zeff*gmcharge(iatom, 1)
            END DO
         END DO
      END IF

      IF (calculate_forces) THEN
         DO iatom = 1, natom
            ikind = kind_of(iatom)
            atom_i = atom_of_kind(iatom)
            CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_kind)
            CALL get_xtb_atom_param(xtb_kind, lmax=ni)
            ! short range
            ni = ni + 1
            DO i = 1, 3
               fij(i) = SUM(charges(iatom, 1:ni)*gchrg(iatom, 1:ni, i + 1))
            END DO
            force(ikind)%rho_elec(1, atom_i) = force(ikind)%rho_elec(1, atom_i) - fij(1)
            force(ikind)%rho_elec(2, atom_i) = force(ikind)%rho_elec(2, atom_i) - fij(2)
            force(ikind)%rho_elec(3, atom_i) = force(ikind)%rho_elec(3, atom_i) - fij(3)
            ! long range
            DO i = 1, 3
               fij(i) = gmcharge(iatom, i + 1)*mcharge(iatom)
            END DO
            force(ikind)%rho_elec(1, atom_i) = force(ikind)%rho_elec(1, atom_i) - fij(1)
            force(ikind)%rho_elec(2, atom_i) = force(ikind)%rho_elec(2, atom_i) - fij(2)
            force(ikind)%rho_elec(3, atom_i) = force(ikind)%rho_elec(3, atom_i) - fij(3)
         END DO
      END IF

      IF (.NOT. just_energy) THEN
         CALL get_qs_env(qs_env=qs_env, matrix_s_kp=matrix_s)
         CALL qs_rho_get(rho, rho_ao_kp=matrix_p)

         nimg = dft_control%nimages
         NULLIFY (cell_to_index)
         IF (nimg > 1) THEN
            NULLIFY (kpoints)
            CALL get_qs_env(qs_env=qs_env, kpoints=kpoints)
            CALL get_kpoint_info(kpoint=kpoints, cell_to_index=cell_to_index)
         END IF

         IF (calculate_forces .AND. SIZE(matrix_p, 1) == 2) THEN
            DO img = 1, nimg
               CALL dbcsr_add(matrix_p(1, img)%matrix, matrix_p(2, img)%matrix, &
                              alpha_scalar=1.0_dp, beta_scalar=1.0_dp)
            END DO
         END IF

         NULLIFY (sap_int)
         IF (do_gamma_stress) THEN
            ! derivative overlap integral (non collapsed)
            CALL xtb_dsint_list(qs_env, sap_int)
         END IF

         IF (nimg == 1) THEN
            ! no k-points; all matrices have been transformed to periodic bsf
            CALL dbcsr_iterator_start(iter, matrix_s(1, 1)%matrix)
            DO WHILE (dbcsr_iterator_blocks_left(iter))
               CALL dbcsr_iterator_next_block(iter, irow, icol, sblock)
               ikind = kind_of(irow)
               jkind = kind_of(icol)

               ! atomic parameters
               CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
               CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
               CALL get_xtb_atom_param(xtb_atom_a, z=za, lao=laoa)
               CALL get_xtb_atom_param(xtb_atom_b, z=zb, lao=laob)

               ni = SIZE(sblock, 1)
               nj = SIZE(sblock, 2)
               ALLOCATE (gcij(ni, nj))
               DO i = 1, ni
                  DO j = 1, nj
                     la = laoa(i) + 1
                     lb = laob(j) + 1
                     gcij(i, j) = 0.5_dp*(gchrg(irow, la, 1) + gchrg(icol, lb, 1))
                  END DO
               END DO
               gmij = 0.5_dp*(gmcharge(irow, 1) + gmcharge(icol, 1))
               DO is = 1, SIZE(ks_matrix, 1)
                  NULLIFY (ksblock)
                  CALL dbcsr_get_block_p(matrix=ks_matrix(is, 1)%matrix, &
                                         row=irow, col=icol, block=ksblock, found=found)
                  CPASSERT(found)
                  ksblock = ksblock - gcij*sblock
                  ksblock = ksblock - gmij*sblock
               END DO
               IF (calculate_forces) THEN
                  atom_i = atom_of_kind(irow)
                  atom_j = atom_of_kind(icol)
                  NULLIFY (pblock)
                  CALL dbcsr_get_block_p(matrix=matrix_p(1, 1)%matrix, &
                                         row=irow, col=icol, block=pblock, found=found)
                  CPASSERT(found)
                  DO i = 1, 3
                     NULLIFY (dsblock)
                     CALL dbcsr_get_block_p(matrix=matrix_s(1 + i, 1)%matrix, &
                                            row=irow, col=icol, block=dsblock, found=found)
                     CPASSERT(found)
                     fij(i) = 0.0_dp
                     ! short range
                     fi = -2.0_dp*SUM(pblock*dsblock*gcij)
                     force(ikind)%rho_elec(i, atom_i) = force(ikind)%rho_elec(i, atom_i) + fi
                     force(jkind)%rho_elec(i, atom_j) = force(jkind)%rho_elec(i, atom_j) - fi
                     fij(i) = fij(i) + fi
                     ! long range
                     fi = -2.0_dp*gmij*SUM(pblock*dsblock)
                     force(ikind)%rho_elec(i, atom_i) = force(ikind)%rho_elec(i, atom_i) + fi
                     force(jkind)%rho_elec(i, atom_j) = force(jkind)%rho_elec(i, atom_j) - fi
                     fij(i) = fij(i) + fi
                  END DO
               END IF
               DEALLOCATE (gcij)
            END DO
            CALL dbcsr_iterator_stop(iter)
            ! stress tensor (needs recalculation of overlap integrals)
            IF (do_gamma_stress) THEN
               DO ikind = 1, nkind
                  DO jkind = 1, nkind
                     iac = ikind + nkind*(jkind - 1)
                     IF (.NOT. ASSOCIATED(sap_int(iac)%alist)) CYCLE
                     ! atomic parameters
                     CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
                     CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
                     CALL get_xtb_atom_param(xtb_atom_a, lao=laoa, natorb=ni)
                     CALL get_xtb_atom_param(xtb_atom_b, lao=laob, natorb=nj)
                     DO ia = 1, sap_int(iac)%nalist
                        IF (.NOT. ASSOCIATED(sap_int(iac)%alist(ia)%clist)) CYCLE
                        iatom = sap_int(iac)%alist(ia)%aatom
                        DO ic = 1, sap_int(iac)%alist(ia)%nclist
                           jatom = sap_int(iac)%alist(ia)%clist(ic)%catom
                           rij = sap_int(iac)%alist(ia)%clist(ic)%rac
                           dr = SQRT(SUM(rij(:)**2))
                           IF (dr > 1.e-6_dp) THEN
                              dsint => sap_int(iac)%alist(ia)%clist(ic)%acint
                              ALLOCATE (gcij(ni, nj))
                              DO i = 1, ni
                                 DO j = 1, nj
                                    la = laoa(i) + 1
                                    lb = laob(j) + 1
                                    gcij(i, j) = 0.5_dp*(gchrg(iatom, la, 1) + gchrg(jatom, lb, 1))
                                 END DO
                              END DO
                              gmij = 0.5_dp*(gmcharge(iatom, 1) + gmcharge(jatom, 1))
                              icol = MAX(iatom, jatom)
                              irow = MIN(iatom, jatom)
                              NULLIFY (pblock)
                              CALL dbcsr_get_block_p(matrix=matrix_p(1, 1)%matrix, &
                                                     row=irow, col=icol, block=pblock, found=found)
                              CPASSERT(found)
                              fij = 0.0_dp
                              DO i = 1, 3
                                 ! short/long range
                                 IF (irow == iatom) THEN
                                    f1 = -2.0_dp*SUM(pblock*dsint(:, :, i)*gcij)
                                    f2 = -2.0_dp*gmij*SUM(pblock*dsint(:, :, i))
                                 ELSE
                                    f1 = -2.0_dp*SUM(TRANSPOSE(pblock)*dsint(:, :, i)*gcij)
                                    f2 = -2.0_dp*gmij*SUM(TRANSPOSE(pblock)*dsint(:, :, i))
                                 END IF
                                 fij(i) = f1 + f2
                              END DO
                              DEALLOCATE (gcij)
                              fi = 1.0_dp
                              IF (iatom == jatom) fi = 0.5_dp
                              CALL virial_pair_force(virial%pv_virial, fi, fij, rij)
                           END IF
                        END DO
                     END DO
                  END DO
               END DO
            END IF
         ELSE
            NULLIFY (n_list)
            CALL get_qs_env(qs_env=qs_env, sab_orb=n_list)
            CALL neighbor_list_iterator_create(nl_iterator, n_list)
            DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
               CALL get_iterator_info(nl_iterator, ikind=ikind, jkind=jkind, &
                                      iatom=iatom, jatom=jatom, r=rij, cell=cellind)

               icol = MAX(iatom, jatom)
               irow = MIN(iatom, jatom)

               ic = cell_to_index(cellind(1), cellind(2), cellind(3))
               CPASSERT(ic > 0)

               NULLIFY (sblock)
               CALL dbcsr_get_block_p(matrix=matrix_s(1, ic)%matrix, &
                                      row=irow, col=icol, block=sblock, found=found)
               CPASSERT(found)

               ! atomic parameters
               CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
               CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
               CALL get_xtb_atom_param(xtb_atom_a, z=za, lao=laoa)
               CALL get_xtb_atom_param(xtb_atom_b, z=zb, lao=laob)

               ni = SIZE(sblock, 1)
               nj = SIZE(sblock, 2)
               ALLOCATE (gcij(ni, nj))
               DO i = 1, ni
                  DO j = 1, nj
                     IF (irow == iatom) THEN
                        la = laoa(i) + 1
                        lb = laob(j) + 1
                        gcij(i, j) = 0.5_dp*(gchrg(iatom, la, 1) + gchrg(jatom, lb, 1))
                     ELSE
                        la = laoa(j) + 1
                        lb = laob(i) + 1
                        gcij(i, j) = 0.5_dp*(gchrg(iatom, la, 1) + gchrg(jatom, lb, 1))
                     END IF
                  END DO
               END DO
               gmij = 0.5_dp*(gmcharge(iatom, 1) + gmcharge(jatom, 1))
               DO is = 1, SIZE(ks_matrix, 1)
                  NULLIFY (ksblock)
                  CALL dbcsr_get_block_p(matrix=ks_matrix(is, ic)%matrix, &
                                         row=irow, col=icol, block=ksblock, found=found)
                  CPASSERT(found)
                  ksblock = ksblock - gcij*sblock
                  ksblock = ksblock - gmij*sblock
               END DO

               IF (calculate_forces) THEN
                  atom_i = atom_of_kind(iatom)
                  atom_j = atom_of_kind(jatom)
                  IF (irow /= iatom) THEN
                     gmij = -gmij
                     gcij = -gcij
                  END IF
                  NULLIFY (pblock)
                  CALL dbcsr_get_block_p(matrix=matrix_p(1, ic)%matrix, &
                                         row=irow, col=icol, block=pblock, found=found)
                  CPASSERT(found)
                  DO i = 1, 3
                     NULLIFY (dsblock)
                     CALL dbcsr_get_block_p(matrix=matrix_s(1 + i, ic)%matrix, &
                                            row=irow, col=icol, block=dsblock, found=found)
                     CPASSERT(found)
                     fij(i) = 0.0_dp
                     ! short range
                     fi = -2.0_dp*SUM(pblock*dsblock*gcij)
                     force(ikind)%rho_elec(i, atom_i) = force(ikind)%rho_elec(i, atom_i) + fi
                     force(jkind)%rho_elec(i, atom_j) = force(jkind)%rho_elec(i, atom_j) - fi
                     fij(i) = fij(i) + fi
                     ! long range
                     fi = -2.0_dp*gmij*SUM(pblock*dsblock)
                     force(ikind)%rho_elec(i, atom_i) = force(ikind)%rho_elec(i, atom_i) + fi
                     force(jkind)%rho_elec(i, atom_j) = force(jkind)%rho_elec(i, atom_j) - fi
                     fij(i) = fij(i) + fi
                  END DO
                  IF (use_virial) THEN
                     dr = SQRT(SUM(rij(:)**2))
                     IF (dr > 1.e-6_dp) THEN
                        fi = 1.0_dp
                        IF (iatom == jatom) fi = 0.5_dp
                        CALL virial_pair_force(virial%pv_virial, fi, fij, rij)
                     END IF
                  END IF
               END IF
               DEALLOCATE (gcij)

            END DO
            CALL neighbor_list_iterator_release(nl_iterator)
         END IF

         IF (calculate_forces .AND. SIZE(matrix_p, 1) == 2) THEN
            DO img = 1, nimg
               CALL dbcsr_add(matrix_p(1, img)%matrix, matrix_p(2, img)%matrix, &
                              alpha_scalar=1.0_dp, beta_scalar=-1.0_dp)
            END DO
         END IF
      END IF

      IF (xtb_control%tb3_interaction) THEN
         CALL get_qs_env(qs_env, nkind=nkind)
         ALLOCATE (zeffk(nkind), xgamma(nkind))
         DO ikind = 1, nkind
            CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_kind)
            CALL get_xtb_atom_param(xtb_kind, xgamma=xgamma(ikind), zeff=zeffk(ikind))
         END DO
         ! Diagonal 3rd order correction (DFTB3)
         CALL build_dftb3_diagonal(qs_env, ks_matrix, rho, mcharge, energy, xgamma, zeffk, &
                                   sap_int, calculate_forces, just_energy)
         DEALLOCATE (zeffk, xgamma)
      END IF

      IF (xtb_control%do_spinpol) THEN
         CALL qs_rho_get(rho, rho_ao_kp=matrix_p)
         CALL build_xtb_spinpol(qs_env, ks_matrix, matrix_p, energy, &
                                sap_int, calculate_forces, just_energy)
      END IF

      ! QMMM
      IF (qs_env%qmmm .AND. qs_env%qmmm_periodic) THEN
         CALL build_tb_coulomb_qmqm(qs_env, ks_matrix, rho, mcharge, energy, &
                                    calculate_forces, just_energy)
      END IF

      IF (do_gamma_stress) THEN
         CALL release_sap_int(sap_int)
      END IF

      CALL timestop(handle)

   END SUBROUTINE build_xtb_coulomb

! **************************************************************************************************
!> \brief Build the periodic GFN1 Klopman-Ohno Coulomb matrix with the generalized
!>        Ewald summation of tblite 0.8.
!>        Adds the short-range residual sr and the Ewald 1/r^3 term to gchrg, while
!>        the ordinary 1/r term is still supplied by the existing long-range Ewald
!>        sum. The resulting on-site and off-site kernels are identical in form to
!>        tblite's get_amat_ko_3d, i.e. sr - 0.5*s3/gamma^2 (+ gamma on-site).
!> \param qs_env ...
!> \param charges ...
!> \param gchrg ...
!> \param calculate_forces ...
! **************************************************************************************************
   SUBROUTINE build_klopman_ohno_ewald_coulomb(qs_env, charges, gchrg, calculate_forces)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      REAL(dp), DIMENSION(:, :), INTENT(in)              :: charges
      REAL(dp), DIMENSION(:, :, :), INTENT(inout)        :: gchrg
      LOGICAL, INTENT(in)                                :: calculate_forces

      CHARACTER(len=*), PARAMETER :: routineN = 'build_klopman_ohno_ewald_coulomb'

      INTEGER :: handle, i, iatom, ikind, itr, j, jatom, jkind, k, lmaxa, lmaxb, &
         natorb_a, natorb_b, natom, ng, nkind, ni, nj, ntrans
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: kind_of
      LOGICAL                                            :: defined
      REAL(dp)                                           :: alpha, c3, deth, dfs, dhdr, &
                                                            ds3(3), dsr(3), dval(3), e, &
                                                            g2inv, gam, invr, invr2, invr3, &
                                                            invr4, k0, onsite_const, ph, pref, &
                                                            r2, sr, s3, sqrtpi, val, vol
      REAL(dp)                                           :: h_inv(3, 3), hmat(3, 3), rij(3)
      REAL(dp), DIMENSION(5)                             :: kappaa, kappab
      REAL(dp)                                           :: etaa, etab
      REAL(dp), PARAMETER                                :: s3_constant = 0.5_dp*(LOG(pi) - 2.0_dp &
                                                                                  + 0.57721566490153286061_dp &
                                                                                  + 2.0_dp*LOG(2.0_dp))
      REAL(dp), ALLOCATABLE, DIMENSION(:)                :: gw, gw3, invr3v, invr5v, invrv, r2v
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: dvec, gvec, trans
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cell_type), POINTER                           :: cell
      TYPE(ewald_environment_type), POINTER              :: ewald_env
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(xtb_atom_type), POINTER                       :: xtb_atom_a, xtb_atom_b

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env=qs_env, &
                      particle_set=particle_set, &
                      cell=cell, &
                      qs_kind_set=qs_kind_set, &
                      atomic_kind_set=atomic_kind_set, &
                      ewald_env=ewald_env, &
                      natom=natom, nkind=nkind)

      CALL get_cell(cell=cell, h=hmat, h_inv=h_inv, deth=deth)
      IF (.NOT. ASSOCIATED(ewald_env)) &
         CPABORT("XTB/KLOPMAN_OHNO_EWALD requires a periodic Poisson/Ewald setup")
      CALL ewald_env_get(ewald_env, alpha=alpha)
      vol = ABS(deth)

      CALL get_atomic_kind_set(atomic_kind_set=atomic_kind_set, kind_of=kind_of)

      CALL get_klopman_ohno_ewald_trans(hmat, h_inv, 40.0_dp, trans)
      ntrans = SIZE(trans, 2)
      CALL get_klopman_ohno_ewald_recip(hmat, h_inv, alpha, vol, &
                                EPSILON(0.0_dp), gvec, gw, gw3)
      ng = SIZE(gw)

      ALLOCATE (dvec(3, ntrans), r2v(ntrans), invrv(ntrans), invr3v(ntrans), invr5v(ntrans))

      sqrtpi = SQRT(pi)
      k0 = alpha/sqrtpi
      pref = 2.0_dp*alpha/sqrtpi
      c3 = 4.0_dp*pi/vol*(LOG(k0) + s3_constant)
      onsite_const = -4.0_dp*pi/3.0_dp*k0**3

      DO iatom = 1, natom
         ikind = kind_of(iatom)
         CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
         CALL get_xtb_atom_param(xtb_atom_a, defined=defined, natorb=natorb_a)
         IF (.NOT. defined .OR. natorb_a < 1) CYCLE
         CALL get_xtb_atom_param(xtb_atom_a, eta=etaa, lmax=lmaxa, kappa=kappaa)
         ni = lmaxa + 1
         DO jatom = 1, iatom
            jkind = kind_of(jatom)
            CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
            CALL get_xtb_atom_param(xtb_atom_b, defined=defined, natorb=natorb_b)
            IF (.NOT. defined .OR. natorb_b < 1) CYCLE
            CALL get_xtb_atom_param(xtb_atom_b, eta=etab, lmax=lmaxb, kappa=kappab)
            nj = lmaxb + 1

            rij(:) = particle_set(iatom)%r - particle_set(jatom)%r
            DO itr = 1, ntrans
               dvec(:, itr) = rij + trans(:, itr)
               r2 = SUM(dvec(:, itr)**2)
               r2v(itr) = r2
               invrv(itr) = 0.0_dp
               invr3v(itr) = 0.0_dp
               invr5v(itr) = 0.0_dp
               IF (r2 > 1.e-20_dp) THEN
                  invr = 1.0_dp/SQRT(r2)
                  invrv(itr) = invr
                  invr3v(itr) = invr**3
                  invr5v(itr) = invr**5
               END IF
            END DO

            ! Geometry-only Ewald 1/r^3 term and its Cartesian derivative.
            s3 = 0.0_dp
            ds3 = 0.0_dp
            DO itr = 1, ntrans
               IF (invrv(itr) == 0.0_dp) CYCLE
               r2 = r2v(itr)
               invr = invrv(itr)
               invr2 = invr*invr
               invr3 = invr3v(itr)
               invr4 = invr*invr3
               e = EXP(-alpha*alpha*r2)
               s3 = s3 + ERFC(alpha*SQRT(r2))*invr3 + pref*e*invr2
               dhdr = -pref*e*invr3 - 3.0_dp*ERFC(alpha*SQRT(r2))*invr4 &
                      + pref*e*(-2.0_dp*alpha*alpha*invr - 2.0_dp*invr3)
               ds3 = ds3 + dhdr*dvec(:, itr)*invr
            END DO
            DO itr = 1, ng
               ph = DOT_PRODUCT(rij, gvec(:, itr))
               s3 = s3 + gw3(itr)*COS(ph)
               ds3 = ds3 - gw3(itr)*SIN(ph)*gvec(:, itr)
            END DO
            s3 = s3 + c3
            IF (iatom == jatom) s3 = s3 + onsite_const

            DO i = 1, ni
               DO j = 1, nj
                  gam = 2.0_dp/(1.0_dp/(etaa*(1.0_dp + kappaa(i))) &
                                + 1.0_dp/(etab*(1.0_dp + kappab(j))))
                  g2inv = 1.0_dp/(gam*gam)
                  ! Klopman-Ohno short-range residual and derivative.
                  sr = 0.0_dp
                  dsr = 0.0_dp
                  DO itr = 1, ntrans
                     IF (invrv(itr) == 0.0_dp) CYCLE
                     r2 = r2v(itr)
                     invr = invrv(itr)
                     invr3 = invr3v(itr)
                     invr4 = invr*invr3
                     sr = sr + 1.0_dp/SQRT(r2 + g2inv) - invr + 0.5_dp*g2inv*invr3
                     dfs = -SQRT(r2)/(r2 + g2inv)**1.5_dp + invr*invr &
                           - 1.5_dp*g2inv*invr4
                     dsr = dsr + dfs*dvec(:, itr)*invr
                  END DO
                  val = sr - 0.5_dp*g2inv*s3
                  dval = dsr - 0.5_dp*g2inv*ds3
                  IF (iatom == jatom) val = val + gam

                  gchrg(iatom, i, 1) = gchrg(iatom, i, 1) + val*charges(jatom, j)
                  IF (iatom /= jatom) THEN
                     gchrg(jatom, j, 1) = gchrg(jatom, j, 1) + charges(iatom, i)*val
                  END IF
                  IF (calculate_forces .AND. iatom /= jatom) THEN
                     DO k = 1, 3
                        gchrg(iatom, i, k + 1) = gchrg(iatom, i, k + 1) &
                                                 + dval(k)*charges(jatom, j)
                        gchrg(jatom, j, k + 1) = gchrg(jatom, j, k + 1) &
                                                 - charges(iatom, i)*dval(k)
                     END DO
                  END IF
               END DO
            END DO
         END DO
      END DO

      DEALLOCATE (dvec, r2v, invrv, invr3v, invr5v)
      DEALLOCATE (trans, gvec, gw, gw3)
      DEALLOCATE (kind_of)

      CALL timestop(handle)

   END SUBROUTINE build_klopman_ohno_ewald_coulomb

! **************************************************************************************************
!> \brief Generate real-space lattice translations inside a cutoff.
!> \param hmat ...
!> \param h_inv ...
!> \param rcut ...
!> \param trans ...
! **************************************************************************************************
   SUBROUTINE get_klopman_ohno_ewald_trans(hmat, h_inv, rcut, trans)
      REAL(dp), DIMENSION(3, 3), INTENT(in)              :: hmat, h_inv
      REAL(dp), INTENT(in)                               :: rcut
      REAL(dp), ALLOCATABLE, DIMENSION(:, :), INTENT(out) :: trans

      INTEGER                                            :: i, itr, j, k
      INTEGER, DIMENSION(3)                              :: rep

      DO i = 1, 3
         rep(i) = CEILING(rcut*SQRT(SUM(h_inv(i, :)**2)))
      END DO
      ALLOCATE (trans(3, PRODUCT(2*rep + 1)))
      itr = 0
      DO k = -rep(3), rep(3)
         DO j = -rep(2), rep(2)
            DO i = -rep(1), rep(1)
               itr = itr + 1
               trans(:, itr) = REAL(i, dp)*hmat(:, 1) + REAL(j, dp)*hmat(:, 2) &
                               + REAL(k, dp)*hmat(:, 3)
            END DO
         END DO
      END DO

   END SUBROUTINE get_klopman_ohno_ewald_trans

! **************************************************************************************************
!> \brief Generate nonzero reciprocal vectors and the 1/r and 1/r^3 Ewald weights.
!> \param hmat ...
!> \param h_inv ...
!> \param alpha ...
!> \param vol ...
!> \param conv ...
!> \param gvec ...
!> \param gw ...
!> \param gw3 ...
! **************************************************************************************************
   SUBROUTINE get_klopman_ohno_ewald_recip(hmat, h_inv, alpha, vol, conv, gvec, gw, gw3)
      REAL(dp), DIMENSION(3, 3), INTENT(in)              :: hmat, h_inv
      REAL(dp), INTENT(in)                               :: alpha, vol, conv
      REAL(dp), ALLOCATABLE, DIMENSION(:, :), INTENT(out) :: gvec
      REAL(dp), ALLOCATABLE, DIMENSION(:), INTENT(out)   :: gw, gw3

      INTEGER                                            :: i, itr, j, k, n, ng
      INTEGER, DIMENSION(3)                              :: rep
      REAL(dp)                                           :: fac, g2, gmax, x
      REAL(dp), DIMENSION(3, 3)                          :: rec

      DO i = 1, 3
         rec(:, i) = 2.0_dp*pi*h_inv(i, :)
      END DO
      gmax = klopman_ohno_ewald_rec_cutoff(alpha, vol, conv)
      DO i = 1, 3
         rep(i) = CEILING(gmax*SQRT(SUM(hmat(:, i)**2))/(2.0_dp*pi))
      END DO
      ng = PRODUCT(2*rep + 1) - 1
      ALLOCATE (gvec(3, ng), gw(ng), gw3(ng))
      fac = 4.0_dp*pi/vol
      itr = 0
      DO k = -rep(3), rep(3)
         DO j = -rep(2), rep(2)
            DO i = -rep(1), rep(1)
               IF (i == 0 .AND. j == 0 .AND. k == 0) CYCLE
               itr = itr + 1
               gvec(:, itr) = rec(:, 1)*REAL(i, dp) + rec(:, 2)*REAL(j, dp) &
                              + rec(:, 3)*REAL(k, dp)
               g2 = SUM(gvec(:, itr)**2)
               x = g2/(4.0_dp*alpha*alpha)
               gw(itr) = fac*EXP(-x)/g2
               gw3(itr) = 0.5_dp*fac*klopman_ohno_ewald_expint_e1(x)
            END DO
         END DO
      END DO

   END SUBROUTINE get_klopman_ohno_ewald_recip

! **************************************************************************************************
!> \brief Reciprocal-space cutoff for the 1/r Ewald term.
!> \param alpha ...
!> \param vol ...
!> \param conv ...
!> \return ...
! **************************************************************************************************
   FUNCTION klopman_ohno_ewald_rec_cutoff(alpha, vol, conv) RESULT(x)
      REAL(dp), INTENT(in)                               :: alpha, vol, conv
      REAL(dp)                                           :: x

      INTEGER                                            :: iter
      REAL(dp)                                           :: xl, xr, y, yl, yr
      REAL(dp), PARAMETER                                :: init = SQRT(EPSILON(0.0_dp))
      INTEGER, PARAMETER                                 :: miter = 30

      x = init
      y = rec_value(x)
      DO WHILE (y > conv .AND. x <= HUGE(1.0_dp))
         x = 2.0_dp*x
         y = rec_value(x)
      END DO
      xl = 0.5_dp*x
      yl = rec_value(xl)
      xr = x
      yr = y
      DO iter = 1, miter
         IF (yl - yr <= conv) EXIT
         x = 0.5_dp*(xl + xr)
         y = rec_value(x)
         IF (y >= conv) THEN
            xl = x
            yl = y
         ELSE
            xr = x
            yr = y
         END IF
      END DO

   CONTAINS

      PURE FUNCTION rec_value(dist) RESULT(val)
         REAL(dp), INTENT(in)                            :: dist
         REAL(dp)                                        :: val

         val = 4.0_dp*pi*EXP(-0.25_dp*dist*dist/(alpha*alpha))/(vol*dist*dist)

      END FUNCTION rec_value

   END FUNCTION klopman_ohno_ewald_rec_cutoff

! **************************************************************************************************
!> \brief Exponential integral E1(x) for positive x (port of the tblite helper).
!> \param x ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION klopman_ohno_ewald_expint_e1(x) RESULT(e1)
      REAL(dp), INTENT(in)                               :: x
      REAL(dp)                                           :: e1

      INTEGER                                            :: iter
      REAL(dp)                                           :: a, b, c, d, delta, h, sum, term
      REAL(dp), PARAMETER                                :: euler_gamma = 0.57721566490153286061_dp, &
                                                            fpmin = 10.0_dp*TINY(1.0_dp)

      IF (x <= 1.0_dp) THEN
         term = 1.0_dp
         sum = 0.0_dp
         DO iter = 1, 100
            term = -term*x/REAL(iter, dp)
            sum = sum + term/REAL(iter, dp)
            IF (ABS(term) < EPSILON(1.0_dp)*ABS(sum)) EXIT
         END DO
         e1 = -euler_gamma - LOG(x) - sum
      ELSE
         b = x + 1.0_dp
         c = 1.0_dp/fpmin
         d = 1.0_dp/b
         h = d
         DO iter = 1, 100
            a = -REAL(iter*iter, dp)
            b = b + 2.0_dp
            d = 1.0_dp/MAX(ABS(a*d + b), fpmin)*SIGN(1.0_dp, a*d + b)
            c = b + a/c
            IF (ABS(c) < fpmin) c = fpmin
            delta = c*d
            h = h*delta
            IF (ABS(delta - 1.0_dp) < EPSILON(1.0_dp)) EXIT
         END DO
         e1 = EXP(-x)*h
      END IF

   END FUNCTION klopman_ohno_ewald_expint_e1

! **************************************************************************************************
!> \brief  Computes the short-range gamma parameter from
!>         Nataga-Mishimoto-Ohno-Klopman formula for xTB
!>         WARNING: The xTB function (gamma - 1/r) has still an l-dependent longrange (1/r^3)
!>                  behaviour. We use a cutoff function to smoothly remove this part.
!>                  However, this will change energies and effect final results.
!>
!> \param gmat ...
!> \param rab ...
!> \param nla ...
!> \param kappaa ...
!> \param etaa ...
!> \param nlb ...
!> \param kappab ...
!> \param etab ...
!> \param kg ...
!> \param rcut ...
!> \par History
!>      10.2018 JGH
!> \version 1.1
! **************************************************************************************************
   SUBROUTINE gamma_rab_sr(gmat, rab, nla, kappaa, etaa, nlb, kappab, etab, kg, rcut)
      REAL(dp), DIMENSION(:, :), INTENT(INOUT)           :: gmat
      REAL(dp), INTENT(IN)                               :: rab
      INTEGER, INTENT(IN)                                :: nla
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: kappaa
      REAL(dp), INTENT(IN)                               :: etaa
      INTEGER, INTENT(IN)                                :: nlb
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: kappab
      REAL(dp), INTENT(IN)                               :: etab, kg, rcut

      REAL(KIND=dp), PARAMETER                           :: rsmooth = 1.0_dp

      INTEGER                                            :: i, j
      REAL(KIND=dp)                                      :: fcut, r, rk, x
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: eta

      ALLOCATE (eta(nla, nlb))
      eta = 0.0_dp

      DO j = 1, nlb
         DO i = 1, nla
            eta(i, j) = 1._dp/(etaa*(1._dp + kappaa(i))) + 1._dp/(etab*(1._dp + kappab(j)))
            eta(i, j) = 2._dp/eta(i, j)
         END DO
      END DO

      gmat = 0.0_dp
      IF (rab < 1.e-6_dp) THEN
         ! on site terms
         gmat(:, :) = eta(:, :)
      ELSE IF (rab > rcut) THEN
         ! do nothing
      ELSE
         rk = rab**kg
         eta = eta**(-kg)
         IF (rab < rcut - rsmooth) THEN
            fcut = 1.0_dp
         ELSE
            r = rab - (rcut - rsmooth)
            x = r/rsmooth
            fcut = -6._dp*x**5 + 15._dp*x**4 - 10._dp*x**3 + 1._dp
         END IF
         gmat(:, :) = fcut*(1._dp/(rk + eta(:, :)))**(1._dp/kg) - fcut/rab
      END IF

      DEALLOCATE (eta)

   END SUBROUTINE gamma_rab_sr

! **************************************************************************************************
!> \brief  Computes the derivative of the short-range gamma parameter from
!>         Nataga-Mishimoto-Ohno-Klopman formula for xTB
!>         WARNING: The xTB function (gamma - 1/r) has still an l-dependent longrange (1/r^3)
!>                  behaviour. We use a cutoff function to smoothly remove this part.
!>                  However, this will change energies and effect final results.
!>
!> \param dgmat ...
!> \param rab ...
!> \param nla ...
!> \param kappaa ...
!> \param etaa ...
!> \param nlb ...
!> \param kappab ...
!> \param etab ...
!> \param kg ...
!> \param rcut ...
!> \par History
!>      10.2018 JGH
!> \version 1.1
! **************************************************************************************************
   SUBROUTINE dgamma_rab_sr(dgmat, rab, nla, kappaa, etaa, nlb, kappab, etab, kg, rcut)
      REAL(dp), DIMENSION(:, :), INTENT(INOUT)           :: dgmat
      REAL(dp), INTENT(IN)                               :: rab
      INTEGER, INTENT(IN)                                :: nla
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: kappaa
      REAL(dp), INTENT(IN)                               :: etaa
      INTEGER, INTENT(IN)                                :: nlb
      REAL(dp), DIMENSION(:), INTENT(IN)                 :: kappab
      REAL(dp), INTENT(IN)                               :: etab, kg, rcut

      REAL(KIND=dp), PARAMETER                           :: eps_rab = 1.0E-06_dp, rsmooth = 1.0_dp

      INTEGER                                            :: i, j
      REAL(KIND=dp)                                      :: dfcut, fcut, r, rk, x
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: eta

      ALLOCATE (eta(nla, nlb))

      DO j = 1, nlb
         DO i = 1, nla
            eta(i, j) = 1._dp/(etaa*(1._dp + kappaa(i))) + 1._dp/(etab*(1._dp + kappab(j)))
            eta(i, j) = 2._dp/eta(i, j)
         END DO
      END DO

      IF (rab < eps_rab) THEN
         ! on site terms
         dgmat(:, :) = 0.0_dp
      ELSE IF (rab > rcut) THEN
         dgmat(:, :) = 0.0_dp
      ELSE
         eta = eta**(-kg)
         rk = rab**kg
         IF (rab < rcut - rsmooth) THEN
            fcut = 1.0_dp
            dfcut = 0.0_dp
         ELSE
            r = rab - (rcut - rsmooth)
            x = r/rsmooth
            fcut = -6._dp*x**5 + 15._dp*x**4 - 10._dp*x**3 + 1._dp
            dfcut = -30._dp*x**4 + 60._dp*x**3 - 30._dp*x**2
            dfcut = dfcut/rsmooth
         END IF
         dgmat(:, :) = dfcut*(1._dp/(rk + eta(:, :)))**(1._dp/kg)
         dgmat(:, :) = dgmat(:, :) - dfcut/rab + fcut/rab**2
         dgmat(:, :) = dgmat(:, :) - fcut/(rk + eta(:, :))*(1._dp/(rk + eta(:, :)))**(1._dp/kg)*rk/rab
      END IF

      DEALLOCATE (eta)

   END SUBROUTINE dgamma_rab_sr

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param sap_int ...
! **************************************************************************************************
   SUBROUTINE xtb_dsint_list(qs_env, sap_int)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(sap_int_type), DIMENSION(:), POINTER          :: sap_int

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'xtb_dsint_list'

      INTEGER :: handle, i, iac, iatom, ikind, ilist, iset, jatom, jkind, jneighbor, jset, ldsab, &
         n1, n2, natorb_a, natorb_b, ncoa, ncob, nkind, nlist, nneighbor, nseta, nsetb, sgfa, sgfb
      INTEGER, DIMENSION(3)                              :: cell
      INTEGER, DIMENSION(:), POINTER                     :: la_max, la_min, lb_max, lb_min, npgfa, &
                                                            npgfb, nsgfa, nsgfb
      INTEGER, DIMENSION(:, :), POINTER                  :: first_sgfa, first_sgfb
      LOGICAL                                            :: defined
      REAL(KIND=dp)                                      :: dr
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: owork
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: oint, sint
      REAL(KIND=dp), DIMENSION(3)                        :: rij
      REAL(KIND=dp), DIMENSION(:), POINTER               :: set_radius_a, set_radius_b
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: rpgfa, rpgfb, scon_a, scon_b, zeta, zetb
      TYPE(clist_type), POINTER                          :: clist
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: nl_iterator
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(xtb_atom_type), POINTER                       :: xtb_atom_a, xtb_atom_b

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env=qs_env, nkind=nkind)
      CPASSERT(.NOT. ASSOCIATED(sap_int))
      ALLOCATE (sap_int(nkind*nkind))
      DO i = 1, nkind*nkind
         NULLIFY (sap_int(i)%alist, sap_int(i)%asort, sap_int(i)%aindex)
         sap_int(i)%nalist = 0
      END DO

      CALL get_qs_env(qs_env=qs_env, &
                      qs_kind_set=qs_kind_set, &
                      dft_control=dft_control, &
                      sab_orb=sab_orb)

      ! set up basis set lists
      ALLOCATE (basis_set_list(nkind))
      CALL basis_set_list_setup(basis_set_list, "ORB", qs_kind_set)

      ! loop over all atom pairs with a non-zero overlap (sab_orb)
      CALL neighbor_list_iterator_create(nl_iterator, sab_orb)
      DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
         CALL get_iterator_info(nl_iterator, ikind=ikind, jkind=jkind, iatom=iatom, &
                                jatom=jatom, nlist=nlist, ilist=ilist, nnode=nneighbor, &
                                inode=jneighbor, cell=cell, r=rij)
         iac = ikind + nkind*(jkind - 1)
         !
         CALL get_qs_kind(qs_kind_set(ikind), xtb_parameter=xtb_atom_a)
         CALL get_xtb_atom_param(xtb_atom_a, defined=defined, natorb=natorb_a)
         IF (.NOT. defined .OR. natorb_a < 1) CYCLE
         CALL get_qs_kind(qs_kind_set(jkind), xtb_parameter=xtb_atom_b)
         CALL get_xtb_atom_param(xtb_atom_b, defined=defined, natorb=natorb_b)
         IF (.NOT. defined .OR. natorb_b < 1) CYCLE

         dr = SQRT(SUM(rij(:)**2))

         ! integral list
         IF (.NOT. ASSOCIATED(sap_int(iac)%alist)) THEN
            sap_int(iac)%a_kind = ikind
            sap_int(iac)%p_kind = jkind
            sap_int(iac)%nalist = nlist
            ALLOCATE (sap_int(iac)%alist(nlist))
            DO i = 1, nlist
               NULLIFY (sap_int(iac)%alist(i)%clist)
               sap_int(iac)%alist(i)%aatom = 0
               sap_int(iac)%alist(i)%nclist = 0
            END DO
         END IF
         IF (.NOT. ASSOCIATED(sap_int(iac)%alist(ilist)%clist)) THEN
            sap_int(iac)%alist(ilist)%aatom = iatom
            sap_int(iac)%alist(ilist)%nclist = nneighbor
            ALLOCATE (sap_int(iac)%alist(ilist)%clist(nneighbor))
            DO i = 1, nneighbor
               sap_int(iac)%alist(ilist)%clist(i)%catom = 0
            END DO
         END IF
         clist => sap_int(iac)%alist(ilist)%clist(jneighbor)
         clist%catom = jatom
         clist%cell = cell
         clist%rac = rij
         ALLOCATE (clist%acint(natorb_a, natorb_b, 3))
         NULLIFY (clist%achint)
         clist%acint = 0._dp
         clist%nsgf_cnt = 0
         NULLIFY (clist%sgf_list)

         ! overlap
         basis_set_a => basis_set_list(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE
         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         scon_a => basis_set_a%scon
         zeta => basis_set_a%zet
         ! basis jkind
         first_sgfb => basis_set_b%first_sgf
         lb_max => basis_set_b%lmax
         lb_min => basis_set_b%lmin
         npgfb => basis_set_b%npgf
         nsetb = basis_set_b%nset
         nsgfb => basis_set_b%nsgf_set
         rpgfb => basis_set_b%pgf_radius
         set_radius_b => basis_set_b%set_radius
         scon_b => basis_set_b%scon
         zetb => basis_set_b%zet

         ldsab = get_memory_usage(qs_kind_set, "ORB", "ORB")
         ALLOCATE (oint(ldsab, ldsab, 4), owork(ldsab, ldsab))
         ALLOCATE (sint(natorb_a, natorb_b, 4))
         sint = 0.0_dp

         DO iset = 1, nseta
            ncoa = npgfa(iset)*ncoset(la_max(iset))
            n1 = npgfa(iset)*(ncoset(la_max(iset)) - ncoset(la_min(iset) - 1))
            sgfa = first_sgfa(1, iset)
            DO jset = 1, nsetb
               IF (set_radius_a(iset) + set_radius_b(jset) < dr) CYCLE
               ncob = npgfb(jset)*ncoset(lb_max(jset))
               n2 = npgfb(jset)*(ncoset(lb_max(jset)) - ncoset(lb_min(jset) - 1))
               sgfb = first_sgfb(1, jset)
               CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                               lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                               rij, sab=oint(:, :, 1), dab=oint(:, :, 2:4))
               ! Contraction
               DO i = 1, 4
                  CALL contraction(oint(:, :, i), owork, ca=scon_a(:, sgfa:), na=n1, ma=nsgfa(iset), &
                                   cb=scon_b(:, sgfb:), nb=n2, mb=nsgfb(jset), fscale=1.0_dp, trans=.FALSE.)
                  CALL block_add("IN", owork, nsgfa(iset), nsgfb(jset), sint(:, :, i), &
                                 sgfa, sgfb, trans=.FALSE.)
               END DO
            END DO
         END DO
         ! update dS/dR matrix
         clist%acint(1:natorb_a, 1:natorb_b, 1:3) = sint(1:natorb_a, 1:natorb_b, 2:4)

         DEALLOCATE (oint, owork, sint)

      END DO
      CALL neighbor_list_iterator_release(nl_iterator)

      DEALLOCATE (basis_set_list)

      CALL timestop(handle)

   END SUBROUTINE xtb_dsint_list

END MODULE xtb_coulomb
