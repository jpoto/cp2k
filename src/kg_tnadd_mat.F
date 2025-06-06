!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
! **************************************************************************************************
!> \brief Calculation of the local potential contribution of the nonadditive kinetic energy
!>         <a|V(local)|b> = <a|Sum e^a*rc**2|b>
!> \par History
!>      - adapted from core_ppl
! **************************************************************************************************
MODULE kg_tnadd_mat
   USE ai_overlap_ppl,                  ONLY: ppl_integral
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: gto_basis_set_p_type,&
                                              gto_basis_set_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_add,&
                                              dbcsr_create,&
                                              dbcsr_distribution_type,&
                                              dbcsr_finalize,&
                                              dbcsr_get_block_p,&
                                              dbcsr_p_type,&
                                              dbcsr_set,&
                                              dbcsr_type_symmetric
   USE cp_dbcsr_cp2k_link,              ONLY: cp_dbcsr_alloc_block_from_nbl
   USE cp_dbcsr_operations,             ONLY: dbcsr_allocate_matrix_set,&
                                              dbcsr_deallocate_matrix_set
   USE external_potential_types,        ONLY: get_potential,&
                                              local_potential_type
   USE kg_environment_types,            ONLY: kg_environment_type
   USE kinds,                           ONLY: dp
   USE orbital_pointers,                ONLY: init_orbital_pointers,&
                                              ncoset
   USE particle_methods,                ONLY: get_particle_set
   USE particle_types,                  ONLY: particle_type
   USE qs_force_types,                  ONLY: qs_force_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              get_qs_kind_set,&
                                              qs_kind_type
   USE qs_neighbor_list_types,          ONLY: get_iterator_info,&
                                              neighbor_list_iterate,&
                                              neighbor_list_iterator_create,&
                                              neighbor_list_iterator_p_type,&
                                              neighbor_list_iterator_release,&
                                              neighbor_list_set_p_type,&
                                              nl_set_sub_iterator,&
                                              nl_sub_iterate
   USE virial_methods,                  ONLY: virial_pair_force
   USE virial_types,                    ONLY: virial_type

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'kg_tnadd_mat'

   PUBLIC :: build_tnadd_mat

CONTAINS

!==========================================================================================================

! **************************************************************************************************
!> \brief ...
!> \param kg_env ...
!> \param matrix_p ...
!> \param force ...
!> \param virial ...
!> \param calculate_forces ...
!> \param use_virial ...
!> \param qs_kind_set ...
!> \param atomic_kind_set ...
!> \param particle_set ...
!> \param sab_orb ...
!> \param dbcsr_dist ...
! **************************************************************************************************
   SUBROUTINE build_tnadd_mat(kg_env, matrix_p, force, virial, calculate_forces, use_virial, &
                              qs_kind_set, atomic_kind_set, particle_set, sab_orb, dbcsr_dist)

      TYPE(kg_environment_type), POINTER                 :: kg_env
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_p
      TYPE(qs_force_type), DIMENSION(:), POINTER         :: force
      TYPE(virial_type), POINTER                         :: virial
      LOGICAL, INTENT(IN)                                :: calculate_forces
      LOGICAL                                            :: use_virial
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      TYPE(dbcsr_distribution_type), POINTER             :: dbcsr_dist

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'build_tnadd_mat'
      INTEGER, PARAMETER                                 :: nexp_max = 10

      INTEGER :: atom_a, atom_b, atom_c, handle, i, iatom, icol, ikind, img, imol, inode, irow, &
         iset, jatom, jkind, jmol, jset, katom, kkind, kmol, last_iatom, last_jatom, ldai, ldsab, &
         maxco, maxder, maxl, maxlgto, maxnset, maxpol, maxsgf, mepos, natom, ncoa, ncob, nder, &
         ngau, nkind, npol, nseta, nsetb, nthread, sgfa, sgfb
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: atom_of_kind
      INTEGER, DIMENSION(:), POINTER                     :: atom_to_molecule, col_blk_sizes, la_max, &
                                                            la_min, lb_max, lb_min, npgfa, npgfb, &
                                                            nsgfa, nsgfb, row_blk_sizes
      INTEGER, DIMENSION(:, :), POINTER                  :: first_sgfa, first_sgfb
      INTEGER, DIMENSION(nexp_max)                       :: nct
      LOGICAL                                            :: found, new_atom_pair
      REAL(KIND=dp)                                      :: dab, dac, dbc, f0, radius
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: work
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: ppl_fwork, ppl_work
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :, :)  :: hab, pab
      REAL(KIND=dp), DIMENSION(3)                        :: force_a, force_b, rab, rac, rbc
      REAL(KIND=dp), DIMENSION(:), POINTER               :: alpha, set_radius_a, set_radius_b
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: ccval, cval, h_block, p_block, rpgfa, &
                                                            rpgfb, sphi_a, sphi_b, zeta, zetb
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_kg
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(local_potential_type), POINTER                :: tnadd_potential
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: ap_iterator, nl_iterator
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sac_kin

      IF (calculate_forces) THEN
         CALL timeset(routineN//"_forces", handle)
      ELSE
         CALL timeset(routineN, handle)
      END IF

      NULLIFY (matrix_kg)
      IF (ASSOCIATED(kg_env%tnadd_mat)) THEN
         CALL dbcsr_deallocate_matrix_set(kg_env%tnadd_mat)
      END IF
      sac_kin => kg_env%sac_kin
      atom_to_molecule => kg_env%atom_to_molecule

      nkind = SIZE(atomic_kind_set)
      natom = SIZE(particle_set)

      CALL get_atomic_kind_set(atomic_kind_set=atomic_kind_set, atom_of_kind=atom_of_kind)

      IF (calculate_forces) THEN
         nder = 1
         IF (SIZE(matrix_p, 1) == 2) THEN
            DO img = 1, SIZE(matrix_p, 2)
               CALL dbcsr_add(matrix_p(1, img)%matrix, matrix_p(2, img)%matrix, &
                              alpha_scalar=1.0_dp, beta_scalar=1.0_dp)
               CALL dbcsr_add(matrix_p(2, img)%matrix, matrix_p(1, img)%matrix, &
                              alpha_scalar=-2.0_dp, beta_scalar=1.0_dp)
            END DO
         END IF
      ELSE
         nder = 0
      END IF

      maxder = ncoset(nder)

      CALL get_qs_kind_set(qs_kind_set, maxpol=maxpol, maxco=maxco, maxlgto=maxlgto, &
                           maxsgf=maxsgf, maxnset=maxnset)

      maxl = MAX(maxlgto, maxpol)
      CALL init_orbital_pointers(maxl + nder + 1)

      ldsab = MAX(maxco, ncoset(maxpol), maxsgf, maxpol)
      ldai = ncoset(maxl + nder + 1)

      ALLOCATE (basis_set_list(nkind))
      DO ikind = 1, nkind
         CALL get_qs_kind(qs_kind_set(ikind), basis_set=basis_set_a)
         IF (ASSOCIATED(basis_set_a)) THEN
            basis_set_list(ikind)%gto_basis_set => basis_set_a
         ELSE
            NULLIFY (basis_set_list(ikind)%gto_basis_set)
         END IF
      END DO

      ! build the matrix
      ALLOCATE (row_blk_sizes(natom), col_blk_sizes(natom))

      CALL get_particle_set(particle_set, qs_kind_set, nsgf=row_blk_sizes)
      CALL get_particle_set(particle_set, qs_kind_set, nsgf=col_blk_sizes)

      CALL dbcsr_allocate_matrix_set(matrix_kg, 1)

      ALLOCATE (matrix_kg(1)%matrix)
      CALL dbcsr_create(matrix=matrix_kg(1)%matrix, &
                        name="Nonadditive kinetic energy potential", &
                        dist=dbcsr_dist, matrix_type=dbcsr_type_symmetric, &
                        row_blk_size=row_blk_sizes, col_blk_size=col_blk_sizes, &
                        reuse_arrays=.TRUE.)
      CALL cp_dbcsr_alloc_block_from_nbl(matrix_kg(1)%matrix, sab_orb)
      CALL dbcsr_set(matrix_kg(1)%matrix, 0.0_dp)

      nthread = 1
!$    nthread = omp_get_max_threads()

      CALL neighbor_list_iterator_create(nl_iterator, sab_orb, nthread=nthread)
      ! iterator for basis/potential list
      CALL neighbor_list_iterator_create(ap_iterator, sac_kin, search=.TRUE., nthread=nthread)

!$OMP PARALLEL &
!$OMP DEFAULT (NONE) &
!$OMP SHARED  (nl_iterator, ap_iterator, basis_set_list, calculate_forces, force, use_virial,&
!$OMP          matrix_kg, matrix_p,virial, atomic_kind_set, qs_kind_set, particle_set,&
!$OMP          sab_orb, sac_kin, nthread, ncoset, nkind,&
!$OMP          atom_of_kind, ldsab,  maxnset, maxder, &
!$OMP          maxlgto, nder, maxco, atom_to_molecule) &
!$OMP PRIVATE (ikind, jkind, inode, iatom, jatom, rab, basis_set_a, basis_set_b, atom_a, &
!$OMP          atom_b, first_sgfa, la_max, la_min, npgfa, nsgfa, sphi_a, &
!$OMP          zeta, first_sgfb, lb_max, lb_min, npgfb, nsetb, rpgfb, set_radius_b, sphi_b, &
!$OMP          zetb, last_iatom, last_jatom, new_atom_pair, dab, irow, icol, h_block, found, iset, ncoa, &
!$OMP          sgfa, jset, ncob, sgfb, nsgfb, p_block, work, pab, hab, kkind, nseta, &
!$OMP          radius, alpha, nct, imol, jmol, kmol,&
!$OMP          npol, ngau, cval, ccval, rac, dac, rbc, dbc, &
!$OMP          set_radius_a,  rpgfa, force_a, force_b, ppl_fwork, mepos, &
!$OMP          f0, katom, ppl_work, atom_c,&
!$OMP          ldai,tnadd_potential)

      mepos = 0
!$    mepos = omp_get_thread_num()

      ALLOCATE (hab(ldsab, ldsab, maxnset, maxnset), work(ldsab, ldsab*maxder))
      ldai = ncoset(2*maxlgto + 2*nder)
      ALLOCATE (ppl_work(ldai, ldai, MAX(maxder, 2*maxlgto + 2*nder + 1)))
      IF (calculate_forces) THEN
         ALLOCATE (pab(maxco, maxco, maxnset, maxnset))
         ldai = ncoset(maxlgto)
         ALLOCATE (ppl_fwork(ldai, ldai, maxder))
      END IF

      last_iatom = 0
      last_jatom = 0
      DO WHILE (neighbor_list_iterate(nl_iterator, mepos=mepos) == 0)

         CALL get_iterator_info(nl_iterator, mepos=mepos, ikind=ikind, jkind=jkind, inode=inode, &
                                iatom=iatom, jatom=jatom, r=rab)

         basis_set_a => basis_set_list(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE

         atom_a = atom_of_kind(iatom)
         atom_b = atom_of_kind(jatom)
         imol = atom_to_molecule(iatom)
         jmol = atom_to_molecule(jatom)
         CPASSERT(imol == jmol)

         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         sphi_a => basis_set_a%sphi
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
         sphi_b => basis_set_b%sphi
         zetb => basis_set_b%zet

         dab = SQRT(SUM(rab*rab))

         IF (iatom == last_iatom .AND. jatom == last_jatom) THEN
            new_atom_pair = .FALSE.
         ELSE
            new_atom_pair = .TRUE.
            last_jatom = jatom
            last_iatom = iatom
         END IF

         ! *** Use the symmetry of the first derivatives ***
         IF (iatom == jatom) THEN
            f0 = 1.0_dp
         ELSE
            f0 = 2.0_dp
         END IF

         ! *** Create matrix blocks for a new matrix block column ***
         IF (new_atom_pair) THEN
            IF (iatom <= jatom) THEN
               irow = iatom
               icol = jatom
            ELSE
               irow = jatom
               icol = iatom
            END IF
            NULLIFY (h_block)
            CALL dbcsr_get_block_p(matrix_kg(1)%matrix, irow, icol, h_block, found)
            IF (ASSOCIATED(h_block)) THEN
            IF (calculate_forces) THEN
               CPASSERT(SIZE(matrix_p, 2) == 1)
               NULLIFY (p_block)
               CALL dbcsr_get_block_p(matrix_p(1, 1)%matrix, irow, icol, p_block, found)
               IF (ASSOCIATED(p_block)) THEN
                  DO iset = 1, nseta
                     ncoa = npgfa(iset)*ncoset(la_max(iset))
                     sgfa = first_sgfa(1, iset)
                     DO jset = 1, nsetb
                        ncob = npgfb(jset)*ncoset(lb_max(jset))
                        sgfb = first_sgfb(1, jset)

                        ! *** Decontract density matrix block ***
                        IF (iatom <= jatom) THEN
                           CALL dgemm("N", "N", ncoa, nsgfb(jset), nsgfa(iset), &
                                      1.0_dp, sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                                      p_block(sgfa, sgfb), SIZE(p_block, 1), &
                                      0.0_dp, work(1, 1), SIZE(work, 1))
                        ELSE
                           CALL dgemm("N", "T", ncoa, nsgfb(jset), nsgfa(iset), &
                                      1.0_dp, sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                                      p_block(sgfb, sgfa), SIZE(p_block, 1), &
                                      0.0_dp, work(1, 1), SIZE(work, 1))
                        END IF

                        CALL dgemm("N", "T", ncoa, ncob, nsgfb(jset), &
                                   1.0_dp, work(1, 1), SIZE(work, 1), &
                                   sphi_b(1, sgfb), SIZE(sphi_b, 1), &
                                   0.0_dp, pab(1, 1, iset, jset), SIZE(pab, 1))
                     END DO
                  END DO
               END IF
            END IF
            END IF
         END IF

         hab = 0._dp

         ! loop over all kinds for nonadditive kinetic energy potential atoms
         DO kkind = 1, nkind

            CALL get_qs_kind(qs_kind_set(kkind), tnadd_potential=tnadd_potential)
            IF (.NOT. ASSOCIATED(tnadd_potential)) CYCLE
            CALL get_potential(potential=tnadd_potential, &
                               alpha=alpha, cval=cval, ngau=ngau, npol=npol, radius=radius)
            nct = npol
            ALLOCATE (ccval(npol, ngau))
            ccval(1:npol, 1:ngau) = TRANSPOSE(cval(1:ngau, 1:npol))

            CALL nl_set_sub_iterator(ap_iterator, ikind, kkind, iatom, mepos)

            DO WHILE (nl_sub_iterate(ap_iterator, mepos) == 0)

               CALL get_iterator_info(ap_iterator, mepos, jatom=katom, r=rac)

               ! this potential only acts on other moleclules
               kmol = atom_to_molecule(katom)
               IF (kmol == imol) CYCLE

               dac = SQRT(SUM(rac*rac))
               rbc(:) = rac(:) - rab(:)
               dbc = SQRT(SUM(rbc*rbc))
               IF ((MAXVAL(set_radius_a(:)) + radius < dac) .OR. &
                   (MAXVAL(set_radius_b(:)) + radius < dbc)) THEN
                  CYCLE
               END IF

               DO iset = 1, nseta
                  IF (set_radius_a(iset) + radius < dac) CYCLE
                  ncoa = npgfa(iset)*ncoset(la_max(iset))
                  sgfa = first_sgfa(1, iset)
                  DO jset = 1, nsetb
                     IF (set_radius_b(jset) + radius < dbc) CYCLE
                     ncob = npgfb(jset)*ncoset(lb_max(jset))
                     sgfb = first_sgfb(1, jset)
                     IF (set_radius_a(iset) + set_radius_b(jset) < dab) CYCLE
                     ! *** Calculate the GTH pseudo potential forces ***
                     IF (calculate_forces) THEN

                        CALL ppl_integral( &
                           la_max(iset), la_min(iset), npgfa(iset), &
                           rpgfa(:, iset), zeta(:, iset), &
                           lb_max(jset), lb_min(jset), npgfb(jset), &
                           rpgfb(:, jset), zetb(:, jset), &
                           ngau, alpha, nct, ccval, radius, &
                           rab, dab, rac, dac, rbc, dbc, &
                           hab(:, :, iset, jset), ppl_work, pab(:, :, iset, jset), &
                           force_a, force_b, ppl_fwork)
                        ! *** The derivatives w.r.t. atomic center c are    ***
                        ! *** calculated using the translational invariance ***
                        ! *** of the first derivatives                      ***
                        atom_c = atom_of_kind(katom)

!$OMP CRITICAL(force_critical)
                        force(ikind)%kinetic(1, atom_a) = force(ikind)%kinetic(1, atom_a) + f0*force_a(1)
                        force(ikind)%kinetic(2, atom_a) = force(ikind)%kinetic(2, atom_a) + f0*force_a(2)
                        force(ikind)%kinetic(3, atom_a) = force(ikind)%kinetic(3, atom_a) + f0*force_a(3)
                        force(kkind)%kinetic(1, atom_c) = force(kkind)%kinetic(1, atom_c) - f0*force_a(1)
                        force(kkind)%kinetic(2, atom_c) = force(kkind)%kinetic(2, atom_c) - f0*force_a(2)
                        force(kkind)%kinetic(3, atom_c) = force(kkind)%kinetic(3, atom_c) - f0*force_a(3)

                        force(jkind)%kinetic(1, atom_b) = force(jkind)%kinetic(1, atom_b) + f0*force_b(1)
                        force(jkind)%kinetic(2, atom_b) = force(jkind)%kinetic(2, atom_b) + f0*force_b(2)
                        force(jkind)%kinetic(3, atom_b) = force(jkind)%kinetic(3, atom_b) + f0*force_b(3)
                        force(kkind)%kinetic(1, atom_c) = force(kkind)%kinetic(1, atom_c) - f0*force_b(1)
                        force(kkind)%kinetic(2, atom_c) = force(kkind)%kinetic(2, atom_c) - f0*force_b(2)
                        force(kkind)%kinetic(3, atom_c) = force(kkind)%kinetic(3, atom_c) - f0*force_b(3)

                        IF (use_virial) THEN
                           CALL virial_pair_force(virial%pv_virial, f0, force_a, rac)
                           CALL virial_pair_force(virial%pv_virial, f0, force_b, rbc)
                        END IF
!$OMP END CRITICAL(force_critical)

                     ELSE
                        CALL ppl_integral( &
                           la_max(iset), la_min(iset), npgfa(iset), &
                           rpgfa(:, iset), zeta(:, iset), &
                           lb_max(jset), lb_min(jset), npgfb(jset), &
                           rpgfb(:, jset), zetb(:, jset), &
                           ngau, alpha, nct, ccval, radius, &
                           rab, dab, rac, dac, rbc, dbc, hab(:, :, iset, jset), ppl_work)
                     END IF
                  END DO
               END DO
            END DO
            DEALLOCATE (ccval)
         END DO

         ! *** Contract integrals
         DO iset = 1, nseta
            ncoa = npgfa(iset)*ncoset(la_max(iset))
            sgfa = first_sgfa(1, iset)
            DO jset = 1, nsetb
               ncob = npgfb(jset)*ncoset(lb_max(jset))
               sgfb = first_sgfb(1, jset)

               CALL dgemm("N", "N", ncoa, nsgfb(jset), ncob, &
                          1.0_dp, hab(1, 1, iset, jset), SIZE(hab, 1), &
                          sphi_b(1, sgfb), SIZE(sphi_b, 1), &
                          0.0_dp, work(1, 1), SIZE(work, 1))

!$OMP CRITICAL(h_block_critical)
               IF (iatom <= jatom) THEN
                  CALL dgemm("T", "N", nsgfa(iset), nsgfb(jset), ncoa, &
                             1.0_dp, sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                             work(1, 1), SIZE(work, 1), &
                             1.0_dp, h_block(sgfa, sgfb), SIZE(h_block, 1))
               ELSE
                  CALL dgemm("T", "N", nsgfb(jset), nsgfa(iset), ncoa, &
                             1.0_dp, work(1, 1), SIZE(work, 1), &
                             sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                             1.0_dp, h_block(sgfb, sgfa), SIZE(h_block, 1))
               END IF
!$OMP END CRITICAL(h_block_critical)

            END DO
         END DO
      END DO

      DEALLOCATE (hab, work, ppl_work)

      IF (calculate_forces) THEN
         DEALLOCATE (pab, ppl_fwork)
      END IF

!$OMP END PARALLEL

      CALL neighbor_list_iterator_release(ap_iterator)
      CALL neighbor_list_iterator_release(nl_iterator)

      DO i = 1, SIZE(matrix_kg)
         CALL dbcsr_finalize(matrix_kg(i)%matrix)
      END DO
      kg_env%tnadd_mat => matrix_kg

      DEALLOCATE (basis_set_list)

      IF (calculate_forces) THEN
         ! *** If LSD, then recover alpha density and beta density     ***
         ! *** from the total density (1) and the spin density (2)     ***
         IF (SIZE(matrix_p, 1) == 2) THEN
            DO img = 1, SIZE(matrix_p, 2)
               CALL dbcsr_add(matrix_p(1, img)%matrix, matrix_p(2, img)%matrix, &
                              alpha_scalar=0.5_dp, beta_scalar=0.5_dp)
               CALL dbcsr_add(matrix_p(2, img)%matrix, matrix_p(1, img)%matrix, &
                              alpha_scalar=-1.0_dp, beta_scalar=1.0_dp)
            END DO
         END IF
      END IF

      CALL timestop(handle)

   END SUBROUTINE build_tnadd_mat

!==========================================================================================================

END MODULE kg_tnadd_mat
