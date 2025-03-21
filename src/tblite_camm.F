!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Calculates the cumulative atomic multipole moments
!> \par History
!> \author JVP (07.03.2025)
! **************************************************************************************************
MODULE tblite_camm
   USE ai_moments,                      ONLY: moment
   USE basis_set_types,                 ONLY: gto_basis_set_p_type,&
                                              gto_basis_set_type
   USE block_p_types,                   ONLY: block_p_type
   USE cell_types,                      ONLY: cell_type,&
                                              pbc
   USE cp_dbcsr_api,                    ONLY: dbcsr_get_block_p,&
                                              dbcsr_p_type
   USE cp_dbcsr_operations,             ONLY: dbcsr_allocate_matrix_set,&
                                              dbcsr_deallocate_matrix_set
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_copy, dbcsr_create, dbcsr_deallocate_matrix, dbcsr_distribution_type, dbcsr_dot, &
        dbcsr_get_block_p, dbcsr_init_p, dbcsr_multiply, dbcsr_p_type, dbcsr_set, dbcsr_trace, &
        dbcsr_type, dbcsr_type_antisymmetric, dbcsr_type_no_symmetry, dbcsr_type_symmetric, dbcsr_print
   USE kinds,                           ONLY: dp
   USE orbital_pointers,                ONLY: ncoset
   USE particle_types,                  ONLY: particle_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              get_qs_kind_set,&
                                              qs_kind_type
   USE qs_moments,                      ONLY: build_local_moment_matrix
   USE qs_neighbor_list_types,          ONLY: get_iterator_info,&
                                              neighbor_list_iterate,&
                                              neighbor_list_iterator_create,&
                                              neighbor_list_iterator_p_type,&
                                              neighbor_list_iterator_release,&
                                              neighbor_list_set_p_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'tblite_camm'

   ! Public subroutines
   PUBLIC :: compute_moments, get_camm

CONTAINS

! **************************************************************************************************
!> \brief compute moments
!> \param qs_env ...
!> \param nmoments ...
!> \param moments ...
!> \param ref_point ...
!> \note JVP 2025
! **************************************************************************************************
   SUBROUTINE get_camm(qs_env)
   
      TYPE(qs_environment_type), POINTER                 :: qs_env
      
      INTEGER                                            :: i, j, k
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: dip_mom
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER        :: quad_mom
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks
      
      NULLIFY (matrix_ks)
      CALL get_qs_env(qs_env=qs_env, matrix_ks=matrix_ks)      
      
      NULLIFY (dip_mom)
      CALL dbcsr_allocate_matrix_set(dip_mom, 3)
      
      DO i = 1, 3
         CALL dbcsr_init_p(dip_mom(i)%matrix)
         CALL dbcsr_copy(dip_mom(i)%matrix, matrix_ks(1)%matrix, "tblite dipole")
         CALL dbcsr_set(dip_mom(i)%matrix, 0.0_dp)
      END DO
      write(*,*) 'ssssssssssssssssssss  multipole moment computation: dipole'
      !CALL build_local_moment_matrix(qs_env, dip_mom, 1, ref_point=[0._dp, 0._dp, 0._dp])
      CALL compute_moments(qs_env, 1, dip_mom)
      
      DO i = 1, 3
         CALL dbcsr_print(dip_mom(i)%matrix)
      END DO
      
      NULLIFY (quad_mom)
      CALL dbcsr_allocate_matrix_set(quad_mom, 9)
      
      DO i = 1, 3
         DO j = 1, 3
            k=(i-1)*3+j
            CALL dbcsr_init_p(quad_mom(k)%matrix)
            CALL dbcsr_copy(quad_mom(k)%matrix, matrix_ks(1)%matrix, "tblite quadrupole")
            CALL dbcsr_set(quad_mom(k)%matrix, 0.0_dp)
         END DO
      END DO
      write(*,*) 'ssssssssssssssssssss  multipole moment computation: quadrupole'
      CALL build_local_moment_matrix(qs_env, quad_mom, 2, ref_point=[0._dp, 0._dp, 0._dp])
      !CALL compute_moments(qs_env, nmoments, moments)
      
      DO i = 1, 3
         DO j = 1, 3
         k=(i-1)*3+j
         CALL dbcsr_print(quad_mom(k)%matrix)
         END DO
      END DO
      
      CALL dbcsr_deallocate_matrix_set(quad_mom)
      
   END SUBROUTINE


! **************************************************************************************************
!> \brief compute moments
!> \param qs_env ...
!> \param nmoments ...
!> \param moments ...
!> \param ref_point ...
!> \note JVP 2025
! **************************************************************************************************
   SUBROUTINE compute_moments(qs_env, nmoments, moments, ref_point)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      INTEGER, INTENT(IN)                                :: nmoments
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(INOUT), &
         POINTER                                         :: moments
      REAL(KIND=dp), DIMENSION(:), INTENT(IN), OPTIONAL  :: ref_point

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'compute_moments'

      INTEGER :: handle, i, iatom, icol, ii, ikind, inode, ipgf, irow, iset, j, jatom, jkind, &
         jpgf, jset, last_jatom, maxco, maxsgf, na, nb, ncoa, ncob, nda, ndb, nkind, nm, &
         nmom_build, nseta, nsetb, sgfa, sgfb
      LOGICAL                                            :: found
      REAL(KIND=dp)                                      :: dab
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: work
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: mab
      REAL(KIND=dp), DIMENSION(3)                        :: ra, rab, rac, rb, rbc, rc
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: mab_tmp
      TYPE(block_p_type), ALLOCATABLE, DIMENSION(:)      :: mom_block
      TYPE(cell_type), POINTER                           :: cell
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: nl_iterator
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_kind_type), POINTER                        :: qs_kind

      write(*,*) "----------------- compute_moments, moments:", nmoments
      nmom_build = nmoments      ! build moments up to order nmom_buiod
      IF (nmom_build < 1) RETURN

      CALL timeset(routineN, handle)

      NULLIFY (qs_kind_set, particle_set, sab_orb, cell)
      CALL get_qs_env(qs_env=qs_env, &
                      qs_kind_set=qs_kind_set, &
                      particle_set=particle_set, &
                      cell=cell, &
                      sab_orb=sab_orb)

      nkind = SIZE(qs_kind_set)

      ! Work storage
      CALL get_qs_kind_set(qs_kind_set=qs_kind_set, &
                           maxco=maxco, maxsgf=maxsgf)

      IF (nmoments > 0) THEN
         nm = (6 + 11*nmoments + 6*nmoments**2 + nmoments**3)/6 - 1
         CPASSERT(SIZE(moments) == nm)
         ! storage for integrals
         ALLOCATE (mab(maxco, maxco, nm))
         ! blocks
         mab(:, :, :) = 0.0_dp
         ALLOCATE (mom_block(nm))
         DO i = 1, nm
            NULLIFY (mom_block(i)%block)
         END DO
      END IF

      ALLOCATE (work(maxco, maxsgf))
      work(:, :) = 0.0_dp

      NULLIFY (basis_set_a, basis_set_b, basis_set_list)
      NULLIFY (qs_kind)
      ALLOCATE (basis_set_list(nkind))
      DO ikind = 1, nkind
         qs_kind => qs_kind_set(ikind)
         CALL get_qs_kind(qs_kind=qs_kind, basis_set=basis_set_a)
         IF (ASSOCIATED(basis_set_a)) THEN
            basis_set_list(ikind)%gto_basis_set => basis_set_a
         ELSE
            NULLIFY (basis_set_list(ikind)%gto_basis_set)
         END IF
      END DO

      ! Calculate derivatives looping over neighbour list
      NULLIFY (nl_iterator)
      CALL neighbor_list_iterator_create(nl_iterator, sab_orb)
      DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
         CALL get_iterator_info(nl_iterator, ikind=ikind, jkind=jkind, inode=inode, &
                                iatom=iatom, jatom=jatom, r=rab)
         basis_set_a => basis_set_list(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE
         ASSOCIATE ( &
            ! basis ikind
            first_sgfa => basis_set_a%first_sgf, &
            la_max => basis_set_a%lmax, &
            la_min => basis_set_a%lmin, &
            npgfa => basis_set_a%npgf, &
            nsgfa => basis_set_a%nsgf_set, &
            rpgfa => basis_set_a%pgf_radius, &
            set_radius_a => basis_set_a%set_radius, &
            sphi_a => basis_set_a%sphi, &
            zeta => basis_set_a%zet, &
            ! basis jkind, &
            first_sgfb => basis_set_b%first_sgf, &
            lb_max => basis_set_b%lmax, &
            lb_min => basis_set_b%lmin, &
            npgfb => basis_set_b%npgf, &
            nsgfb => basis_set_b%nsgf_set, &
            rpgfb => basis_set_b%pgf_radius, &
            set_radius_b => basis_set_b%set_radius, &
            sphi_b => basis_set_b%sphi, &
            zetb => basis_set_b%zet)

            nseta = basis_set_a%nset
            nsetb = basis_set_b%nset

            ! reference point
            IF (PRESENT(ref_point)) THEN
               rc(:) = ref_point(:)
            ELSE
               rc(:) = 0._dp
            END IF
            ! using PBC here might screw a molecule that fits the box (but e.g. hasn't been shifted by center_molecule)
            ! by folding around the center, such screwing can be avoided for a proper choice of center.
            ra(:) = pbc(particle_set(iatom)%r(:) - rc, cell) + rc
            rb(:) = pbc(particle_set(jatom)%r(:) - rc, cell) + rc
            ! we dont use PBC at this point
            rab(:) = ra(:) - rb(:)
            rac(:) = ra(:) - rc(:)
            rbc(:) = rb(:) - rc(:)
            dab = NORM2(rab)

            ! get blocks
            IF (inode == 1) last_jatom = 0

            IF (jatom == last_jatom) THEN
               CYCLE
            END IF

            last_jatom = jatom

            IF (iatom <= jatom) THEN
               irow = iatom
               icol = jatom
            ELSE
               irow = jatom
               icol = iatom
            END IF
            flush(6)
            IF (nmoments > 0) THEN
               DO i = 1, nm
                  NULLIFY (mom_block(i)%block)
                  ! get block from pre calculated overlap matrix
                  CALL dbcsr_get_block_p(matrix=moments(i)%matrix, &
                                         row=irow, col=icol, BLOCK=mom_block(i)%block, found=found)
                  CPASSERT(found .AND. ASSOCIATED(mom_block(i)%block))
                  mom_block(i)%block = 0._dp
               END DO
            END IF

            DO iset = 1, nseta

               ncoa = npgfa(iset)*ncoset(la_max(iset))
               sgfa = first_sgfa(1, iset)

               DO jset = 1, nsetb

                  IF (set_radius_a(iset) + set_radius_b(jset) < dab) CYCLE

                  ncob = npgfb(jset)*ncoset(lb_max(jset))
                  sgfb = first_sgfb(1, jset)

                  NULLIFY (mab_tmp)
                  ALLOCATE (mab_tmp(npgfa(iset)*ncoset(la_max(iset) + 1), &
                                    npgfb(jset)*ncoset(lb_max(jset) + 1), ncoset(nmom_build) - 1))

                  ! Calculate the primitive integrals (need l+1 for derivatives)
                  CALL moment(la_max(iset) + 1, npgfa(iset), zeta(:, iset), &
                              rpgfa(:, iset), la_min(iset), &
                              lb_max(jset) + 1, npgfb(jset), zetb(:, jset), &
                              rpgfb(:, jset), nmom_build, rac, rbc, mab_tmp)
                              
                  !write(*,'(6F18.9)') mab_tmp

                  IF (nmoments > 0) THEN
                     ! copy subset of mab_tmp (l+1) to mab (l)
                     mab = 0.0_dp
                     DO ii = 1, nm
                        na = 0
                        nda = 0
                        DO ipgf = 1, npgfa(iset)
                           nb = 0
                           ndb = 0
                           DO jpgf = 1, npgfb(jset)
                              DO j = 1, ncoset(lb_max(jset))
                                 DO i = 1, ncoset(la_max(iset))
                                    mab(i + na, j + nb, ii) = mab_tmp(i + nda, j + ndb, ii)
                                 END DO ! i
                              END DO ! j
                              nb = nb + ncoset(lb_max(jset))
                              ndb = ndb + ncoset(lb_max(jset) + 1)
                           END DO ! jpgf
                           na = na + ncoset(la_max(iset))
                           nda = nda + ncoset(la_max(iset) + 1)
                        END DO ! ipgf
                     END DO
                     ! Contraction step
                     DO i = 1, nm

                        CALL dgemm("N", "N", ncoa, nsgfb(jset), ncob, &
                                   1.0_dp, mab(1, 1, i), SIZE(mab, 1), &
                                   sphi_b(1, sgfb), SIZE(sphi_b, 1), &
                                   0.0_dp, work(1, 1), SIZE(work, 1))

                        IF (iatom <= jatom) THEN
                           CALL dgemm("T", "N", nsgfa(iset), nsgfb(jset), ncoa, &
                                      1.0_dp, sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                                      work(1, 1), SIZE(work, 1), &
                                      1.0_dp, mom_block(i)%block(sgfa, sgfb), &
                                      SIZE(mom_block(i)%block, 1))
                        ELSE
                           CALL dgemm("T", "N", nsgfb(jset), nsgfa(iset), ncoa, &
                                      1.0_dp, work(1, 1), SIZE(work, 1), &
                                      sphi_a(1, sgfa), SIZE(sphi_a, 1), &
                                      1.0_dp, mom_block(i)%block(sgfb, sgfa), &
                                      SIZE(mom_block(i)%block, 1))
                        END IF
                     END DO
                  END IF

                  DEALLOCATE (mab_tmp)
               END DO
            END DO
         END ASSOCIATE
      END DO
      CALL neighbor_list_iterator_release(nl_iterator)

      ! deallocations
      DEALLOCATE (basis_set_list)
      DEALLOCATE (work)
      IF (nmoments > 0) THEN
         DEALLOCATE (mab)
         DO i = 1, nm
            NULLIFY (mom_block(i)%block)
         END DO
         DEALLOCATE (mom_block)
      END IF

      CALL timestop(handle)

   END SUBROUTINE compute_moments

END MODULE tblite_camm
