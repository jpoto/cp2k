!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2024 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Read pTB parameters.
!> \author JVP (08.2024)
! **************************************************************************************************
MODULE ptb_parameters

   USE basis_set_types,                 ONLY: allocate_gto_basis_set,&
                                              gto_basis_set_type,&
                                              orthofun
   USE cp_parser_methods,               ONLY: parser_get_next_line,&
                                              parser_get_object
   USE cp_parser_types,                 ONLY: cp_parser_type,&
                                              parser_create
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE memory_utilities,                ONLY: reallocate
   USE message_passing,                 ONLY: mp_para_env_type
   USE orbital_pointers,                ONLY: indco,&
                                              nco,&
                                              ncoset,&
                                              nso
   USE orbital_symbols,                 ONLY: cgf_symbol,&
                                              sgf_symbol
   USE periodic_table,                  ONLY: nelem,&
                                              ptable
   USE ptb_types,                       ONLY: max_elem,&
                                              max_prim,&
                                              ptb_atom_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ptb_parameters'

! *** Public data types ***

   PUBLIC :: ptb_parameters_init, init_ptb_basis

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param param ...
!> \param element_symbol ...
!> \param parameter_file_path ...
!> \param parameter_file_name ...
!> \param para_env ...
! **************************************************************************************************
   SUBROUTINE ptb_parameters_init(param, element_symbol, parameter_file_path, parameter_file_name, &
                                  para_env)

      TYPE(ptb_atom_type), POINTER                       :: param
      CHARACTER(LEN=2), INTENT(IN)                       :: element_symbol
      CHARACTER(LEN=*), INTENT(IN)                       :: parameter_file_path, parameter_file_name
      TYPE(mp_para_env_type), POINTER                    :: para_env

      CHARACTER(len=default_string_length)               :: filename, hline
      INTEGER                                            :: icoef, ielem, ishell, k, nshell, z_num
      LOGICAL                                            :: at_end, found
      REAL(KIND=dp)                                      :: valence
      TYPE(cp_parser_type)                               :: parser

      filename = ADJUSTL(TRIM(parameter_file_path))//ADJUSTL(TRIM(parameter_file_name))

      !identify element
      ielem = 999
      DO k = 1, nelem
         IF (element_symbol == ptable(k)%symbol) THEN
            ielem = k
            EXIT
         END IF
      END DO

      IF (ielem > max_elem) CPABORT('Element not available for pTB')
      !storing information
      param%typ = "ORB"
      param%symbol = element_symbol
      param%z = ielem

      !parse informaiton from file
      CALL parser_create(parser, filename, para_env=para_env)
      found = .FALSE.
      DO
         at_end = .FALSE.
         CALL parser_get_next_line(parser, 1, at_end)
         IF (at_end) EXIT
         CALL parser_get_object(parser, hline)
         CALL parser_get_object(parser, z_num)
         CALL parser_get_object(parser, valence)
         CALL parser_get_object(parser, nshell)
         IF (z_num == ielem) THEN
            param%nshell = nshell
            param%valence = valence
            DO ishell = 1, nshell
               CALL parser_get_next_line(parser, 1, at_end)
               CALL parser_get_object(parser, param%ind(ishell))
               CALL parser_get_object(parser, param%nprim(ishell))
               CALL parser_get_object(parser, param%lval(ishell))
               CALL parser_get_next_line(parser, 1, at_end)
               DO icoef = 1, param%nprim(ishell)
                  CALL parser_get_object(parser, param%kappa(ishell, icoef))
               END DO
               CALL parser_get_next_line(parser, 1, at_end)
               DO icoef = 1, param%nprim(ishell)
                  CALL parser_get_object(parser, param%coef(ishell, icoef))
               END DO
            END DO
         ELSE
            CALL parser_get_next_line(parser, nshell*3, at_end)
         END IF
      END DO

      param%defined = .TRUE.

   END SUBROUTINE ptb_parameters_init

! **************************************************************************************************
!> \brief ...
!> \param param ...
!> \param gto_basis_set ...
!> \param ortho ...
! **************************************************************************************************
   SUBROUTINE init_ptb_basis(param, gto_basis_set, ortho)

      TYPE(ptb_atom_type), POINTER                       :: param
      TYPE(gto_basis_set_type), POINTER                  :: gto_basis_set
      LOGICAL, INTENT(IN), OPTIONAL                      :: ortho

      INTEGER                                            :: ipgf, iset, maxl, ng, nset
      LOGICAL                                            :: do_ortho

      CALL allocate_gto_basis_set(gto_basis_set)

      do_ortho = .FALSE.
      IF (PRESENT(ortho)) do_ortho = ortho

      IF (ASSOCIATED(param)) THEN
         IF (param%defined) THEN
            !set gto basis set information
            gto_basis_set%nset = param%nshell

            gto_basis_set%name = "pTB"

            nset = param%nshell
            ng = max_prim
            gto_basis_set%nset = nset
            maxl = MAXVAL(param%lval)
            CALL reallocate(gto_basis_set%lmax, 1, nset)
            CALL reallocate(gto_basis_set%lmin, 1, nset)
            CALL reallocate(gto_basis_set%npgf, 1, nset)
            CALL reallocate(gto_basis_set%nshell, 1, nset)
            CALL reallocate(gto_basis_set%n, 1, 1, 1, nset)
            CALL reallocate(gto_basis_set%l, 1, 1, 1, nset)
            CALL reallocate(gto_basis_set%zet, 1, ng, 1, nset)
            CALL reallocate(gto_basis_set%gcc, 1, ng, 1, 1, 1, nset)

            gto_basis_set%gcc = 0.00_dp
            gto_basis_set%zet = 0.00_dp
            DO iset = 1, nset
               gto_basis_set%lmax(iset) = param%lval(iset)
               gto_basis_set%lmin(iset) = param%lval(iset)
               gto_basis_set%npgf(iset) = param%nprim(iset)
               gto_basis_set%nshell(iset) = 1
               gto_basis_set%n(1, iset) = param%lval(iset) + 1
               gto_basis_set%l(1, iset) = param%lval(iset)
               DO ipgf = 1, param%nprim(iset)
                  gto_basis_set%gcc(ipgf, 1, iset) = param%coef(iset, ipgf)
                  gto_basis_set%zet(ipgf, iset) = param%kappa(iset, ipgf)
               END DO
            END DO

            IF (do_ortho) THEN
               CALL orthogonalize(gto_basis_set, nset, ng, maxl)
            END IF

            CALL set_cart_spher(gto_basis_set, nset)

            gto_basis_set%norm_type = -1

         ELSE
            CPWARN("atom basis not defined in init_ptb_basis !")
         END IF
      ELSE
         CPABORT("The pointer param is not associated.")
      END IF

   END SUBROUTINE init_ptb_basis

! **************************************************************************************************
!> \brief ...
!> \param gto_basis_set ...
!> \param nset ...
! **************************************************************************************************
   SUBROUTINE set_cart_spher(gto_basis_set, nset)

      TYPE(gto_basis_set_type), POINTER                  :: gto_basis_set
      INTEGER                                            :: nset

      INTEGER                                            :: ico, iset, lshell, m, maxco, ncgf, ngs, &
                                                            np, nsgf

      write(*,*) 'set cart spher'

      ngs = MAXVAL(gto_basis_set%npgf(1:nset))
      CALL reallocate(gto_basis_set%set_radius, 1, nset)
      CALL reallocate(gto_basis_set%pgf_radius, 1, ngs, 1, nset)
      CALL reallocate(gto_basis_set%first_cgf, 1, 1, 1, nset)
      CALL reallocate(gto_basis_set%first_sgf, 1, 1, 1, nset)
      CALL reallocate(gto_basis_set%last_cgf, 1, 1, 1, nset)
      CALL reallocate(gto_basis_set%last_sgf, 1, 1, 1, nset)
      CALL reallocate(gto_basis_set%ncgf_set, 1, nset)
      CALL reallocate(gto_basis_set%nsgf_set, 1, nset)

      maxco = 0
      ncgf = 0
      nsgf = 0

      DO iset = 1, nset
         gto_basis_set%ncgf_set(iset) = 0
         gto_basis_set%nsgf_set(iset) = 0
         lshell = gto_basis_set%l(1, iset)
         gto_basis_set%first_cgf(1, iset) = ncgf + 1
         ncgf = ncgf + nco(lshell)

         gto_basis_set%last_cgf(1, iset) = ncgf
         gto_basis_set%ncgf_set(iset) = &
            gto_basis_set%ncgf_set(iset) + nco(lshell)
         gto_basis_set%first_sgf(1, iset) = nsgf + 1
         nsgf = nsgf + nso(lshell)
         gto_basis_set%last_sgf(1, iset) = nsgf
         gto_basis_set%nsgf_set(iset) = &
            gto_basis_set%nsgf_set(iset) + nso(lshell)
         ngs = gto_basis_set%npgf(iset)
         maxco = MAX(maxco, ngs*ncoset(lshell))
      END DO

      gto_basis_set%ncgf = ncgf
      gto_basis_set%nsgf = nsgf

      CALL reallocate(gto_basis_set%cphi, 1, maxco, 1, ncgf)
      CALL reallocate(gto_basis_set%sphi, 1, maxco, 1, nsgf)
      CALL reallocate(gto_basis_set%scon, 1, maxco, 1, nsgf)
      CALL reallocate(gto_basis_set%lx, 1, ncgf)
      CALL reallocate(gto_basis_set%ly, 1, ncgf)
      CALL reallocate(gto_basis_set%lz, 1, ncgf)
      CALL reallocate(gto_basis_set%m, 1, nsgf)
      CALL reallocate(gto_basis_set%norm_cgf, 1, ncgf)
      ALLOCATE (gto_basis_set%cgf_symbol(ncgf))
      ALLOCATE (gto_basis_set%sgf_symbol(nsgf))

      ncgf = 0
      nsgf = 0

      DO iset = 1, nset
         lshell = gto_basis_set%l(1, iset)
         np = lshell + 1
         DO ico = ncoset(lshell - 1) + 1, ncoset(lshell)
            ncgf = ncgf + 1
            gto_basis_set%lx(ncgf) = indco(1, ico)
            gto_basis_set%ly(ncgf) = indco(2, ico)
            gto_basis_set%lz(ncgf) = indco(3, ico)
            gto_basis_set%cgf_symbol(ncgf) = &
               cgf_symbol(np, (/gto_basis_set%lx(ncgf), &
                                gto_basis_set%ly(ncgf), &
                                gto_basis_set%lz(ncgf)/))
         END DO
         DO m = -lshell, lshell
            nsgf = nsgf + 1
            gto_basis_set%m(nsgf) = m
            gto_basis_set%sgf_symbol(nsgf) = sgf_symbol(np, lshell, m)
         END DO
      END DO

   END SUBROUTINE set_cart_spher

! **************************************************************************************************
!> \brief ...
!> \param gto_basis_set ...
!> \param nset ...
!> \param ng ...
!> \param maxl ...
! **************************************************************************************************
   SUBROUTINE orthogonalize(gto_basis_set, nset, ng, maxl)

      TYPE(gto_basis_set_type), POINTER                  :: gto_basis_set
      INTEGER                                            :: nset, ng, maxl

      INTEGER                                            :: i1, i2, iset, jset, l, m
      INTEGER, DIMENSION(0:10)                           :: mxf
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: gal, zal, zll

      write(*,*) "+++++ ortho"

      mxf = 0
      DO iset = 1, nset
         l = gto_basis_set%l(1, iset)
         mxf(l) = mxf(l) + 1
      END DO
      m = MAXVAL(mxf)
      IF (m > 1) THEN
         ALLOCATE (gal(ng, nset), zal(ng, nset), zll(m*ng, 0:maxl))
         DO iset = 1, nset
            zal(1:ng, iset) = gto_basis_set%zet(1:ng, iset)
            gal(1:ng, iset) = gto_basis_set%gcc(1:ng, 1, iset)
         END DO
         CALL reallocate(gto_basis_set%zet, 1, m*ng, 1, nset)
         CALL reallocate(gto_basis_set%gcc, 1, m*ng, 1, 1, 1, nset)
         DO iset = 1, nset
            l = gto_basis_set%l(1, iset)
            gto_basis_set%npgf(iset) = ng*mxf(l)
         END DO
         gto_basis_set%zet = 0.0_dp
         gto_basis_set%gcc = 0.0_dp
         zll = 0.0_dp
         mxf = 0
         DO iset = 1, nset
            l = gto_basis_set%l(1, iset)
            mxf(l) = mxf(l) + 1
            i1 = mxf(l)*ng - ng + 1
            i2 = mxf(l)*ng
            zll(i1:i2, l) = zal(1:ng, iset)
            gto_basis_set%gcc(i1:i2, 1, iset) = gal(1:ng, iset)
         END DO
         DO iset = 1, nset
            l = gto_basis_set%l(1, iset)
            gto_basis_set%zet(:, iset) = zll(:, l)
         END DO
         DO iset = 1, nset
            l = gto_basis_set%l(1, iset)
            DO jset = 1, iset - 1
               IF (gto_basis_set%l(1, iset) == l) THEN
                  m = mxf(l)*ng
                  CALL orthofun(gto_basis_set%zet(1:m, iset), gto_basis_set%gcc(1:m, 1, iset), &
                                gto_basis_set%gcc(1:m, 1, jset), l)
               END IF
            END DO
         END DO
         DEALLOCATE (gal, zal, zll)
      END IF

   END SUBROUTINE orthogonalize

END MODULE ptb_parameters

