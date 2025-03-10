!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
! **************************************************************************************************
MODULE hartree_local_types

   USE kinds,                           ONLY: dp
   USE qs_rho_atom_types,               ONLY: rho_atom_coeff
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters (only in this module)

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'hartree_local_types'

! *** Define the ecoul_1center_type ***

! **************************************************************************************************
   TYPE ecoul_1center_type
      TYPE(rho_atom_coeff), POINTER   :: Vh1_h => NULL(), Vh1_s => NULL()
      REAL(dp)                        :: ecoul_1_h = 0.0_dp, &
                                         ecoul_1_s = 0.0_dp, &
                                         ecoul_1_z = 0.0_dp, &
                                         ecoul_1_0 = 0.0_dp
   END TYPE ecoul_1center_type

! **************************************************************************************************
   TYPE hartree_local_type
      TYPE(ecoul_1center_type), &
         DIMENSION(:), POINTER   :: ecoul_1c => NULL()
   END TYPE hartree_local_type

! *** Public subroutines ***

   PUBLIC :: allocate_ecoul_1center, &
             get_hartree_local, hartree_local_create, &
             hartree_local_release, set_ecoul_1c, &
             set_hartree_local

! *** Public data types ***

   PUBLIC :: ecoul_1center_type, hartree_local_type

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param ecoul_1c ...
!> \param natom ...
! **************************************************************************************************
   SUBROUTINE allocate_ecoul_1center(ecoul_1c, natom)

      TYPE(ecoul_1center_type), DIMENSION(:), POINTER    :: ecoul_1c
      INTEGER, INTENT(IN)                                :: natom

      INTEGER                                            :: iat

      IF (ASSOCIATED(ecoul_1c)) THEN
         CALL deallocate_ecoul_1center(ecoul_1c)
      END IF

      ALLOCATE (ecoul_1c(natom))

      DO iat = 1, natom
         ALLOCATE (ecoul_1c(iat)%Vh1_h)
         NULLIFY (ecoul_1c(iat)%Vh1_h%r_coef)
         ALLOCATE (ecoul_1c(iat)%Vh1_s)
         NULLIFY (ecoul_1c(iat)%Vh1_s%r_coef)
      END DO

   END SUBROUTINE allocate_ecoul_1center

! **************************************************************************************************
!> \brief ...
!> \param ecoul_1c ...
! **************************************************************************************************
   SUBROUTINE deallocate_ecoul_1center(ecoul_1c)

      TYPE(ecoul_1center_type), DIMENSION(:), POINTER    :: ecoul_1c

      INTEGER                                            :: iat, natom

      natom = SIZE(ecoul_1c, 1)

      DO iat = 1, natom
         IF (ASSOCIATED(ecoul_1c(iat)%Vh1_h%r_coef)) THEN
            DEALLOCATE (ecoul_1c(iat)%Vh1_h%r_coef)
         END IF
         DEALLOCATE (ecoul_1c(iat)%Vh1_h)

         IF (ASSOCIATED(ecoul_1c(iat)%Vh1_s%r_coef)) THEN
            DEALLOCATE (ecoul_1c(iat)%Vh1_s%r_coef)
         END IF
         DEALLOCATE (ecoul_1c(iat)%Vh1_s)

      END DO

      DEALLOCATE (ecoul_1c)

   END SUBROUTINE deallocate_ecoul_1center

! **************************************************************************************************
!> \brief ...
!> \param hartree_local ...
!> \param ecoul_1c ...
! **************************************************************************************************
   SUBROUTINE get_hartree_local(hartree_local, ecoul_1c)

      TYPE(hartree_local_type), POINTER                  :: hartree_local
      TYPE(ecoul_1center_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: ecoul_1c

      IF (PRESENT(ecoul_1c)) ecoul_1c => hartree_local%ecoul_1c

   END SUBROUTINE get_hartree_local

! **************************************************************************************************
!> \brief ...
!> \param hartree_local ...
! **************************************************************************************************
   SUBROUTINE hartree_local_create(hartree_local)

      TYPE(hartree_local_type), POINTER                  :: hartree_local

      ALLOCATE (hartree_local)

      NULLIFY (hartree_local%ecoul_1c)

   END SUBROUTINE hartree_local_create

! **************************************************************************************************
!> \brief ...
!> \param hartree_local ...
! **************************************************************************************************
   SUBROUTINE hartree_local_release(hartree_local)

      TYPE(hartree_local_type), POINTER                  :: hartree_local

      IF (ASSOCIATED(hartree_local)) THEN
         IF (ASSOCIATED(hartree_local%ecoul_1c)) THEN
            CALL deallocate_ecoul_1center(hartree_local%ecoul_1c)
         END IF

         DEALLOCATE (hartree_local)
      END IF

   END SUBROUTINE hartree_local_release

! **************************************************************************************************
!> \brief ...
!> \param ecoul_1c ...
!> \param iatom ...
!> \param ecoul_1_h ...
!> \param ecoul_1_s ...
!> \param ecoul_1_z ...
!> \param ecoul_1_0 ...
! **************************************************************************************************
   SUBROUTINE set_ecoul_1c(ecoul_1c, iatom, ecoul_1_h, ecoul_1_s, ecoul_1_z, ecoul_1_0)

      TYPE(ecoul_1center_type), DIMENSION(:), POINTER    :: ecoul_1c
      INTEGER, INTENT(IN), OPTIONAL                      :: iatom
      REAL(dp), INTENT(IN), OPTIONAL                     :: ecoul_1_h, ecoul_1_s, ecoul_1_z, &
                                                            ecoul_1_0

      IF (PRESENT(iatom)) THEN
         IF (PRESENT(ecoul_1_h)) ecoul_1c(iatom)%ecoul_1_h = ecoul_1_h
         IF (PRESENT(ecoul_1_s)) ecoul_1c(iatom)%ecoul_1_s = ecoul_1_s
         IF (PRESENT(ecoul_1_0)) ecoul_1c(iatom)%ecoul_1_0 = ecoul_1_0
         IF (PRESENT(ecoul_1_z)) ecoul_1c(iatom)%ecoul_1_z = ecoul_1_z
      END IF

   END SUBROUTINE set_ecoul_1c

! **************************************************************************************************
!> \brief ...
!> \param hartree_local ...
!> \param ecoul_1c ...
! **************************************************************************************************
   SUBROUTINE set_hartree_local(hartree_local, ecoul_1c)

      TYPE(hartree_local_type), POINTER                  :: hartree_local
      TYPE(ecoul_1center_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: ecoul_1c

      IF (PRESENT(ecoul_1c)) hartree_local%ecoul_1c => ecoul_1c

   END SUBROUTINE set_hartree_local

END MODULE hartree_local_types

