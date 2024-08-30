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
MODULE ptb_types


#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   INTEGER, PARAMETER, PRIVATE :: max_elem = 86
   INTEGER, PARAMETER, PRIVATE :: max_shell = 7
   INTEGER, PARAMETER, PRIVATE :: max_prim = 5


END MODULE ptb_types

