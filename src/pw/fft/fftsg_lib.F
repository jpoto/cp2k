!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
MODULE fftsg_lib
   USE fft_kinds,                       ONLY: dp

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fftsg_lib'

   PUBLIC :: fftsg_do_init, fftsg_do_cleanup, fftsg_get_lengths

CONTAINS

! **************************************************************************************************
!> \brief ...
! **************************************************************************************************
   SUBROUTINE fftsg_do_init()

      ! no init needed

   END SUBROUTINE fftsg_do_init

! **************************************************************************************************
!> \brief ...
! **************************************************************************************************
   SUBROUTINE fftsg_do_cleanup()

      ! no cleanup needed

   END SUBROUTINE fftsg_do_cleanup

! **************************************************************************************************
!> \brief ...
!> \param DATA ...
!> \param max_length ...
!> \par History
!>      Adapted to new interface structure
!> \author JGH
! **************************************************************************************************
   SUBROUTINE fftsg_get_lengths(DATA, max_length)

      INTEGER, DIMENSION(*)                              :: DATA
      INTEGER, INTENT(INOUT)                             :: max_length

      INTEGER, PARAMETER                                 :: rlen = 81
      INTEGER, DIMENSION(rlen), PARAMETER :: radix = [2, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 25, 27&
         , 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135&
         , 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 256, 270, 288, 300, 320, 324, 360&
         , 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720&
         , 729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024]

      INTEGER                                            :: ndata

!------------------------------------------------------------------------------

      ndata = MIN(max_length, rlen)
      DATA(1:ndata) = RADIX(1:ndata)
      max_length = ndata

   END SUBROUTINE fftsg_get_lengths
   
END MODULE fftsg_lib

