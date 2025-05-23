!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \par History
!>      09.2004 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
MODULE qmmm_util
   USE cell_types,                      ONLY: cell_type
   USE cp_log_handling,                 ONLY: cp_logger_get_default_io_unit
   USE cp_subsys_types,                 ONLY: cp_subsys_type
   USE fist_environment_types,          ONLY: fist_env_get
   USE force_env_types,                 ONLY: force_env_type,&
                                              use_qmmm,&
                                              use_qmmmx
   USE input_constants,                 ONLY: do_qmmm_wall_none,&
                                              do_qmmm_wall_quadratic,&
                                              do_qmmm_wall_reflective
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: pi
   USE particle_methods,                ONLY: write_fist_particle_coordinates,&
                                              write_qs_particle_coordinates
   USE particle_types,                  ONLY: particle_type
   USE qmmm_types,                      ONLY: qmmm_env_type
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment_types,            ONLY: get_qs_env
   USE qs_kind_types,                   ONLY: qs_kind_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .FALSE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qmmm_util'
   PUBLIC :: apply_qmmm_walls_reflective, &
             apply_qmmm_walls, &
             apply_qmmm_translate, &
             apply_qmmm_wrap, &
             apply_qmmm_unwrap, &
             spherical_cutoff_factor

CONTAINS

! **************************************************************************************************
!> \brief Apply QM quadratic walls in order to avoid QM atoms escaping from
!>      the QM Box
!> \param qmmm_env ...
!> \par History
!>      02.2008 created
!> \author Benjamin G Levine
! **************************************************************************************************
   SUBROUTINE apply_qmmm_walls(qmmm_env)
      TYPE(qmmm_env_type), POINTER                       :: qmmm_env

      INTEGER                                            :: iwall_type
      LOGICAL                                            :: do_qmmm_force_mixing, explicit
      TYPE(section_vals_type), POINTER                   :: qmmmx_section, walls_section

      walls_section => section_vals_get_subs_vals(qmmm_env%qs_env%input, "QMMM%WALLS")
      qmmmx_section => section_vals_get_subs_vals(qmmm_env%qs_env%input, "QMMM%FORCE_MIXING")
      CALL section_vals_get(qmmmx_section, explicit=do_qmmm_force_mixing)
      CALL section_vals_get(walls_section, explicit=explicit)
      IF (explicit) THEN
         CALL section_vals_val_get(walls_section, "TYPE", i_val=iwall_type)
         SELECT CASE (iwall_type)
         CASE (do_qmmm_wall_quadratic)
            IF (do_qmmm_force_mixing) THEN
               CALL cp_warn(__LOCATION__, &
                            "Quadratic walls for QM/MM are not implemented (or useful), when "// &
                            "force mixing is active.  Skipping!")
            ELSE
               CALL apply_qmmm_walls_quadratic(qmmm_env, walls_section)
            END IF
         CASE (do_qmmm_wall_reflective)
            ! Do nothing.. reflective walls are applied directly in the integrator
         END SELECT
      END IF

   END SUBROUTINE apply_qmmm_walls

! **************************************************************************************************
!> \brief Apply reflective QM walls in order to avoid QM atoms escaping from
!>      the QM Box
!> \param force_env ...
!> \par History
!>      08.2007 created [tlaino] - Zurich University
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE apply_qmmm_walls_reflective(force_env)
      TYPE(force_env_type), POINTER                      :: force_env

      INTEGER                                            :: ip, iwall_type, qm_index
      INTEGER, DIMENSION(:), POINTER                     :: qm_atom_index
      LOGICAL                                            :: explicit, is_x(2), is_y(2), is_z(2)
      REAL(KIND=dp), DIMENSION(3)                        :: coord, qm_cell_diag, skin
      REAL(KIND=dp), DIMENSION(:), POINTER               :: list
      TYPE(cell_type), POINTER                           :: mm_cell, qm_cell
      TYPE(cp_subsys_type), POINTER                      :: subsys_mm, subsys_qm
      TYPE(particle_type), DIMENSION(:), POINTER         :: particles_mm
      TYPE(section_vals_type), POINTER                   :: walls_section

      NULLIFY (subsys_mm, subsys_qm, qm_atom_index, particles_mm, qm_cell, mm_cell, &
               walls_section)

      IF (force_env%in_use /= use_qmmm .AND. force_env%in_use /= use_qmmmx) RETURN

      walls_section => section_vals_get_subs_vals(force_env%root_section, "FORCE_EVAL%QMMM%WALLS")
      CALL section_vals_get(walls_section, explicit=explicit)
      IF (explicit) THEN
         NULLIFY (list)
         CALL section_vals_val_get(walls_section, "WALL_SKIN", r_vals=list)
         CALL section_vals_val_get(walls_section, "TYPE", i_val=iwall_type)
         skin(:) = list(:)
      ELSE
         ![NB]
         iwall_type = do_qmmm_wall_reflective
         skin(:) = 0.0_dp
      END IF

      IF (force_env%in_use == use_qmmmx) THEN
         IF (iwall_type /= do_qmmm_wall_none) &
            CALL cp_warn(__LOCATION__, &
                         "Reflective walls for QM/MM are not implemented (or useful) when "// &
                         "force mixing is active.  Skipping!")
         RETURN
      END IF

      ! from here on we can be sure that it's conventional QM/MM
      CPASSERT(ASSOCIATED(force_env%qmmm_env))

      CALL fist_env_get(force_env%qmmm_env%fist_env, cell=mm_cell, subsys=subsys_mm)
      CALL get_qs_env(force_env%qmmm_env%qs_env, cell=qm_cell, cp_subsys=subsys_qm)
      qm_atom_index => force_env%qmmm_env%qm%qm_atom_index
      CPASSERT(ASSOCIATED(qm_atom_index))

      qm_cell_diag = (/qm_cell%hmat(1, 1), &
                       qm_cell%hmat(2, 2), &
                       qm_cell%hmat(3, 3)/)
      particles_mm => subsys_mm%particles%els
      DO ip = 1, SIZE(qm_atom_index)
         qm_index = qm_atom_index(ip)
         coord = particles_mm(qm_index)%r
         IF (ANY(coord < skin) .OR. ANY(coord > (qm_cell_diag - skin))) THEN
            IF (explicit) THEN
               IF (iwall_type == do_qmmm_wall_reflective) THEN
                  ! Apply Walls
                  is_x(1) = (coord(1) < skin(1))
                  is_x(2) = (coord(1) > (qm_cell_diag(1) - skin(1)))
                  is_y(1) = (coord(2) < skin(2))
                  is_y(2) = (coord(2) > (qm_cell_diag(2) - skin(2)))
                  is_z(1) = (coord(3) < skin(3))
                  is_z(2) = (coord(3) > (qm_cell_diag(3) - skin(3)))
                  IF (ANY(is_x)) THEN
                     ! X coordinate
                     IF (is_x(1)) THEN
                        particles_mm(qm_index)%v(1) = ABS(particles_mm(qm_index)%v(1))
                     ELSE IF (is_x(2)) THEN
                        particles_mm(qm_index)%v(1) = -ABS(particles_mm(qm_index)%v(1))
                     END IF
                  END IF
                  IF (ANY(is_y)) THEN
                     ! Y coordinate
                     IF (is_y(1)) THEN
                        particles_mm(qm_index)%v(2) = ABS(particles_mm(qm_index)%v(2))
                     ELSE IF (is_y(2)) THEN
                        particles_mm(qm_index)%v(2) = -ABS(particles_mm(qm_index)%v(2))
                     END IF
                  END IF
                  IF (ANY(is_z)) THEN
                     ! Z coordinate
                     IF (is_z(1)) THEN
                        particles_mm(qm_index)%v(3) = ABS(particles_mm(qm_index)%v(3))
                     ELSE IF (is_z(2)) THEN
                        particles_mm(qm_index)%v(3) = -ABS(particles_mm(qm_index)%v(3))
                     END IF
                  END IF
               END IF
            ELSE
               ! Otherwise print a warning and continue crossing cp2k's finger..
               CALL cp_warn(__LOCATION__, &
                            "One or few QM atoms are within the SKIN of the quantum box. Check your run "// &
                            "and you may possibly consider: the activation of the QMMM WALLS "// &
                            "around the QM box, switching ON the centering of the QM box or increase "// &
                            "the size of the QM cell. CP2K CONTINUE but results could be meaningless. ")
            END IF
         END IF
      END DO

   END SUBROUTINE apply_qmmm_walls_reflective

! **************************************************************************************************
!> \brief Apply QM quadratic walls in order to avoid QM atoms escaping from
!>      the QM Box
!> \param qmmm_env ...
!> \param walls_section ...
!> \par History
!>      02.2008 created
!> \author Benjamin G Levine
! **************************************************************************************************
   SUBROUTINE apply_qmmm_walls_quadratic(qmmm_env, walls_section)
      TYPE(qmmm_env_type), POINTER                       :: qmmm_env
      TYPE(section_vals_type), POINTER                   :: walls_section

      INTEGER                                            :: ip, qm_index
      INTEGER, DIMENSION(:), POINTER                     :: qm_atom_index
      LOGICAL                                            :: is_x(2), is_y(2), is_z(2)
      REAL(KIND=dp)                                      :: k, wallenergy, wallforce
      REAL(KIND=dp), DIMENSION(3)                        :: coord, qm_cell_diag, skin
      REAL(KIND=dp), DIMENSION(:), POINTER               :: list
      TYPE(cell_type), POINTER                           :: mm_cell, qm_cell
      TYPE(cp_subsys_type), POINTER                      :: subsys_mm, subsys_qm
      TYPE(particle_type), DIMENSION(:), POINTER         :: particles_mm
      TYPE(qs_energy_type), POINTER                      :: energy

      NULLIFY (list)
      CALL section_vals_val_get(walls_section, "WALL_SKIN", r_vals=list)
      CALL section_vals_val_get(walls_section, "K", r_val=k)
      CPASSERT(ASSOCIATED(qmmm_env))

      CALL fist_env_get(qmmm_env%fist_env, cell=mm_cell, subsys=subsys_mm)
      CALL get_qs_env(qmmm_env%qs_env, cell=qm_cell, cp_subsys=subsys_qm)

      qm_atom_index => qmmm_env%qm%qm_atom_index
      CPASSERT(ASSOCIATED(qm_atom_index))

      skin(:) = list(:)

      qm_cell_diag = (/qm_cell%hmat(1, 1), &
                       qm_cell%hmat(2, 2), &
                       qm_cell%hmat(3, 3)/)
      particles_mm => subsys_mm%particles%els
      wallenergy = 0.0_dp
      DO ip = 1, SIZE(qm_atom_index)
         qm_index = qm_atom_index(ip)
         coord = particles_mm(qm_index)%r
         IF (ANY(coord < skin) .OR. ANY(coord > (qm_cell_diag - skin))) THEN
            is_x(1) = (coord(1) < skin(1))
            is_x(2) = (coord(1) > (qm_cell_diag(1) - skin(1)))
            is_y(1) = (coord(2) < skin(2))
            is_y(2) = (coord(2) > (qm_cell_diag(2) - skin(2)))
            is_z(1) = (coord(3) < skin(3))
            is_z(2) = (coord(3) > (qm_cell_diag(3) - skin(3)))
            IF (is_x(1)) THEN
               wallforce = 2.0_dp*k*(skin(1) - coord(1))
               particles_mm(qm_index)%f(1) = particles_mm(qm_index)%f(1) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*(skin(1) - coord(1))*0.5_dp
            END IF
            IF (is_x(2)) THEN
               wallforce = 2.0_dp*k*((qm_cell_diag(1) - skin(1)) - coord(1))
               particles_mm(qm_index)%f(1) = particles_mm(qm_index)%f(1) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*((qm_cell_diag(1) - skin(1)) - &
                                                    coord(1))*0.5_dp
            END IF
            IF (is_y(1)) THEN
               wallforce = 2.0_dp*k*(skin(2) - coord(2))
               particles_mm(qm_index)%f(2) = particles_mm(qm_index)%f(2) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*(skin(2) - coord(2))*0.5_dp
            END IF
            IF (is_y(2)) THEN
               wallforce = 2.0_dp*k*((qm_cell_diag(2) - skin(2)) - coord(2))
               particles_mm(qm_index)%f(2) = particles_mm(qm_index)%f(2) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*((qm_cell_diag(2) - skin(2)) - &
                                                    coord(2))*0.5_dp
            END IF
            IF (is_z(1)) THEN
               wallforce = 2.0_dp*k*(skin(3) - coord(3))
               particles_mm(qm_index)%f(3) = particles_mm(qm_index)%f(3) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*(skin(3) - coord(3))*0.5_dp
            END IF
            IF (is_z(2)) THEN
               wallforce = 2.0_dp*k*((qm_cell_diag(3) - skin(3)) - coord(3))
               particles_mm(qm_index)%f(3) = particles_mm(qm_index)%f(3) + &
                                             wallforce
               wallenergy = wallenergy + wallforce*((qm_cell_diag(3) - skin(3)) - &
                                                    coord(3))*0.5_dp
            END IF
         END IF
      END DO

      CALL get_qs_env(qs_env=qmmm_env%qs_env, energy=energy)
      energy%total = energy%total + wallenergy

   END SUBROUTINE apply_qmmm_walls_quadratic

! **************************************************************************************************
!> \brief wrap positions (with mm periodicity)
!> \param subsys_mm ...
!> \param mm_cell ...
!> \param subsys_qm ...
!> \param qm_atom_index ...
!> \param saved_pos ...
! **************************************************************************************************
   SUBROUTINE apply_qmmm_wrap(subsys_mm, mm_cell, subsys_qm, qm_atom_index, saved_pos)
      TYPE(cp_subsys_type), POINTER                      :: subsys_mm
      TYPE(cell_type), POINTER                           :: mm_cell
      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys_qm
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: qm_atom_index
      REAL(dp), ALLOCATABLE                              :: saved_pos(:, :)

      INTEGER                                            :: i_dim, ip
      REAL(dp)                                           :: r_lat(3)

      ALLOCATE (saved_pos(3, subsys_mm%particles%n_els))
      DO ip = 1, subsys_mm%particles%n_els
         saved_pos(1:3, ip) = subsys_mm%particles%els(ip)%r(1:3)
         r_lat = MATMUL(mm_cell%h_inv, subsys_mm%particles%els(ip)%r)
         DO i_dim = 1, 3
            IF (mm_cell%perd(i_dim) /= 1) THEN
               r_lat(i_dim) = 0.0_dp
            END IF
         END DO
         subsys_mm%particles%els(ip)%r = subsys_mm%particles%els(ip)%r - MATMUL(mm_cell%hmat, FLOOR(r_lat))
      END DO

      IF (PRESENT(subsys_qm) .AND. PRESENT(qm_atom_index)) THEN
         DO ip = 1, SIZE(qm_atom_index)
            subsys_qm%particles%els(ip)%r = subsys_mm%particles%els(qm_atom_index(ip))%r
         END DO
      END IF
   END SUBROUTINE apply_qmmm_wrap

! **************************************************************************************************
!> \brief ...
!> \param subsys_mm ...
!> \param subsys_qm ...
!> \param qm_atom_index ...
!> \param saved_pos ...
! **************************************************************************************************
   SUBROUTINE apply_qmmm_unwrap(subsys_mm, subsys_qm, qm_atom_index, saved_pos)
      TYPE(cp_subsys_type), POINTER                      :: subsys_mm
      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys_qm
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: qm_atom_index
      REAL(dp), ALLOCATABLE                              :: saved_pos(:, :)

      INTEGER                                            :: ip

      DO ip = 1, subsys_mm%particles%n_els
         subsys_mm%particles%els(ip)%r(1:3) = saved_pos(1:3, ip)
      END DO

      IF (PRESENT(subsys_qm) .AND. PRESENT(qm_atom_index)) THEN
         DO ip = 1, SIZE(qm_atom_index)
            subsys_qm%particles%els(ip)%r = subsys_mm%particles%els(qm_atom_index(ip))%r
         END DO
      END IF

      DEALLOCATE (saved_pos)
   END SUBROUTINE apply_qmmm_unwrap

! **************************************************************************************************
!> \brief Apply translation to the full system in order to center the QM
!>      system into the QM box
!> \param qmmm_env ...
!> \par History
!>      08.2007 created [tlaino] - Zurich University
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE apply_qmmm_translate(qmmm_env)
      TYPE(qmmm_env_type), POINTER                       :: qmmm_env

      INTEGER                                            :: bigger_ip, i_dim, ip, max_ip, min_ip, &
                                                            smaller_ip, tmp_ip, unit_nr
      INTEGER, DIMENSION(:), POINTER                     :: qm_atom_index
      LOGICAL, ALLOCATABLE                               :: avoid(:)
      REAL(DP) :: bigger_lat_dv, center_p(3), lat_dv, lat_dv3(3), lat_min(3), lat_p(3), &
         max_coord_lat(3), min_coord_lat(3), smaller_lat_dv
      REAL(DP), POINTER                                  :: charges(:)
      REAL(KIND=dp), DIMENSION(3)                        :: max_coord, min_coord, transl_v
      TYPE(cell_type), POINTER                           :: mm_cell, qm_cell
      TYPE(cp_subsys_type), POINTER                      :: subsys_mm, subsys_qm
      TYPE(particle_type), DIMENSION(:), POINTER         :: particles_mm, particles_qm
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(section_vals_type), POINTER                   :: subsys_section

      NULLIFY (subsys_mm, subsys_qm, qm_atom_index, particles_mm, particles_qm, &
               subsys_section, qm_cell, mm_cell, qs_kind_set)

      CPASSERT(ASSOCIATED(qmmm_env))

      CALL fist_env_get(qmmm_env%fist_env, cell=mm_cell, subsys=subsys_mm)
      CALL get_qs_env(qmmm_env%qs_env, cell=qm_cell, cp_subsys=subsys_qm)
      qm_atom_index => qmmm_env%qm%qm_atom_index
      CPASSERT(ASSOCIATED(qm_atom_index))

      particles_qm => subsys_qm%particles%els
      particles_mm => subsys_mm%particles%els
      IF (.NOT. qmmm_env%qm%center_qm_subsys0) qmmm_env%qm%do_translate = .FALSE.
      IF (qmmm_env%qm%do_translate) THEN
         IF (.NOT. qmmm_env%qm%center_qm_subsys_pbc_aware) THEN
            ! naive coordinate based min-max
            min_coord = HUGE(0.0_dp)
            max_coord = -HUGE(0.0_dp)
            DO ip = 1, SIZE(qm_atom_index)
               min_coord = MIN(min_coord, particles_mm(qm_atom_index(ip))%r)
               max_coord = MAX(max_coord, particles_mm(qm_atom_index(ip))%r)
            END DO
         ELSE
            !! periodic based min max (uses complex number based mean)
            center_p = qmmm_pbc_aware_mean(particles_mm, mm_cell, qm_atom_index)
            ALLOCATE (avoid(SIZE(qm_atom_index)))
            DO i_dim = 1, 3
               IF (mm_cell%perd(i_dim) /= 1) THEN
                  ! find absolute min and max positions (along i_dim direction) in lattice coordinates
                  min_coord_lat(i_dim) = HUGE(0.0_dp)
                  max_coord_lat(i_dim) = -HUGE(0.0_dp)
                  DO ip = 1, SIZE(qm_atom_index)
                     lat_p = MATMUL(mm_cell%h_inv, particles_mm(qm_atom_index(ip))%r)
                     min_coord_lat(i_dim) = MIN(lat_p(i_dim), min_coord_lat(i_dim))
                     max_coord_lat(i_dim) = MAX(lat_p(i_dim), max_coord_lat(i_dim))
                  END DO
               ELSE
                  ! find min_ip closest to (pbc-aware) mean pos
                  avoid = .FALSE.
                  min_ip = qmmm_find_closest(particles_mm, mm_cell, qm_atom_index, avoid, center_p, i_dim, 0)
                  avoid(min_ip) = .TRUE.
                  ! find max_ip closest to min_ip
                  max_ip = qmmm_find_closest(particles_mm, mm_cell, qm_atom_index, avoid, &
                                             particles_mm(qm_atom_index(min_ip))%r, i_dim, 0, lat_dv)
                  avoid(max_ip) = .TRUE.
                  ! switch min and max if necessary
                  IF (lat_dv < 0.0) THEN
                     tmp_ip = min_ip
                     min_ip = max_ip
                     max_ip = tmp_ip
                  END IF
                  ! loop over all other atoms
                  DO WHILE (.NOT. ALL(avoid))
                     ! find smaller below min, bigger after max
                     smaller_ip = qmmm_find_closest(particles_mm, mm_cell, qm_atom_index, &
                                                    avoid, particles_mm(qm_atom_index(min_ip))%r, i_dim, -1, smaller_lat_dv)
                     bigger_ip = qmmm_find_closest(particles_mm, mm_cell, qm_atom_index, &
                                                   avoid, particles_mm(qm_atom_index(max_ip))%r, i_dim, 1, bigger_lat_dv)
                     ! move min or max, not both
                     IF (ABS(smaller_lat_dv) < ABS(bigger_lat_dv)) THEN
                        min_ip = smaller_ip
                        avoid(min_ip) = .TRUE.
                     ELSE
                        max_ip = bigger_ip
                        avoid(max_ip) = .TRUE.
                     END IF
                  END DO
                  ! find min and max coordinates in lattice positions (i_dim ! only)
                  lat_dv3 = qmmm_lat_dv(mm_cell, particles_mm(qm_atom_index(min_ip))%r, particles_mm(qm_atom_index(max_ip))%r)
                  IF (lat_dv3(i_dim) < 0.0_dp) lat_dv3(i_dim) = lat_dv3(i_dim) + 1.0_dp
                  lat_min = MATMUL(mm_cell%h_inv, particles_mm(qm_atom_index(min_ip))%r)
                  min_coord_lat(i_dim) = lat_min(i_dim)
                  max_coord_lat(i_dim) = lat_min(i_dim) + lat_dv3(i_dim)
               END IF ! periodic
            END DO ! i_dim
            ! min and max coordinates from lattice positions to Cartesian
            min_coord = MATMUL(mm_cell%hmat, min_coord_lat)
            max_coord = MATMUL(mm_cell%hmat, max_coord_lat)
            DEALLOCATE (avoid)
         END IF ! pbc aware center
         transl_v = (max_coord + min_coord)/2.0_dp

         !
         ! The first time we always translate all the system in order
         ! to centre the QM system in the box.
         !
         transl_v(:) = transl_v(:) - SUM(qm_cell%hmat, 2)/2.0_dp

         IF (ANY(qmmm_env%qm%utrasl /= 1.0_dp)) THEN
            transl_v = REAL(FLOOR(transl_v/qmmm_env%qm%utrasl), KIND=dp)* &
                       qmmm_env%qm%utrasl
         END IF
         qmmm_env%qm%transl_v = qmmm_env%qm%transl_v + transl_v
         particles_mm => subsys_mm%particles%els
         DO ip = 1, subsys_mm%particles%n_els
            particles_mm(ip)%r = particles_mm(ip)%r - transl_v
         END DO
         IF (qmmm_env%qm%added_shells%num_mm_atoms .GT. 0) THEN
            DO ip = 1, qmmm_env%qm%added_shells%num_mm_atoms
               qmmm_env%qm%added_shells%added_particles(ip)%r = qmmm_env%qm%added_shells%added_particles(ip)%r - transl_v
               qmmm_env%qm%added_shells%added_cores(ip)%r = qmmm_env%qm%added_shells%added_cores(ip)%r - transl_v
            END DO
         END IF
         unit_nr = cp_logger_get_default_io_unit()
         IF (unit_nr > 0) WRITE (unit=unit_nr, fmt='(/1X,A)') &
            " Translating the system in order to center the QM fragment in the QM box."
         IF (.NOT. qmmm_env%qm%center_qm_subsys) qmmm_env%qm%do_translate = .FALSE.
      END IF
      particles_mm => subsys_mm%particles%els
      DO ip = 1, SIZE(qm_atom_index)
         particles_qm(ip)%r = particles_mm(qm_atom_index(ip))%r
      END DO

      subsys_section => section_vals_get_subs_vals(qmmm_env%qs_env%input, "SUBSYS")

      CALL get_qs_env(qs_env=qmmm_env%qs_env, qs_kind_set=qs_kind_set)
      CALL write_qs_particle_coordinates(particles_qm, qs_kind_set, subsys_section, "QM/MM first QM, then MM (0 charges)")
      ALLOCATE (charges(SIZE(particles_mm)))
      charges = 0.0_dp
      CALL write_fist_particle_coordinates(particles_mm, subsys_section, charges)
      DEALLOCATE (charges)

   END SUBROUTINE apply_qmmm_translate

! **************************************************************************************************
!> \brief pbc-aware mean QM atom position
!> \param particles_mm ...
!> \param mm_cell ...
!> \param qm_atom_index ...
!> \return ...
! **************************************************************************************************
   FUNCTION qmmm_pbc_aware_mean(particles_mm, mm_cell, qm_atom_index)
      TYPE(particle_type), DIMENSION(:), POINTER         :: particles_mm
      TYPE(cell_type), POINTER                           :: mm_cell
      INTEGER, DIMENSION(:), POINTER                     :: qm_atom_index
      REAL(dp)                                           :: qmmm_pbc_aware_mean(3)

      COMPLEX(dp)                                        :: mean_z(3)
      INTEGER                                            :: ip

      mean_z = 0.0_dp
      DO ip = 1, SIZE(qm_atom_index)
         mean_z = mean_z + EXP(CMPLX(0.0_dp, 1.0_dp, KIND=dp)*2.0*pi* &
                               MATMUL(mm_cell%h_inv, particles_mm(qm_atom_index(ip))%r))
      END DO
      mean_z = mean_z/ABS(mean_z)
      qmmm_pbc_aware_mean = MATMUL(mm_cell%hmat, &
                                   REAL(LOG(mean_z)/(CMPLX(0.0_dp, 1.0_dp, KIND=dp)*2.0_dp*pi), dp))
   END FUNCTION

! **************************************************************************************************
!> \brief minimum image lattice coordinates difference vector
!> \param mm_cell ...
!> \param p1 ...
!> \param p2 ...
!> \return ...
! **************************************************************************************************
   FUNCTION qmmm_lat_dv(mm_cell, p1, p2)
      TYPE(cell_type), POINTER                           :: mm_cell
      REAL(dp)                                           :: p1(3), p2(3), qmmm_lat_dv(3)

      REAL(dp)                                           :: lat_v1(3), lat_v2(3)

      lat_v1 = MATMUL(mm_cell%h_inv, p1)
      lat_v2 = MATMUL(mm_cell%h_inv, p2)

      qmmm_lat_dv = lat_v2 - lat_v1
      qmmm_lat_dv = qmmm_lat_dv - FLOOR(qmmm_lat_dv)
   END FUNCTION qmmm_lat_dv

! **************************************************************************************************
!> \brief find closest QM particle, in position/negative direction
!>        if dir is 1 or -1, respectively
!> \param particles_mm ...
!> \param mm_cell ...
!> \param qm_atom_index ...
!> \param avoid ...
!> \param p ...
!> \param i_dim ...
!> \param dir ...
!> \param closest_dv ...
!> \return ...
! **************************************************************************************************
   FUNCTION qmmm_find_closest(particles_mm, mm_cell, qm_atom_index, avoid, p, i_dim, dir, closest_dv) RESULT(closest_ip)
      TYPE(particle_type), DIMENSION(:), POINTER         :: particles_mm
      TYPE(cell_type), POINTER                           :: mm_cell
      INTEGER, DIMENSION(:), POINTER                     :: qm_atom_index
      LOGICAL                                            :: avoid(:)
      REAL(dp)                                           :: p(3)
      INTEGER                                            :: i_dim, dir
      REAL(dp), OPTIONAL                                 :: closest_dv
      INTEGER                                            :: closest_ip

      INTEGER                                            :: ip, shift
      REAL(dp)                                           :: lat_dv3(3), lat_dv_shifted, my_closest_dv

      closest_ip = -1
      my_closest_dv = HUGE(0.0)
      DO ip = 1, SIZE(qm_atom_index)
         IF (avoid(ip)) CYCLE
         lat_dv3 = qmmm_lat_dv(mm_cell, p, particles_mm(qm_atom_index(ip))%r)
         DO shift = -1, 1
            lat_dv_shifted = lat_dv3(i_dim) + shift*1.0_dp
            IF (ABS(lat_dv_shifted) < ABS(my_closest_dv) .AND. (dir*lat_dv_shifted >= 0.0)) THEN
               my_closest_dv = lat_dv_shifted
               closest_ip = ip
            END IF
         END DO
      END DO

      IF (PRESENT(closest_dv)) THEN
         closest_dv = my_closest_dv
      END IF

   END FUNCTION qmmm_find_closest

! **************************************************************************************************
!> \brief Computes a spherical cutoff factor for the QMMM interactions
!> \param spherical_cutoff ...
!> \param rij ...
!> \param factor ...
!> \par History
!>      08.2008 created
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE spherical_cutoff_factor(spherical_cutoff, rij, factor)
      REAL(KIND=dp), DIMENSION(2), INTENT(IN)            :: spherical_cutoff
      REAL(KIND=dp), DIMENSION(3), INTENT(IN)            :: rij
      REAL(KIND=dp), INTENT(OUT)                         :: factor

      REAL(KIND=dp)                                      :: r, r0

      r = SQRT(DOT_PRODUCT(rij, rij))
      r0 = spherical_cutoff(1) - 20.0_dp*spherical_cutoff(2)
      factor = 0.5_dp*(1.0_dp - TANH((r - r0)/spherical_cutoff(2)))

   END SUBROUTINE spherical_cutoff_factor

END MODULE qmmm_util
