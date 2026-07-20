!--------------------------------------------------------------------------------------------------!
! Interface between Fortran and C for MP2 RI routines intent (2)
!--------------------------------------------------------------------------------------------------!

MODULE mp2_ri_c_interface
    USE ISO_C_BINDING, ONLY: C_INT, C_DOUBLE, C_BOOL, C_PTR, C_FUNPTR, C_NULL_PTR
    USE kinds, ONLY: dp
    USE mp2_types, ONLY: mp2_type
    USE group_dist_types, ONLY: group_dist_d1_type, maxsize
    USE message_passing, ONLY: mp_para_env_type
    

    IMPLICIT NONE
    PRIVATE

    PUBLIC :: mp2_ri_get_integ_group_size_c_wrapper

    ! Forward declaration of C function
    INTERFACE
        SUBROUTINE c_mp2_ri_get_integ_group_size( &
            integ_group_size_out, ngroup_out, num_integ_group_out, &
            ngroup, num_integ_group, integ_group_size, &
            mp2_memory, number_integration_groups, &
            homo, homo_size, virtual_arr, virtual_size, &
            dimen_RI, calc_forces, unit_nr, &
            gd_array_sizes, gd_array_sizes_size, &
            gd_B_virtual_sizes, gd_B_virtual_sizes_size, &
            maxsize_gd_array, maxsize_gd_B_virtual, &
            maxval_gd_B_virtual, maxval_virtual, &
            max_homo, sum_homo_virtual, product_homo, &
            nspins, calc_group_size, comm_handle &
        ) BIND(C, name='c_mp2_ri_get_integ_group_size')
            USE ISO_C_BINDING, ONLY: C_INT, C_DOUBLE, C_BOOL, C_PTR
            INTEGER(C_INT), INTENT(OUT) :: integ_group_size_out
            INTEGER(C_INT), INTENT(OUT) :: ngroup_out
            INTEGER(C_INT), INTENT(OUT) :: num_integ_group_out
            INTEGER(C_INT), INTENT(IN)  :: ngroup
            INTEGER(C_INT), INTENT(IN)  :: num_integ_group
            INTEGER(C_INT), INTENT(IN)  :: integ_group_size
            REAL(C_DOUBLE), INTENT(IN)  :: mp2_memory
            INTEGER(C_INT), INTENT(IN)  :: number_integration_groups
            INTEGER(C_INT), INTENT(IN)  :: homo_size
            INTEGER(C_INT), INTENT(IN)  :: homo(homo_size)
            INTEGER(C_INT), INTENT(IN)  :: virtual_size
            INTEGER(C_INT), INTENT(IN)  :: virtual_arr(virtual_size)
            INTEGER(C_INT), INTENT(IN)  :: dimen_RI
            INTEGER(C_INT), INTENT(IN)  :: calc_forces
            INTEGER(C_INT), INTENT(IN)  :: unit_nr
            INTEGER(C_INT), INTENT(IN)  :: gd_array_sizes_size
            INTEGER(C_INT), INTENT(IN)  :: gd_array_sizes(gd_array_sizes_size)
            INTEGER(C_INT), INTENT(IN)  :: gd_B_virtual_sizes_size
            INTEGER(C_INT), INTENT(IN)  :: gd_B_virtual_sizes(gd_B_virtual_sizes_size)
            INTEGER(C_INT), INTENT(IN)  :: maxsize_gd_array
            INTEGER(C_INT), INTENT(IN)  :: maxsize_gd_B_virtual
            INTEGER(C_INT), INTENT(IN)  :: maxval_gd_B_virtual
            INTEGER(C_INT), INTENT(IN)  :: maxval_virtual
            INTEGER(C_INT), INTENT(IN)  :: max_homo
            INTEGER(C_INT), INTENT(IN)  :: sum_homo_virtual
            INTEGER(C_INT), INTENT(IN)  :: product_homo
            INTEGER(C_INT), INTENT(IN)  :: nspins
            INTEGER(C_INT), INTENT(IN)  :: calc_group_size
            TYPE(C_PTR), VALUE, INTENT(IN) :: comm_handle
        END SUBROUTINE c_mp2_ri_get_integ_group_size
    END INTERFACE

CONTAINS

    SUBROUTINE mp2_ri_get_integ_group_size_c_wrapper(&
        mp2_env, para_env, para_env_sub, &
        gd_array, gd_B_virtual, &
        homo, dimen_RI, unit_nr, &
        integ_group_size, &
        ngroup, num_integ_group, &
        virtual, calc_forces &
    )
        TYPE(mp2_type), INTENT(IN) :: mp2_env
        TYPE(mp_para_env_type), INTENT(IN) :: para_env, para_env_sub
        TYPE(group_dist_d1_type), INTENT(IN) :: gd_array
        TYPE(group_dist_d1_type), DIMENSION(:), INTENT(IN) :: gd_B_virtual
        INTEGER, DIMENSION(:), INTENT(IN) :: homo
        INTEGER, INTENT(IN) :: dimen_RI, unit_nr
        INTEGER, INTENT(OUT) :: integ_group_size, ngroup, num_integ_group
        INTEGER, DIMENSION(:), INTENT(IN) :: virtual
        LOGICAL, INTENT(IN) :: calc_forces

        INTEGER :: nspins, homo_size, virtual_size, gd_array_size, gd_B_size, i
        INTEGER, ALLOCATABLE :: gd_array_sizes(:), gd_B_sizes(:)
        LOGICAL :: calc_group_size_local
        INTEGER :: max_homo, sum_homo_virtual, product_homo
        INTEGER :: integ_group_size_out, ngroup_out, num_integ_group_out
        TYPE(C_PTR) :: comm_handle

        ! Prepare data for C call
        nspins = SIZE(homo)
        homo_size = SIZE(homo)
        virtual_size = SIZE(virtual)
        gd_array_size = SIZE(gd_array%sizes)
        
        ALLOCATE(gd_array_sizes(gd_array_size))
        gd_array_sizes = gd_array%sizes
        
        ALLOCATE(gd_B_sizes(SIZE(gd_B_virtual)))
        DO i = 1, SIZE(gd_B_virtual)
            gd_B_sizes(i) = gd_B_virtual(i)%sizes(0)  ! Simplified - needs proper handling
        END DO

        ! Calculate derived values
        max_homo = MAXVAL(homo)
        sum_homo_virtual = SUM(homo * virtual)
        product_homo = PRODUCT(homo)

        calc_group_size_local = (mp2_env%ri_mp2%number_integration_groups <= 0) .OR. &
                                        (MOD(ngroup, mp2_env%ri_mp2%number_integration_groups) /= 0)

        comm_handle = C_NULL_PTR

        CALL c_mp2_ri_get_integ_group_size( &
            integ_group_size_out, ngroup_out, num_integ_group_out, &
            ngroup, num_integ_group, integ_group_size, &
            mp2_env%mp2_memory, mp2_env%ri_mp2%number_integration_groups, &
            homo, homo_size, virtual, virtual_size, &
            dimen_RI, MERGE(1, 0, calc_forces), unit_nr, &
            gd_array_sizes, gd_array_size, &
            gd_B_sizes, SIZE(gd_B_sizes), &
            maxsize(gd_array), MAXVAL(maxsize(gd_B_virtual)), &
            MAXVAL(maxsize(gd_B_virtual)), MAXVAL(virtual), &
            max_homo, sum_homo_virtual, product_homo, &
            nspins, MERGE(1, 0, calc_group_size_local), &
            comm_handle &
        )

        ! Return values
        integ_group_size = integ_group_size_out
        ngroup = ngroup_out
        num_integ_group = num_integ_group_out

        DEALLOCATE(gd_array_sizes, gd_B_sizes)
    END SUBROUTINE mp2_ri_get_integ_group_size_c_wrapper

END MODULE mp2_ri_c_interface