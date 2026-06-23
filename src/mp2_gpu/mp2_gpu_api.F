MODULE mp2_gpu_api
   USE ISO_C_BINDING,             ONLY: C_DOUBLE, C_INT
   USE kinds,                     ONLY: dp
   USE message_passing,           ONLY: mp_comm_type

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: calc_ri_mp2_energy

CONTAINS

   SUBROUTINE calc_ri_mp2_energy(E_cou, E_ex, E_s, E_t, BIb_C, comm_all, comm_sub, eigenval, n_homo, &
                                 virtual_start, virtual_size, aux_start, aux_size, n_aux)
      REAL(KIND=dp), INTENT(OUT) :: E_cou, E_ex, E_s, E_t
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN) :: BIb_C
      CLASS(mp_comm_type), INTENT(IN) :: comm_all, comm_sub
      REAL(KIND=dp), DIMENSION(:), INTENT(IN) :: eigenval
      INTEGER, INTENT(IN) :: n_homo, virtual_start, virtual_size, aux_start, aux_size, n_aux

      INTERFACE
         SUBROUTINE calc_ri_mp2_energy_c(E_cou, E_ex, E_s, E_t, BIb_C, comm_all, comm_sub, eigenval, n_homo, &
                                         virtual_start, virtual_size, aux_start, aux_size, n_aux) &
                                         BIND(C, name="calc_ri_mp2_energy")
            IMPORT :: C_DOUBLE, C_INT
            REAL(C_DOUBLE) :: E_cou, E_ex, E_s, E_t
            REAL(C_DOUBLE), DIMENSION(*) :: BIb_C
            INTEGER(C_INT), VALUE :: comm_all, comm_sub
            REAL(C_DOUBLE), DIMENSION(*) :: eigenval
            INTEGER(C_INT), VALUE :: n_homo, virtual_start, virtual_size, aux_start, aux_size, n_aux
         END SUBROUTINE calc_ri_mp2_energy_c
      END INTERFACE

      CALL calc_ri_mp2_energy_c(E_cou, E_ex, E_s, E_t, BIb_C, comm_all%get_handle(), comm_sub%get_handle(), eigenval, n_homo, &
                                virtual_start, virtual_size, aux_start, aux_size, n_aux)

   END SUBROUTINE
END MODULE mp2_gpu_api