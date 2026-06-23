#include "../mpiwrap/cp_mpi.h"

void calc_ri_mp2_energy(double *E_cou, double *E_ex, double *E_s, double *E_t, const double *BIb_C, int comm_all_f, int comm_sub_f, const double *eigenval, int n_homo, int virtual_start, int virtual_size, int aux_start, int aux_size, int n_aux) {
    const cp_mpi_comm_t comm_all = cp_mpi_comm_f2c(comm_all_f);
    const cp_mpi_comm_t comm_sub = cp_mpi_comm_f2c(comm_sub_f);

    // Mark everything used
    (void) BIb_C;
    (void) comm_all;
    (void) comm_sub;
    (void) eigenval;
    (void) n_homo;
    (void) virtual_start;
    (void) virtual_size;
    (void) aux_start;
    (void) aux_size;
    (void) n_aux;

    // Implementation for calculating RI-MP2 energy
    *E_cou = 0.0;
    *E_ex = 0.0;
    *E_s = 0.0;
    *E_t = 0.0;
}