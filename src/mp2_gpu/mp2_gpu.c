#include <stdlib.h>
#include <string.h>
#include "../mpiwrap/cp_mpi.h"
#include "gemm_c_api.h"

void calc_ri_mp2_energy(double *E_cou, double *E_ex, double *E_s, double *E_t, const double *BIb_C, int comm_all_f, int comm_sub_f, const double *eigenval, int n_homo, int virtual_start, int virtual_size, int aux_start, int aux_size, int n_aux) {
    const cp_mpi_comm_t comm_all = cp_mpi_comm_f2c(comm_all_f);
    const cp_mpi_comm_t comm_sub = cp_mpi_comm_f2c(comm_sub_f);

    (void) comm_all;
    (void) comm_sub;
    (void) eigenval;
    (void) virtual_start;
    (void) aux_start;
    (void) n_aux;

    gemm_ctx_t *ctx = gemm_ctx_create(GEMM_PU_HOST, GEMM_LIB_BLAS);

    const int M = aux_size;
    const int N = n_homo;

    if (M > 0 && N > 0) {
        double *A = (double*)malloc(M * N * sizeof(double));
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                A[i + j * M] = BIb_C[i + j * M];
            }
        }

        double *C = (double*)calloc(M * M, sizeof(double));
        gemm_ctx_dgemm(ctx, 'N', 'T', M, M, N, 1.0, A, M, A, M, 0.0, C, M);

        free(A);
        double sum = 0.0;
        for (int i = 0; i < M * M; i++) sum += C[i];
        free(C);
        gemm_ctx_destroy(ctx);

        *E_cou = sum;
    } else {
        gemm_ctx_destroy(ctx);
        *E_cou = 0.0;
    }

    *E_ex = 0.0;
    *E_s = 0.0;
    *E_t = 0.0;
}