/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/**
 * @file gemm_c_api.h
 *
 * C wrapper for GEMM (matrix-matrix multiply) with pluggable backends.
 *
 * Supports three backends:
 *   - GEMM_LIB_BLAS   : Standard BLAS (CPU only, always available)
 *   - GEMM_LIB_CUBLAS : NVIDIA cuBLAS (GPU direct)
 *   - GEMM_LIB_SPLA   : Intel SPLA (GPU offload via cuBLAS/cutlass underneath)
 *
 * Example usage:
 * @code
 *   gemm_ctx_t *ctx = gemm_ctx_create(GEMM_PU_GPU, GEMM_LIB_CUBLAS);
 *   gemm_ctx_set_threshold(ctx, 128*128*128*2);
 *   gemm_ctx_dgemm(ctx, 'N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *   gemm_ctx_destroy(ctx);
 * @endcode
 */

#ifndef GEMM_C_API_H
#define GEMM_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Processing unit (where computation runs)
 ******************************************************************************/
typedef enum {
    GEMM_PU_HOST = 0, /**< CPU */
    GEMM_PU_GPU = 1   /**< GPU accelerator */
} gemm_pu_t;

/*******************************************************************************
 * BLAS library backend
 ******************************************************************************/
typedef enum {
    GEMM_LIB_BLAS = 0,   /**< Standard BLAS (dgemm), CPU only */
    GEMM_LIB_CUBLAS = 1, /**< NVIDIA cuBLAS, GPU direct */
    GEMM_LIB_SPLA = 2    /**< Intel SPLA, GPU offload */
} gemm_lib_t;

/*******************************************************************************
 * Opaque GEMM context
 ******************************************************************************/
typedef struct gemm_ctx gemm_ctx_t;

/*******************************************************************************
 * Create a GEMM context for the given processing unit and backend.
 *
 * @param[in] pu   Processing unit: GEMM_PU_HOST or GEMM_PU_GPU
 * @param[in] lib  Backend library: GEMM_LIB_BLAS, GEMM_LIB_CUBLAS, GEMM_LIB_SPLA
 * @return         Opaque context handle, or NULL on failure (abort on GPU).
 *
 * Notes:
 *   - GEMM_LIB_BLAS ignores \p pu and always runs on CPU.
 *   - GEMM_LIB_CUBLAS requires CUDA runtime and GPU device.
 *   - GEMM_LIB_SPLA requires SPLA compiled with GPU support
 *     (SPLA_GPU_BACKEND=CUDA|ROCM) and __SPLA + __OFFLOAD_GEMM defines.
 ******************************************************************************/
gemm_ctx_t *gemm_ctx_create(gemm_pu_t pu, gemm_lib_t lib);

/*******************************************************************************
 * Destroy a GEMM context and release associated resources.
 *
 * @param[in,out] ctx  Context to destroy (may be NULL)
 ******************************************************************************/
void gemm_ctx_destroy(gemm_ctx_t *ctx);

/*******************************************************************************
 * Set the operation-size threshold below which the GPU is bypassed
 * and the computation runs on the host/CPU instead.
 *
 * Only meaningful for GPU backends. The threshold is compared against
 * the product m * n * k. Small matrices may be faster on CPU due to
 * GPU kernel launch overhead.
 *
 * @param[in,out] ctx       Context
 * @param[in]     threshold Product m*n*k below which to use CPU
 *
 * Notes:
 *   - cuBLAS: threshold is advisory; implementation may still use GPU.
 *   - SPLA:   threshold is enforced by SPLA internally.
 ******************************************************************************/
void gemm_ctx_set_threshold(gemm_ctx_t *ctx, size_t threshold);

/*******************************************************************************
 * Set the cuBLAS compute math mode (e.g. Tensor Core enablement).
 *
 * Only meaningful when the backend is GEMM_LIB_CUBLAS.
 *
 * @param[in,out] ctx   Context
 * @param[in]     mode  cuBLAS math mode (e.g. CUBLAS_DEFAULT_MATH,
 *                      CUBLAS_TENSOR_OP_MATH, CUBLAS_FAST_MATH).
 *                      Pass 0 to leave at default.
 ******************************************************************************/
void gemm_ctx_set_cublas_math_mode(gemm_ctx_t *ctx, unsigned int mode);

/*******************************************************************************
 * Double-precision real GEMM: C = alpha * op(A) * op(B) + beta * C
 *
 * @param[in,out] ctx    Context (created with gemm_ctx_create)
 * @param[in]     transa 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * @param[in]     transb 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * @param[in]     m      Number of rows of op(A) and C
 * @param[in]     n      Number of columns of op(B) and C
 * @param[in]     k      Number of columns of op(A) and rows of op(B)
 * @param[in]     alpha  Scalar multiplier for A*B
 * @param[in]     A      Left matrix (size lda x k, or lda x m if not transposed)
 * @param[in]     lda    Leading dimension of A (lda >= max(1,m) or max(1,k))
 * @param[in]     B      Right matrix (size ldb x n, or ldb x k if not transposed)
 * @param[in]     ldb    Leading dimension of B (ldb >= max(1,k) or max(1,n))
 * @param[in]     beta   Scalar multiplier for C
 * @param[in,out] C      Output matrix (size ldc x n, or ldc x m if not transposed)
 * @param[in]     ldc    Leading dimension of C (ldc >= max(1,m) or max(1,n))
 *
 * Memory layout:
 *   - BLAS/CUBLAS: Column-major (Fortran order).
 *   - A, B, C are always host pointers. For GPU backends the wrapper handles
 *     host-to-device and device-to-host transfers internally.
 ******************************************************************************/
void gemm_ctx_dgemm(gemm_ctx_t *ctx,
                    char transa, char transb,
                    int m, int n, int k,
                    double alpha,
                    const double *A, int lda,
                    const double *B, int ldb,
                    double beta,
                    double *C, int ldc);

/*******************************************************************************
 * Single-precision real GEMM (same signature as dgemm, but float).
 *
 * Only supported by GEMM_LIB_CUBLAS and GEMM_LIB_SPLA backends.
 * With GEMM_LIB_BLAS this calls sgemm from standard BLAS.
 ******************************************************************************/
void gemm_ctx_sgemm(gemm_ctx_t *ctx,
                    char transa, char transb,
                    int m, int n, int k,
                    float alpha,
                    const float *A, int lda,
                    const float *B, int ldb,
                    float beta,
                    float *C, int ldc);

/*******************************************************************************
 * Get a human-readable string describing the active backend.
 * Useful for debugging and logging.
 *
 * @param[in] ctx  Context
 * @return         Static string (do not free), e.g. "BLAS", "cuBLAS", "SPLA"
 ******************************************************************************/
const char *gemm_ctx_get_backend_name(gemm_ctx_t *ctx);

/*******************************************************************************
 * Query whether the context is using GPU acceleration.
 *
 * @param[in] ctx  Context
 * @return         1 if GPU is used for computation, 0 otherwise
 ******************************************************************************/
int gemm_ctx_uses_gpu(gemm_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_C_API_H */