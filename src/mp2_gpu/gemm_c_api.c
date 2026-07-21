/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "gemm_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../offload/offload_library.h"

#if defined(__CUBLAS)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
#include <spla/spla.h>
#endif

/*******************************************************************************
 * Error-checking macros
 ******************************************************************************/
#if defined(__CUBLAS)
#define CUBLAS_CHECK(cmd)                                                      \
  do {                                                                         \
    cublasStatus_t status__ = (cmd);                                           \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "CUBLAS_ERROR: %s:%d ", __FILE__, __LINE__);             \
      switch (status__) {                                                      \
      case CUBLAS_STATUS_NOT_INITIALIZED:                                      \
        fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");                    \
        break;                                                                 \
      case CUBLAS_STATUS_ALLOC_FAILED:                                         \
        fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n");                       \
        break;                                                                 \
      case CUBLAS_STATUS_INVALID_VALUE:                                        \
        fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_ARCH_MISMATCH:                                        \
        fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_MAPPING_ERROR:                                        \
        fprintf(stderr, "CUBLAS_STATUS_MAPPING_ERROR\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_EXECUTION_FAILED:                                     \
        fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n");                   \
        break;                                                                 \
      case CUBLAS_STATUS_INTERNAL_ERROR:                                       \
        fprintf(stderr, "CUBLAS_STATUS_INTERNAL_ERROR\n");                     \
        break;                                                                 \
      default:                                                                 \
        fprintf(stderr, "unknown error %d\n", status__);                       \
      }                                                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t status__ = (cmd);                                              \
    if (status__ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA_ERROR: %s %s:%d\n", cudaGetErrorString(status__),  \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define CUBLAS_CHECK(cmd) ((void)0)
#define CUDA_CHECK(cmd) ((void)0)
#endif

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
#define SPLA_CHECK(cmd)                                                        \
  do {                                                                         \
    int ok__ = (cmd);                                                          \
    if (ok__ != 0) {                                                           \
      fprintf(stderr, "SPLA_ERROR: %s:%d (code %d)\n", __FILE__, __LINE__,     \
              ok__);                                                           \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define SPLA_CHECK(cmd) ((void)0)
#endif

/*******************************************************************************
 * BLAS declaration (standard C BLAS interface via cblas.h)
 ******************************************************************************/
void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *A, const int *lda,
            const double *B, const int *ldb, const double *beta, double *C,
            const int *ldc);

void sgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const float *alpha, const float *A, const int *lda,
            const float *B, const int *ldb, const float *beta, float *C,
            const int *ldc);

/*******************************************************************************
 * Internal context structure
 ******************************************************************************/
struct gemm_ctx {
  gemm_lib_t lib; /**< Active backend */
  gemm_pu_t pu;   /**< Processing unit */
  int uses_gpu;   /**< 1 if GPU is used for computation */

#if defined(__CUBLAS)
  cublasHandle_t cublas_handle; /**< cuBLAS handle */
#endif

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  void *spla_ctx;   /**< SPLA context (opaque) */
  size_t threshold; /**< m*n*k threshold for GPU dispatch */
#endif
};

/*******************************************************************************
 * Backend: Standard BLAS (always compiled)
 ******************************************************************************/
static void blas_dgemm(char transa, char transb, int m, int n, int k,
                       double alpha, const double *A, int lda, const double *B,
                       int ldb, double beta, double *C, int ldc) {
  dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
         &ldc);
}

static void blas_sgemm(char transa, char transb, int m, int n, int k,
                       float alpha, const float *A, int lda, const float *B,
                       int ldb, float beta, float *C, int ldc) {
  sgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
         &ldc);
}

/*******************************************************************************
 * Backend: cuBLAS (compiled when __CUBLAS is defined)
 ******************************************************************************/
#if defined(__CUBLAS)

static void cublas_activate_device(void) { offload_activate_chosen_device(); }

static void cublas_dgemm_impl(gemm_ctx_t *ctx, char transa, char transb, int m,
                              int n, int k, double alpha, const double *A,
                              int lda, const double *B, int ldb, double beta,
                              double *C, int ldc) {
  cublasOperation_t opA =
      (transa == 'N' || transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB =
      (transb == 'N' || transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  double *d_A = NULL, *d_B = NULL, *d_C = NULL;
  double *h_A = NULL, *h_B = NULL;

  const int A_rows = (opA == CUBLAS_OP_N) ? m : k;
  const int A_cols = (opA == CUBLAS_OP_N) ? k : m;
  const int B_rows = (opB == CUBLAS_OP_N) ? k : n;
  const int B_cols = (opB == CUBLAS_OP_N) ? n : k;

  const size_t size_A = (size_t)A_rows * A_cols * sizeof(double);
  const size_t size_B = (size_t)B_rows * B_cols * sizeof(double);
  const size_t size_C = (size_t)m * n * sizeof(double);

  h_A = (double *)malloc(size_A);
  h_B = (double *)malloc(size_B);
  if (!h_A || !h_B) {
    fprintf(stderr, "gemm_c_api: malloc failed\n");
    abort();
  }

  memcpy(h_A, A, size_A);
  memcpy(h_B, B, size_B);

  CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void **)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void **)&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  if (beta != 0.0) {
    CUDA_CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
  }

  CUBLAS_CHECK(cublasDgemm(ctx->cublas_handle, opA, opB, m, n, k, &alpha, d_A,
                           lda, d_B, ldb, &beta, d_C, ldc));

  CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
}

static void cublas_sgemm_impl(gemm_ctx_t *ctx, char transa, char transb, int m,
                              int n, int k, float alpha, const float *A,
                              int lda, const float *B, int ldb, float beta,
                              float *C, int ldc) {
  cublasOperation_t opA =
      (transa == 'N' || transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB =
      (transb == 'N' || transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  float *h_A = NULL, *h_B = NULL;

  const int A_rows = (opA == CUBLAS_OP_N) ? m : k;
  const int A_cols = (opA == CUBLAS_OP_N) ? k : m;
  const int B_rows = (opB == CUBLAS_OP_N) ? k : n;
  const int B_cols = (opB == CUBLAS_OP_N) ? n : k;

  const size_t size_A = (size_t)A_rows * A_cols * sizeof(float);
  const size_t size_B = (size_t)B_rows * B_cols * sizeof(float);
  const size_t size_C = (size_t)m * n * sizeof(float);

  h_A = (float *)malloc(size_A);
  h_B = (float *)malloc(size_B);
  if (!h_A || !h_B) {
    fprintf(stderr, "gemm_c_api: malloc failed\n");
    abort();
  }

  memcpy(h_A, A, size_A);
  memcpy(h_B, B, size_B);

  CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void **)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void **)&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  if (beta != 0.0f) {
    CUDA_CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
  }

  CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle, opA, opB, m, n, k, &alpha, d_A,
                           lda, d_B, ldb, &beta, d_C, ldc));

  CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
}

#endif /* __CUBLAS */

/*******************************************************************************
 * Backend: SPLA (compiled when __SPLA and __OFFLOAD_GEMM are defined)
 ******************************************************************************/
#if defined(__SPLA) && defined(__OFFLOAD_GEMM)

static void spla_dgemm_impl(gemm_ctx_t *ctx, char transa, char transb, int m,
                            int n, int k, double alpha, const double *A,
                            int lda, const double *B, int ldb, double beta,
                            double *C, int ldc) {
  SPLA spLA_op_A =
      (transa == 'N' || transa == 'n') ? SPLA_OP_NONE : SPLA_OP_TRANSPOSE;
  SPLA spLA_op_B =
      (transb == 'N' || transb == 'n') ? SPLA_OP_NONE : SPLA_OP_TRANSPOSE;

  SPLA_CHECK(spla_dgemm(spLA_op_A, spLA_op_B, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc, ctx->spla_ctx));
}

#endif /* __SPLA && __OFFLOAD_GEMM */

/*******************************************************************************
 * Public API
 ******************************************************************************/

gemm_ctx_t *gemm_ctx_create(gemm_pu_t pu, gemm_lib_t lib) {
  gemm_ctx_t *ctx = (gemm_ctx_t *)calloc(1, sizeof(gemm_ctx_t));
  if (!ctx) {
    fprintf(stderr, "gemm_ctx_create: calloc failed\n");
    abort();
  }

  ctx->lib = lib;
  ctx->pu = pu;
  ctx->uses_gpu = 0;

  switch (lib) {

#if defined(__CUBLAS)
  case GEMM_LIB_CUBLAS:
    if (pu == GEMM_PU_GPU) {
      cublas_activate_device();
      CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
      ctx->uses_gpu = 1;
    } else {
      ctx->uses_gpu = 0;
    }
    break;
#endif

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  case GEMM_LIB_SPLA: {
    if (pu == GEMM_PU_GPU) {
      offload_activate_chosen_device();
      SPLA_CHECK(spla_ctx_create(&ctx->spla_ctx, SPLA_PU_GPU));
      ctx->uses_gpu = 1;
      ctx->threshold = 128 * 128 * 128 * 2;
    } else {
      SPLA_CHECK(spla_ctx_create(&ctx->spla_ctx, SPLA_PU_HOST));
      ctx->uses_gpu = 0;
      ctx->threshold = 0;
    }
    break;
  }
#endif

  case GEMM_LIB_BLAS:
    ctx->uses_gpu = 0;
    break;

  default:
    fprintf(stderr, "gemm_ctx_create: unknown library %d\n", lib);
    free(ctx);
    return NULL;
  }

  return ctx;
}

void gemm_ctx_destroy(gemm_ctx_t *ctx) {
  if (!ctx)
    return;

  switch (ctx->lib) {

#if defined(__CUBLAS)
  case GEMM_LIB_CUBLAS:
    CUBLAS_CHECK(cublasDestroy(ctx->cublas_handle));
    break;
#endif

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  case GEMM_LIB_SPLA:
    SPLA_CHECK(spla_ctx_destroy(ctx->spla_ctx));
    break;
#endif

  case GEMM_LIB_BLAS:
    break;

  default:
    fprintf(stderr, "gemm_ctx_destroy: unknown library %d\n", ctx->lib);
  }

  free(ctx);
}

void gemm_ctx_set_threshold(gemm_ctx_t *ctx, size_t threshold) {
#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  if (ctx && ctx->lib == GEMM_LIB_SPLA) {
    ctx->threshold = threshold;
    if (ctx->uses_gpu) {
      offload_activate_chosen_device();
      SPLA_CHECK(spla_ctx_set_op_threshold_gpu(ctx->spla_ctx, threshold));
    }
  }
#else
  (void)ctx;
  (void)threshold;
#endif
}

void gemm_ctx_set_cublas_math_mode(gemm_ctx_t *ctx, unsigned int mode) {
#if defined(__CUBLAS)
  if (ctx && ctx->lib == GEMM_LIB_CUBLAS && mode != 0) {
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas_handle, mode));
  }
#else
  (void)ctx;
  (void)mode;
#endif
}

void gemm_ctx_dgemm(gemm_ctx_t *ctx, char transa, char transb, int m, int n,
                    int k, double alpha, const double *A, int lda,
                    const double *B, int ldb, double beta, double *C, int ldc) {
  if (!ctx) {
    fprintf(stderr, "gemm_ctx_dgemm: NULL context\n");
    abort();
  }

  switch (ctx->lib) {

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  case GEMM_LIB_SPLA:
    spla_dgemm_impl(ctx, transa, transb, m, n, k, (double)alpha,
                    (const double *)A, lda, (const double *)B, ldb,
                    (double)beta, (double *)C, ldc);
    break;
#endif

#if defined(__CUBLAS)
  case GEMM_LIB_CUBLAS:
    if (ctx->uses_gpu) {
      cublas_dgemm_impl(ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
    } else {
      blas_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    break;
#endif

  case GEMM_LIB_BLAS:
  default:
    blas_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  }
}

void gemm_ctx_sgemm(gemm_ctx_t *ctx, char transa, char transb, int m, int n,
                    int k, float alpha, const float *A, int lda, const float *B,
                    int ldb, float beta, float *C, int ldc) {
  if (!ctx) {
    fprintf(stderr, "gemm_ctx_sgemm: NULL context\n");
    abort();
  }

  switch (ctx->lib) {

#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  case GEMM_LIB_SPLA:
    fprintf(stderr, "gemm_ctx_sgemm: SPLA C API has no sgemm; "
                    "falling back to BLAS\n");
    blas_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
#endif

#if defined(__CUBLAS)
  case GEMM_LIB_CUBLAS:
    if (ctx->uses_gpu) {
      cublas_sgemm_impl(ctx, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
    } else {
      blas_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    break;
#endif

  case GEMM_LIB_BLAS:
  default:
    blas_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  }
}

const char *gemm_ctx_get_backend_name(gemm_ctx_t *ctx) {
  if (!ctx)
    return "NULL";

  switch (ctx->lib) {
#if defined(__CUBLAS)
  case GEMM_LIB_CUBLAS:
    return ctx->uses_gpu ? "cuBLAS" : "cuBLAS(CPU fallback)";
#endif
#if defined(__SPLA) && defined(__OFFLOAD_GEMM)
  case GEMM_LIB_SPLA:
    return ctx->uses_gpu ? "SPLA" : "SPLA(CPU fallback)";
#endif
  case GEMM_LIB_BLAS:
    return "BLAS";
  default:
    return "unknown";
  }
}

int gemm_ctx_uses_gpu(gemm_ctx_t *ctx) { return ctx ? ctx->uses_gpu : 0; }