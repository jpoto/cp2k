/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)

#include "fft_gpu_kernels.h"

#if defined(_OMP_H)
#error "OpenMP should not be used in .cu files to accommodate HIP."
#endif

/*******************************************************************************
 * \brief   Performs a out-of-place copy of a double precision vector (first
 *          half filled) into a double precision complex vector on the GPU.
 *          It requires a global double precision vector 'zout' of length '2n'.
 *          [memory (shared):  none Byte
 *           memory (private): 4 Byte
 *           memory (global):  16*n Byte]
 *          n - size of double precision input vector
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_real_to_complex(const double *din, double *zout,
                                   const int ngpts) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    zout[2 * igpt] = din[igpt];
    zout[2 * igpt + 1] = 0.0;
  }
}

/*******************************************************************************
 * \brief Launcher for pw_real_to_complex kernel.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_launch_real_to_complex(const double *din, double *zout,
                                    const int ngpts, offloadStream_t stream) {
  const int threadsPerBlock = 1024;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_real_to_complex<<<numBlocks, threadsPerBlock, 0, stream>>>(din, zout,
                                                                ngpts);
}

/*******************************************************************************
 * \brief   Performs a out-of-place copy of a double precision complex vector
 *          (real part) into a double precision vector on the GPU.
 *          It requires a global double precision vector 'dout' of length 'n'.
 *          [memory (shared):  none Byte
 *           memory (private): 4 Byte
 *           memory (global):  16*n Byte]
 *          n - size of double precision output vector
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_complex_to_real(const double *zin, double *dout,
                                   const int ngpts) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    dout[igpt] = zin[2 * igpt];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_complex_to_real kernel.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_launch_complex_to_real(const double *zin, double *dout,
                                    const int ngpts, offloadStream_t stream) {
  const int threadsPerBlock = 1024;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_complex_to_real<<<numBlocks, threadsPerBlock, 0, stream>>>(zin, dout,
                                                                ngpts);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) gather and scale on the GPU.
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_gather(double *pwcc, const double *c, const double scale,
                          const int ngpts, const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    pwcc[2 * igpt] = scale * c[2 * ghatmap[igpt]];
    pwcc[2 * igpt + 1] = scale * c[2 * ghatmap[igpt] + 1];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_gather kernel.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_launch_gather(double *pwcc, const double *c, const double scale,
                           const int ngpts, const int *ghatmap,
                           offloadStream_t stream) {
  const int threadsPerBlock = 32;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_gather<<<numBlocks, threadsPerBlock, 0, stream>>>(pwcc, c, scale, ngpts,
                                                       ghatmap);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) scatter and scale on the GPU.
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_scatter(double *c, const double *pwcc, const double scale,
                           const int ngpts, const int nmaps,
                           const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    c[2 * ghatmap[igpt]] = scale * pwcc[2 * igpt];
    c[2 * ghatmap[igpt] + 1] = scale * pwcc[2 * igpt + 1];
    if (nmaps == 2) {
      c[2 * ghatmap[igpt + ngpts]] = scale * pwcc[2 * igpt];
      c[2 * ghatmap[igpt + ngpts] + 1] = -scale * pwcc[2 * igpt + 1];
    }
  }
}

/*******************************************************************************
 * \brief Launcher for pw_scatter kernel.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_launch_scatter(double *c, const double *pwcc, const double scale,
                            const int ngpts, const int nmaps,
                            const int *ghatmap, offloadStream_t stream) {
  const int threadsPerBlock = 32;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_scatter<<<numBlocks, threadsPerBlock, 0, stream>>>(c, pwcc, scale, ngpts,
                                                        nmaps, ghatmap);
}

/*******************************************************************************
 * \brief Elementwise complex multiply a *= b (each of n complex numbers).
 ******************************************************************************/
__global__ void pw_multiply_c1d(double *a, const double *b, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const double ar = a[2 * i], ai = a[2 * i + 1];
    const double br = b[2 * i], bi = b[2 * i + 1];
    a[2 * i] = ar * br - ai * bi;
    a[2 * i + 1] = ar * bi + ai * br;
  }
}

/*******************************************************************************
 * \brief Launcher for pw_multiply_c1d kernel.
 ******************************************************************************/
void fft_gpu_launch_multiply_c1d(double *a, const double *b, const int n,
                                 offloadStream_t stream) {
  const int threadsPerBlock = 32;
  const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  pw_multiply_c1d<<<numBlocks, threadsPerBlock, 0, stream>>>(a, b, n);
}

/*******************************************************************************
 * \brief Complex vector axpy: y = a*x + b*y.
 ******************************************************************************/
__global__ void pw_axpy_c1d(double *y, const double *x, const double a,
                            const double b, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[2 * i] = a * x[2 * i] + b * y[2 * i];
    y[2 * i + 1] = a * x[2 * i + 1] + b * y[2 * i + 1];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_axpy_c1d kernel.
 ******************************************************************************/
void fft_gpu_launch_axpy_c1d(double *y, const double *x, const double a,
                             const double b, const int n,
                             offloadStream_t stream) {
  const int threadsPerBlock = 32;
  const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  pw_axpy_c1d<<<numBlocks, threadsPerBlock, 0, stream>>>(y, x, a, b, n);
}

/*******************************************************************************
 * \brief Real-vector axpy: y = a*x + b*y.
 ******************************************************************************/
__global__ void pw_axpy_r3d(double *y, const double *x, const double a,
                            const double b, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + b * y[i];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_axpy_r3d kernel.
 ******************************************************************************/
void fft_gpu_launch_axpy_r3d(double *y, const double *x, const double a,
                             const double b, const int n,
                             offloadStream_t stream) {
  const int threadsPerBlock = 256;
  const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  pw_axpy_r3d<<<numBlocks, threadsPerBlock, 0, stream>>>(y, x, a, b, n);
}

/*******************************************************************************
 * \brief Block reduction of Re(conj(a)*b); accumulates into *out via atomicAdd.
 ******************************************************************************/
__global__ void pw_dot_c1d_kernel(const double *a, const double *b, double *out,
                                  const int n) {
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    acc += a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1];
  }
  __shared__ double sh[256];
  const int t = threadIdx.x;
  sh[t] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s)
      sh[t] += sh[t + s];
    __syncthreads();
  }
  if (t == 0)
    atomicAdd(out, sh[0]);
}

/*******************************************************************************
 * \brief Launcher for pw_dot_c1d_kernel.
 ******************************************************************************/
void fft_gpu_launch_dot_c1d(const double *a, const double *b, double *out,
                            const int n, offloadStream_t stream) {
  const int tpb = 256;
  int blocks = (n + tpb - 1) / tpb;
  if (blocks > 4096)
    blocks = 4096;
  if (blocks < 1)
    blocks = 1;
  pw_dot_c1d_kernel<<<blocks, tpb, 0, stream>>>(a, b, out, n);
}

/*******************************************************************************
 * \brief Block reduction of a real vector sum; accumulates into *out via
 *        atomicAdd.
 ******************************************************************************/
__global__ void pw_sum_r3d_kernel(const double *a, double *out, const int n) {
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    acc += a[i];
  }
  __shared__ double sh[256];
  const int t = threadIdx.x;
  sh[t] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s)
      sh[t] += sh[t + s];
    __syncthreads();
  }
  if (t == 0)
    atomicAdd(out, sh[0]);
}

/*******************************************************************************
 * \brief Launcher for pw_sum_r3d_kernel.
 ******************************************************************************/
void fft_gpu_launch_sum_r3d(const double *a, double *out, const int n,
                            offloadStream_t stream) {
  const int tpb = 256;
  int blocks = (n + tpb - 1) / tpb;
  if (blocks > 4096)
    blocks = 4096;
  if (blocks < 1)
    blocks = 1;
  pw_sum_r3d_kernel<<<blocks, tpb, 0, stream>>>(a, out, n);
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)

// EOF
