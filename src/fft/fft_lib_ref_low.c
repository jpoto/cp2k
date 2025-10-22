/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib_ref.h"
#include "fft_utils.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// We need these definitions for the recursion within this module
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size, const int number_of_ffts);
void fft_ref_1d_bw_local_internal(double *grid_in_real, double *grid_in_imag,
                                  double *grid_out_real, double *grid_out_imag,
                                  const int fft_size, const int number_of_ffts);

void reorder_input(const double complex *restrict grid_in,
                   double *restrict grid_out_real,
                   double *restrict grid_out_imag, const int fft_size,
                   const int number_of_ffts, const int stride_in,
                   const int distance_in) {
  if (distance_in == 1 && stride_in == number_of_ffts) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index * number_of_ffts + fft]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index * number_of_ffts + fft]);
      }
    }
  } else if (distance_in == fft_size && stride_in == 1) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      for (int index = 0; index < fft_size; index++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index + fft * fft_size]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index + fft * fft_size]);
      }
    }
  } else {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index * stride_in + fft * distance_in]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index * stride_in + fft * distance_in]);
      }
    }
  }
}

void reorder_input_r2c(const double *restrict grid_in,
                       double *restrict grid_out, const int fft_size,
                       const int number_of_ffts, const int stride_in,
                       const int distance_in) {
  if (distance_in == 1 && stride_in == number_of_ffts) {
    memcpy(grid_out, grid_in, number_of_ffts * fft_size * sizeof(double));
  } else if (distance_in == fft_size && stride_in == 1) {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_in, distance_in, fft_size)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      for (int index = 0; index < fft_size; index++) {
        grid_out[fft + index * number_of_ffts] =
            grid_in[index + fft * fft_size];
      }
    }
  } else {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_in, distance_in, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index * number_of_ffts] =
            grid_in[index * stride_in + fft * distance_in];
      }
    }
  }
}

void reorder_input_r2c_for_c2c(const double *restrict grid_in,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size[2], const int number_of_ffts,
                               const int stride_in, const int distance_in) {
#pragma omp parallel for default(none) collapse(3)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
  for (int index_large = 0; index_large < fft_size[0] / 2; index_large++) {
    for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[(index_large * fft_size[1] + index_2) * number_of_ffts +
                      fft] =
            grid_in[(2 * index_large * fft_size[1] + index_2) * stride_in +
                    fft * distance_in];
        grid_out_imag[(index_large * fft_size[1] + index_2) * number_of_ffts +
                      fft] =
            grid_in[((2 * index_large + 1) * fft_size[1] + index_2) *
                        stride_in +
                    fft * distance_in];
      }
    }
  }
}

void reorder_input_c2r_2d(const double complex *restrict grid_in,
                          double *restrict grid_out_real,
                          double *restrict grid_out_imag, const int fft_size[2],
                          const int number_of_ffts, const int stride_in,
                          const int distance_in) {
  // If the distance is 1, we can use a simple copy
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
  for (int index_0 = 0; index_0 < fft_size[0] / 2 + 1; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + (index_1 * (fft_size[0] / 2 + 1) + index_0) *
                                number_of_ffts] =
            creal(grid_in[fft * distance_in +
                          (index_0 * fft_size[1] + index_1) * stride_in]);
        grid_out_imag[fft + (index_1 * (fft_size[0] / 2 + 1) + index_0) *
                                number_of_ffts] =
            cimag(grid_in[fft * distance_in +
                          (index_0 * fft_size[1] + index_1) * stride_in]);
      }
    }
  }
}

void reorder_input_c2r_for_c2c(const double complex *restrict grid_in,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int large_factor, const int number_of_ffts,
                               const int stride_in, const int distance_in) {
  const double pi = acos(-1);
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, large_factor, pi)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    const double complex phase_factor =
        cexp(pi * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 1.0 - cimag(phase_factor);
    const double factor_minus_real = 1.0 + cimag(phase_factor);
    const double half_factor_real = creal(phase_factor);
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[index_large * number_of_ffts + fft] =
          factor_plus_real *
              creal(grid_in[index_large * stride_in + fft * distance_in]) -
          half_factor_real *
              cimag(grid_in[index_large * stride_in + fft * distance_in]) +
          factor_minus_real *
              creal(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]) -
          half_factor_real *
              cimag(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]);
      grid_out_imag[index_large * number_of_ffts + fft] =
          half_factor_real *
              creal(grid_in[index_large * stride_in + fft * distance_in]) +
          factor_plus_real *
              cimag(grid_in[index_large * stride_in + fft * distance_in]) -
          half_factor_real *
              creal(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]) -
          factor_minus_real *
              cimag(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]);
    }
  }
}

void reorder_input_c2r_for_c2c_2d(const double *restrict grid_in_real,
                                  const double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size[2],
                                  const int number_of_ffts) {
  assert(fft_size[0] % 2 == 0);
  const double pi = acos(-1);
  const int large_factor = fft_size[0] / 2;
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, large_factor, fft_size, pi)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    const double complex phase_factor =
        cexp(pi * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 1.0 - cimag(phase_factor);
    const double factor_minus_real = 1.0 + cimag(phase_factor);
    const double half_factor_real = creal(phase_factor);
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[(index_1 + index_large * fft_size[1]) * number_of_ffts +
                      fft] =
            factor_plus_real *
                grid_in_real[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] -
            half_factor_real *
                grid_in_imag[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] +
            factor_minus_real * grid_in_real[((large_factor - index_large) +
                                              index_1 * (large_factor + 1)) *
                                                 number_of_ffts +
                                             fft] -
            half_factor_real * grid_in_imag[((large_factor - index_large) +
                                             index_1 * (large_factor + 1)) *
                                                number_of_ffts +
                                            fft];
        grid_out_imag[(index_1 + index_large * fft_size[1]) * number_of_ffts +
                      fft] =
            half_factor_real *
                grid_in_real[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] +
            factor_plus_real *
                grid_in_imag[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] -
            half_factor_real * grid_in_real[((large_factor - index_large) +
                                             index_1 * (large_factor + 1)) *
                                                number_of_ffts +
                                            fft] -
            factor_minus_real * grid_in_imag[((large_factor - index_large) +
                                              index_1 * (large_factor + 1)) *
                                                 number_of_ffts +
                                             fft];
      }
    }
  }
}

void reorder_output(const double *grid_in_real, const double *grid_in_imag,
                    double complex *grid_out, const int fft_size,
                    const int number_of_ffts, const int stride_out,
                    const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size)
  for (int index = 0; index < fft_size; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          CMPLX(grid_in_real[fft + index * number_of_ffts],
                grid_in_imag[fft + index * number_of_ffts]);
    }
  }
}

void reorder_output_2d(const double *grid_in_real, const double *grid_in_imag,
                       double complex *grid_out, const int fft_size[2],
                       const int number_of_ffts, const int stride_out,
                       const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size)
  for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft * distance_out +
                 (index_0 * fft_size[1] + index_1) * stride_out] =
            CMPLX(grid_in_real[fft + (index_1 * fft_size[0] + index_0) *
                                         number_of_ffts],
                  grid_in_imag[fft + (index_1 * fft_size[0] + index_0) *
                                         number_of_ffts]);
      }
    }
  }
}

void reorder_output_c2r(const double *grid_in, double *grid_out,
                        const int fft_size, const int number_of_ffts,
                        const int stride_out, const int distance_out) {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_out, distance_out, fft_size)
  for (int index = 0; index < fft_size; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          grid_in[fft + index * number_of_ffts];
    }
  }
}

void reorder_output_c2r_from_c2c(const double *grid_in_real,
                                 const double *grid_in_imag, double *grid_out,
                                 const int large_factor,
                                 const int number_of_ffts, const int stride_out,
                                 const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, large_factor)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[2 * index_large * stride_out + fft * distance_out] =
          grid_in_real[index_large * number_of_ffts + fft];
      grid_out[(2 * index_large + 1) * stride_out + fft * distance_out] =
          grid_in_imag[index_large * number_of_ffts + fft];
    }
  }
}

void reorder_output_c2r_from_c2c_2d(
    const double *grid_in_real, const double *grid_in_imag, double *grid_out,
    const int large_factor, const int fft_size_2, const int number_of_ffts,
    const int stride_out, const int distance_out) {
#pragma omp parallel for default(none) collapse(3)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, large_factor, fft_size_2)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    for (int index_2 = 0; index_2 < fft_size_2; index_2++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[(2 * index_large * fft_size_2 + index_2) * stride_out +
                 fft * distance_out] =
            grid_in_real[(index_large * fft_size_2 + index_2) * number_of_ffts +
                         fft];
        grid_out[((2 * index_large + 1) * fft_size_2 + index_2) * stride_out +
                 fft * distance_out] =
            grid_in_imag[(index_large * fft_size_2 + index_2) * number_of_ffts +
                         fft];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts) {
  // Perform FFTs along the first dimension
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real,
           fft_size * number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag,
           fft_size * number_of_ffts * sizeof(double));
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] =
          grid_in_real[fft] + grid_in_real[number_of_ffts + fft];
      grid_out_imag[fft] =
          grid_in_imag[fft] + grid_in_imag[number_of_ffts + fft];
      grid_out_real[number_of_ffts + fft] =
          grid_in_real[fft] - grid_in_real[number_of_ffts + fft];
      grid_out_imag[number_of_ffts + fft] =
          grid_in_imag[fft] - grid_in_imag[number_of_ffts + fft];
    }
  } else if (fft_size == 4) {
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum1 =
          grid_in_real[fft] + grid_in_real[fft + 2 * number_of_ffts];
      const double real_sum2 = grid_in_real[fft + number_of_ffts] +
                               grid_in_real[fft + 3 * number_of_ffts];
      const double real_diff1 =
          grid_in_real[fft] - grid_in_real[fft + 2 * number_of_ffts];
      const double real_diff2 = grid_in_real[fft + number_of_ffts] -
                                grid_in_real[fft + 3 * number_of_ffts];
      const double imag_sum1 =
          grid_in_imag[fft] + grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_sum2 = grid_in_imag[fft + number_of_ffts] +
                               grid_in_imag[fft + 3 * number_of_ffts];
      const double imag_diff1 =
          grid_in_imag[fft] - grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_diff2 = grid_in_imag[fft + number_of_ffts] -
                                grid_in_imag[fft + 3 * number_of_ffts];
      grid_out_real[fft] = real_sum1 + real_sum2;
      grid_out_real[fft + number_of_ffts] = real_diff1 + imag_diff2;
      grid_out_real[fft + 2 * number_of_ffts] = real_sum1 - real_sum2;
      grid_out_real[fft + 3 * number_of_ffts] = real_diff1 - imag_diff2;
      grid_out_imag[fft] = imag_sum1 + imag_sum2;
      grid_out_imag[fft + number_of_ffts] = imag_diff1 - real_diff2;
      grid_out_imag[fft + 2 * number_of_ffts] = imag_sum1 - imag_sum2;
      grid_out_imag[fft + 3 * number_of_ffts] = imag_diff1 + real_diff2;
    }
  } else if (fft_size == 3) {
    const double half_sqrt3 = 0.5 * sqrt(3.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, half_sqrt3)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum = grid_in_real[fft + number_of_ffts] +
                              grid_in_real[fft + 2 * number_of_ffts];
      const double imag_sum = grid_in_imag[fft + number_of_ffts] +
                              grid_in_imag[fft + 2 * number_of_ffts];
      const double real_diff = grid_in_real[fft + number_of_ffts] -
                               grid_in_real[fft + 2 * number_of_ffts];
      const double imag_diff = grid_in_imag[fft + number_of_ffts] -
                               grid_in_imag[fft + 2 * number_of_ffts];
      grid_out_real[fft] = grid_in_real[fft] + real_sum;
      grid_out_imag[fft] = grid_in_imag[fft] + imag_sum;
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum + half_sqrt3 * imag_diff;
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] - half_sqrt3 * real_diff - 0.5 * imag_sum;
      grid_out_real[fft + 2 * number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum - half_sqrt3 * imag_diff;
      grid_out_imag[fft + 2 * number_of_ffts] =
          grid_in_imag[fft] + half_sqrt3 * real_diff - 0.5 * imag_sum;
    }
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  creal(phase_factor) -
              grid_in_imag[fft + index_in * number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  cimag(phase_factor) +
              grid_in_imag[fft + index_in * number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_naive(const double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size,
                                   const int number_of_ffts) {
  double *grid_out_real = grid_out;
  double *grid_out_imag = grid_out_real + (fft_size / 2 + 1) * number_of_ffts;
  // Perform FFTs along the first dimension
  const double pi = acos(-1.0);
  memset(grid_out_real, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));
  memset(grid_out_imag, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none) shared(                                 \
        grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts, pi)
  for (int index_out = 0; index_out < fft_size / 2 + 1; index_out++) {
    for (int index_in = 0; index_in < fft_size; index_in++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * creal(phase_factor);
        grid_out_imag[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * cimag(phase_factor);
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts) {
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real, number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag, number_of_ffts * sizeof(double));
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] =
          grid_in_real[fft] + grid_in_real[fft + number_of_ffts];
      grid_out_imag[fft] =
          grid_in_imag[fft] + grid_in_imag[fft + number_of_ffts];
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - grid_in_real[fft + number_of_ffts];
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] - grid_in_imag[fft + number_of_ffts];
    }
  } else if (fft_size == 4) {
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum1 =
          grid_in_real[fft] + grid_in_real[fft + 2 * number_of_ffts];
      const double real_sum2 = grid_in_real[fft + number_of_ffts] +
                               grid_in_real[fft + 3 * number_of_ffts];
      const double real_diff1 =
          grid_in_real[fft] - grid_in_real[fft + 2 * number_of_ffts];
      const double real_diff2 = grid_in_real[fft + number_of_ffts] -
                                grid_in_real[fft + 3 * number_of_ffts];
      const double imag_sum1 =
          grid_in_imag[fft] + grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_sum2 = grid_in_imag[fft + number_of_ffts] +
                               grid_in_imag[fft + 3 * number_of_ffts];
      const double imag_diff1 =
          grid_in_imag[fft] - grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_diff2 = grid_in_imag[fft + number_of_ffts] -
                                grid_in_imag[fft + 3 * number_of_ffts];
      grid_out_real[fft] = real_sum1 + real_sum2;
      grid_out_real[fft + number_of_ffts] = real_diff1 - imag_diff2;
      grid_out_real[fft + 2 * number_of_ffts] = real_sum1 - real_sum2;
      grid_out_real[fft + 3 * number_of_ffts] = real_diff1 + imag_diff2;
      grid_out_imag[fft] = imag_sum1 + imag_sum2;
      grid_out_imag[fft + number_of_ffts] = imag_diff1 + real_diff2;
      grid_out_imag[fft + 2 * number_of_ffts] = imag_sum1 - imag_sum2;
      grid_out_imag[fft + 3 * number_of_ffts] = imag_diff1 - real_diff2;
    }
  } else if (fft_size == 3) {
    const double half_sqrt3 = 0.5 * sqrt(3.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, half_sqrt3)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum = grid_in_real[fft + number_of_ffts] +
                              grid_in_real[fft + 2 * number_of_ffts];
      const double imag_sum = grid_in_imag[fft + number_of_ffts] +
                              grid_in_imag[fft + 2 * number_of_ffts];
      const double real_diff = grid_in_real[fft + number_of_ffts] -
                               grid_in_real[fft + 2 * number_of_ffts];
      const double imag_diff = grid_in_imag[fft + number_of_ffts] -
                               grid_in_imag[fft + 2 * number_of_ffts];
      grid_out_real[fft] = grid_in_real[fft] + real_sum;
      grid_out_imag[fft] = grid_in_imag[fft] + imag_sum;
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum - half_sqrt3 * imag_diff;
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] + half_sqrt3 * real_diff - 0.5 * imag_sum;
      grid_out_real[fft + 2 * number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum + half_sqrt3 * imag_diff;
      grid_out_imag[fft + 2 * number_of_ffts] =
          grid_in_imag[fft] - half_sqrt3 * real_diff - 0.5 * imag_sum;
    }
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  creal(phase_factor) -
              grid_in_imag[fft + index_in * number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  cimag(phase_factor) +
              grid_in_imag[fft + index_in * number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_naive(double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size,
                                   const int number_of_ffts) {

  double *grid_in_real = grid_in;
  double *grid_in_imag = grid_in + (fft_size / 2 + 1) * number_of_ffts;

  for (int index_out = 0; index_out < fft_size; index_out++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft + index_out * number_of_ffts] = 0.0;
    }
    for (int index_in = 0; index_in < fft_size / 2 + 1; index_in++) {
      double complex phase_factor =
          cexp(2.0 * I * acos(-1) * index_in * index_out / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in_real[fft + index_in * number_of_ffts] *
                creal(phase_factor) -
            grid_in_imag[fft + index_in * number_of_ffts] * cimag(phase_factor);
      }
    }
    for (int index_in = fft_size / 2 + 1; index_in < fft_size; index_in++) {
      double complex phase_factor2 =
          cexp(2.0 * I * acos(-1) * index_in * index_out / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in_real[fft + (fft_size - index_in) * number_of_ffts] *
                creal(phase_factor2) +
            grid_in_imag[fft + (fft_size - index_in) * number_of_ffts] *
                cimag(phase_factor2);
      }
    }
  }
}

void fft_twiddle_r2c_fw_for_c2c(const double *grid_in_real,
                                const double *grid_in_imag, double *grid_out,
                                const int fft_size, const int number_of_ffts,
                                const int stride_out, const int distance_out,
                                const int imag_shift) {
  const int large_factor = fft_size / 2;
  for (int index_large = 0; index_large < large_factor + 1; index_large++) {
    const double complex phase_factor =
        cexp(-acos(-1) * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 0.5 - 0.5 * cimag(phase_factor);
    const double factor_minus_real = 0.5 + 0.5 * cimag(phase_factor);
    const double half_factor_real = 0.5 * creal(phase_factor);
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[2 * index_large * stride_out + 2 * fft * distance_out] =
          factor_minus_real *
              grid_in_real[index_large % large_factor * number_of_ffts + fft] +
          half_factor_real *
              grid_in_imag[index_large % large_factor * number_of_ffts + fft] +
          factor_plus_real * grid_in_real[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft] +
          half_factor_real * grid_in_imag[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft];
      grid_out[2 * index_large * stride_out + 2 * fft * distance_out +
               imag_shift] =
          -half_factor_real *
              grid_in_real[index_large % large_factor * number_of_ffts + fft] +
          factor_minus_real *
              grid_in_imag[index_large % large_factor * number_of_ffts + fft] +
          half_factor_real * grid_in_real[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft] -
          factor_plus_real * grid_in_imag[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft];
    }
  }
}

void fft_twiddle_r2c_fw_for_c2c_2d(const double *grid_in_real,
                                   const double *grid_in_imag, double *grid_out,
                                   const int fft_size[2],
                                   const int number_of_ffts) {
  const int large_factor = fft_size[0] / 2;
  for (int index_large = 0; index_large < large_factor + 1; index_large++) {
    const double complex phase_factor =
        cexp(-acos(-1) * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 0.5 - 0.5 * cimag(phase_factor);
    const double factor_minus_real = 0.5 + 0.5 * cimag(phase_factor);
    const double half_factor_real = 0.5 * creal(phase_factor);
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[(index_1 * (large_factor + 1) + index_large) * number_of_ffts +
                 fft] =
            factor_minus_real *
                grid_in_real[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            half_factor_real *
                grid_in_imag[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            factor_plus_real * grid_in_real[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft] +
            half_factor_real * grid_in_imag[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft];
        grid_out[(large_factor + 1) * number_of_ffts * fft_size[1] +
                 (index_1 * (large_factor + 1) + index_large) * number_of_ffts +
                 fft] =
            -half_factor_real *
                grid_in_real[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            factor_minus_real *
                grid_in_imag[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            half_factor_real * grid_in_real[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft] -
            factor_plus_real * grid_in_imag[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  if (small_factor == 2 && fft_size % 4 == 0)
    small_factor = 4;
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_fw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_fw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
    fft_ref_1d_fw_local_internal(grid_in_real, grid_in_imag, grid_out_real,
                                 grid_out_imag, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Cooley-Tukey step (For internal use only, special format).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  if (small_factor == 2 && fft_size % 4 == 0)
    small_factor = 4;
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_bw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_bw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
    fft_ref_1d_bw_local_internal(grid_in_real, grid_in_imag, grid_out_real,
                                 grid_out_imag, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size, const int number_of_ffts,
                                 const int stride_in, const int stride_out,
                                 const int distance_in,
                                 const int distance_out) {

  double *grid_in_real = grid_in;
  double *grid_out_real = (double *)grid_out;
  double *grid_in_imag = grid_in_real + (fft_size / 2 + 1) * number_of_ffts;
  double *grid_out_imag = grid_out_real + (fft_size / 2 + 1) * number_of_ffts;

  if (fft_size % 2 == 0) {
    grid_in_imag = grid_in_real + (fft_size / 2 + 1) * number_of_ffts;
    grid_out_imag = grid_out_real + (fft_size / 2 + 1) * number_of_ffts;
    const int large_factor = fft_size / 2;
    reorder_input_r2c_for_c2c(grid_in, grid_out_real, grid_out_imag,
                              (const int[2]){fft_size, 1}, number_of_ffts,
                              stride_in, distance_in);
    fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor, number_of_ffts);
    fft_twiddle_r2c_fw_for_c2c(grid_in_real, grid_in_imag, grid_out_real,
                               fft_size, number_of_ffts, stride_out,
                               distance_out, 1);
  } else {
    grid_in_imag = grid_in_real + fft_size * (number_of_ffts / 2);
    grid_out_imag = grid_out_real + fft_size * (number_of_ffts / 2);
    // Calculate two R2C FFTs as a single complex FFT
    for (int fft = 0; fft < number_of_ffts / 2 * 2; fft += 2) {
      for (int index = 0; index < fft_size; index++) {
        grid_out_real[index * (number_of_ffts / 2) + fft / 2] =
            grid_in[index * stride_in + fft * distance_in];
        grid_out_imag[index * (number_of_ffts / 2) + fft / 2] =
            grid_in[index * stride_in + (fft + 1) * distance_in];
      }
    }
    // If the number of FFTs is odd, we take care of the last FFT at the end
    if (number_of_ffts % 2 == 1) {
      for (int index = 0; index < fft_size; index++) {
        grid_out_imag[fft_size * (number_of_ffts / 2) + index] =
            grid_in[index * stride_in + (number_of_ffts - 1) * distance_in];
      }
    }
    // Perform the actual FFTs
    fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, fft_size, number_of_ffts / 2);
    if (number_of_ffts % 2 == 1) {
      fft_ref_1d_fw_local_r2c_naive(
          grid_out_imag + fft_size * (number_of_ffts / 2),
          grid_in_imag + fft_size * (number_of_ffts / 2), fft_size, 1);
    }
    // Reorder the data to the final layout
    for (int fft = 0; fft < number_of_ffts / 2 * 2; fft += 2) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        grid_out[index * stride_out + fft * distance_out] =
            CMPLX(0.5 * (grid_in_real[index * (number_of_ffts / 2) + fft / 2] +
                         grid_in_real[(fft_size - index) % fft_size *
                                          (number_of_ffts / 2) +
                                      fft / 2]),
                  0.5 * (grid_in_imag[index * (number_of_ffts / 2) + fft / 2] -
                         grid_in_imag[(fft_size - index) % fft_size *
                                          (number_of_ffts / 2) +
                                      fft / 2]));
        grid_out[index * stride_out + (fft + 1) * distance_out] =
            CMPLX(0.5 * (grid_in_imag[index * (number_of_ffts / 2) + fft / 2] +
                         grid_in_imag[(fft_size - index) % fft_size *
                                          (number_of_ffts / 2) +
                                      fft / 2]),
                  -0.5 * (grid_in_real[index * (number_of_ffts / 2) + fft / 2] -
                          grid_in_real[(fft_size - index) % fft_size *
                                           (number_of_ffts / 2) +
                                       fft / 2]));
      }
    }
    if (number_of_ffts % 2 == 1) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        grid_out[index * stride_out + (number_of_ffts - 1) * distance_out] =
            CMPLX(grid_in_imag[fft_size * (number_of_ffts / 2) + index],
                  grid_in_imag[fft_size * (number_of_ffts / 2) +
                               (fft_size / 2 + 1) + index]);
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out, const int fft_size,
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

  if (fft_size % 2 == 0) {
    const int large_factor = fft_size / 2;
    double *grid_in_real = (double *)grid_in;
    double *grid_in_imag = grid_in_real + large_factor * number_of_ffts;
    double *grid_out_real = grid_out;
    double *grid_out_imag = grid_out + large_factor * number_of_ffts;

    reorder_input_c2r_for_c2c(grid_in, grid_out_real, grid_out_imag,
                              large_factor, number_of_ffts, stride_in,
                              distance_in);
    fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor, number_of_ffts);
    reorder_output_c2r_from_c2c(grid_in_real, grid_in_imag, grid_out,
                                large_factor, number_of_ffts, stride_out,
                                distance_out);
  } else {
    double *grid_in_real = (double *)grid_in;
    double *grid_in_imag = grid_in_real + fft_size * (number_of_ffts / 2);
    double *grid_out_real = grid_out;
    double *grid_out_imag = grid_out_real + fft_size * (number_of_ffts / 2);
    // The target is given by H_k=F_k(fft)+I*F_k(fft+1)
    for (int fft = 0; fft < number_of_ffts / 2 * 2; fft += 2) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        grid_out_real[index * (number_of_ffts / 2) + fft / 2] =
            creal(grid_in[index * stride_in + fft * distance_in]) -
            cimag(grid_in[index * stride_in + (fft + 1) * distance_in]);
        grid_out_imag[index * (number_of_ffts / 2) + fft / 2] =
            cimag(grid_in[index * stride_in + fft * distance_in]) +
            creal(grid_in[index * stride_in + (fft + 1) * distance_in]);
        grid_out_real[(fft_size - index) % fft_size * (number_of_ffts / 2) +
                      fft / 2] =
            creal(grid_in[index * stride_in + fft * distance_in]) +
            cimag(grid_in[index * stride_in + (fft + 1) * distance_in]);
        grid_out_imag[(fft_size - index) % fft_size * (number_of_ffts / 2) +
                      fft / 2] =
            -cimag(grid_in[index * stride_in + fft * distance_in]) +
            creal(grid_in[index * stride_in + (fft + 1) * distance_in]);
      }
    }
    if (number_of_ffts % 2 == 1) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        grid_out_imag[fft_size * (number_of_ffts / 2) + index] = creal(
            grid_in[index * stride_in + (number_of_ffts - 1) * distance_in]);
        grid_out_imag[fft_size * (number_of_ffts / 2) + (fft_size / 2 + 1) +
                      index] =
            cimag(grid_in[index * stride_in +
                          (number_of_ffts - 1) * distance_in]);
      }
    }
    // Perform the actual FFTs
    fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, fft_size, number_of_ffts / 2);
    if (number_of_ffts % 2 == 1) {
      fft_ref_1d_bw_local_c2r_naive(
          grid_out_imag + fft_size * (number_of_ffts / 2),
          grid_in_imag + fft_size * (number_of_ffts / 2), fft_size, 1);
    }
    // Calculate two R2C FFTs as a single complex FFT
    for (int fft = 0; fft < number_of_ffts / 2 * 2; fft += 2) {
      for (int index = 0; index < fft_size; index++) {
        grid_out[index * stride_out + fft * distance_out] =
            grid_in_real[index * (number_of_ffts / 2) + fft / 2];
        grid_out[index * stride_out + (fft + 1) * distance_out] =
            grid_in_imag[index * (number_of_ffts / 2) + fft / 2];
      }
    }
    // If the number of FFTs is odd, we take care of the last FFT at the end
    if (number_of_ffts % 2 == 1) {
      for (int index = 0; index < fft_size; index++) {
        grid_out[index * stride_out + (number_of_ffts - 1) * distance_out] =
            grid_in_imag[fft_size * (number_of_ffts / 2) + index];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1], number_of_ffts, stride_in,
                distance_in);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               number_of_ffts * fft_size[1]);
  // Transpose the data
  transpose_local_double_block(grid_in_real, grid_out_real, fft_size[1],
                               fft_size[0], number_of_ffts);
  transpose_local_double_block(grid_in_imag, grid_out_imag, fft_size[1],
                               fft_size[0], number_of_ffts);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double complex *grid_in_complex = (double complex *)grid_in;
  double *grid_out_real = (double *)grid_out;

  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[(index_0 * number_of_ffts + fft) * fft_size[1] +
                      index_1] =
            grid_in[(index_0 * fft_size[1] + index_1) * stride_in +
                    fft * distance_in];
      }
    }
  }
  fft_ref_1d_fw_local_r2c_low(grid_out_real, grid_in_complex, fft_size[1],
                              fft_size[0] * number_of_ffts, 1, 1, fft_size[1],
                              fft_size[1] / 2 + 1);
  fft_ref_1d_fw_local_low(grid_in_complex, grid_out, fft_size[0],
                          (fft_size[1] / 2 + 1) * number_of_ffts,
                          (fft_size[1] / 2 + 1) * number_of_ffts,
                          (fft_size[1] / 2 + 1) * number_of_ffts, 1, 1);
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_in_complex[(index_0 * (fft_size[1] / 2 + 1) + index_1) *
                            stride_out +
                        fft * distance_out] =
            grid_out[(index_0 * number_of_ffts + fft) * (fft_size[1] / 2 + 1) +
                     index_1];
      }
    }
  }
  memcpy(grid_out, grid_in_complex,
         fft_size[0] * (fft_size[1] / 2 + 1) * number_of_ffts *
             sizeof(double complex));
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1], number_of_ffts, stride_in,
                distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               number_of_ffts * fft_size[1]);
  // Transpose the data
  transpose_local_double_block(grid_in_real, grid_out_real, fft_size[1],
                               fft_size[0], number_of_ffts);
  transpose_local_double_block(grid_in_imag, grid_out_imag, fft_size[1],
                               fft_size[0], number_of_ffts);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double complex *grid_out_complex = ((double complex *)grid_out);

  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_complex[(index_0 * number_of_ffts + fft) *
                             (fft_size[1] / 2 + 1) +
                         index_1] =
            grid_in[(index_0 * (fft_size[1] / 2 + 1) + index_1) * stride_in +
                    fft * distance_in];
      }
    }
  }
  fft_ref_1d_bw_local_low(grid_out_complex, grid_in, fft_size[0],
                          (fft_size[1] / 2 + 1) * number_of_ffts,
                          (fft_size[1] / 2 + 1) * number_of_ffts,
                          (fft_size[1] / 2 + 1) * number_of_ffts, 1, 1);
  fft_ref_1d_bw_local_c2r_low(grid_in, grid_out, fft_size[1],
                              fft_size[0] * number_of_ffts, 1, 1,
                              fft_size[1] / 2 + 1, fft_size[1]);
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_in_real[(index_0 * fft_size[1] + index_1) * stride_out +
                     fft * distance_out] =
            grid_out[(index_0 * number_of_ffts + fft) * fft_size[1] + index_1];
      }
    }
  }
  memcpy(grid_out, grid_in_real,
         fft_size[0] * fft_size[1] * number_of_ffts * sizeof(double));
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size[0],
                fft_size[1] * fft_size[2], fft_size[1] * fft_size[2], 1);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * fft_size[2]);
  transpose_local_double(grid_in_real, grid_out_real, fft_size[1] * fft_size[2],
                         fft_size[0], fft_size[1] * fft_size[2], fft_size[0]);
  transpose_local_double(grid_in_imag, grid_out_imag, fft_size[1] * fft_size[2],
                         fft_size[0], fft_size[1] * fft_size[2], fft_size[0]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * fft_size[2]);
  transpose_local_double(grid_in_real, grid_out_real, fft_size[0] * fft_size[2],
                         fft_size[1], fft_size[0] * fft_size[2], fft_size[1]);
  transpose_local_double(grid_in_imag, grid_out_imag, fft_size[0] * fft_size[2],
                         fft_size[1], fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[2],
                               fft_size[0] * fft_size[1]);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size[2],
                 fft_size[0] * fft_size[1], 1, fft_size[2]);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size[3]) {

  fft_ref_1d_fw_local_r2c_low(grid_in, grid_out, fft_size[2],
                              fft_size[0] * fft_size[1], 1,
                              fft_size[0] * fft_size[1], fft_size[2], 1);
  fft_ref_1d_fw_local_low(grid_out, (double complex *)grid_in, fft_size[1],
                          fft_size[0] * (fft_size[2] / 2 + 1), 1,
                          fft_size[0] * (fft_size[2] / 2 + 1), fft_size[1], 1);
  fft_ref_1d_fw_local_low((double complex *)grid_in, grid_out, fft_size[0],
                          fft_size[1] * (fft_size[2] / 2 + 1), 1,
                          fft_size[1] * (fft_size[2] / 2 + 1), fft_size[0], 1);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size[0],
                fft_size[1] * fft_size[2], fft_size[1] * fft_size[2], 1);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * fft_size[2]);
  transpose_local_double(grid_in_real, grid_out_real, fft_size[1] * fft_size[2],
                         fft_size[0], fft_size[1] * fft_size[2], fft_size[0]);
  transpose_local_double(grid_in_imag, grid_out_imag, fft_size[1] * fft_size[2],
                         fft_size[0], fft_size[1] * fft_size[2], fft_size[0]);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * fft_size[2]);
  transpose_local_double(grid_in_real, grid_out_real, fft_size[0] * fft_size[2],
                         fft_size[1], fft_size[0] * fft_size[2], fft_size[1]);
  transpose_local_double(grid_in_imag, grid_out_imag, fft_size[0] * fft_size[2],
                         fft_size[1], fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[2],
                               fft_size[0] * fft_size[1]);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size[2],
                 fft_size[0] * fft_size[1], 1, fft_size[2]);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out,
                                 const int fft_size[3]) {

  fft_ref_1d_bw_local_low(grid_in, (double complex *)grid_out, fft_size[0],
                          fft_size[1] * (fft_size[2] / 2 + 1),
                          fft_size[1] * (fft_size[2] / 2 + 1), 1, 1,
                          fft_size[0]);
  fft_ref_1d_bw_local_low((double complex *)grid_out, grid_in, fft_size[1],
                          fft_size[0] * (fft_size[2] / 2 + 1),
                          fft_size[0] * (fft_size[2] / 2 + 1), 1, 1,
                          fft_size[1]);
  fft_ref_1d_bw_local_c2r_low(grid_in, grid_out, fft_size[2],
                              fft_size[0] * fft_size[1],
                              fft_size[0] * fft_size[1], 1, 1, fft_size[2]);
}

// EOF
