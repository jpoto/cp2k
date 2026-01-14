/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib_test.h"
#include "fft_utils.h"

#include "../mpiwrap/cp_mpi.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_redistribution.h"
#include "fft_utils.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1d_local_low(const int fft_size, const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(fft_size * number_of_ffts, &input_array);
  fft_allocate_complex(fft_size * number_of_ffts, &output_array);

  memset(input_array, 0, fft_size * number_of_ffts * sizeof(double complex));

  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) * number_of_ffts + number_of_fft] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) + number_of_fft * fft_size] = 1.0;
    }
  }

  fft_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  input_array, output_array);

  double max_error = 0.0;
  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double complex my_value =
            output_array[number_of_fft + index * number_of_ffts];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1.0e-4)
          printf("ERROR %i %i/%i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double complex my_value =
            output_array[number_of_fft * fft_size + index];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1.0e-4)
          printf("ERROR %i %i/%i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-4) {
    if (my_process == 0) {
      printf("The fw 1D-FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0, fft_size * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft + number_of_fft % fft_size * number_of_ffts] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft * fft_size + number_of_fft % fft_size] = 1.0;
    }
  }

  fft_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double complex my_value =
            input_array[index * number_of_ffts + number_of_fft];
        const double complex ref_value =
            cexp(2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1.0e-12)
          printf("ERROR %i %i/%i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double complex my_value =
            input_array[index + number_of_fft * fft_size];
        const double complex ref_value =
            cexp(2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1.0e-12)
          printf("ERROR %i %i/%i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 1D FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 1D FFT does work correctly (%i %i)!\n", fft_size,
           number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1d_local_r2c_low(const int fft_size, const int number_of_ffts,
                              const int transpose_rs, const int transpose_gs) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double *input_array = NULL;
  double complex *output_array = NULL;
  fft_allocate_double(2 * (fft_size / 2 + 1) * number_of_ffts, &input_array);
  fft_allocate_complex((fft_size / 2 + 1) * number_of_ffts + 4, &output_array);

  memset(input_array, 0, fft_size * number_of_ffts * sizeof(double));
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) * number_of_ffts + number_of_fft] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) + number_of_fft * fft_size] = 1.0;
    }
  }

  fft_1d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      input_array, output_array);

  double max_error = 0.0;
  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        const double complex my_value =
            output_array[number_of_fft + index * number_of_ffts];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-12)
          printf("Error %i %i / %i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        const double complex my_value =
            output_array[number_of_fft * (fft_size / 2 + 1) + index];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-12)
          printf("Error %i %i / %i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-8) {
    if (my_process == 0) {
      printf("The fw R2C 1D-FFT does not work correctly (%i %i): %f!\n",
             fft_size, number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        output_array[number_of_fft + index * number_of_ffts] =
            cexp(-2.0 * I * acos(-1) * index * number_of_fft / fft_size);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        output_array[number_of_fft * (fft_size / 2 + 1) + index] =
            cexp(-2.0 * I * acos(-1) * index * number_of_fft / fft_size);
      }
    }
  }

  fft_1d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, my_process)                  \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double my_value =
            input_array[index * number_of_ffts + number_of_fft];
        const double ref_value =
            (number_of_fft % fft_size == index ? (double)fft_size : 0.0);
        const double current_error = fabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-4)
          printf("ERROR %i %i / %i %i : %f %f\n", number_of_fft, index,
                 number_of_ffts, fft_size, my_value, ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, my_process)                  \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double my_value = input_array[index + number_of_fft * fft_size];
        const double ref_value =
            (number_of_fft % fft_size == index ? (double)fft_size : 0.0);
        const double current_error = fabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-4)
          printf("ERROR %i %i / %i %i : %f %f\n", number_of_fft, index,
                 number_of_ffts, fft_size, my_value, ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  if (max_error > 1e-8) {
    if (my_process == 0) {
      printf("The bw C2R-1D FFT does not work correctly (%i %i): %f!\n",
             fft_size, number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  fft_free_double(input_array);
  fft_free_complex(output_array);

  if (errors == 0 && my_process == 0)
    printf("The 1D R/C FFT does work correctly (%i %i)!\n", fft_size,
           number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_local_low(const int fft_size[2], const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  const int elements_per_fft = fft_size[0] * fft_size[1];
  fft_allocate_complex(elements_per_fft * number_of_ffts, &input_array);
  fft_allocate_complex(elements_per_fft * number_of_ffts, &output_array);

  memset(input_array, 0,
         elements_per_fft * number_of_ffts * sizeof(double complex));
  double max_error = 0.0;
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, elements_per_fft, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % elements_per_fft * number_of_ffts +
                  number_of_fft] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, elements_per_fft, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % elements_per_fft +
                  number_of_fft * elements_per_fft] = 1.0;
    }
  }

  fft_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  input_array, output_array);

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
          const double complex my_value =
              output_array[number_of_fft +
                           (index_0 * fft_size[1] + index_1) * number_of_ffts];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1.0e-4)
            printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_0,
                   index_1, number_of_fft, fft_size[0], fft_size[1],
                   number_of_ffts, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
          const double complex my_value =
              output_array[(number_of_fft * fft_size[0] + index_0) *
                               fft_size[1] +
                           index_1];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1.0e-3)
            printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_0,
                   index_1, number_of_fft, fft_size[0], fft_size[1],
                   number_of_ffts, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-3) {
    if (my_process == 0) {
      printf("The fw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0,
         elements_per_fft * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, elements_per_fft, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft +
                   (number_of_fft % elements_per_fft) * number_of_ffts] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, elements_per_fft, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft * elements_per_fft +
                   number_of_fft % elements_per_fft] = 1.0;
    }
  }

  fft_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
          const double complex my_value =
              input_array[(index_0 * fft_size[1] + index_1) * number_of_ffts +
                          number_of_fft];
          const double complex ref_value = cexp(
              2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-12)
            printf("Error %i %i %i: (%f %f) (%f %f)\n", index_0, index_1,
                   number_of_fft, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
          const double complex my_value =
              input_array[index_0 * fft_size[1] + index_1 +
                          number_of_fft * fft_size[0] * fft_size[1]];
          const double complex ref_value = cexp(
              2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-12)
            printf("Error %i %i %i: (%f %f) (%f %f)\n", index_0, index_1,
                   number_of_fft, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 2D FFT does work correctly (%i %i/%i)!\n", fft_size[0],
           fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_local_r2c_low(const int fft_size[2], const int number_of_ffts,
                              const int transpose_rs, const int transpose_gs) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double *real_buffer = NULL;
  double complex *complex_buffer = NULL;
  fft_allocate_double(2 * (fft_size[1] / 2 + 1) * fft_size[0] * number_of_ffts,
                      &real_buffer);
  fft_allocate_complex((fft_size[1] / 2 + 1) * fft_size[0] * number_of_ffts,
                       &complex_buffer);
  memset(real_buffer, 0,
         2 * (fft_size[1] / 2 + 1) * fft_size[0] * number_of_ffts *
             sizeof(double));

  double max_error = 0.0;
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(real_buffer, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      real_buffer[(number_of_fft / fft_size[1] % fft_size[0] * fft_size[1] +
                   number_of_fft % fft_size[1]) *
                      number_of_ffts +
                  number_of_fft] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(real_buffer, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      real_buffer[(number_of_fft / fft_size[1] % fft_size[0] * fft_size[1] +
                   number_of_fft % fft_size[1]) +
                  number_of_fft * (fft_size[0] * fft_size[1])] = 1.0;
    }
  }

  fft_2d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      real_buffer, complex_buffer);

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, number_of_ffts, pi, my_process)           \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
          const double complex my_value =
              complex_buffer[number_of_fft +
                             (index_0 * (fft_size[1] / 2 + 1) + index_1) *
                                 number_of_ffts];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-6)
            printf("Error %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_0,
                   index_1, number_of_fft, fft_size[0], fft_size[1],
                   number_of_ffts, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, number_of_ffts, pi, my_process)           \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
          const double complex my_value =
              complex_buffer[number_of_fft * fft_size[0] *
                                 (fft_size[1] / 2 + 1) +
                             index_0 * (fft_size[1] / 2 + 1) + index_1];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-6)
            printf("Error %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_0,
                   index_1, number_of_fft, fft_size[0], fft_size[1],
                   number_of_ffts, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw R2C 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(complex_buffer, 0,
         fft_size[0] * (fft_size[1] / 2 + 1) * number_of_ffts *
             sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, number_of_ffts, pi)                       \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
          complex_buffer[number_of_fft +
                         (index_0 * (fft_size[1] / 2 + 1) + index_1) *
                             number_of_ffts] =
              cexp(-2.0 * I * pi *
                   ((double)(number_of_fft / fft_size[1]) * index_0 /
                        fft_size[0] +
                    (double)(number_of_fft % fft_size[1]) * index_1 /
                        fft_size[1]));
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, number_of_ffts, pi)                       \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1] / 2 + 1; index_1++) {
          complex_buffer[number_of_fft * fft_size[0] * (fft_size[1] / 2 + 1) +
                         index_0 * (fft_size[1] / 2 + 1) + index_1] =
              cexp(-2.0 * I * pi *
                   ((double)(number_of_fft / fft_size[1]) * index_0 /
                        fft_size[0] +
                    (double)(number_of_fft % fft_size[1]) * index_1 /
                        fft_size[1]));
        }
      }
    }
  }

  fft_2d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      complex_buffer, real_buffer);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(real_buffer, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
        for (int number_of_fft = 0; number_of_fft < number_of_ffts;
             number_of_fft++) {
          const double my_value =
              real_buffer[(index_0 * fft_size[1] + index_1) * number_of_ffts +
                          number_of_fft];
          const double ref_value =
              index_0 == number_of_fft / fft_size[1] % fft_size[0] &&
                      index_1 == number_of_fft % fft_size[1]
                  ? (double)(fft_size[0] * fft_size[1])
                  : 0.0;
          const double current_error = fabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-6)
            printf("Error %i %i %i/%i %i %i: %f %f\n", index_0, index_1,
                   number_of_fft, fft_size[0], fft_size[1], number_of_ffts,
                   my_value, ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(real_buffer, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
        for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
          const double my_value =
              real_buffer[index_0 * fft_size[1] + index_1 +
                          number_of_fft * fft_size[0] * fft_size[1]];
          const double ref_value =
              index_0 == number_of_fft / fft_size[1] % fft_size[0] &&
                      index_1 == number_of_fft % fft_size[1]
                  ? (double)(fft_size[0] * fft_size[1])
                  : 0.0;
          double current_error = fabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-6)
            printf("Error %i %i %i/%i %i %i: %f %f\n", index_0, index_1,
                   number_of_fft, fft_size[0], fft_size[1], number_of_ffts,
                   my_value, ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  fft_free_double(real_buffer);
  fft_free_complex(complex_buffer);

  if (max_error > 1e-8) {
    if (my_process == 0) {
      printf("The bw C2R 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 2D R2C/C2R FFT does work correctly (%i %i/%i)!\n", fft_size[0],
           fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_local_low(const int fft_size[3], const int test_every) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(fft_size[0] * fft_size[1] * fft_size[2], &input_array);
  fft_allocate_complex(fft_size[0] * fft_size[1] * fft_size[2], &output_array);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(input_array, 0,
               fft_size[0] * fft_size[1] * fft_size[2] *
                   sizeof(double complex));
        input_array[(mx * fft_size[1] + my) * fft_size[2] + mz] = 1.0;
        fft_3d_fw_local(fft_size, input_array, output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz) reduction(max : max_error)  \
    collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  output_array[(nx * fft_size[1] + ny) * fft_size[2] + nz];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(output_array, 0, product3(fft_size) * sizeof(double complex));
        output_array[(mx * fft_size[1] + my) * fft_size[2] + mz] = 1.0;
        fft_3d_bw_local(fft_size, output_array, input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, pi, mx, my, mz) reduction(max : max_error)   \
    collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  input_array[(nx * fft_size[1] + ny) * fft_size[2] + nz];
              const double complex ref_value =
                  cexp(2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT does work correctly (%i %i %i)!\n", fft_size[0],
           fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_local_r2c_low(const int fft_size[3], const int test_every) {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  const double pi = acos(-1);

  double *double_buffer = NULL;
  double complex *complex_buffer = NULL;
  fft_allocate_double(2 * fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1),
                      &double_buffer);
  fft_allocate_complex(fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1),
                       &complex_buffer);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(double_buffer, 0, product3(fft_size) * sizeof(double));
        double_buffer[(mx * fft_size[1] + my) * fft_size[2] + mz] = 1.0;
        fft_3d_fw_local_r2c(fft_size, double_buffer, complex_buffer);

#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, pi, mx, my, mz, my_process)               \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
              const double complex my_value =
                  complex_buffer[(nx * fft_size[1] + ny) *
                                     (fft_size[2] / 2 + 1) +
                                 nz];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (my_process == 0 && current_error > 1e-6) {
                printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", nx, ny, nz,
                       mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-6) {
    if (my_process == 0) {
      printf("The fw R2C 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;

#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, pi, mx, my, mz, my_process) collapse(3)
        for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nx = 0; nx < fft_size[0]; nx++) {
              complex_buffer[(nx * fft_size[1] + ny) * (fft_size[2] / 2 + 1) +
                             nz] = cexp(-2.0 * I * pi *
                                        (((double)mx) * nx / fft_size[0] +
                                         ((double)my) * ny / fft_size[1] +
                                         ((double)mz) * nz / fft_size[2]));
            }
          }
        }
        fft_3d_bw_local_c2r(fft_size, complex_buffer, double_buffer);

#pragma omp parallel for default(none)                                         \
    shared(double_buffer, fft_size, pi, mx, my, mz, my_process)                \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double my_value =
                  double_buffer[(nx * fft_size[1] + ny) * fft_size[2] + nz];
              const double ref_value = (nx == mx && ny == my && nz == mz)
                                           ? (double)product3(fft_size)
                                           : 0.0;
              double current_error = fabs(my_value - ref_value);
              if (my_process == 0 && current_error > 1e-6) {
                printf("ERROR %i %i %i/%i %i %i: %f %f\n", nx, ny, nz, mx, my,
                       mz, my_value, ref_value);
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  fft_free_double(double_buffer);
  fft_free_complex(complex_buffer);

  if (max_error > 1e-6) {
    if (my_process == 0) {
      printf("The bw 3D C2R FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D R2C/C2R FFT does work correctly (%i %i %i)!\n", fft_size[0],
           fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend (1-3D).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_local() {
  int errors = 0;

  clock_t begin = clock();
  errors += fft_test_1d_local_low(15, 26, true, true);
  errors += fft_test_1d_local_low(18, 22, true, false);
  errors += fft_test_1d_local_low(20, 28, false, true);
  errors += fft_test_1d_local_low(14, 13, false, false);

  errors += fft_test_1d_local_r2c_low(15, 26, true, false);
  errors += fft_test_1d_local_r2c_low(18, 22, false, false);
  errors += fft_test_1d_local_r2c_low(20, 28, false, true);
  errors += fft_test_1d_local_r2c_low(14, 13, true, true);

  errors += fft_test_2d_local_low((const int[2]){10, 10}, 10, true, true);
  errors += fft_test_2d_local_low((const int[2]){15, 9}, 10, true, false);
  errors += fft_test_2d_local_low((const int[2]){7, 20}, 11, false, true);
  errors += fft_test_2d_local_low((const int[2]){12, 14}, 10, false, false);

  errors += fft_test_2d_local_r2c_low((const int[2]){10, 10}, 10, true, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){15, 9}, 10, true, false);
  errors += fft_test_2d_local_r2c_low((const int[2]){7, 20}, 10, false, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){12, 14}, 11, false, false);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_low((const int[3]){8, 8, 8}, 10);
  errors += fft_test_3d_local_low((const int[3]){7, 5, 3}, 10);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_r2c_low((const int[3]){8, 8, 8}, 10);
  errors += fft_test_3d_local_r2c_low((const int[3]){7, 5, 3}, 10);
  clock_t end = clock();
  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0)
    printf("Time to test local FFTs with planning: %f\n",
           (double)(end - begin) / CLOCKS_PER_SEC);

  begin = clock();
  errors += fft_test_1d_local_low(15, 26, true, true);
  errors += fft_test_1d_local_low(18, 22, true, false);
  errors += fft_test_1d_local_low(20, 28, false, true);
  errors += fft_test_1d_local_low(14, 13, false, false);

  errors += fft_test_1d_local_r2c_low(15, 26, true, false);
  errors += fft_test_1d_local_r2c_low(18, 22, false, false);
  errors += fft_test_1d_local_r2c_low(20, 28, false, true);
  errors += fft_test_1d_local_r2c_low(14, 13, true, true);

  errors += fft_test_2d_local_low((const int[2]){10, 10}, 10, true, true);
  errors += fft_test_2d_local_low((const int[2]){15, 9}, 10, true, false);
  errors += fft_test_2d_local_low((const int[2]){7, 20}, 11, false, true);
  errors += fft_test_2d_local_low((const int[2]){12, 14}, 11, false, false);

  errors += fft_test_2d_local_r2c_low((const int[2]){10, 10}, 10, true, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){15, 9}, 10, true, false);
  errors += fft_test_2d_local_r2c_low((const int[2]){7, 20}, 11, false, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){12, 14}, 11, false, false);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_low((const int[3]){8, 8, 8}, 10);
  errors += fft_test_3d_local_low((const int[3]){7, 5, 3}, 10);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_r2c_low((const int[3]){8, 8, 8}, 10);
  errors += fft_test_3d_local_r2c_low((const int[3]){7, 5, 3}, 10);
  end = clock();
  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0)
    printf("Time to test local FFTs without planning: %f\n",
           (double)(end - begin) / CLOCKS_PER_SEC);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_distributed_low(const int fft_size[2],
                                const int number_of_ffts) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n0, local_n0_start;
  int local_n1, local_n1_start;
  const int buffer_size =
      fft_2d_distributed_sizes(fft_size, number_of_ffts, comm, &local_n0,
                               &local_n0_start, &local_n1, &local_n1_start);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(buffer_size, &input_array);
  fft_allocate_complex(buffer_size, &output_array);

  memset(input_array, 0, buffer_size * sizeof(double complex));
  double max_error = 0.0;
  // Check the forward FFT
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, local_n0, local_n0_start)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    const int index_0 = number_of_fft / fft_size[1] % fft_size[0];
    const int index_1 = number_of_fft % fft_size[1];
    if (index_0 >= local_n0_start && index_0 < local_n0_start + local_n0) {
      input_array[number_of_fft +
                  ((index_0 - local_n0_start) * fft_size[1] + index_1) *
                      number_of_ffts] = 1.0;
    }
  }

  fft_2d_fw_distributed(fft_size, number_of_ffts, comm, input_array,
                        output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n1,               \
               local_n1_start) reduction(max : max_error) collapse(3)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int index_1 = 0; index_1 < local_n1; index_1++) {
        const double complex my_value =
            output_array[number_of_fft +
                         (index_1 * fft_size[0] + index_0) * number_of_ffts];
        const double complex ref_value = cexp(
            -2.0 * I * pi *
            ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
             (double)(number_of_fft % fft_size[1]) *
                 (index_1 + local_n1_start) / fft_size[1]));
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("Error %i %i %i: (%f %f) (%f %f)\n", index_0, index_1,
                 number_of_fft, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The distributed fw 2D-FFT does not work correctly (%i %i/%i): "
             "%f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
    errors++;
  }

  // Check the backward FFT
  memset(input_array, 0, buffer_size * sizeof(double complex));
  memset(output_array, 0, buffer_size * sizeof(double complex));
  // Check the forward FFT
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, local_n1, local_n1_start)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    const int index_0 = number_of_fft / fft_size[1];
    const int index_1 = number_of_fft % fft_size[1];
    if (index_1 >= local_n1_start && index_1 < local_n1_start + local_n1) {
      input_array[number_of_fft +
                  ((index_1 - local_n1_start) * fft_size[0] + index_0) *
                      number_of_ffts] = 1.0;
    }
  }

  fft_2d_bw_distributed(fft_size, number_of_ffts, comm, input_array,
                        output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n0,               \
               local_n0_start) reduction(max : max_error) collapse(3)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index_0 = 0; index_0 < local_n0; index_0++) {
      for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
        const double complex my_value =
            output_array[number_of_fft +
                         (index_0 * fft_size[1] + index_1) * number_of_ffts];
        const double complex ref_value = cexp(
            2.0 * I * pi *
            ((double)(number_of_fft / fft_size[1]) *
                 (index_0 + local_n0_start) / fft_size[0] +
             (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("Error %i %i %i: (%f %f) (%f %f)\n", index_0, index_1,
                 number_of_fft, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf(
          "The distributed bw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
          fft_size[0], fft_size[1], number_of_ffts, max_error);
    errors++;
  }

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (errors == 0 && my_process == 0)
    printf("The distributed 2D FFT does work correctly (%i %i/%i)!\n",
           fft_size[0], fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_distributed_r2c_low(const int fft_size[2],
                                    const int number_of_ffts) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n0, local_n0_start;
  int local_n1, local_n1_start;
  const int buffer_size =
      fft_2d_distributed_sizes_r2c(fft_size, number_of_ffts, comm, &local_n0,
                                   &local_n0_start, &local_n1, &local_n1_start);

  double *input_array = NULL;
  double complex *output_array = NULL;
  fft_allocate_double(2 * buffer_size, &input_array);
  fft_allocate_complex(buffer_size, &output_array);

  double max_error = 0.0;
  // Check the forward FFT
  memset(input_array, 0, 2 * buffer_size * sizeof(double));
  memset(output_array, 0, buffer_size * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, local_n0, local_n0_start,    \
               my_process)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    const int index_0 = (number_of_fft / fft_size[1]) % fft_size[0];
    const int index_1 = number_of_fft % fft_size[1];
    if (index_0 >= local_n0_start && index_0 < local_n0_start + local_n0) {
      input_array[((index_0 - local_n0_start) * (fft_size[1] / 2 + 1) * 2 +
                   index_1) *
                      number_of_ffts +
                  number_of_fft] = 1.0;
    }
  }

  fft_2d_fw_distributed_r2c(fft_size, number_of_ffts, comm, input_array,
                            output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n1,               \
               local_n1_start, my_process) reduction(max : max_error)          \
    collapse(3)
  for (int index_1 = 0; index_1 < local_n1; index_1++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int number_of_fft = 0; number_of_fft < number_of_ffts;
           number_of_fft++) {
        const double complex my_value =
            output_array[(index_1 * fft_size[0] + index_0) * number_of_ffts +
                         number_of_fft];
        const double complex ref_value =
            cexp(-2.0 * I * pi *
                 ((double)((number_of_fft / fft_size[1]) % fft_size[0]) *
                      index_0 / fft_size[0] +
                  (double)(number_of_fft % fft_size[1]) *
                      (index_1 + local_n1_start) / fft_size[1]));
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("%i Error %i %i %i / %i %i %i: (%f %f) (%f %f)\n", my_process,
                 index_0, index_1 + local_n1_start, number_of_fft, fft_size[0],
                 fft_size[1], number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf(
          "The distributed fw R2C 2D-FFT does not work correctly (%i %i/%i): "
          "%f!\n",
          fft_size[0], fft_size[1], number_of_ffts, max_error);
    fflush(stdout);
    cp_mpi_barrier(comm);
    errors++;
  }

  // Check the backward FFT
  memset(input_array, 0, 2 * buffer_size * sizeof(double));
  memset(output_array, 0, buffer_size * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n1,               \
               local_n1_start) reduction(max : max_error) collapse(3)
  for (int index_1 = 0; index_1 < local_n1; index_1++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int number_of_fft = 0; number_of_fft < number_of_ffts;
           number_of_fft++) {
        output_array[number_of_fft +
                     (index_1 * fft_size[0] + index_0) * number_of_ffts] =
            cexp(-2.0 * I * pi *
                 ((double)((number_of_fft / fft_size[1]) % fft_size[0]) *
                      index_0 / fft_size[0] +
                  (double)(number_of_fft % fft_size[1]) *
                      (index_1 + local_n1_start) / fft_size[1]));
      }
    }
  }

  fft_2d_bw_distributed_c2r(fft_size, number_of_ffts, comm, output_array,
                            input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, local_n0,                \
               local_n0_start) reduction(max : max_error) collapse(3)
  for (int index_0 = 0; index_0 < local_n0; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int number_of_fft = 0; number_of_fft < number_of_ffts;
           number_of_fft++) {
        const double my_value =
            input_array[number_of_fft +
                        (index_0 * (fft_size[1] / 2 + 1) * 2 + index_1) *
                            number_of_ffts];
        const double ref_value =
            (index_0 + local_n0_start ==
                 (number_of_fft / fft_size[1]) % fft_size[0] &&
             index_1 == number_of_fft % fft_size[1])
                ? (double)(fft_size[0] * fft_size[1])
                : 0.0;
        double current_error = fabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("Error %i %i %i / %i %i %i: %f %f\n", index_0 + local_n0_start,
                 index_1, number_of_fft, fft_size[0], fft_size[1],
                 number_of_ffts, my_value, ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The distributed bw C2R 2D-FFT does not work correctly (%i "
             "%i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
    errors++;
  }

  fft_free_double(input_array);
  fft_free_complex(output_array);

  if (errors == 0 && my_process == 0)
    printf("The distributed 2D R2C/C2R FFT does work correctly (%i %i/%i)!\n",
           fft_size[0], fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_distributed_low(const int fft_size[3], const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n2, local_n2_start;
  int local_n1, local_n1_start;
  const int buffer_size = fft_3d_distributed_sizes(
      (const int[3]){fft_size[2], fft_size[1], fft_size[0]}, comm, &local_n2,
      &local_n2_start, &local_n1, &local_n1_start);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(buffer_size, &input_array);
  fft_allocate_complex(buffer_size, &output_array);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(input_array, 0, buffer_size * sizeof(double complex));
        if (mz >= local_n2_start && mz < local_n2_start + local_n2)
          input_array[(mz - local_n2_start) * fft_size[0] * fft_size[1] +
                      my * fft_size[0] + mx] = 1.0;
        fft_3d_fw_distributed(
            (const int[3]){fft_size[2], fft_size[1], fft_size[0]}, comm,
            input_array, output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz, local_n1, local_n1_start)   \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < local_n1; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  output_array[ny * fft_size[0] * fft_size[2] +
                               nz * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * (ny + local_n1_start) / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: (%f %f) (%f %f)\n", nx, ny,
                       nz, mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The distributed fw 3D-FFT does not work correctly (%i %i %i): "
             "%f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(output_array, 0, buffer_size * sizeof(double complex));
        if (my >= local_n1_start && my < local_n1_start + local_n1)
          output_array[((my - local_n1_start) * fft_size[2] + mz) *
                           fft_size[0] +
                       mx] = 1.0;

        fft_3d_bw_distributed(
            (const int[3]){fft_size[2], fft_size[1], fft_size[0]}, comm,
            output_array, input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, pi, mx, my, mz, local_n2, local_n2_start)    \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < local_n2; nz++) {
              const double complex my_value =
                  input_array[(nz * fft_size[1] + ny) * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * (nz + local_n2_start) / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: (%f %f) (%f %f)\n", nx, ny,
                       nz, mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The distributed bw 3D-FFT does not work correctly (%i %i %i): "
             "%f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (errors == 0 && my_process == 0)
    printf("The distributed 3D FFT does work correctly (%i %i %i)!\n",
           fft_size[0], fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_distributed_r2c_low(const int fft_size[3],
                                    const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n0, local_n0_start;
  int local_n1, local_n1_start;
  const int buffer_size = fft_3d_distributed_sizes_r2c(
      fft_size, comm, &local_n0, &local_n0_start, &local_n1, &local_n1_start);

  double *real_buffer = NULL;
  double complex *complex_buffer = NULL;
  fft_allocate_double(2 * buffer_size, &real_buffer);
  fft_allocate_complex(buffer_size, &complex_buffer);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(real_buffer, 0, 2 * buffer_size * sizeof(double));
        if (mx >= local_n0_start && mx < local_n0_start + local_n0)
          real_buffer[(mx - local_n0_start) * fft_size[1] *
                          (fft_size[2] / 2 + 1) * 2 +
                      my * (fft_size[2] / 2 + 1) * 2 + mz] = 1.0;
        fft_3d_fw_distributed_r2c(fft_size, comm, real_buffer, complex_buffer);

#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, pi, mx, my, mz, local_n1, local_n1_start) \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < local_n1; ny++) {
            for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
              const double complex my_value =
                  complex_buffer[ny * fft_size[0] * (fft_size[2] / 2 + 1) +
                                 nx * (fft_size[2] / 2 + 1) + nz];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * (ny + local_n1_start) / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: (%f %f) (%f %f)\n", nx, ny,
                       nz, mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf(
          "The distributed fw R2C 3D-FFT does not work correctly (%i %i %i): "
          "%f!\n",
          fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
#pragma omp parallel for default(none)                                         \
    shared(complex_buffer, fft_size, pi, mx, my, mz, local_n1, local_n1_start) \
    collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < local_n1; ny++) {
            for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
              complex_buffer[(ny * fft_size[0] + nx) * (fft_size[2] / 2 + 1) +
                             nz] =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * (ny + local_n1_start) / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
            }
          }
        }

        fft_3d_bw_distributed_c2r(fft_size, comm, complex_buffer, real_buffer);

#pragma omp parallel for default(none)                                         \
    shared(real_buffer, fft_size, pi, mx, my, mz, local_n0, local_n0_start)    \
    reduction(max : max_error) collapse(3)
        for (int nz = 0; nz < fft_size[2]; nz++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nx = 0; nx < local_n0; nx++) {
              const double my_value =
                  real_buffer[nx * fft_size[1] * (fft_size[2] / 2 + 1) * 2 +
                              ny * (fft_size[2] / 2 + 1) * 2 + nz];
              const double ref_value =
                  (nx + local_n0_start == mx && ny == my && nz == mz)
                      ? (double)(fft_size[0] * fft_size[1] * fft_size[2])
                      : 0.0;
              double current_error = fabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: %f %f\n", nx + local_n0_start,
                       ny, nz, mx, my, mz, my_value, ref_value);
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf(
          "The distributed bw C2R 3D-FFT does not work correctly (%i %i %i): "
          "%f!\n",
          fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  fft_free_double(real_buffer);
  fft_free_complex(complex_buffer);

  if (errors == 0 && my_process == 0)
    printf("The distributed 3D R2C FFT does work correctly (%i %i %i)!\n",
           fft_size[0], fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the distributed FFT backend (2-3D, 1D not used).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_distributed() {
  int errors = 0;

  const bool do_print = cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0;

  if (!fft_lib_use_mpi()) {
    if (do_print)
      printf("Skipped testing the distributed FFT backend!\n");
    return 0;
  }

  clock_t begin = clock();
  errors += fft_test_2d_distributed_low((const int[2]){15, 9}, 51);
  errors += fft_test_2d_distributed_low((const int[2]){12, 14}, 23);

  errors += fft_test_2d_distributed_r2c_low((const int[2]){15, 9}, 51);
  errors += fft_test_2d_distributed_r2c_low((const int[2]){12, 14}, 23);

  errors += fft_test_3d_distributed_low((const int[3]){8, 8, 8}, 19);
  errors += fft_test_3d_distributed_low((const int[3]){7, 5, 3}, 11);

  errors += fft_test_3d_distributed_r2c_low((const int[3]){8, 8, 8}, 19);
  errors += fft_test_3d_distributed_r2c_low((const int[3]){7, 5, 3}, 11);
  clock_t end = clock();
  if (do_print)
    printf("Time to test distributed FFTs with planning: %f\n",
           (double)(end - begin) / CLOCKS_PER_SEC);

  begin = clock();
  errors += fft_test_2d_distributed_low((const int[2]){15, 9}, 51);
  errors += fft_test_2d_distributed_low((const int[2]){12, 14}, 23);

  errors += fft_test_2d_distributed_r2c_low((const int[2]){15, 9}, 51);
  errors += fft_test_2d_distributed_r2c_low((const int[2]){12, 14}, 23);

  errors += fft_test_3d_distributed_low((const int[3]){8, 8, 8}, 19);
  errors += fft_test_3d_distributed_low((const int[3]){7, 5, 3}, 11);

  errors += fft_test_3d_distributed_r2c_low((const int[3]){8, 8, 8}, 19);
  errors += fft_test_3d_distributed_r2c_low((const int[3]){7, 5, 3}, 11);
  end = clock();
  if (do_print)
    printf("Time to test distributed FFTs without planning: %f\n",
           (double)(end - begin) / CLOCKS_PER_SEC);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the local transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose() {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());
  // Check a few fft sizes
  const int fft_sizes[2] = {16, 18};

  int max_size = fft_sizes[0] * fft_sizes[1];

  double complex *input_array = calloc(max_size, sizeof(double complex));
  double complex *output_array = calloc(max_size, sizeof(double complex));

#pragma omp parallel for default(none) shared(input_array, fft_sizes)          \
    collapse(2)
  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      input_array[index_1 * fft_sizes[1] + index_2] =
          1.0 * index_1 - index_2 * I;
    }
  }

  transpose_local_complex(input_array, output_array, fft_sizes[1], fft_sizes[0],
                          fft_sizes[1], fft_sizes[0]);

  double error = 0.0;

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_sizes, my_process) reduction(max : error)         \
    collapse(2)
  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      const double complex my_value =
          output_array[index_2 * fft_sizes[0] + index_1];
      const double complex ref_value = 1.0 * index_1 - index_2 * I;
      const double current_error = cabs(my_value - ref_value);
      if (current_error > 1e-12 && my_process == 0) {
        printf("Error %i %i/ %i %i: (%f %f) (%f %f)\n", index_1, index_2,
               fft_sizes[0], fft_sizes[1], creal(my_value), cimag(my_value),
               creal(ref_value), cimag(ref_value));
      }
      error = fmax(error, current_error);
    }
  }

  free(input_array);
  free(output_array);

  if (error > 1e-12) {
    if (my_process == 0)
      printf("The low-level transpose does not work correctly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("The local transpose does work correctly!\n");
    return 0;
  }
}

// EOF
