/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid_test.h"

#include "../mpiwrap/cp_mpi.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_redistribution.h"
#include "fft_utils.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_cartesian(const int npts_global[3], const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv,
                              false, -1.0, NULL, NULL);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1];
  const int my_number_of_elements_gs =
      imax(product3(my_sizes_gs), fft_grid_layout->npts_gs_local);

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double complex *rs_data = NULL;
  fft_allocate_complex(my_number_of_elements_rs, &rs_data);
  double complex *gs_data = NULL;
  fft_allocate_complex(my_number_of_elements_gs, &gs_data);

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(rs_data, 0, my_number_of_elements_rs * sizeof(double complex));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          rs_data[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                   my_bounds_rs[1][0]) *
                      my_sizes_rs[2] +
                  nz - my_bounds_rs[2][0]] = 1.0;

        fft_3d_fw_with_layout(rs_data, gs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(gs_data, fft_grid_layout, nx, ny, nz, npts_global, scale)           \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int *index_g = fft_grid_layout->index_to_g[index];
          const double complex my_value = gs_data[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_g[0]) * nx / npts_global[0] +
                            ((double)index_g[1]) * ny / npts_global[1] +
                            ((double)index_g[2]) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n",
                   index_g[0], index_g[1], index_g[2], nx, ny, nz, index,
                   creal(my_value), cimag(my_value), creal(ref_value),
                   cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw 3D FFT (non-cartesian) does not work correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check forward 3D FFTs
  max_error = 0.0;
  number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(gs_data, 0, my_number_of_elements_gs * sizeof(double complex));

#pragma omp parallel for default(none)                                         \
    shared(gs_data, fft_grid_layout, nx, ny, nz, npts_global)                  \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int *index_g = fft_grid_layout->index_to_g[index];
          gs_data[index] = cexp(-2.0 * I * pi *
                                (((double)index_g[0]) * nx / npts_global[0] +
                                 ((double)index_g[1]) * ny / npts_global[1] +
                                 ((double)index_g[2]) * nz / npts_global[2]));
        }

        fft_3d_bw_with_layout(gs_data, rs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(rs_data, my_sizes_rs, my_number_of_elements_rs, nx, ny, nz,         \
               npts_global, my_bounds_rs, stdout) reduction(max : max_error)
        for (int index = 0; index < my_number_of_elements_rs; index++) {
          const double complex my_value = rs_data[index];
          const int index_z = index % my_sizes_rs[2] + my_bounds_rs[2][0];
          const int index_y =
              index / my_sizes_rs[2] % my_sizes_rs[1] + my_bounds_rs[1][0];
          const int index_x =
              index / my_sizes_rs[2] / my_sizes_rs[1] + my_bounds_rs[0][0];
          const double complex ref_value =
              index_x == nx && index_y == ny && index_z == nz
                  ? ((double)product3(npts_global))
                  : 0.0;
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12) {
            printf("ERROR %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n", index_x,
                   index_y, index_z, nx, ny, nz, index, creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
            fflush(stdout);
          }
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 3D FFT (non-cartesian) does not work correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  fft_free_complex(rs_data);
  fft_free_complex(gs_data);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf(
        "The 3D FFT with non-cartesian layout does work correctly (sizes %i %i "
        "%i)!\n",
        npts_global[0], npts_global[1], npts_global[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_cartesian_cart(const int npts_global[3], const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv,
                              false, -1.0, NULL, NULL);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1];
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double complex *rs_data = NULL;
  fft_allocate_complex(my_number_of_elements_rs, &rs_data);
  double complex *gs_data = NULL;
  fft_allocate_complex(my_number_of_elements_gs, &gs_data);

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(rs_data, 0, my_number_of_elements_rs * sizeof(double complex));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          rs_data[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                   my_bounds_rs[1][0]) *
                      my_sizes_rs[2] +
                  (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_with_layout_to_cart(rs_data, gs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(gs_data, my_bounds_gs, my_sizes_gs, nx, ny, nz, npts_global,        \
               my_number_of_elements_gs, scale) reduction(max : max_error)
        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx =
              my_number_of_elements_gs / my_sizes_gs[1] / my_sizes_gs[2] +
              my_bounds_gs[0][0];
          const int my =
              my_number_of_elements_gs / my_sizes_gs[2] % my_sizes_gs[1] +
              my_bounds_gs[1][0];
          const int mz =
              my_number_of_elements_gs % my_sizes_gs[2] + my_bounds_gs[0][0];
          const double complex my_value = gs_data[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)mx) * nx / npts_global[0] +
                            ((double)my) * ny / npts_global[1] +
                            ((double)mz) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n", mx, my,
                   mz, nx, ny, nz, index, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw 3D FFT (cartesian) does not work correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check forward 3D FFTs
  max_error = 0.0;
  number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(gs_data, 0, my_number_of_elements_gs * sizeof(double complex));

#pragma omp parallel for default(none)                                         \
    shared(gs_data, my_sizes_gs, my_bounds_gs, my_number_of_elements_gs, nx,   \
               ny, nz, npts_global) reduction(max : max_error)
        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx =
              my_number_of_elements_gs / my_sizes_gs[1] / my_sizes_gs[2] +
              my_bounds_gs[0][0];
          const int my =
              my_number_of_elements_gs / my_sizes_gs[2] % my_sizes_gs[1] +
              my_bounds_gs[1][0];
          const int mz =
              my_number_of_elements_gs % my_sizes_gs[2] + my_bounds_gs[0][0];
          gs_data[index] = cexp(-2.0 * I * pi *
                                (((double)mx) * nx / npts_global[0] +
                                 ((double)my) * ny / npts_global[1] +
                                 ((double)mz) * nz / npts_global[2]));
        }

        fft_3d_bw_with_layout_from_cart(gs_data, rs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(rs_data, my_sizes_rs, my_number_of_elements_rs, nx, ny, nz,         \
               my_bounds_rs) reduction(max : max_error)
        for (int index = 0; index < my_number_of_elements_rs; index++) {
          const double complex my_value = rs_data[index];
          const int index_x = index % my_sizes_rs[2] + my_bounds_rs[2][0];
          const int index_y =
              index / my_sizes_rs[2] % my_sizes_rs[1] + my_bounds_rs[1][0];
          const int index_z =
              index / my_sizes_rs[2] / my_sizes_rs[1] + my_bounds_rs[0][0];
          const double complex ref_value =
              index_x == nx && index_y == ny && index_z == nz ? 1.0 : 0.0;
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n", index_x,
                   index_y, index_z, nx, ny, nz, index, creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 3D FFT (cartesian) does not work correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  fft_free_complex(rs_data);
  fft_free_complex(gs_data);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT with cartesian layout does work correctly "
           "(sizes %i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_r2c_cartesian(const int npts_global[3], const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv,
                              false, -1.0, NULL, NULL);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1];
  const int my_number_of_elements_gs =
      imax(product3(my_sizes_gs), fft_grid_layout->npts_gs_local);

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double *rs_data = NULL;
  fft_allocate_double(my_number_of_elements_rs, &rs_data);
  double complex *gs_data = NULL;
  fft_allocate_complex(my_number_of_elements_gs, &gs_data);

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(rs_data, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          rs_data[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                   my_bounds_rs[1][0]) *
                      my_sizes_rs[2] +
                  (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_r2c_with_layout(rs_data, gs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(gs_data, fft_grid_layout, nx, ny, nz, npts_global, scale)           \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int *index_g = fft_grid_layout->index_to_g[index];
          const double complex my_value = gs_data[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_g[0]) * nx / npts_global[0] +
                            ((double)index_g[1]) * ny / npts_global[1] +
                            ((double)index_g[2]) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n",
                   index_g[0], index_g[1], index_g[2], nx, ny, nz, index,
                   creal(my_value), cimag(my_value), creal(ref_value),
                   cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw R2C 3D FFT (non-cartesian) to ordered layout does not "
             "work correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(gs_data, 0, my_number_of_elements_gs * sizeof(double complex));

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx = fft_grid_layout->index_to_g[index][0];
          const int my = fft_grid_layout->index_to_g[index][1];
          const int mz = fft_grid_layout->index_to_g[index][2];
          gs_data[index] = cexp(-2.0 * I * pi *
                                (((double)mx) * nx / npts_global[0] +
                                 ((double)my) * ny / npts_global[1] +
                                 ((double)mz) * nz / npts_global[2]));
        }

        fft_3d_bw_c2r_with_layout(gs_data, rs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(rs_data, my_bounds_rs, my_sizes_rs, nx, ny, nz, npts_global)        \
    collapse(3) reduction(max : max_error)
        for (int mx = 0; mx < npts_global[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  rs_data[(mx * my_sizes_rs[1] + my) * my_sizes_rs[2] + mz];
              const double ref_value = (mx + my_bounds_rs[0][0] == nx &&
                                        my + my_bounds_rs[1][0] == ny &&
                                        mz + my_bounds_rs[2][0] == nz)
                                           ? (double)product3(npts_global)
                                           : 0.0;
              double current_error = fabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("ERROR %i %i %i/%i %i %i: (%f) (%f)\n",
                       mx + my_bounds_rs[0][0], my + my_bounds_rs[1][0],
                       mz + my_bounds_rs[2][0], nx, ny, nz, my_value,
                       ref_value);
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
          "The bw R2C 3D FFT (non-cartesian) from ordered layout does not work "
          "correctly "
          "(%i %i %i): %f!\n",
          npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  fft_free_double(rs_data);
  fft_free_complex(gs_data);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The 3D R2C FFT with non-cartesian layout does work correctly "
           "(sizes %i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_r2c_cartesian_halfspace(const int npts_global[3],
                                        const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv,
                              true, -1.0, NULL, NULL);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1];
  const int my_number_of_elements_gs =
      imax(product3(my_sizes_gs), fft_grid_layout->npts_gs_local);

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double *rs_data = NULL;
  fft_allocate_double(my_number_of_elements_rs, &rs_data);
  double complex *gs_data = NULL;
  fft_allocate_complex(my_number_of_elements_gs, &gs_data);

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = -1;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        number_of_tests++;
        if (test_every > 0 && number_of_tests % test_every != 0) {
          continue;
        }
        memset(rs_data, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          rs_data[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                   my_bounds_rs[1][0]) *
                      my_sizes_rs[2] +
                  (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_r2c_with_layout(rs_data, gs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(gs_data, fft_grid_layout, nx, ny, nz, npts_global, scale,           \
               my_process) reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int *index_g = fft_grid_layout->index_to_g[index];
          const double complex my_value = gs_data[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_g[0]) * nx / npts_global[0] +
                            ((double)index_g[1]) * ny / npts_global[1] +
                            ((double)index_g[2]) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR (%i) %i %i %i/%i %i %i (%i): (%f %f) (%f %f)\n",
                   my_process, index_g[0], index_g[1], index_g[2], nx, ny, nz,
                   index, creal(my_value), cimag(my_value), creal(ref_value),
                   cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw R2C 3D FFT (non-cartesian, halfspace) to ordered layout "
             "does not work "
             "correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(gs_data, 0, my_number_of_elements_gs * sizeof(double complex));

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx = fft_grid_layout->index_to_g[index][0];
          const int my = fft_grid_layout->index_to_g[index][1];
          const int mz = fft_grid_layout->index_to_g[index][2];
          gs_data[index] = cexp(-2.0 * I * pi *
                                (((double)mx) * nx / npts_global[0] +
                                 ((double)my) * ny / npts_global[1] +
                                 ((double)mz) * nz / npts_global[2]));
        }

        fft_3d_bw_c2r_with_layout(gs_data, rs_data, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(rs_data, my_bounds_rs, my_sizes_rs, nx, ny, nz, npts_global)        \
    collapse(3) reduction(max : max_error)
        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  rs_data[(mx * my_sizes_rs[1] + my) * my_sizes_rs[2] + mz];
              const double ref_value = ((mx + my_bounds_rs[0][0] == nx &&
                                         my + my_bounds_rs[1][0] == ny &&
                                         mz + my_bounds_rs[2][0] == nz)
                                            ? (double)product3(npts_global)
                                            : 0.0);
              double current_error = fabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("ERROR %i %i %i/%i %i %i: (%f) (%f)\n", mx, my, mz, nx,
                       ny, nz, my_value, ref_value);
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
      printf("The bw R2C 3D FFT (non-cartesian, halfspace) from ordered layout "
             "does "
             "not work correctly (%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  fft_free_double(rs_data);
  fft_free_complex(gs_data);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The 3D R2C FFT with non-cartesian, halfspace layout does work "
           "correctly "
           "(sizes %i %i %i)!\n",
           npts_global[0], npts_global[1], npts_global[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_ray(const int npts_global[3], const int npts_global_ref[3],
                    const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv,
                              false, -1.0, NULL, NULL);

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_layout, npts_global, -1.0, NULL,
                                             ref_grid_layout);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs =
      fft_grid_layout->rays_per_process[my_process] * npts_global[2];

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid_layout->rays_per_process[process];

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double complex *buffer_1 =
      calloc(my_number_of_elements_rs, sizeof(double complex));
  double complex *buffer_2 =
      calloc(my_number_of_elements_gs, sizeof(double complex));

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(buffer_1, 0, my_number_of_elements_rs * sizeof(double complex));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          buffer_1[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                    my_bounds_rs[1][0]) *
                       my_sizes_rs[2] +
                   (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_with_layout(buffer_1, buffer_2, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(fft_grid_layout, npts_global, nx, ny, nz, buffer_2, scale)          \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int index_x = fft_grid_layout->index_to_g[index][0];
          const int index_y = fft_grid_layout->index_to_g[index][1];
          const int index_z = fft_grid_layout->index_to_g[index][2];
          const double complex my_value = buffer_2[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_x) * nx / npts_global[0] +
                            ((double)index_y) * ny / npts_global[1] +
                            ((double)index_z) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_x,
                   index_y, index_z, nx, ny, nz, creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw 3D FFT with ray layout does not work correctly (%i %i "
             "%i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }
  fflush(stdout);

  // Check backwards 3D FFTs
  int total_number_of_rays = 0;
  for (int process = 0; process < cp_mpi_comm_size(comm); process++)
    total_number_of_rays += fft_grid_layout->rays_per_process[process];
  max_error = 0.0;
  number_of_tests = 0;
  for (int nz = 0; nz < npts_global[2]; nz++) {
    for (int nxy = 0; nxy < total_number_of_rays; nxy++) {
      if (test_every > 0 && number_of_tests % test_every != 0) {
        number_of_tests++;
        continue;
      }
      number_of_tests++;
      const int nx = fft_grid_layout->ray_to_xy[nxy][0];
      const int ny = fft_grid_layout->ray_to_xy[nxy][1];

      memset(buffer_2, 0,
             fft_grid_layout->npts_gs_local * sizeof(double complex));

      if (nxy >= my_ray_offset &&
          nxy < my_ray_offset + fft_grid_layout->rays_per_process[my_process]) {
        for (int index = 0; index < my_number_of_elements_gs; index++) {
          if (fft_grid_layout->index_to_g[index][0] == nx &&
              fft_grid_layout->index_to_g[index][1] == ny &&
              fft_grid_layout->index_to_g[index][2] == nz) {
            buffer_2[index] = 1.0;
            break;
          }
        }
      }

      fft_3d_bw_with_layout(buffer_2, buffer_1, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_rs, my_bounds_rs, npts_global, nx, ny, nz)       \
    reduction(max : max_error) collapse(3)
      for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
        for (int my = 0; my < my_sizes_rs[1]; my++) {
          for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
            const double complex my_value =
                buffer_1[(mx * my_sizes_rs[1] + my) * my_sizes_rs[2] + mz];
            const double complex ref_value =
                cexp(2.0 * I * pi *
                     (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                      ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                      ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
            double current_error = cabs(my_value - ref_value);
            if (current_error > 1e-12)
              printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", mx, my, mz,
                     nx, ny, nz, creal(my_value), cimag(my_value),
                     creal(ref_value), cimag(ref_value));
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 3D FFT with ray layout does not work correctly (%i "
             "%i %i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_fft_grid_layout(ref_grid_layout);
  free(buffer_1);
  free(buffer_2);

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT with ray layout does work correctly (%i %i %i)/(%i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
           npts_global_ref[1], npts_global_ref[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_r2c_ray(const int npts_global[3], const int npts_global_ref[3],
                        const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv,
                              false, -1.0, NULL, NULL);

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_layout, npts_global, -1.0, NULL,
                                             ref_grid_layout);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs =
      fft_grid_layout->rays_per_process[my_process] * npts_global[2];

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid_layout->rays_per_process[process];

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double *buffer_1 = calloc(my_number_of_elements_rs, sizeof(double));
  double complex *buffer_2 =
      calloc(my_number_of_elements_gs, sizeof(double complex));

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(buffer_1, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          buffer_1[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                    my_bounds_rs[1][0]) *
                       my_sizes_rs[2] +
                   (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_r2c_with_layout(buffer_1, buffer_2, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(fft_grid_layout, npts_global, nx, ny, nz, buffer_2, scale)          \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int index_x = fft_grid_layout->index_to_g[index][0];
          const int index_y = fft_grid_layout->index_to_g[index][1];
          const int index_z = fft_grid_layout->index_to_g[index][2];
          const double complex my_value = buffer_2[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_x) * nx / npts_global[0] +
                            ((double)index_y) * ny / npts_global[1] +
                            ((double)index_z) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_x,
                   index_y, index_z, nx, ny, nz, creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw R2C-3D FFT with ray layout does not work correctly (%i %i "
             "%i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }
  fflush(stdout);

  // Check backwards 3D FFTs
  int total_number_of_rays = 0;
  for (int process = 0; process < cp_mpi_comm_size(comm); process++)
    total_number_of_rays += fft_grid_layout->rays_per_process[process];
  max_error = 0.0;
  number_of_tests = 0;
  for (int nz = 0; nz < npts_global[2]; nz++) {
    for (int nxy = 0; nxy < total_number_of_rays; nxy++) {
      if (test_every > 0 && number_of_tests % test_every != 0) {
        number_of_tests++;
        continue;
      }
      number_of_tests++;
      const int nx = fft_grid_layout->ray_to_xy[nxy][0];
      const int ny = fft_grid_layout->ray_to_xy[nxy][1];

      memset(buffer_2, 0,
             fft_grid_layout->npts_gs_local * sizeof(double complex));

      if (nxy >= my_ray_offset &&
          nxy < my_ray_offset + fft_grid_layout->rays_per_process[my_process]) {
        for (int index = 0; index < my_number_of_elements_gs; index++) {
          if (fft_grid_layout->index_to_g[index][0] == nx &&
              fft_grid_layout->index_to_g[index][1] == ny &&
              fft_grid_layout->index_to_g[index][2] == nz) {
            buffer_2[index] = 1.0;
            break;
          }
        }
      }

      fft_3d_bw_c2r_with_layout(buffer_2, buffer_1, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_rs, my_bounds_rs, npts_global, nx, ny, nz)       \
    reduction(max : max_error) collapse(3)
      for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
        for (int my = 0; my < my_sizes_rs[1]; my++) {
          for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
            const double my_value =
                buffer_1[(mx * my_sizes_rs[1] + my) * my_sizes_rs[2] + mz];
            const double ref_value =
                cos(2.0 * pi *
                    (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                     ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                     ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
            double current_error = fabs(my_value - ref_value);
            if (current_error > 1e-12)
              printf("ERROR %i %i %i/%i %i %i: (%f) (%f)\n", mx, my, mz, nx, ny,
                     nz, my_value, ref_value);
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw C2R-3D FFT with ray layout does not work correctly (%i "
             "%i %i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_fft_grid_layout(ref_grid_layout);
  free(buffer_1);
  free(buffer_2);

  if (errors == 0 && my_process == 0)
    printf(
        "The R2C-3D FFT with ray layout does work correctly (%i %i %i)/(%i %i "
        "%i)!\n",
        npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
        npts_global_ref[1], npts_global_ref[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_r2c_ray_halfspace(const int npts_global[3],
                                  const int npts_global_ref[3],
                                  const int test_every) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv,
                              true, -1.0, NULL, NULL);

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_layout, npts_global, -1.0, NULL,
                                             ref_grid_layout);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs = fft_grid_layout->npts_gs_local;

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid_layout->rays_per_process[process];

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  double *buffer_1 = calloc(my_number_of_elements_rs, sizeof(double));
  double complex *buffer_2 =
      calloc(my_number_of_elements_gs, sizeof(double complex));

  // Check forward 3D FFTs
  double max_error = 0.0;
  int number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(buffer_1, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] &&
            nx <= my_bounds_rs[0][0] + my_bounds_rs[0][1] - 1 &&
            ny >= my_bounds_rs[1][0] &&
            ny <= my_bounds_rs[1][0] + my_bounds_rs[1][1] - 1 &&
            nz >= my_bounds_rs[2][0] &&
            nz <= my_bounds_rs[2][0] + my_bounds_rs[2][1] - 1)
          buffer_1[((nx - my_bounds_rs[0][0]) * my_sizes_rs[1] + ny -
                    my_bounds_rs[1][0]) *
                       my_sizes_rs[2] +
                   (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_r2c_with_layout(buffer_1, buffer_2, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(fft_grid_layout, npts_global, nx, ny, nz, buffer_2, scale)          \
    reduction(max : max_error)
        for (int index = 0; index < fft_grid_layout->npts_gs_local; index++) {
          const int index_x = fft_grid_layout->index_to_g[index][0];
          const int index_y = fft_grid_layout->index_to_g[index][1];
          const int index_z = fft_grid_layout->index_to_g[index][2];
          const double complex my_value = buffer_2[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)index_x) * nx / npts_global[0] +
                            ((double)index_y) * ny / npts_global[1] +
                            ((double)index_z) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", index_x,
                   index_y, index_z, nx, ny, nz, creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The fw R2C-3D FFT with ray layout does not work correctly (%i %i "
             "%i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  } else {
    if (my_process == 0)
      printf("The fw R2C-3D FFT with ray layout works correctly (%i %i "
             "%i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
  }
  fflush(stdout);

  // Check backwards 3D FFTs
  int total_number_of_rays = 0;
  for (int process = 0; process < cp_mpi_comm_size(comm); process++)
    total_number_of_rays += fft_grid_layout->rays_per_process[process];
  max_error = 0.0;
  number_of_tests = 0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;

        memset(buffer_2, 0,
               fft_grid_layout->npts_gs_local * sizeof(double complex));

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx = fft_grid_layout->index_to_g[index][0];
          const int my = fft_grid_layout->index_to_g[index][1];
          const int mz = fft_grid_layout->index_to_g[index][2];
          buffer_2[index] = cexp(-2.0 * I * pi *
                                 (((double)mx) * nx / npts_global[0] +
                                  ((double)my) * ny / npts_global[1] +
                                  ((double)mz) * nz / npts_global[2]));
        }

        fft_3d_bw_c2r_with_layout(buffer_2, buffer_1, fft_grid_layout);

#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_rs, my_bounds_rs, npts_global, nx, ny, nz)       \
    reduction(max : max_error) collapse(3)
        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  buffer_1[(mx * my_sizes_rs[1] + my) * my_sizes_rs[2] + mz];
              const double ref_value = (mx + my_bounds_rs[0][0] == nx &&
                                        my + my_bounds_rs[1][0] == ny &&
                                        mz + my_bounds_rs[2][0] == nz)
                                           ? (double)product3(npts_global)
                                           : 0.0;
              double current_error = fabs(my_value - ref_value);
              if (current_error > 1e-12)
                printf("ERROR %i %i %i/%i %i %i: (%f) (%f)\n", mx, my, mz, nx,
                       ny, nz, my_value, ref_value);
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
      printf("The bw C2R-3D FFT with ray layout (halfspace) does not work "
             "correctly (%i "
             "%i %i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_fft_grid_layout(ref_grid_layout);
  free(buffer_1);
  free(buffer_2);

  if (errors == 0 && my_process == 0)
    printf(
        "The R2C-3D FFT with ray layout does work correctly (%i %i %i)/(%i %i "
        "%i)!\n",
        npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
        npts_global_ref[1], npts_global_ref[2]);
  return errors;
}

int fft_test_3d() {
  const int my_process = cp_mpi_comm_rank(cp_mpi_get_comm_world());

  int errors = 0;

  // Grid sizes to be checked
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  clock_t begin = clock();
  // Check the cartesian layout
  errors += fft_test_3d_cartesian(npts_global, 5);
  errors += fft_test_3d_cartesian(npts_global_small, 13);
  errors += fft_test_3d_cartesian(npts_global_reverse, 11);
  errors += fft_test_3d_cartesian(npts_global_small_reverse, 7);

  // Check the cartesian layout
  errors += fft_test_3d_r2c_cartesian(npts_global, 13);
  errors += fft_test_3d_r2c_cartesian(npts_global_small, 17);
  errors += fft_test_3d_r2c_cartesian(npts_global_reverse, 19);
  errors += fft_test_3d_r2c_cartesian(npts_global_small_reverse, 11);

  // Check the cartesian layout with the halfspace flavor
  errors += fft_test_3d_r2c_cartesian_halfspace(npts_global, 13);
  errors += fft_test_3d_r2c_cartesian_halfspace(npts_global_small, 17);
  errors += fft_test_3d_r2c_cartesian_halfspace(npts_global_reverse, 19);
  errors += fft_test_3d_r2c_cartesian_halfspace(npts_global_small_reverse, 11);

  // Check the ray layout with the same grid sizes
  errors += fft_test_3d_ray(npts_global, npts_global, 19);
  errors += fft_test_3d_ray(npts_global_small, npts_global_small, 11);
  errors += fft_test_3d_ray(npts_global_small, npts_global, 13);
  errors += fft_test_3d_ray(npts_global_reverse, npts_global_reverse, 17);

  // Check the ray layout with the same grid sizes
  errors += fft_test_3d_r2c_ray(npts_global, npts_global, 19);
  errors += fft_test_3d_r2c_ray(npts_global_small, npts_global_small, 11);
  errors += fft_test_3d_r2c_ray(npts_global_small, npts_global, 13);
  errors += fft_test_3d_r2c_ray(npts_global_reverse, npts_global_reverse, 17);
  errors +=
      fft_test_3d_r2c_ray(npts_global_small_reverse, npts_global_reverse, 17);

  // Check the ray layout with the same grid sizes in half space
  errors += fft_test_3d_r2c_ray_halfspace(npts_global, npts_global, 19);
  errors +=
      fft_test_3d_r2c_ray_halfspace(npts_global_small, npts_global_small, 11);
  errors += fft_test_3d_r2c_ray_halfspace(npts_global_small, npts_global, 13);
  errors += fft_test_3d_r2c_ray_halfspace(npts_global_reverse,
                                          npts_global_reverse, 17);
  errors += fft_test_3d_r2c_ray_halfspace(npts_global_small_reverse,
                                          npts_global_reverse, 17);
  clock_t end = clock();

  if (errors == 0 && my_process == 0)
    fprintf(stdout, "\n The 3D FFT routines work correctly!\n");
  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0)
    printf("Time to test high-level FFTs: %f\n",
           (double)(end - begin) / CLOCKS_PER_SEC);
  return errors;
}

// EOF
