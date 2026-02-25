/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_redistribution_test.h"

#include "../mpiwrap/cp_mpi.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_redistribution.h"
#include "fft_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Function to test the parallel transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose_blocked(const int npts_global[3],
                               const bool use_halfspace) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv,
                              use_halfspace);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1];
  // For redistribution, the first size is local entirely and taken from g-space
  // as the bounds used here are for the real space before any partial FFT
  const int my_number_of_elements_rs =
      fft_grid_layout->npts_global_gspace[0] * my_sizes_rs[1] * my_sizes_rs[2];

  const int(*my_bounds_ms)[2] = fft_grid_layout->proc2local_ms[my_process];
  int my_sizes_ms[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms[dir] = my_bounds_ms[dir][1];
  const int my_number_of_elements_ms = product3(my_sizes_ms);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1];
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  ensure_buffer_size(fft_grid_layout->buffer_size);
  double complex *buffer_1 = get_buffer_1();
  double complex *buffer_2 = get_buffer_2();

  // Collect the maximum error
  double max_error;

  // With a 1D distribution, we have another distribution in mixed space (y is
  // distributed after the second step instead of z)
#pragma omp parallel for default(none)                                         \
    shared(buffer_2, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        buffer_2[(nx * my_sizes_ms[1] + ny) * my_sizes_ms[2] + nz] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }
  memset(buffer_1, 0, my_number_of_elements_gs * sizeof(double complex));

  // Check the reverse direction
  collect_z_and_distribute_y_blocked_transpose_pack(
      buffer_2, buffer_1, fft_grid_layout->redistribution,
      fft_grid_layout->proc2local_y_gs);
  collect_z_and_distribute_y_blocked_comm(buffer_1, buffer_2,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[0]);

  // Check forward RS->MS FFTs
  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(buffer_2, my_sizes_gs, my_bounds_gs, npts_global) collapse(3)       \
    reduction(max : max_error)
  for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
    for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
      for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
        const double complex my_value =
            buffer_2[(nz * my_sizes_gs[0] + nx) * my_sizes_gs[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_gs[0][0],
                 ny + my_bounds_gs[1][0], nz + my_bounds_gs[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_barrier(comm);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xz_to_xy_blocked_transpose does not work correctly "
             "(%i %i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

  // Check the reverse direction
#pragma omp parallel for default(none)                                         \
    shared(buffer_2, my_sizes_gs, my_bounds_gs, npts_global, use_halfspace)    \
    collapse(3)
  for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
    for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
      for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
        buffer_2[(nz * my_sizes_gs[0] + nx) * my_sizes_gs[1] + ny] =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
      }
    }
  }
  memset(buffer_1, 0, my_number_of_elements_ms * sizeof(double complex));

  // Check the reverse direction
  collect_y_and_distribute_z_blocked_comm(buffer_2, buffer_1,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[0]);
  collect_y_and_distribute_z_blocked_transpose_unpack(
      buffer_1, buffer_2, fft_grid_layout->redistribution,
      fft_grid_layout->proc2local_y_gs);

  // Check forward RS->MS FFTs
  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(buffer_2, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)       \
    reduction(max : max_error)
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            buffer_2[(nx * my_sizes_ms[1] + ny) * my_sizes_ms[2] + nz];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_ms[0][0],
                 ny + my_bounds_ms[1][0], nz + my_bounds_ms[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xy_to_xz_blocked_transpose does not work correctly "
             "(%i %i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

// Check forward RS->MS FFTs
#pragma omp parallel for default(none)                                         \
    shared(fft_grid_layout, buffer_1, my_sizes_rs, my_bounds_rs, npts_global,  \
               use_halfspace) collapse(3)
  for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
    for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
      for (int nx = 0; nx < fft_grid_layout->npts_global_gspace[0]; nx++) {
        buffer_1[(ny * my_sizes_rs[2] + nz) *
                     fft_grid_layout->npts_global_gspace[0] +
                 nx] = ((nx + my_bounds_rs[0][0]) * npts_global[1] +
                        (ny + my_bounds_rs[1][0])) +
                       I * (nz + my_bounds_rs[2][0]);
      }
    }
  }

  collect_y_and_distribute_x_blocked_pack(buffer_1, buffer_2,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->proc2local_x_gs);

  collect_y_and_distribute_x_blocked_comm(buffer_2, buffer_1,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[1]);

  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)       \
    reduction(max : max_error)
  for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
    for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            buffer_1[(ny * my_sizes_ms[2] + nz) * my_sizes_ms[0] + nx];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_ms[0][0],
                 ny + my_bounds_ms[1][0], nz + my_bounds_ms[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose yz_to_xz_blocked does not work correctly (%i %i "
             "%i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

#pragma omp parallel for default(none)                                         \
    shared(buffer_2, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)
  for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
    for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
      for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
        buffer_2[(ny * my_sizes_ms[2] + nz) * my_sizes_ms[0] + nx] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }
  memset(buffer_1, 0, my_number_of_elements_rs * sizeof(double complex));

  // Check the reverse direction
  collect_x_and_distribute_y_blocked_comm(buffer_2, buffer_1,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[1]);
  collect_x_and_distribute_y_blocked_unpack(buffer_1, buffer_2,
                                            fft_grid_layout->redistribution,
                                            fft_grid_layout->proc2local_x_gs);

  // Check forward RS->MS FFTs
  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(fft_grid_layout, buffer_2, my_sizes_rs, my_bounds_rs, npts_global,  \
               use_halfspace) collapse(3) reduction(max : max_error)
  for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
    for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
      for (int nx = 0; nx < fft_grid_layout->npts_global_gspace[0]; nx++) {
        const double complex my_value =
            buffer_2[(ny * my_sizes_rs[2] + nz) *
                         fft_grid_layout->npts_global_gspace[0] +
                     nx];
        const double complex ref_value =
            ((nx + my_bounds_rs[0][0]) * npts_global[1] +
             (ny + my_bounds_rs[1][0])) +
            I * (nz + my_bounds_rs[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_rs[0][0],
                 ny + my_bounds_rs[1][0], nz + my_bounds_rs[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xz_to_yz_blocked does not work correctly (%i %i "
             "%i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)
  for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
    for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
      for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
        buffer_1[(nz * my_sizes_ms[0] + nx) * my_sizes_ms[1] + ny] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }
  memset(buffer_2, 0, my_number_of_elements_gs * sizeof(double complex));

  // Check the MS/GS direction
  collect_z_and_distribute_y_blocked_pack(buffer_1, buffer_2,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->proc2local_y_gs);
  collect_z_and_distribute_y_blocked_comm(buffer_2, buffer_1,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[0]);

  // Check forward RS->MS FFTs
  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_gs, my_bounds_gs, npts_global) collapse(3)       \
    reduction(max : max_error)
  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
        const double complex my_value =
            buffer_1[(nz * my_sizes_gs[0] + nx) * my_sizes_gs[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_gs[0][0],
                 ny + my_bounds_gs[1][0], nz + my_bounds_gs[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xz_to_xy_blocked does not work correctly (%i %i "
             "%i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_gs, my_bounds_gs, npts_global) collapse(3)
  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
      for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
        buffer_1[(nz * my_sizes_gs[0] + nx) * my_sizes_gs[1] + ny] =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
      }
    }
  }
  memset(buffer_2, 0, my_number_of_elements_ms);

  // Check the MS/GS direction
  collect_y_and_distribute_z_blocked_comm(buffer_1, buffer_2,
                                          fft_grid_layout->redistribution,
                                          fft_grid_layout->sub_comm[0]);
  collect_y_and_distribute_z_blocked_unpack(buffer_2, buffer_1,
                                            fft_grid_layout->redistribution,
                                            fft_grid_layout->proc2local_y_gs);

  // Check forward RS->MS FFTs
  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(buffer_1, my_sizes_ms, my_bounds_ms, npts_global) collapse(3)       \
    reduction(max : max_error)
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
      for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
        const double complex my_value =
            buffer_1[(nz * my_sizes_ms[0] + nx) * my_sizes_ms[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("ERROR %i %i %i: %i %i %i\n", nx + my_bounds_ms[0][0],
                 ny + my_bounds_ms[1][0], nz + my_bounds_ms[2][0],
                 ((int)creal(my_value)) / npts_global[1],
                 ((int)creal(my_value)) % npts_global[1],
                 ((int)cimag(my_value)));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xy_to_xz_blocked does not work correctly (%i %i "
             "%i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], use_halfspace,
             max_error);
    errors++;
  }

  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The transpose from the blocked distribution does work correctly "
           "(%i %i %i, HS: %i)!\n",
           npts_global[0], npts_global[1], npts_global[2], use_halfspace);

  return errors;
}

int fft_test_transpose_ray(const int npts_global[3],
                           const int npts_global_ref[3],
                           const bool use_halfspace) {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  double max_error = 0.0;
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // Build the reference grid
  fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv,
                              use_halfspace);

  // Test ray transpositiond,
  fft_grid_layout *fft_grid_ray_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_ray_layout, npts_global,
                                             ref_grid_layout);

  ensure_buffer_size(fft_grid_ray_layout->buffer_size);
  double complex *buffer_1 = get_buffer_1();
  double complex *buffer_2 = get_buffer_2();

  int my_bounds_ms_ray[3][2];
  memcpy(my_bounds_ms_ray, fft_grid_ray_layout->proc2local_ms[my_process],
         sizeof(int[3][2]));
  int my_sizes_ms_ray[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms_ray[dir] = my_bounds_ms_ray[dir][1];

#pragma omp parallel for default(none)                                         \
    shared(fft_grid_ray_layout, buffer_1, my_sizes_ms_ray, my_bounds_ms_ray)   \
    collapse(3)
  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
        buffer_1[(index_x * my_sizes_ms_ray[1] + index_y) * my_sizes_ms_ray[2] +
                 index_z] = ((index_y + my_bounds_ms_ray[1][0]) *
                                 fft_grid_ray_layout->npts_global[2] +
                             (index_z + my_bounds_ms_ray[2][0])) +
                            I * (index_x + my_bounds_ms_ray[0][0]);
      }
    }
  }

  collect_z_and_distribute_xy_ray_transpose(
      buffer_1, buffer_2, fft_grid_ray_layout->npts_global_gspace,
      fft_grid_ray_layout->proc2local_ms, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_xy, fft_grid_ray_layout->comm);

  max_error = 0.0;
  int ray_index_offset = 0;
  for (int process = 0; process < my_process; process++)
    ray_index_offset += fft_grid_ray_layout->rays_per_process[process];
#pragma omp parallel for default(none) collapse(2)                             \
    shared(fft_grid_ray_layout, buffer_2, my_sizes_ms_ray, npts_global,        \
               ray_index_offset, my_process) reduction(max : max_error)
  for (int index_z = 0; index_z < fft_grid_ray_layout->npts_global[2];
       index_z++) {
    for (int xy_ray = 0;
         xy_ray < fft_grid_ray_layout->rays_per_process[my_process]; xy_ray++) {
      const int index_x =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][0];
      const int index_y =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][1];
      const double complex my_value =
          buffer_2[xy_ray * npts_global[2] + index_z];
      const double complex ref_value =
          (index_y * npts_global[2] + index_z) + I * index_x;
      double current_error = cabs(my_value - ref_value);
      if (current_error > 1e-12)
        printf("ERROR %i %i %i: (%f %f) (%f %f)\n", index_x, index_y, index_z,
               creal(my_value), cimag(my_value), creal(ref_value),
               cimag(ref_value));
      max_error = fmax(max_error, current_error);
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The transpose yz_to_x_ray_transpose does not work correctly "
             "(%i %i %i/%i "
             "%i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], use_halfspace, max_error);
    }
    errors++;
  }

  memset(buffer_1, 0,
         fft_grid_ray_layout->npts_global_gspace[2] *
             fft_grid_ray_layout->rays_per_process[my_process] *
             sizeof(double complex));
  memset(buffer_2, 0, product3(my_sizes_ms_ray) * sizeof(double complex));

  ray_index_offset = 0;
  for (int process = 0; process < my_process; process++)
    ray_index_offset += fft_grid_ray_layout->rays_per_process[process];
#pragma omp parallel for default(none) collapse(2)                             \
    shared(fft_grid_ray_layout, buffer_1, my_sizes_ms_ray, my_process,         \
               ray_index_offset)
  for (int index_z = 0; index_z < fft_grid_ray_layout->npts_global[2];
       index_z++) {
    for (int xy_ray = 0;
         xy_ray < fft_grid_ray_layout->rays_per_process[my_process]; xy_ray++) {
      const int index_x =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][0];
      const int index_y =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][1];
      buffer_1[xy_ray * fft_grid_ray_layout->npts_global[2] + index_z] =
          (index_y * fft_grid_ray_layout->npts_global[2] + index_z) +
          I * index_x;
    }
  }
  collect_xy_and_distribute_z_ray_transpose(
      buffer_1, buffer_2, fft_grid_ray_layout->npts_global_gspace,
      fft_grid_ray_layout->proc2local_ms, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_xy, fft_grid_ray_layout->comm);

  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(fft_grid_ray_layout, buffer_2, my_sizes_ms_ray, my_bounds_ms_ray)   \
    collapse(2) reduction(max : max_error)
  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      // Check whether there is a ray with the given index pair
      if (fft_grid_ray_layout
              ->xy_to_process[(index_x + my_bounds_ms_ray[0][0]) *
                                  fft_grid_ray_layout->npts_global_gspace[1] +
                              (index_y + my_bounds_ms_ray[1][0])] >= 0) {
        for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
          const double complex my_value =
              buffer_2[(index_x * my_sizes_ms_ray[1] + index_y) *
                           my_sizes_ms_ray[2] +
                       index_z];
          const double complex ref_value =
              ((index_y + my_bounds_ms_ray[1][0]) *
                   fft_grid_ray_layout->npts_global[2] +
               (index_z + my_bounds_ms_ray[2][0])) +
              I * (index_x + my_bounds_ms_ray[0][0]);
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i: (%f %f) (%f %f)\n",
                   index_x + my_bounds_ms_ray[0][0],
                   index_y + my_bounds_ms_ray[1][0],
                   index_z + my_bounds_ms_ray[2][0], creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      } else {
        for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
          const double complex my_value =
              buffer_2[(index_x * my_sizes_ms_ray[1] * index_y) *
                           my_sizes_ms_ray[2] +
                       index_z];
          // The value is assumed to be zero if the ray absent
          const double complex ref_value = 0.0;
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i: (%f %f) (%f %f)\n",
                   index_x + my_bounds_ms_ray[0][0],
                   index_y + my_bounds_ms_ray[1][0],
                   index_z + my_bounds_ms_ray[2][0], creal(my_value),
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
      printf("The transpose z_to_xy_ray_transpose does not work correctly "
             "(%i %i %i/%i "
             "%i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], use_halfspace, max_error);
    errors++;
  }

#pragma omp parallel for default(none)                                         \
    shared(npts_global, buffer_1, my_sizes_ms_ray, my_bounds_ms_ray)           \
    collapse(3)
  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
        buffer_1[(index_z * my_sizes_ms_ray[0] + index_x) * my_sizes_ms_ray[1] +
                 index_y] =
            ((index_y + my_bounds_ms_ray[1][0]) * npts_global[2] +
             (index_z + my_bounds_ms_ray[2][0])) +
            I * (index_x + my_bounds_ms_ray[0][0]);
      }
    }
  }

  collect_z_and_distribute_xy_ray(
      buffer_1, buffer_2, fft_grid_ray_layout->npts_global_gspace,
      fft_grid_ray_layout->proc2local_ms, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_xy, fft_grid_ray_layout->redistribution, fft_grid_ray_layout->comm);

  max_error = 0.0;
  ray_index_offset = 0;
  for (int process = 0; process < my_process; process++)
    ray_index_offset += fft_grid_ray_layout->rays_per_process[process];
#pragma omp parallel for default(none) collapse(2)                             \
    shared(fft_grid_ray_layout, buffer_2, my_sizes_ms_ray, npts_global,        \
               ray_index_offset, my_process) reduction(max : max_error)
  for (int index_z = 0; index_z < fft_grid_ray_layout->npts_global_gspace[2];
       index_z++) {
    for (int xy_ray = 0;
         xy_ray < fft_grid_ray_layout->rays_per_process[my_process]; xy_ray++) {
      const int index_x =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][0];
      const int index_y =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][1];
      const double complex my_value =
          buffer_2[xy_ray * fft_grid_ray_layout->npts_global[2] + index_z];
      const double complex ref_value =
          (index_y * npts_global[2] + index_z) + I * index_x;
      double current_error = cabs(my_value - ref_value);
      if (current_error > 1e-12)
        printf("ERROR %i %i %i: (%f %f) (%f %f)\n", index_x, index_y, index_z,
               creal(my_value), cimag(my_value), creal(ref_value),
               cimag(ref_value));
      max_error = fmax(max_error, current_error);
    }
  }
  fflush(stdout);
  cp_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The transpose yz_to_x_ray does not work correctly (%i %i %i/%i "
             "%i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], use_halfspace, max_error);
    }
    errors++;
  }

  memset(buffer_1, 0, product3(my_sizes_ms_ray) * sizeof(double complex));
  memset(buffer_2, 0,
         npts_global[2] * fft_grid_ray_layout->rays_per_process[my_process] *
             sizeof(double complex));

#pragma omp parallel for default(none) collapse(2)                             \
    shared(fft_grid_ray_layout, buffer_1, my_sizes_ms_ray, my_process,         \
               ray_index_offset)
  for (int index_z = 0; index_z < fft_grid_ray_layout->npts_global[2];
       index_z++) {
    for (int xy_ray = 0;
         xy_ray < fft_grid_ray_layout->rays_per_process[my_process]; xy_ray++) {
      const int index_x =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][0];
      const int index_y =
          fft_grid_ray_layout->ray_to_xy[ray_index_offset + xy_ray][1];
      buffer_1[xy_ray * fft_grid_ray_layout->npts_global[2] + index_z] =
          (index_y * fft_grid_ray_layout->npts_global[2] + index_z) +
          I * index_x;
    }
  }
  collect_xy_and_distribute_z_ray(
      buffer_1, buffer_2, fft_grid_ray_layout->npts_global_gspace,
      fft_grid_ray_layout->proc2local_ms, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_xy, fft_grid_ray_layout->comm);

  max_error = 0.0;
#pragma omp parallel for default(none)                                         \
    shared(fft_grid_ray_layout, npts_global, buffer_2, my_sizes_ms_ray,        \
               my_bounds_ms_ray, my_process) collapse(2)                       \
    reduction(max : max_error)
  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      // Check whether there is a ray with the given index pair
      if (fft_grid_ray_layout
              ->xy_to_process[(index_x + my_bounds_ms_ray[0][0]) *
                                  npts_global[1] +
                              (index_y + my_bounds_ms_ray[1][0])] >= 0) {
        for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
          const double complex my_value =
              buffer_2[(index_z * my_sizes_ms_ray[0] + index_x) *
                           my_sizes_ms_ray[1] +
                       index_y];
          const double complex ref_value =
              ((index_y + my_bounds_ms_ray[1][0]) * npts_global[2] +
               (index_z + my_bounds_ms_ray[2][0])) +
              I * (index_x + my_bounds_ms_ray[0][0]);
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i: (%f %f) (%f %f)\n",
                   index_x + my_bounds_ms_ray[0][0],
                   index_y + my_bounds_ms_ray[1][0],
                   index_z + my_bounds_ms_ray[2][0], creal(my_value),
                   cimag(my_value), creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      } else {
        for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
          const double complex my_value =
              buffer_2[(index_z * my_sizes_ms_ray[0] + index_x) *
                           my_sizes_ms_ray[1] +
                       index_y];
          // The value is assumed to be zero if the ray absent
          const double complex ref_value = 0.0;
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("ERROR %i %i %i: (%f %f) (%f %f)\n",
                   index_x + my_bounds_ms_ray[0][0],
                   index_y + my_bounds_ms_ray[1][0],
                   index_z + my_bounds_ms_ray[2][0], creal(my_value),
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
      printf("The transpose z_to_xy_ray does not work correctly (%i %i %i/%i "
             "%i %i, HS: %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], use_halfspace, max_error);
    errors++;
  }

  grid_free_fft_grid_layout(fft_grid_ray_layout);
  grid_free_fft_grid_layout(ref_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The transpose from the ray distribution works correctly (%i %i "
           "%i/%i %i %i, HS: %i)!\n",
           npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
           npts_global_ref[1], npts_global_ref[2], use_halfspace);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose_parallel() {
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int my_process = cp_mpi_comm_rank(comm);

  int errors = 0;

  // Grid sizes to be checked
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  // Check the blocked layout
  errors += fft_test_transpose_blocked(npts_global, false);
  errors += fft_test_transpose_blocked(npts_global_small, false);
  errors += fft_test_transpose_blocked(npts_global_reverse, false);
  errors += fft_test_transpose_blocked(npts_global_small_reverse, false);

  // Check the ray layout with the same grid sizes
  errors += fft_test_transpose_ray(npts_global, npts_global, false);
  errors += fft_test_transpose_ray(npts_global_small, npts_global_small, false);
  errors +=
      fft_test_transpose_ray(npts_global_reverse, npts_global_reverse, false);
  errors += fft_test_transpose_ray(npts_global_small_reverse,
                                   npts_global_small_reverse, false);
  // Check the ray layout with different grid sizes
  errors += fft_test_transpose_ray(npts_global_small, npts_global, false);
  errors += fft_test_transpose_ray(npts_global_small_reverse,
                                   npts_global_reverse, false);

  // Check the blocked layout
  errors += fft_test_transpose_blocked(npts_global, true);
  errors += fft_test_transpose_blocked(npts_global_small, true);
  errors += fft_test_transpose_blocked(npts_global_reverse, true);
  errors += fft_test_transpose_blocked(npts_global_small_reverse, true);

  // Check the ray layout with the same grid sizes
  errors += fft_test_transpose_ray(npts_global, npts_global, true);
  errors += fft_test_transpose_ray(npts_global_small, npts_global_small, true);
  errors +=
      fft_test_transpose_ray(npts_global_reverse, npts_global_reverse, true);
  errors += fft_test_transpose_ray(npts_global_small_reverse,
                                   npts_global_small_reverse, true);
  // Check the ray layout with different grid sizes
  errors += fft_test_transpose_ray(npts_global_small, npts_global, true);
  errors += fft_test_transpose_ray(npts_global_small_reverse,
                                   npts_global_reverse, true);

  if (errors == 0 && my_process == 0)
    printf("\n The parallel transposition routines work correctly!\n");
  return errors;
}

// EOF
