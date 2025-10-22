/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid_layout.h"
#include "fft_grid.h"
#include "fft_lib.h"
#include "fft_reorder.h"
#include "fft_utils.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int current_grid_id = 1;

// Could be reformulated with Lapack or calculated
// For orthorhombic cells, this is at the order of
// 3*eps(multiplication)+6*eps(addition) For non-orthorhombic cells, this
// depends on the cell shape
const double max_rel_error_for_equivalence_g_squared = 1e-12;

typedef struct {
  double value;
  int index;
} double_index_pair;

double squared_length_of_g_vector(const int g[3], const double h_inv[3][3]) {
  if (g[0] == 0 && g[1] == 0 && g[2] == 0) {
    return 0.0;
  }
  const double two_pi = 2.0 * acos(-1.0);
  double length_g_squared = 0.0;
  for (int dir = 0; dir < 3; dir++) {
    double length_g_dir = 0.0;
    for (int dir2 = 0; dir2 < 3; dir2++) {
      length_g_dir += g[dir] * h_inv[dir2][dir];
    }
    length_g_dir *= two_pi;
    length_g_squared += length_g_dir * length_g_dir;
  }
  return length_g_squared;
}

int compare_double(const void *a, const void *b) {
  const double a_value = ((const double_index_pair *)a)->value;
  const double b_value = ((const double_index_pair *)b)->value;
  return (a_value > b_value ? 1 : (a_value < b_value ? -1 : 0));
}

int compare_shell(const void *a, const void *b) {
  for (int index = 0; index < 3; index++) {
    const int a_value = ((const int *)a)[index];
    const int b_value = ((const int *)b)[index];
    const double max_allowed_error =
        max_rel_error_for_equivalence_g_squared * fmax(a_value, b_value);
    if (a_value > b_value + max_allowed_error) {
      return 1;
    } else if (a_value + max_allowed_error < b_value) {
      return -1;
    }
  }
  return 0;
}

void sort_shell(int (*shell)[3], const int shell_size) {
  qsort(shell, shell_size, sizeof(int[3]), compare_shell);
}

void sort_g_vectors(fft_grid_layout *my_fft_grid) {
  assert(my_fft_grid != NULL);
  assert(my_fft_grid->npts_gs_local >= 0);

  int *local_index2g_squared = calloc(my_fft_grid->npts_gs_local, sizeof(int));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, local_index2g_squared)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    local_index2g_squared[index] = squared_length_of_g_vector(
        my_fft_grid->index_to_g[index], my_fft_grid->h_inv);
  }

  // Sort the indices according to the length of the vectors
  double_index_pair *g_square_index_pair =
      calloc(my_fft_grid->npts_gs_local, sizeof(double_index_pair));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, g_square_index_pair, local_index2g_squared)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    g_square_index_pair[index].value = local_index2g_squared[index];
    g_square_index_pair[index].index = index;
  }
  qsort(g_square_index_pair, my_fft_grid->npts_gs_local,
        sizeof(double_index_pair), compare_double);

  // Apply the sorting to the index_to_g array
  {
    int(*index_to_g_sorted)[3] =
        calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, index_to_g_sorted, g_square_index_pair,                \
               local_index2g_squared)
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      memcpy(index_to_g_sorted[index],
             my_fft_grid->index_to_g[g_square_index_pair[index].index],
             3 * sizeof(int));
      local_index2g_squared[index] = g_square_index_pair[index].value;
    }
    memcpy(my_fft_grid->index_to_g, &index_to_g_sorted[0][0],
           my_fft_grid->npts_gs_local * sizeof(int[3]));
    free(index_to_g_sorted);
  }

  // Sort the vectors with the same length according to the x-, then y-, then
  // z-coordinate
  {
    double last_g_squared = g_square_index_pair[0].value;
    int start_index = 0;
    for (int end_index = 1; end_index < my_fft_grid->npts_gs_local;
         end_index++) {
      if (fabs(g_square_index_pair[end_index].value - last_g_squared) >
          fmax(g_square_index_pair[end_index].value, last_g_squared) *
              max_rel_error_for_equivalence_g_squared) {
        // If the length of the current vector is different from the previous
        // one, sort the vectors with the same length according to the x-, then
        // y-, then z-coordinate
        sort_shell(my_fft_grid->index_to_g + start_index,
                   end_index - start_index);
        start_index = end_index;
        last_g_squared = g_square_index_pair[end_index].value;
      }
    }
    // At the end, we need to sort the last shell
    sort_shell(my_fft_grid->index_to_g + start_index,
               my_fft_grid->npts_gs_local - start_index);
  }
  free(g_square_index_pair);
  free(local_index2g_squared);
}

void grid_free_fft_grid_layout(fft_grid_layout *fft_grid) {
  if (fft_grid != NULL) {
    if (cp_mpi_comm_rank(fft_grid->comm) == 0)
      assert((fft_grid->ref_counter) > 0);
    fft_grid->ref_counter--;
    if (fft_grid->ref_counter == 0) {
      cp_mpi_comm_free(&fft_grid->comm);
      cp_mpi_comm_free(&fft_grid->sub_comm[0]);
      cp_mpi_comm_free(&fft_grid->sub_comm[1]);
      free(fft_grid->proc2local_rs);
      free(fft_grid->proc2local_ms);
      free(fft_grid->proc2local_gs);
      fft_free_complex(fft_grid->buffer_1);
      fft_free_complex(fft_grid->buffer_2);
      free(fft_grid->yz_to_process);
      free(fft_grid->ray_to_yz);
      free(fft_grid->rays_per_process);
      free(fft_grid->index_to_g);
      free(fft_grid->local_index_to_ref_grid);
      free(fft_grid);
    }
  }
}

void setup_proc2local(fft_grid_layout *my_fft_grid) {
  const int number_of_processes = cp_mpi_comm_size(my_fft_grid->comm);
  const int my_process = cp_mpi_comm_rank(my_fft_grid->comm);

  my_fft_grid->proc2local_rs = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_ms = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_gs = malloc(number_of_processes * sizeof(int[3][2]));
  if (fft_lib_use_mpi() && cp_mpi_comm_size(my_fft_grid->comm) > 1) {
    // The data distribution is taken optimized for use with FFTW
    // The first index is distributed, the others are not
    // We ask for output data with the first two indices swapped (transposed
    // mode) to save communication within the library Several distributed 2D
    // FFTs require the distance between elements of adjacent FFTs to be 1
    // Starting from the order (z, y, x), we have to distribute the first index
    // and, in case of a pencil distribution, the LAST index (this should be
    // related to improved vectorization within the library)
    if (my_fft_grid->proc_grid[1] > 1) {
      // Start with a distributed FFT using the first sub-communicator in y- and
      // z-direction
      int local_n1_rs, local_n1_start_rs, local_n2_gs, local_n2_start_gs,
          my_bounds[3][2];
      // We need to pre-distribute the x-direction
      const int block_size_x_rs =
          (my_fft_grid->npts_global[0] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // In the last step, y is distributed in the second direction
      const int block_size_y_gs =
          (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // Determine a maximum buffer size
      // With half-space, we need different routines
      // With ray-distribution, we have a different number of partial FFTs
      if (my_fft_grid->ray_distribution) {
        if (my_fft_grid->use_halfspace) {
          my_fft_grid->buffer_size = fft_2d_distributed_sizes_r2c(
              (const int[2]){my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              block_size_x_rs, my_fft_grid->sub_comm[1], &local_n1_rs,
              &local_n1_start_rs, &local_n2_gs, &local_n2_start_gs);
        } else {
          my_fft_grid->buffer_size = fft_2d_distributed_sizes(
              (const int[2]){my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              block_size_x_rs, my_fft_grid->sub_comm[1], &local_n1_rs,
              &local_n1_start_rs, &local_n2_gs, &local_n2_start_gs);
        }
      } else {
        if (my_fft_grid->use_halfspace) {
          my_fft_grid->buffer_size = fft_2d_distributed_sizes_r2c(
              (const int[2]){my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              block_size_x_rs, my_fft_grid->sub_comm[1], &local_n1_rs,
              &local_n1_start_rs, &local_n2_gs, &local_n2_start_gs);
        } else {
          my_fft_grid->buffer_size = fft_2d_distributed_sizes(
              (const int[2]){my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              block_size_x_rs, my_fft_grid->sub_comm[1], &local_n1_rs,
              &local_n1_start_rs, &local_n2_gs, &local_n2_start_gs);
        }
      }
      // Setup the bounds in real space
      // In x-direction, we need to define them ourselves
      my_bounds[0][0] = imin(block_size_x_rs * my_fft_grid->proc_coords[0],
                             my_fft_grid->npts_global[0]);
      my_bounds[0][1] =
          imin(block_size_x_rs * (my_fft_grid->proc_coords[0] + 1) - 1,
               my_fft_grid->npts_global[0] - 1);
      // The distributions in y- and z-direction are provided by the FFT library
      // With y (second index) required to be locally available
      my_bounds[1][0] = local_n1_start_rs;
      my_bounds[1][1] = local_n1_start_rs + local_n1_rs - 1;
      my_bounds[2][0] = 0;
      my_bounds[2][1] = my_fft_grid->npts_global[2] - 1;
      // Exchange the distribution with the other processes
      cp_mpi_allgather_int((const int *)my_bounds, 6,
                           (int *)my_fft_grid->proc2local_rs,
                           6 * number_of_processes, my_fft_grid->comm);
      // The result has the same data distribution but with y now being
      // distributed as first index and z locally (the original distribution is
      // possible but requires more communication by the library)
      my_bounds[1][0] = 0;
      my_bounds[1][1] = my_fft_grid->npts_global[1] - 1;
      my_bounds[2][0] = local_n2_start_gs;
      my_bounds[2][1] = local_n2_start_gs + local_n2_gs - 1;
      // Exchange the bounds
      cp_mpi_allgather_int((const int *)my_bounds, 6,
                           (int *)my_fft_grid->proc2local_ms,
                           6 * number_of_processes, my_fft_grid->comm);
// The last FFT step is performed locally in x-direction
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, block_size_y_gs, number_of_processes)
      for (int process = 0; process < number_of_processes; process++) {
        int proc_coords[2];
        cp_mpi_cart_coords(my_fft_grid->comm, process, 2, proc_coords);
        // x needs to be available entirely
        my_fft_grid->proc2local_gs[process][0][0] = 0;
        my_fft_grid->proc2local_gs[process][0][1] =
            my_fft_grid->npts_global[0] - 1;
        // y is redistributed now
        my_fft_grid->proc2local_gs[process][1][0] =
            imin(block_size_y_gs * proc_coords[0],
                 my_fft_grid->npts_global_gspace[1]);
        my_fft_grid->proc2local_gs[process][1][1] =
            imin(block_size_y_gs * (proc_coords[0] + 1) - 1,
                 my_fft_grid->npts_global_gspace[1] - 1);
        // z is taken from the preceding distribution
        my_fft_grid->proc2local_gs[process][2][0] =
            my_fft_grid->proc2local_ms[process][2][0];
        my_fft_grid->proc2local_gs[process][2][1] =
            my_fft_grid->proc2local_ms[process][2][1];
      }
      // We need to consider the last transformation step for the buffer size
      my_fft_grid->buffer_size =
          imax(my_fft_grid->buffer_size,
               (my_fft_grid->proc2local_gs[my_process][0][1] -
                my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                   (my_fft_grid->proc2local_gs[my_process][1][1] -
                    my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
                   (my_fft_grid->proc2local_gs[my_process][2][1] -
                    my_fft_grid->proc2local_gs[my_process][2][0] + 1));
    } else {
      // With distributed 3D FFTs, we ask the library to perform all FFT steps
      // This data distribution is obtained from the 2D case without data
      // distribution in the second process direction
      int local_n0_rs, local_n0_start_rs, local_n1_gs, local_n1_start_gs,
          my_bounds[3][2];
      // With ray distribution, we perform only a distributed 2D FFT to be able
      // to perform the final local FFT of the own rays only
      if (my_fft_grid->ray_distribution) {
        const int block_size_x_rs =
            (my_fft_grid->npts_global[0] + my_fft_grid->proc_grid[0] - 1) /
            my_fft_grid->proc_grid[0];
        const int block_size_y_gs = (my_fft_grid->npts_global_gspace[1] +
                                     my_fft_grid->proc_grid[0] - 1) /
                                    my_fft_grid->proc_grid[0];
        // No distributed FFT necessary, communication only before the final FFT
        local_n0_start_rs = imin(my_fft_grid->proc_coords[0] * block_size_x_rs,
                                 my_fft_grid->npts_global[0]);
        local_n0_rs =
            imax(0, imin(block_size_x_rs,
                         my_fft_grid->npts_global[0] - local_n0_start_rs));
        my_fft_grid->buffer_size = local_n0_rs * my_fft_grid->npts_global[1] *
                                   (my_fft_grid->npts_global[2] / 2 + 1) * 2;
        local_n1_start_gs = imin(my_fft_grid->proc_coords[0] * block_size_y_gs,
                                 my_fft_grid->npts_global_gspace[1]);
        local_n1_gs =
            imax(0, imin(block_size_y_gs,
                         my_fft_grid->npts_global[1] - local_n1_start_gs));
        my_fft_grid->buffer_size =
            imax(my_fft_grid->buffer_size,
                 local_n1_gs * my_fft_grid->npts_global[0] *
                     (my_fft_grid->npts_global[2] / 2 + 1) * 2);
      } else {
        if (my_fft_grid->use_halfspace) {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes_r2c(
              (const int[3]){my_fft_grid->npts_global[0],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              my_fft_grid->sub_comm[0], &local_n0_rs, &local_n0_start_rs,
              &local_n1_gs, &local_n1_start_gs);
        } else {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes(
              (const int[3]){my_fft_grid->npts_global[0],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[2]},
              my_fft_grid->sub_comm[0], &local_n0_rs, &local_n0_start_rs,
              &local_n1_gs, &local_n1_start_gs);
        }
      }
      my_bounds[0][0] = local_n0_start_rs;
      my_bounds[0][1] = local_n0_start_rs + local_n0_rs - 1;
      my_bounds[1][0] = 0;
      my_bounds[1][1] = my_fft_grid->npts_global[1] - 1;
      my_bounds[2][0] = 0;
      my_bounds[2][1] = my_fft_grid->npts_global[2] - 1;
      cp_mpi_allgather_int((const int *)my_bounds, 6,
                           (int *)my_fft_grid->proc2local_rs,
                           6 * number_of_processes, my_fft_grid->comm);
      my_bounds[0][0] = 0;
      my_bounds[0][1] = my_fft_grid->npts_global[0] - 1;
      my_bounds[1][0] = local_n1_start_gs;
      my_bounds[1][1] = local_n1_start_gs + local_n1_gs - 1;
      my_bounds[2][0] = 0;
      my_bounds[2][1] = my_fft_grid->npts_global_gspace[2] - 1;
      cp_mpi_allgather_int((const int *)my_bounds, 6,
                           (int *)my_fft_grid->proc2local_gs,
                           6 * number_of_processes, my_fft_grid->comm);
      // For tests
      for (int process = 0; process < number_of_processes; process++) {
        int proc_coords[2];
        cp_mpi_cart_coords(my_fft_grid->comm, process, 2, proc_coords);
        my_fft_grid->proc2local_ms[process][0][0] =
            my_fft_grid->proc2local_rs[process][0][0];
        my_fft_grid->proc2local_ms[process][0][1] =
            my_fft_grid->proc2local_rs[process][0][1];
        my_fft_grid->proc2local_ms[process][1][0] = 0;
        my_fft_grid->proc2local_ms[process][1][1] =
            my_fft_grid->npts_global_gspace[1] - 1;
        my_fft_grid->proc2local_ms[process][2][0] =
            my_fft_grid->proc2local_gs[process][2][0];
        my_fft_grid->proc2local_ms[process][2][1] =
            my_fft_grid->proc2local_gs[process][2][1];
      }
    }
  } else {
    // Right now, we cannot make use of the Guru interface. So, the data
    // distribution is different in real space here, distribute in y, and z
    // directions (z_D,y_D,x) (->rs) In mixed space I, distribute in x and z
    // directions (x_D,z_D,y) (->ms) In mixed space II, distribute in x and y
    // directions (y_D,x_D,z) (->gs, it is the same distribution, just a
    // different order of indices) To g-space, transpose the data (z, y_D, x_D)
    // (blocked format) In ray-distribution, it is even (z, yx_D) In case of a
    // 1D data distribution, the last two directions are locally available to
    // enable 2D FFT plans
    const int block_size_x_rs =
        (my_fft_grid->npts_global[0] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    const int block_size_y_rs =
        (my_fft_grid->npts_global[1] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_y_gs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    const int block_size_z_gs =
        (my_fft_grid->npts_global_gspace[2] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    // OMP parallelization requires a multi-threaded MPI
    for (int proc = 0; proc < number_of_processes; proc++) {
      int proc_coords[2];
      cp_mpi_cart_coords(my_fft_grid->comm, proc, 2, proc_coords);
      // Determine the bounds in real space
      my_fft_grid->proc2local_rs[proc][0][0] =
          imin(block_size_x_rs * proc_coords[0], my_fft_grid->npts_global[0]);
      my_fft_grid->proc2local_rs[proc][0][1] =
          imin(block_size_x_rs * (proc_coords[0] + 1) - 1,
               my_fft_grid->npts_global[0] - 1);
      my_fft_grid->proc2local_rs[proc][1][0] =
          imin(block_size_y_rs * proc_coords[1], my_fft_grid->npts_global[1]);
      my_fft_grid->proc2local_rs[proc][1][1] =
          imin(block_size_y_rs * (proc_coords[1] + 1) - 1,
               my_fft_grid->npts_global[1] - 1);
      my_fft_grid->proc2local_rs[proc][2][0] = 0;
      my_fft_grid->proc2local_rs[proc][2][1] = my_fft_grid->npts_global[2] - 1;
      // Determine the bounds in mixed space: we keep the distribution in the
      // first direction to reduce communication
      my_fft_grid->proc2local_ms[proc][0][0] =
          my_fft_grid->proc2local_rs[proc][0][0];
      my_fft_grid->proc2local_ms[proc][0][1] =
          my_fft_grid->proc2local_rs[proc][0][1];
      my_fft_grid->proc2local_ms[proc][1][0] = 0;
      my_fft_grid->proc2local_ms[proc][1][1] = my_fft_grid->npts_global[1] - 1;
      my_fft_grid->proc2local_ms[proc][2][0] = imin(
          block_size_z_gs * proc_coords[1], my_fft_grid->npts_global_gspace[2]);
      my_fft_grid->proc2local_ms[proc][2][1] =
          imin(block_size_z_gs * (proc_coords[1] + 1) - 1,
               my_fft_grid->npts_global_gspace[2] - 1);
      // Determine the bounds in mixed space: we keep the distribution in the
      // third direction to reduce communication
      my_fft_grid->proc2local_gs[proc][0][0] = 0;
      my_fft_grid->proc2local_gs[proc][0][1] =
          my_fft_grid->npts_global_gspace[0] - 1;
      my_fft_grid->proc2local_gs[proc][1][0] = imin(
          block_size_y_gs * proc_coords[0], my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_gs[proc][1][1] =
          imin(block_size_y_gs * (proc_coords[0] + 1) - 1,
               my_fft_grid->npts_global_gspace[1] - 1);
      my_fft_grid->proc2local_gs[proc][2][0] =
          my_fft_grid->proc2local_ms[proc][2][0];
      my_fft_grid->proc2local_gs[proc][2][1] =
          my_fft_grid->proc2local_ms[proc][2][1];
    }
    int buffer_size = 0;
    buffer_size = imax(buffer_size,
                       (my_fft_grid->proc2local_rs[my_process][0][1] -
                        my_fft_grid->proc2local_rs[my_process][0][0] + 1) *
                           (my_fft_grid->proc2local_rs[my_process][1][1] -
                            my_fft_grid->proc2local_rs[my_process][1][0] + 1) *
                           ((my_fft_grid->proc2local_rs[my_process][2][1] -
                             my_fft_grid->proc2local_rs[my_process][2][0] + 1) /
                                2 +
                            1) *
                           2);
    buffer_size = imax(buffer_size,
                       (my_fft_grid->proc2local_ms[my_process][0][1] -
                        my_fft_grid->proc2local_ms[my_process][0][0] + 1) *
                           (my_fft_grid->proc2local_ms[my_process][1][1] -
                            my_fft_grid->proc2local_ms[my_process][1][0] + 1) *
                           (my_fft_grid->proc2local_ms[my_process][2][1] -
                            my_fft_grid->proc2local_ms[my_process][2][0] + 1));
    buffer_size = imax(buffer_size,
                       (my_fft_grid->proc2local_gs[my_process][0][1] -
                        my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                           (my_fft_grid->proc2local_gs[my_process][1][1] -
                            my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
                           (my_fft_grid->proc2local_gs[my_process][2][1] -
                            my_fft_grid->proc2local_gs[my_process][2][0] + 1));
    my_fft_grid->buffer_size = buffer_size;
  }
}

void allocate_fft_buffers(fft_grid_layout *my_fft_grid) {
  // Determine the maximum buffer size
  int buffer_size = my_fft_grid->buffer_size;
  buffer_size = imax(buffer_size, my_fft_grid->npts_gs_local);
  // Allocate the buffers
  my_fft_grid->buffer_1 = NULL;
  my_fft_grid->buffer_2 = NULL;
  fft_allocate_complex(buffer_size, &my_fft_grid->buffer_1);
  fft_allocate_complex(buffer_size, &my_fft_grid->buffer_2);
}

void grid_create_fft_grid_layout(fft_grid_layout **fft_grid,
                                 const cp_mpi_comm_t comm,
                                 const int npts_global[3],
                                 const double dh_inv[3][3],
                                 const bool use_halfspace) {
  fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(fft_grid_layout));

  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = current_grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;
  my_fft_grid->ray_distribution = false;

  // Split the last dimension in real-space
  if (npts_global[0] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
    // Swap dimension if the large process dimension is not on the large global
    // dimension
    if ((npts_global[0] - npts_global[1]) *
            (my_fft_grid->proc_grid[0] - my_fft_grid->proc_grid[1]) <
        0) {
      const int proc_grid_0 = my_fft_grid->proc_grid[0];
      my_fft_grid->proc_grid[0] = my_fft_grid->proc_grid[1];
      my_fft_grid->proc_grid[1] = proc_grid_0;
    }
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  my_fft_grid->use_halfspace = use_halfspace;
  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));
  memcpy(my_fft_grid->npts_global_gspace, npts_global, 3 * sizeof(int));
  if (my_fft_grid->use_halfspace)
    my_fft_grid->npts_global_gspace[2] = npts_global[2] / 2 + 1;
  for (int dir = 0; dir < 3; dir++) {
    for (int dir2 = 0; dir2 < 3; dir2++) {
      my_fft_grid->h_inv[dir][dir2] =
          dh_inv[dir][dir2] / ((double)npts_global[dir2]);
    }
  }

  my_fft_grid->periodic[0] = 1;
  my_fft_grid->periodic[1] = 1;
  my_fft_grid->comm = cp_mpi_cart_create(comm, 2, my_fft_grid->proc_grid,
                                         my_fft_grid->periodic, true);

  cp_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid,
                  my_fft_grid->periodic, my_fft_grid->proc_coords);

  my_fft_grid->sub_comm[0] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0});
  my_fft_grid->sub_comm[1] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1});
  assert(cp_mpi_comm_size(my_fft_grid->sub_comm[0]) ==
         my_fft_grid->proc_grid[0]);
  assert(cp_mpi_comm_size(my_fft_grid->sub_comm[1]) ==
         my_fft_grid->proc_grid[1]);

  setup_proc2local(my_fft_grid);

  my_fft_grid->npts_gs_local =
      (my_fft_grid->proc2local_gs[my_process][0][1] -
       my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][1][1] -
       my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][2][1] -
       my_fft_grid->proc2local_gs[my_process][2][0] + 1);

  allocate_fft_buffers(my_fft_grid);

  my_fft_grid->yz_to_process = NULL;
  my_fft_grid->ray_to_yz = NULL;
  my_fft_grid->rays_per_process = NULL;
  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
#pragma omp parallel for default(none) shared(my_fft_grid, my_process)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->index_to_g[index][0] =
        my_fft_grid->proc2local_gs[my_process][0][0] +
        index % (my_fft_grid->proc2local_gs[my_process][0][1] -
                 my_fft_grid->proc2local_gs[my_process][0][0] + 1);
    my_fft_grid->index_to_g[index][1] =
        my_fft_grid->proc2local_gs[my_process][1][0] +
        (index / (my_fft_grid->proc2local_gs[my_process][0][1] -
                  my_fft_grid->proc2local_gs[my_process][0][0] + 1)) %
            (my_fft_grid->proc2local_gs[my_process][1][1] -
             my_fft_grid->proc2local_gs[my_process][1][0] + 1);
    my_fft_grid->index_to_g[index][2] =
        my_fft_grid->proc2local_gs[my_process][2][0] +
        index / ((my_fft_grid->proc2local_gs[my_process][0][1] -
                  my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][1][1] -
                  my_fft_grid->proc2local_gs[my_process][1][0] + 1));
  }

  my_fft_grid->local_index_to_ref_grid =
      calloc(my_fft_grid->npts_gs_local, sizeof(int));
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->local_index_to_ref_grid[index] = index;
  }

  sort_g_vectors(my_fft_grid);

  *fft_grid = my_fft_grid;
}

void grid_create_fft_grid_layout_from_reference(
    fft_grid_layout **fft_grid, const int npts_global[3],
    const fft_grid_layout *fft_grid_ref) {
  assert(fft_grid_ref != NULL &&
         "Grid creation from reference grid requires a valid reference grid!");
  // Current restriction of the code.
  assert(!fft_grid_ref->ray_distribution &&
         "The reference grid has to have a blocked distribution!");
  // We will use the reference grid to collect the data from other grids. To
  // prevent loss of accuracy, we require the new grid to be coarser than or as
  // coarse as the reference grid.
  assert(npts_global[0] <= fft_grid_ref->npts_global[0] &&
         npts_global[1] <= fft_grid_ref->npts_global[1] &&
         npts_global[2] <= fft_grid_ref->npts_global[2] &&
         "The new grid cannot have more grid points in any direction than the "
         "reference grid!");

  const int number_of_processes = cp_mpi_comm_size(fft_grid_ref->comm);
  const int my_process = cp_mpi_comm_rank(fft_grid_ref->comm);

  fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(fft_grid_layout));

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = fft_grid_ref->grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;

  my_fft_grid->ray_distribution = true;

  if (npts_global[0] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
    // Swap dimension if the large process dimension is not on the large global
    // dimension
    if ((npts_global[0] - npts_global[1]) *
            (my_fft_grid->proc_grid[0] - my_fft_grid->proc_grid[1]) <
        0) {
      const int proc_grid_0 = my_fft_grid->proc_grid[0];
      my_fft_grid->proc_grid[0] = my_fft_grid->proc_grid[1];
      my_fft_grid->proc_grid[1] = proc_grid_0;
    }
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  my_fft_grid->use_halfspace = fft_grid_ref->use_halfspace;
  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));
  memcpy(my_fft_grid->npts_global_gspace, npts_global, 3 * sizeof(int));
  if (my_fft_grid->use_halfspace)
    my_fft_grid->npts_global_gspace[2] = npts_global[2] / 2 + 1;
  for (int dir = 0; dir < 3; dir++) {
    for (int dir2 = 0; dir2 < 3; dir2++) {
      my_fft_grid->h_inv[dir][dir2] =
          fft_grid_ref->h_inv[dir][dir2] *
          ((double)fft_grid_ref->npts_global[dir2]) /
          ((double)npts_global[dir2]);
    }
  }

  my_fft_grid->periodic[0] = 1;
  my_fft_grid->periodic[1] = 1;
  my_fft_grid->comm =
      cp_mpi_cart_create(fft_grid_ref->comm, 2, my_fft_grid->proc_grid,
                         my_fft_grid->periodic, false);

  cp_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid,
                  my_fft_grid->periodic, my_fft_grid->proc_coords);

  my_fft_grid->sub_comm[0] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0});
  my_fft_grid->sub_comm[1] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1});
  assert(cp_mpi_comm_size(my_fft_grid->sub_comm[0]) ==
         my_fft_grid->proc_grid[0]);
  assert(cp_mpi_comm_size(my_fft_grid->sub_comm[1]) ==
         my_fft_grid->proc_grid[1]);

  setup_proc2local(my_fft_grid);

  // Assign the (yz)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->yz_to_process =
      malloc(my_fft_grid->npts_global_gspace[1] *
             my_fft_grid->npts_global_gspace[2] * sizeof(int));
  memset(my_fft_grid->yz_to_process, -1,
         my_fft_grid->npts_global_gspace[1] *
             my_fft_grid->npts_global_gspace[2] * sizeof(int));
  // Count the number of rays on each process
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  int total_number_of_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, fft_grid_ref, npts_global, number_of_processes,        \
               my_process, stdout) reduction(+ : total_number_of_rays)
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_z = fft_grid_ref->proc2local_gs[process][2][0];
         index_z <= fft_grid_ref->proc2local_gs[process][2][1]; index_z++) {
      // The right half of the indices is shifted
      const int index_z_shifted = convert_c_index_to_shifted_index(
          index_z, fft_grid_ref->npts_global[2]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (!is_on_grid(index_z_shifted, npts_global[2]))
        continue;
      const int index_z_new =
          convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
      for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
           index_y <= fft_grid_ref->proc2local_gs[process][1][1]; index_y++) {
        // The right half of the indices is shifted
        const int index_y_shifted = convert_c_index_to_shifted_index(
            index_y, fft_grid_ref->npts_global[1]);
        // Same check for z-coordinate
        if (!is_on_grid(index_y_shifted, npts_global[1]))
          continue;
        const int index_y_new =
            convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
        assert(
            my_fft_grid->yz_to_process[index_z_new *
                                           my_fft_grid->npts_global_gspace[1] +
                                       index_y_new] < 0);
        my_fft_grid
            ->yz_to_process[index_z_new * my_fft_grid->npts_global_gspace[1] +
                            index_y_new] = process;
        my_fft_grid->rays_per_process[process]++;
        total_number_of_rays++;
      }
    }
  }
  my_fft_grid->npts_gs_local = my_fft_grid->npts_global_gspace[0] *
                               my_fft_grid->rays_per_process[my_process];
  my_fft_grid->buffer_size =
      imax(my_fft_grid->buffer_size, my_fft_grid->npts_gs_local);

  int *ray_offsets = calloc(number_of_processes, sizeof(int));
  int *ray_index_per_process = calloc(number_of_processes, sizeof(int));
  for (int process = 1; process < number_of_processes; process++) {
    ray_offsets[process] =
        ray_offsets[process - 1] + my_fft_grid->rays_per_process[process - 1];
  }
  assert(ray_offsets[number_of_processes - 1] +
             my_fft_grid->rays_per_process[number_of_processes - 1] ==
         total_number_of_rays);

  // Create the map of yz index to the yz coordinates and the x-values required
  // for the mixed space
  my_fft_grid->ray_to_yz = malloc(total_number_of_rays * sizeof(int[2]));
  memset(my_fft_grid->ray_to_yz, -1, total_number_of_rays * sizeof(int[2]));
  for (int index_z = 0; index_z < fft_grid_ref->npts_global_gspace[2];
       index_z++) {
    const int index_z_shifted =
        convert_c_index_to_shifted_index(index_z, fft_grid_ref->npts_global[2]);
    if (!is_on_grid(index_z_shifted, npts_global[2]))
      continue;
    const int index_z_new =
        convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
    for (int index_y = 0; index_y < fft_grid_ref->npts_global_gspace[1];
         index_y++) {
      const int index_y_shifted = convert_c_index_to_shifted_index(
          index_y, fft_grid_ref->npts_global[1]);
      // Same check for y-coordinate
      if (!is_on_grid(index_y_shifted, npts_global[1]))
        continue;
      const int index_y_new =
          convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
      const int current_process =
          my_fft_grid
              ->yz_to_process[index_z_new * my_fft_grid->npts_global_gspace[1] +
                              index_y_new];
      const int current_ray_index = ray_index_per_process[current_process];
      const int current_ray_offset = ray_offsets[current_process];
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][0] =
          index_y_new;
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][1] =
          index_z_new;
      ray_index_per_process[current_process]++;
    }
  }
  for (int process = 0; process < number_of_processes; process++) {
    assert(ray_index_per_process[process] ==
               my_fft_grid->rays_per_process[process] &&
           "The number of rays does not match the expected number of rays!");
    const int current_ray_offset = ray_offsets[process];
    for (int ray_index = 0; ray_index < my_fft_grid->rays_per_process[process];
         ray_index++) {
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] >= 0 &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] >= 0 &&
             "The ray has to be assigned to a valid yz index!");
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] <
                 my_fft_grid->npts_global_gspace[1] &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] <
                 my_fft_grid->npts_global_gspace[2] &&
             "The ray has to be assigned to a valid yz index!");
    }
  }

  free(ray_offsets);
  free(ray_index_per_process);

  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
  // This grid is smaller in all directions such that all points of the new grid
  // should be available on the reference grid
  my_fft_grid->local_index_to_ref_grid =
      calloc(my_fft_grid->npts_gs_local, sizeof(int));

  int own_index = 0;
  for (int ref_index = 0; ref_index < fft_grid_ref->npts_gs_local;
       ref_index++) {
    const int shifted_indices[3] = {
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][0],
                                         fft_grid_ref->npts_global[0]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][1],
                                         fft_grid_ref->npts_global[1]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][2],
                                         fft_grid_ref->npts_global[2])};
    if (is_on_grid(shifted_indices[0], my_fft_grid->npts_global[0]) &&
        is_on_grid(shifted_indices[1], my_fft_grid->npts_global[1]) &&
        is_on_grid(shifted_indices[2], my_fft_grid->npts_global[2])) {
      for (int dir = 0; dir < 3; dir++) {
        my_fft_grid->index_to_g[own_index][dir] =
            convert_shifted_index_to_c_index(shifted_indices[dir],
                                             my_fft_grid->npts_global[dir]);
        my_fft_grid->local_index_to_ref_grid[own_index] = ref_index;
      }
      own_index++;
    }
  }
  assert(own_index == my_fft_grid->npts_gs_local);

  allocate_fft_buffers(my_fft_grid);

  *fft_grid = my_fft_grid;
}

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void grid_retain_fft_grid_layout(fft_grid_layout *fft_grid) {
  assert(fft_grid != NULL);
  assert(fft_grid->ref_counter > 0);
  fft_grid->ref_counter++;
}

/*******************************************************************************
 * \brief Print some information on a grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_print_grid_layout_info(const fft_grid_layout *layout,
                                 bool print_distribution) {
  if (cp_mpi_comm_rank(layout->comm) == 0) {
    fprintf(stdout, "Grid ID: %i\n", layout->grid_id);
    fprintf(stdout, "Reference Grid ID: %i\n", layout->ref_grid_id);
    fprintf(stdout, "Global sizes: %i %i %i\n", layout->npts_global[0],
            layout->npts_global[1], layout->npts_global[2]);
    for (int dir = 0; dir < 3; dir++)
      fprintf(stdout, "Grid spacing %i: %f %f %f\n", dir, layout->h_inv[dir][0],
              layout->h_inv[dir][1], layout->h_inv[dir][2]);
    fprintf(stdout, "Use half space: %i\n", layout->use_halfspace);
    fprintf(stdout, "Use ray distribution: %i\n", layout->ray_distribution);
    fprintf(stdout, "Process grid: %i %i\n", layout->proc_grid[0],
            layout->proc_grid[1]);
    if (print_distribution) {
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions RS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_rs[process][0][0],
                layout->proc2local_rs[process][0][1],
                layout->proc2local_rs[process][1][0],
                layout->proc2local_rs[process][1][1],
                layout->proc2local_rs[process][2][0],
                layout->proc2local_rs[process][2][1]);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions MS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_ms[process][0][0],
                layout->proc2local_ms[process][0][1],
                layout->proc2local_ms[process][1][0],
                layout->proc2local_ms[process][1][1],
                layout->proc2local_ms[process][2][0],
                layout->proc2local_ms[process][2][1]);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions GS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_gs[process][0][0],
                layout->proc2local_gs[process][0][1],
                layout->proc2local_gs[process][1][0],
                layout->proc2local_gs[process][1][1],
                layout->proc2local_gs[process][2][0],
                layout->proc2local_gs[process][2][1]);
      }
      if (layout->ray_distribution) {
        for (int index_y = 0; index_y < layout->npts_global_gspace[1];
             index_y++) {
          for (int index_z = 0; index_z < layout->npts_global_gspace[2];
               index_z++) {
            fprintf(
                stdout, "yz-pair %i %i is on process %i\n", index_y, index_z,
                layout->yz_to_process[index_z * layout->npts_global_gspace[1] +
                                      index_y]);
          }
        }
      }
    }
    fflush(stdout);
  }
  cp_mpi_barrier(layout->comm);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked_low(double complex *restrict grid_buffer_1,
                           double complex *restrict grid_buffer_2,
                           const int npts_global[3],
                           const int (*proc2local_rs)[3][2],
                           const int (*proc2local_ms)[3][2],
                           const int (*proc2local_gs)[3][2],
                           const cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT
    if (fft_lib_use_mpi()) {
      // Perform the first two FFTs in y- and z-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)
      transpose_local_complex(
          grid_buffer_1, grid_buffer_2, fft_sizes_rs[1] * fft_sizes_rs[2],
          fft_sizes_rs[0], fft_sizes_rs[1] * fft_sizes_rs[2], fft_sizes_rs[0]);
      // Copy back (we do not have in-place transposition implemented)
      memcpy(grid_buffer_1, grid_buffer_2,
             product3(fft_sizes_rs) * sizeof(double complex));
      fft_2d_fw_distributed((const int[2]){npts_global[1], npts_global[2]},
                            fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                            grid_buffer_2);

      // Perform second redistribution and transpose
      // (z_d,y,x_d) -> (x,z_d,y_d)
      collect_x_and_distribute_y_blocked_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          proc2local_gs, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1], false,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform redistribution
      collect_y_and_distribute_z_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2],
          proc2local_rs, proc2local_ms, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second redistribution
      collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_ms,
                                         proc2local_gs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    if (fft_lib_use_mpi()) {
      // Perform the distributed 3D FFT in one shot (x_D, y, z)->(y_D,x, z)
      // Returns transposed layout
      fft_3d_fw_distributed(npts_global, comm, grid_buffer_1, grid_buffer_2);
      // Transpose the data (y_D,x,z) -> (x,y_D,z)
      transpose_local_complex_block(grid_buffer_2, grid_buffer_1,
                                    fft_sizes_gs[0], fft_sizes_gs[1],
                                    fft_sizes_gs[2]);
      // Copy the data back to the output buffer (we do not have an in-place
      // transposition available)
      memcpy(grid_buffer_2, grid_buffer_1,
             product3(fft_sizes_gs) * sizeof(double complex));
    } else {
      // Perform the first FFT
      fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                      fft_sizes_rs[0], false, true, grid_buffer_1,
                      grid_buffer_2);

      // Perform second transpose
      collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_ms,
                                         proc2local_gs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else {
    fft_3d_fw_local(npts_global, grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_blocked_low(double complex *restrict grid_buffer_1,
                               double complex *restrict grid_buffer_2,
                               const int npts_global[3],
                               const int (*proc2local_rs)[3][2],
                               const int (*proc2local_ms)[3][2],
                               const int (*proc2local_gs)[3][2],
                               const cp_mpi_comm_t comm,
                               const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT
    if (fft_lib_use_mpi()) {
      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)
      // Padd the z-direction as required by FFTW
      for (int index_x = 0; index_x < fft_sizes_rs[0]; index_x++) {
        for (int index_y = 0; index_y < fft_sizes_rs[1]; index_y++) {
          for (int index_z = 0; index_z < fft_sizes_rs[2]; index_z++) {
            ((double *)grid_buffer_2)[(index_y * (npts_global[2] / 2 + 1) * 2 +
                                       index_z) *
                                          fft_sizes_rs[0] +
                                      index_x] =
                ((double *)
                     grid_buffer_1)[(index_x * fft_sizes_rs[1] + index_y) *
                                        npts_global[2] +
                                    index_z];
          }
        }
      }
      memcpy((double *)grid_buffer_1, (double *)grid_buffer_2,
             fft_sizes_rs[0] * fft_sizes_rs[1] * (npts_global[2] / 2 + 1) * 2 *
                 sizeof(double));
      fft_2d_fw_distributed_r2c((const int[2]){npts_global[1], npts_global[2]},
                                fft_sizes_rs[0], sub_comm[1],
                                (double *)grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (y_d,z,x_d) -> (x,z_d,y_d)
      collect_x_and_distribute_y_blocked_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          proc2local_gs, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    } else {
      fft_1d_fw_local_r2c(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                          false, true, (double *)grid_buffer_1, grid_buffer_2);

      // Perform redistribution
      collect_y_and_distribute_z_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2] / 2 + 1,
          proc2local_rs, proc2local_ms, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second redistribution
      collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_ms,
                                         proc2local_gs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    if (fft_lib_use_mpi()) {
      // We need to reorder the data because the data is padded for the
      // distributed case
      memset((double *)grid_buffer_2, 0,
             fft_sizes_rs[0] * fft_sizes_rs[1] * (npts_global[2] / 2 + 1) * 2 *
                 sizeof(double));
      for (int index_xy = 0; index_xy < fft_sizes_rs[0] * fft_sizes_rs[1];
           index_xy++) {
        memcpy(((double *)grid_buffer_2) +
                   (npts_global[2] / 2 + 1) * 2 * index_xy,
               ((double *)grid_buffer_1) + npts_global[2] * index_xy,
               npts_global[2] * sizeof(double));
      }
      // Perform the distributed 3D FFT in one shot (x_D,y,z)->(y_D,x,z)
      // Returns transposed layout
      fft_3d_fw_distributed_r2c(npts_global, comm, (double *)grid_buffer_2,
                                grid_buffer_1);

      // Exchange the first two dimensions to arrive at the correct layout
      // Transpose the data (y_D,x,z) -> (x,y_D,z)
      transpose_local_complex_block(grid_buffer_1, grid_buffer_2,
                                    fft_sizes_gs[0], fft_sizes_gs[1],
                                    fft_sizes_gs[2]);
    } else {
      // Perform the first FFT
      fft_2d_fw_local_r2c((const int[2]){npts_global[1], npts_global[2]},
                          fft_sizes_rs[0], false, true, (double *)grid_buffer_1,
                          grid_buffer_2);

      // Perform second transpose
      collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_ms,
                                         proc2local_gs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else {
    fft_3d_fw_local_r2c(npts_global, (double *)grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked_low(double complex *restrict grid_buffer_1,
                           double complex *restrict grid_buffer_2,
                           const int npts_global[3],
                           const int (*proc2local_rs)[3][2],
                           const int (*proc2local_ms)[3][2],
                           const int (*proc2local_gs)[3][2],
                           const cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in x-direction
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (x,z_d,y_d) -> (y_d,z,x_d)
      collect_y_and_distribute_x_blocked_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_gs,
          proc2local_ms, comm, sub_comm);

      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)
      fft_2d_bw_distributed((const int[2]){npts_global[1], npts_global[2]},
                            fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                            grid_buffer_2);
      transpose_local_complex(grid_buffer_2, grid_buffer_1, fft_sizes_rs[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2],
                              fft_sizes_rs[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2]);
      memcpy(grid_buffer_2, grid_buffer_1,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction
      collect_y_and_distribute_x_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_gs,
                                         proc2local_ms, comm, sub_comm);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in z-direction and distribute y-direction
      collect_z_and_distribute_y_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2],
          proc2local_ms, proc2local_rs, comm, sub_comm);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      fft_1d_bw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1], false,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    if (fft_lib_use_mpi()) {
      // Exchange the first two dimensions
      transpose_local_complex_block(grid_buffer_1, grid_buffer_2,
                                    fft_sizes_gs[1], fft_sizes_gs[0],
                                    fft_sizes_gs[2]);
      fft_3d_bw_distributed(npts_global, comm, grid_buffer_2, grid_buffer_1);
      memcpy(grid_buffer_2, grid_buffer_1,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction
      collect_y_and_distribute_x_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_gs,
                                         proc2local_ms, comm, sub_comm);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                      fft_sizes_ms[0], false, true, grid_buffer_1,
                      grid_buffer_2);
    }
  } else {
    fft_3d_bw_local(npts_global, grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_blocked_low(double complex *restrict grid_buffer_1,
                               double complex *restrict grid_buffer_2,
                               const int npts_global[3],
                               const int (*proc2local_rs)[3][2],
                               const int (*proc2local_ms)[3][2],
                               const int (*proc2local_gs)[3][2],
                               const cp_mpi_comm_t comm,
                               const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in x-direction
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (x,z_d,y_d) -> (y_d,z,x_d)
      collect_y_and_distribute_x_blocked_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_gs,
          proc2local_ms, comm, sub_comm);

      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)
      fft_2d_bw_distributed_c2r((const int[2]){npts_global[1], npts_global[2]},
                                fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                                (double *)grid_buffer_2);
      for (int index_x = 0; index_x < fft_sizes_rs[0]; index_x++) {
        for (int index_y = 0; index_y < fft_sizes_rs[1]; index_y++) {
          for (int index_z = 0; index_z < fft_sizes_rs[2]; index_z++) {
            ((double *)grid_buffer_1)[(index_x * fft_sizes_rs[1] + index_y) *
                                          fft_sizes_rs[2] +
                                      index_z] =
                ((double *)
                     grid_buffer_2)[(index_y * (fft_sizes_rs[2] / 2 + 1) * 2 +
                                     index_z) *
                                        fft_sizes_rs[0] +
                                    index_x];
          }
        }
      }
      memcpy((double *)grid_buffer_2, (double *)grid_buffer_1,
             product3(fft_sizes_rs) * sizeof(double));
    } else {
      // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction
      collect_y_and_distribute_x_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_gs,
                                         proc2local_ms, comm, sub_comm);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in z-direction and distribute y-direction
      collect_z_and_distribute_y_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2] / 2 + 1,
          proc2local_ms, proc2local_rs, comm, sub_comm);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      fft_1d_bw_local_c2r(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                          false, true, grid_buffer_1, (double *)grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    if (fft_lib_use_mpi()) {
      // Exchange the first two dimensions
      transpose_local_complex_block(grid_buffer_1, grid_buffer_2,
                                    fft_sizes_gs[1], fft_sizes_gs[0],
                                    fft_sizes_gs[2]);
      fft_3d_bw_distributed_c2r(npts_global, comm, grid_buffer_2,
                                (double *)grid_buffer_1);
      for (int index_xy = 0; index_xy < fft_sizes_rs[0] * fft_sizes_rs[1];
           index_xy++) {
        memcpy(((double *)grid_buffer_2) + npts_global[2] * index_xy,
               ((double *)grid_buffer_1) +
                   (npts_global[2] / 2 + 1) * 2 * index_xy,
               npts_global[2] * sizeof(double));
      }
    } else {
      // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
      fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction
      collect_y_and_distribute_x_blocked(grid_buffer_2, grid_buffer_1,
                                         npts_global, proc2local_gs,
                                         proc2local_ms, comm, sub_comm);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_2d_bw_local_c2r((const int[2]){npts_global[1], npts_global[2]},
                          fft_sizes_ms[0], false, true, grid_buffer_1,
                          (double *)grid_buffer_2);
    }
  } else {
    fft_3d_bw_local_c2r(npts_global, grid_buffer_1, (double *)grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_ray_low(double complex *restrict grid_buffer_1,
                       double complex *restrict grid_buffer_2,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int *rays_per_process, const int (*ray_to_yz)[2],
                       const cp_mpi_comm_t comm,
                       const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int number_of_local_yz_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)
      transpose_local_complex(
          grid_buffer_1, grid_buffer_2, fft_sizes_rs[1] * fft_sizes_rs[2],
          fft_sizes_rs[0], fft_sizes_rs[1] * fft_sizes_rs[2], fft_sizes_rs[0]);
      memcpy(grid_buffer_1, grid_buffer_2,
             product3(fft_sizes_rs) * sizeof(double complex));
      fft_2d_fw_distributed((const int[2]){npts_global[1], npts_global[2]},
                            fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                            grid_buffer_2);

      // Perform second redistribution and transpose
      // (y_d,z,x_d) -> (x,z_d,y_d)
      collect_x_and_distribute_yz_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_yz, comm);

      // Perform the final FFT
      fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);
    } else {
      // Perform the first FFT
      fft_1d_fw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1], false,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform transpose
      collect_y_and_distribute_z_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2],
          proc2local_rs, proc2local_ms, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second transpose
      collect_x_and_distribute_yz_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                      proc2local_ms, rays_per_process,
                                      ray_to_yz, comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    // Depending on the use of a distributed FFT library, we have different
    // data distributions Perform the first FFT (z_d,y,x)->(x,z_d_y)
    fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                    fft_sizes_ms[0], false, true, grid_buffer_1, grid_buffer_2);

    // Perform second transpose
    collect_x_and_distribute_yz_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_yz,
                                    comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
  } else {
    fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global[0], false, false, grid_buffer_1, grid_buffer_2);
// Copy to the ray format
// Maybe, a 2D FFT, redistribution to rays and final FFT is faster
#pragma omp parallel for default(none)                                         \
    shared(npts_global, grid_buffer_1, ray_to_yz, grid_buffer_2,               \
               number_of_local_yz_rays) collapse(2)
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      for (int ray_yz = 0; ray_yz < number_of_local_yz_rays; ray_yz++) {
        const int index_y = ray_to_yz[ray_yz][0];
        const int index_z = ray_to_yz[ray_yz][1];
        grid_buffer_1[index_x * number_of_local_yz_rays + ray_yz] =
            grid_buffer_2[index_x * npts_global[1] * npts_global[2] +
                          index_y * npts_global[2] + index_z];
      }
    }
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_ray_low(double complex *restrict grid_buffer_1,
                           double complex *restrict grid_buffer_2,
                           const int npts_global[3],
                           const int (*proc2local_rs)[3][2],
                           const int (*proc2local_ms)[3][2],
                           const int *rays_per_process,
                           const int (*ray_to_yz)[2], const cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int number_of_local_yz_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (z_d,y,x_d) -> (y_d,z,x_d)

      for (int index_x = 0; index_x < fft_sizes_rs[0]; index_x++) {
        for (int index_y = 0; index_y < fft_sizes_rs[1]; index_y++) {
          for (int index_z = 0; index_z < fft_sizes_rs[2]; index_z++) {
            ((double *)grid_buffer_2)[(index_y * (npts_global[2] / 2 + 1) * 2 +
                                       index_z) *
                                          fft_sizes_rs[0] +
                                      index_x] =
                ((double *)
                     grid_buffer_1)[(index_x * fft_sizes_rs[1] + index_y) *
                                        npts_global[2] +
                                    index_z];
          }
        }
      }
      memcpy((double *)grid_buffer_1, (double *)grid_buffer_2,
             fft_sizes_rs[0] * fft_sizes_rs[1] * (npts_global[2] / 2 + 1) * 2 *
                 sizeof(double));
      fft_2d_fw_distributed_r2c((const int[2]){npts_global[1], npts_global[2]},
                                fft_sizes_rs[0], sub_comm[1],
                                (double *)grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (y_d,z,x_d) -> (x,z_d,y_d)
      collect_x_and_distribute_yz_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_yz, comm);

      // Perform the final FFT
      fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);
    } else {
      // Perform the first FFT
      fft_1d_fw_local_r2c(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                          false, true, (double *)grid_buffer_1, grid_buffer_2);

      // Perform transpose
      collect_y_and_distribute_z_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2] / 2 + 1,
          proc2local_rs, proc2local_ms, comm, sub_comm);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second transpose
      collect_x_and_distribute_yz_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                      proc2local_ms, rays_per_process,
                                      ray_to_yz, comm);

      // Perform the third FFT
      fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    // The first two FFTs can be performed locally
    // Perform the first FFT (z_d,y,x)->(x,z_d_y)
    fft_2d_fw_local_r2c((const int[2]){npts_global[1], npts_global[2]},
                        fft_sizes_ms[0], false, true, (double *)grid_buffer_1,
                        grid_buffer_2);

    // but we need to redistribute to rays
    collect_x_and_distribute_yz_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_yz,
                                    comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
  } else {
    fft_2d_fw_local_r2c((const int[2]){npts_global[1], npts_global[2]},
                        npts_global[0], false, false, (double *)grid_buffer_1,
                        grid_buffer_2);
// Copy to the ray format
// Maybe, a 2D FFT, redistribution to rays and final FFT is faster
#pragma omp parallel for default(none)                                         \
    shared(npts_global, grid_buffer_1, ray_to_yz, grid_buffer_2,               \
               number_of_local_yz_rays) collapse(2)
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      for (int ray_yz = 0; ray_yz < number_of_local_yz_rays; ray_yz++) {
        const int index_y = ray_to_yz[ray_yz][0];
        const int index_z = ray_to_yz[ray_yz][1];
        grid_buffer_1[index_x * number_of_local_yz_rays + ray_yz] =
            grid_buffer_2[index_x * npts_global[1] * (npts_global[2] / 2 + 1) +
                          index_y * (npts_global[2] / 2 + 1) + index_z];
      }
    }
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_ray_low(double complex *restrict grid_buffer_1,
                       double complex *restrict grid_buffer_2,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int *rays_per_process, const int (*ray_to_yz)[2],
                       const cp_mpi_comm_t comm,
                       const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int number_of_local_yz_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in x-direction
      fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (x,zy_d) -> (y_d,z,x_d)
      collect_yz_and_distribute_x_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_yz, comm);

      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (y_d,z,x_d) -> (z_d,y,x_d)
      fft_2d_bw_distributed((const int[2]){npts_global[1], npts_global[2]},
                            fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                            grid_buffer_2);
      transpose_local_complex(grid_buffer_2, grid_buffer_1, fft_sizes_rs[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2],
                              fft_sizes_rs[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2]);
      memcpy(grid_buffer_2, grid_buffer_1,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      // Perform the first FFT
      fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);

      // Perform transpose
      collect_yz_and_distribute_x_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                      proc2local_ms, rays_per_process,
                                      ray_to_yz, comm);

      // Perform the second FFT
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second transpose
      collect_z_and_distribute_y_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2],
          proc2local_ms, proc2local_rs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_bw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1], false,
                      true, grid_buffer_1, grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose
    collect_yz_and_distribute_x_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_yz,
                                    comm);

    // Perform the second FFT
    fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                    fft_sizes_rs[0], false, true, grid_buffer_1, grid_buffer_2);
  } else {
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
    // Copy to the new format
    // Maybe, the order 1D FFT, redistribution to blocks and 2D FFT is
    // faster
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_local_yz_rays, grid_buffer_2, ray_to_yz,     \
               grid_buffer_1) collapse(2)
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      for (int yz_ray = 0; yz_ray < number_of_local_yz_rays; yz_ray++) {
        const int index_y = ray_to_yz[yz_ray][0];
        const int index_z = ray_to_yz[yz_ray][1];

        grid_buffer_1[index_x * npts_global[1] * npts_global[2] +
                      index_y * npts_global[2] + index_z] =
            grid_buffer_2[index_x * number_of_local_yz_rays + yz_ray];
      }
    }
    fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global[0], false, false, grid_buffer_1, grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_ray_low(double complex *restrict grid_buffer_1,
                           double complex *restrict grid_buffer_2,
                           const int npts_global[3],
                           const int (*proc2local_rs)[3][2],
                           const int (*proc2local_ms)[3][2],
                           const int *rays_per_process,
                           const int (*ray_to_yz)[2], const cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  const int my_process = cp_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  int number_of_local_yz_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in x-direction
      fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (x,zy_d) -> (y_d,z,x_d)
      collect_yz_and_distribute_x_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_yz, comm);

      // Perform the first two FFTs in x- and y-direction
      // transpose the last two indices (is cheaper)
      // (y_d,z,x_d) -> (z_d,y,x_d)
      fft_2d_bw_distributed_c2r((const int[2]){npts_global[1], npts_global[2]},
                                fft_sizes_rs[0], sub_comm[1], grid_buffer_1,
                                (double *)grid_buffer_2);
      for (int index_x = 0; index_x < fft_sizes_rs[0]; index_x++) {
        for (int index_y = 0; index_y < fft_sizes_rs[1]; index_y++) {
          for (int index_z = 0; index_z < fft_sizes_rs[2]; index_z++) {
            ((double *)grid_buffer_1)[(index_x * fft_sizes_rs[1] + index_y) *
                                          npts_global[2] +
                                      index_z] =
                ((double *)
                     grid_buffer_2)[(index_y * (npts_global[2] / 2 + 1) * 2 +
                                     index_z) *
                                        fft_sizes_rs[0] +
                                    index_x];
          }
        }
      }
      memcpy((double *)grid_buffer_2, (double *)grid_buffer_1,
             fft_sizes_rs[0] * fft_sizes_rs[1] * npts_global[2] *
                 sizeof(double));
    } else {
      // Perform the first FFT
      fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                      grid_buffer_1, grid_buffer_2);

      // Perform transpose
      collect_yz_and_distribute_x_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                      proc2local_ms, rays_per_process,
                                      ray_to_yz, comm);

      // Perform the second FFT
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      true, grid_buffer_1, grid_buffer_2);

      // Perform second transpose
      collect_z_and_distribute_y_blocked(
          grid_buffer_2, grid_buffer_1, npts_global, npts_global[2] / 2 + 1,
          proc2local_ms, proc2local_rs, comm, sub_comm);

      // Perform the third FFT
      fft_1d_bw_local_c2r(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                          false, true, grid_buffer_1, (double *)grid_buffer_2);
    }
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose
    collect_yz_and_distribute_x_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_yz,
                                    comm);

    // Perform the second FFT
    fft_2d_bw_local_c2r((const int[2]){npts_global[1], npts_global[2]},
                        fft_sizes_rs[0], false, true, grid_buffer_1,
                        (double *)grid_buffer_2);
  } else {
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, true, true,
                    grid_buffer_1, grid_buffer_2);
    // Copy to the new format
    // Maybe, the order 1D FFT, redistribution to blocks and 2D FFT is
    // faster
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_local_yz_rays, grid_buffer_2, ray_to_yz,     \
               grid_buffer_1) collapse(2)
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      for (int yz_ray = 0; yz_ray < number_of_local_yz_rays; yz_ray++) {
        const int index_y = ray_to_yz[yz_ray][0];
        const int index_z = ray_to_yz[yz_ray][1];

        grid_buffer_1[index_x * npts_global[1] * (npts_global[2] / 2 + 1) +
                      index_y * (npts_global[2] / 2 + 1) + index_z] =
            grid_buffer_2[index_x * number_of_local_yz_rays + yz_ray];
      }
    }
    fft_2d_bw_local_c2r((const int[2]){npts_global[1], npts_global[2]},
                        npts_global[0], false, false, grid_buffer_1,
                        (double *)grid_buffer_2);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs complex-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout(const double complex *restrict grid_rs,
                           double complex *restrict grid_gs,
                           const fft_grid_layout *grid_layout) {
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  memcpy(grid_layout->buffer_1, grid_rs,
         product3(local_sizes_rs) * sizeof(double complex));
  if (grid_layout->ray_distribution) {
    fft_3d_fw_ray_low(grid_layout->buffer_1, grid_layout->buffer_2,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->rays_per_process,
                      grid_layout->ray_to_yz, grid_layout->comm,
                      grid_layout->sub_comm);
    const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += grid_layout->rays_per_process[process];
    }
    const int my_number_of_rays = grid_layout->rays_per_process[my_process];
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, my_ray_to_yz, grid_gs, my_number_of_rays)
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      const int *index_g = grid_layout->index_to_g[index];
      for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          grid_gs[index] =
              grid_layout->buffer_2[index_g[0] * my_number_of_rays + yz_ray];
          break;
        }
      }
    }
  } else {
    fft_3d_fw_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      grid_gs[index] =
          grid_layout->buffer_2
              [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
                   (local_sizes_gs[1] * local_sizes_gs[2]) +
               (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                   local_sizes_gs[2] +
               (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])];
    }
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs complex-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout_to_cart(const double complex *restrict grid_rs,
                                   double complex *restrict grid_gs,
                                   const fft_grid_layout *grid_layout) {
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  memcpy(grid_layout->buffer_1, grid_rs,
         product3(local_sizes_rs) * sizeof(double complex));
  fft_3d_fw_blocked_low(grid_layout->buffer_1, grid_layout->buffer_2,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                        grid_layout->comm, grid_layout->sub_comm);
  int local_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                          grid_layout->proc2local_gs[my_process][dir][0] + 1;
  }
  memcpy(grid_gs, grid_layout->buffer_2,
         product3(local_sizes_gs) * sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout_to_cart(const double *restrict grid_rs,
                                       double complex *restrict grid_gs,
                                       const fft_grid_layout *grid_layout) {
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  if (grid_layout->use_halfspace) {
    memcpy((double *)grid_layout->buffer_1, grid_rs,
           product3(local_sizes_rs) * sizeof(double));
    fft_3d_fw_r2c_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
  } else {
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_rs, grid_rs)
    for (int i = 0; i < product3(local_sizes_rs); i++) {
      grid_layout->buffer_1[i] = CMPLX(grid_rs[i], 0.0);
    }
    fft_3d_fw_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
  }
  int local_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                          grid_layout->proc2local_gs[my_process][dir][0] + 1;
  }
  memcpy(grid_gs, grid_layout->buffer_2,
         product3(local_sizes_gs) * sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout(const double *restrict grid_rs,
                               double complex *restrict grid_gs,
                               const fft_grid_layout *grid_layout) {
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  if (grid_layout->use_halfspace) {
    memcpy((double *)grid_layout->buffer_1, grid_rs,
           product3(local_sizes_rs) * sizeof(double));
    if (grid_layout->ray_distribution) {
      fft_3d_fw_r2c_ray_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->rays_per_process,
          grid_layout->ray_to_yz, grid_layout->comm, grid_layout->sub_comm);
      const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
      for (int process = 0; process < my_process; process++) {
        my_ray_to_yz += grid_layout->rays_per_process[process];
      }
      const int my_number_of_rays = grid_layout->rays_per_process[my_process];
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, my_ray_to_yz, grid_gs, my_number_of_rays)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        const int *index_g = grid_layout->index_to_g[index];
        for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
          if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
              my_ray_to_yz[yz_ray][1] == index_g[2]) {
            grid_gs[index] =
                grid_layout->buffer_2[index_g[0] * my_number_of_rays + yz_ray];
            break;
          }
        }
      }
    } else {
      fft_3d_fw_r2c_blocked_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->comm, grid_layout->sub_comm);
      int local_sizes_gs[3];
      for (int dir = 0; dir < 3; dir++) {
        local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                              grid_layout->proc2local_gs[my_process][dir][0] +
                              1;
      }
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        grid_gs[index] =
            grid_layout->buffer_2
                [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
                     local_sizes_gs[1] * local_sizes_gs[2] +
                 (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                     local_sizes_gs[2] +
                 (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])];
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_rs, grid_rs)
    for (int i = 0; i < product3(local_sizes_rs); i++) {
      grid_layout->buffer_1[i] = CMPLX(grid_rs[i], 0.0);
    }
    if (grid_layout->ray_distribution) {
      fft_3d_fw_ray_low(grid_layout->buffer_1, grid_layout->buffer_2,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms,
                        grid_layout->rays_per_process, grid_layout->ray_to_yz,
                        grid_layout->comm, grid_layout->sub_comm);
      const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
      for (int process = 0; process < my_process; process++) {
        my_ray_to_yz += grid_layout->rays_per_process[process];
      }
      const int my_number_of_rays = grid_layout->rays_per_process[my_process];
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, my_ray_to_yz, grid_gs, my_number_of_rays)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        const int *index_g = grid_layout->index_to_g[index];
        for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
          if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
              my_ray_to_yz[yz_ray][1] == index_g[2]) {
            grid_gs[index] =
                grid_layout->buffer_2[index_g[0] * my_number_of_rays + yz_ray];
            break;
          }
        }
      }
    } else {
      fft_3d_fw_blocked_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->comm, grid_layout->sub_comm);
      int local_sizes_gs[3];
      for (int dir = 0; dir < 3; dir++) {
        local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                              grid_layout->proc2local_gs[my_process][dir][0] +
                              1;
      }
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        grid_gs[index] =
            grid_layout->buffer_2
                [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
                     local_sizes_gs[1] * local_sizes_gs[2] +
                 (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                     local_sizes_gs[2] +
                 (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])];
      }
    }
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout(const double complex *restrict grid_gs,
                           double complex *restrict grid_rs,
                           const fft_grid_layout *grid_layout) {
  assert(grid_gs != NULL);
  assert(grid_rs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  if (grid_layout->ray_distribution) {
    const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += grid_layout->rays_per_process[process];
    }
    const int my_number_of_rays = grid_layout->rays_per_process[my_process];
    memset(grid_layout->buffer_1, 0,
           my_number_of_rays * grid_layout->npts_global_gspace[0] *
               sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, grid_gs, my_ray_to_yz, my_number_of_rays)
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          grid_layout->buffer_1[index_g[0] * my_number_of_rays + yz_ray] =
              grid_gs[index];
          break;
        }
      }
    }
    fft_3d_bw_ray_low(grid_layout->buffer_1, grid_layout->buffer_2,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->rays_per_process,
                      grid_layout->ray_to_yz, grid_layout->comm,
                      grid_layout->sub_comm);
  } else {
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    memset(grid_layout->buffer_1, 0, product3(local_sizes_gs));
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_gs, grid_gs, my_process)
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      grid_layout->buffer_1
          [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
               local_sizes_gs[1] * local_sizes_gs[2] +
           (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
               local_sizes_gs[2] +
           (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])] =
          grid_gs[index];
    }
    fft_3d_bw_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
  }
  memcpy(grid_rs, grid_layout->buffer_2,
         product3(local_sizes_rs) * sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout_from_cart(const double complex *restrict grid_gs,
                                     double complex *restrict grid_rs,
                                     const fft_grid_layout *grid_layout) {
  assert(grid_gs != NULL);
  assert(grid_rs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                          grid_layout->proc2local_gs[my_process][dir][0] + 1;
  }
  memcpy(grid_layout->buffer_1, grid_gs,
         product3(local_sizes_gs) * sizeof(double complex));
  fft_3d_bw_blocked_low(grid_layout->buffer_1, grid_layout->buffer_2,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                        grid_layout->comm, grid_layout->sub_comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  memcpy(grid_rs, grid_layout->buffer_2,
         product3(local_sizes_rs) * sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout(const double complex *restrict grid_gs,
                               double *restrict grid_rs,
                               const fft_grid_layout *grid_layout) {
  assert(grid_gs != NULL);
  assert(grid_rs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
      for (int process = 0; process < my_process; process++) {
        my_ray_to_yz += grid_layout->rays_per_process[process];
      }
      const int my_number_of_rays = grid_layout->rays_per_process[my_process];
      memset(grid_layout->buffer_1, 0,
             grid_layout->npts_global_gspace[0] * my_number_of_rays);
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, grid_gs, my_ray_to_yz, my_number_of_rays)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
          if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
              my_ray_to_yz[yz_ray][1] == index_g[2]) {
            grid_layout->buffer_1[index_g[0] * my_number_of_rays + yz_ray] =
                grid_gs[index];
            break;
          }
        }
      }
      fft_3d_bw_c2r_ray_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->rays_per_process,
          grid_layout->ray_to_yz, grid_layout->comm, grid_layout->sub_comm);
    } else {
      int local_sizes_gs[3];
      for (int dir = 0; dir < 3; dir++) {
        local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                              grid_layout->proc2local_gs[my_process][dir][0] +
                              1;
      }
      memset(grid_layout->buffer_1, 0, product3(local_sizes_gs));
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_gs, grid_gs, my_process)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        grid_layout->buffer_1
            [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
                 local_sizes_gs[1] * local_sizes_gs[2] +
             (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                 local_sizes_gs[2] +
             (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])] =
            grid_gs[index];
      }
      fft_3d_bw_c2r_blocked_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->comm, grid_layout->sub_comm);
    }
    memcpy(grid_rs, (double *)grid_layout->buffer_2,
           product3(local_sizes_rs) * sizeof(double));
  } else {
    if (grid_layout->ray_distribution) {
      const int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
      for (int process = 0; process < my_process; process++) {
        my_ray_to_yz += grid_layout->rays_per_process[process];
      }
      const int my_number_of_rays = grid_layout->rays_per_process[my_process];
      memset(grid_layout->buffer_1, 0,
             grid_layout->npts_global_gspace[0] * my_number_of_rays);
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, grid_gs, my_ray_to_yz, my_number_of_rays)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
          if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
              my_ray_to_yz[yz_ray][1] == index_g[2]) {
            grid_layout->buffer_1[index_g[0] * my_number_of_rays + yz_ray] =
                grid_gs[index];
            break;
          }
        }
      }
      fft_3d_bw_ray_low(grid_layout->buffer_1, grid_layout->buffer_2,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms,
                        grid_layout->rays_per_process, grid_layout->ray_to_yz,
                        grid_layout->comm, grid_layout->sub_comm);
    } else {
      int local_sizes_gs[3];
      for (int dir = 0; dir < 3; dir++) {
        local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                              grid_layout->proc2local_gs[my_process][dir][0] +
                              1;
      }
      memset(grid_layout->buffer_1, 0, product3(local_sizes_gs));
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_gs, grid_gs, my_process)
      for (int index = 0; index < grid_layout->npts_gs_local; index++) {
        int *index_g = grid_layout->index_to_g[index];
        grid_layout->buffer_1
            [(index_g[0] - grid_layout->proc2local_gs[my_process][0][0]) *
                 local_sizes_gs[1] * local_sizes_gs[2] +
             (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                 local_sizes_gs[2] +
             (index_g[2] - grid_layout->proc2local_gs[my_process][2][0])] =
            grid_gs[index];
      }
      fft_3d_bw_blocked_low(
          grid_layout->buffer_1, grid_layout->buffer_2,
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->comm, grid_layout->sub_comm);
    }
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_rs, grid_rs)
    for (int i = 0; i < product3(local_sizes_rs); i++) {
      grid_rs[i] = creal(grid_layout->buffer_2[i]);
    }
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout_from_cart(const double complex *restrict grid_gs,
                                         double *restrict grid_rs,
                                         const fft_grid_layout *grid_layout) {
  assert(grid_gs != NULL);
  assert(grid_rs != NULL);
  assert(grid_layout != NULL);
  assert(grid_layout->ref_counter > 0);

  const int my_process = cp_mpi_comm_rank(grid_layout->comm);
  if (grid_layout->use_halfspace) {
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    memcpy(grid_layout->buffer_1, grid_gs,
           product3(local_sizes_gs) * sizeof(double complex));
    fft_3d_bw_c2r_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
    int local_sizes_rs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                            grid_layout->proc2local_rs[my_process][dir][0] + 1;
    }
    memcpy(grid_rs, (double *)grid_layout->buffer_2,
           product3(local_sizes_rs) * sizeof(double));
  } else {
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    memcpy(grid_layout->buffer_1, grid_gs,
           product3(local_sizes_gs) * sizeof(double complex));
    fft_3d_bw_blocked_low(
        grid_layout->buffer_1, grid_layout->buffer_2, grid_layout->npts_global,
        grid_layout->proc2local_rs, grid_layout->proc2local_ms,
        grid_layout->proc2local_gs, grid_layout->comm, grid_layout->sub_comm);
    int local_sizes_rs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                            grid_layout->proc2local_rs[my_process][dir][0] + 1;
    }
#pragma omp parallel for default(none)                                         \
    shared(grid_layout, local_sizes_rs, grid_rs)
    for (int i = 0; i < product3(local_sizes_rs); i++) {
      grid_rs[i] = creal(grid_layout->buffer_2[i]);
    }
  }
}

// EOF
