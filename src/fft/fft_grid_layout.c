/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid_layout.h"
#include "fft_driver.h"
#include "fft_lib.h"
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
        (const int[3]){
            convert_c_index_to_shifted_index(my_fft_grid->index_to_g[index][0],
                                             my_fft_grid->npts_global[0]),
            convert_c_index_to_shifted_index(my_fft_grid->index_to_g[index][1],
                                             my_fft_grid->npts_global[1]),
            convert_c_index_to_shifted_index(my_fft_grid->index_to_g[index][2],
                                             my_fft_grid->npts_global[2])},
        my_fft_grid->h_inv);
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
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_free_grid_layout");
  const int handle = fft_start_timer(routine_name);
  if (fft_grid != NULL) {
    assert((fft_grid->ref_counter) > 0);
    fft_grid->ref_counter--;
    if (fft_grid->ref_counter == 0) {
      cp_mpi_comm_free(&fft_grid->comm);
      cp_mpi_comm_free(&fft_grid->sub_comm[0]);
      cp_mpi_comm_free(&fft_grid->sub_comm[1]);
      cleanup_redistribution(fft_grid->redistribution);
      free(fft_grid->redistribution);
      free(fft_grid->proc2local_rs);
      free(fft_grid->proc2local_ms);
      free(fft_grid->proc2local_gs);
      free(fft_grid->proc2local_y_rs);
      free(fft_grid->proc2local_z_rs);
      free(fft_grid->proc2local_x_gs);
      free(fft_grid->proc2local_y_gs);
      free(fft_grid->xy_to_process);
      free(fft_grid->ray_to_xy);
      free(fft_grid->xy_to_ray);
      free(fft_grid->rays_per_process);
      free(fft_grid->index_to_g);
      free(fft_grid->local_index_to_ref_grid);
      free(fft_grid->index_to_cart);
      free(fft_grid->index_to_cart_neg);
      free(fft_grid->index_to_cart_pos);
      free(fft_grid->index_to_ray);
      free(fft_grid->index_to_ray_neg);
      free(fft_grid->index_to_ray_pos);
      free(fft_grid);
    }
  }
  fft_stop_timer(handle);
}

void setup_proc2local(fft_grid_layout *my_fft_grid) {
  const int number_of_processes = cp_mpi_comm_size(my_fft_grid->comm);
  const int my_process = cp_mpi_comm_rank(my_fft_grid->comm);

  my_fft_grid->proc2local_rs = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_ms = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_gs = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_y_rs =
      calloc(2 * my_fft_grid->proc_grid[1], sizeof(int));
  my_fft_grid->proc2local_z_rs =
      calloc(2 * my_fft_grid->proc_grid[0], sizeof(int));
  my_fft_grid->proc2local_x_gs =
      calloc(2 * my_fft_grid->proc_grid[1], sizeof(int));
  my_fft_grid->proc2local_y_gs =
      calloc(2 * my_fft_grid->proc_grid[0], sizeof(int));

  // With distributed FFT libraries, we thereby determine a buffer size which is
  // refined later
  int buffer_size = 0, my_bounds[2] = {0, 0};
  if (fft_lib_use_mpi() && cp_mpi_comm_size(my_fft_grid->comm) > 1) {
    // The data distribution is taken optimized for use with FFTW
    // The first index is distributed, the others are not
    // We ask for output data with the first two indices swapped (transposed
    // mode) to save communication within the library Several distributed 2D
    // FFTs require the distance between elements of adjacent FFTs to be 1
    // Starting from the order (z, y, x), we have to distribute the first index
    // and, in case of a pencil distribution, the LAST index (this should be
    // related to improved vectorization within the library)

    // As a first step, we determine the data distribution in each process
    // dimension
    if (my_fft_grid->proc_grid[1] > 1) {
      // Start with a distributed FFT using the first sub-communicator in y- and
      // z-direction
      int local_y_rs, local_y_start_rs, local_x_gs, local_x_start_gs;
      // We need to pre-distribute the z-direction
      const int block_size_z_rs =
          (my_fft_grid->npts_global_gspace[2] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // In the last step, y is distributed in the second direction
      const int block_size_y_gs =
          (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // Determine a maximum buffer size
      // With half-space, we need different routines
      if (my_fft_grid->use_halfspace) {
        my_fft_grid->buffer_size = fft_2d_distributed_sizes_r2c(
            (const int[2]){my_fft_grid->npts_global[1],
                           my_fft_grid->npts_global[0]},
            block_size_z_rs, my_fft_grid->sub_comm[1], &local_y_rs,
            &local_y_start_rs, &local_x_gs, &local_x_start_gs);
      } else {
        my_fft_grid->buffer_size = fft_2d_distributed_sizes(
            (const int[2]){my_fft_grid->npts_global[1],
                           my_fft_grid->npts_global[0]},
            block_size_z_rs, my_fft_grid->sub_comm[1], &local_y_rs,
            &local_y_start_rs, &local_x_gs, &local_x_start_gs);
      }
      // Setup the bounds in real space
      // In z-direction, we need to define them ourselves
      // The distributions in x- and y-direction are provided by the FFT library
      // With y (second index) required to be locally available
      my_bounds[0] = local_y_start_rs;
      my_bounds[1] = local_y_rs;
      // Exchange the distribution with the other processes
      cp_mpi_allgather_int((const int *)my_bounds, 2,
                           (int *)my_fft_grid->proc2local_y_rs, 2,
                           my_fft_grid->sub_comm[1]);
      my_bounds[0] = local_x_start_gs;
      my_bounds[1] = local_x_gs;
      // Exchange the distribution with the other processes
      cp_mpi_allgather_int((const int *)my_bounds, 2,
                           (int *)my_fft_grid->proc2local_x_gs, 2,
                           my_fft_grid->sub_comm[1]);
      for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
        my_fft_grid->proc2local_z_rs[process][0] =
            imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
        my_fft_grid->proc2local_z_rs[process][1] =
            imin((process + 1) * block_size_z_rs - 1,
                 my_fft_grid->npts_global_gspace[2] - 1) -
            imin(process * block_size_z_rs,
                 my_fft_grid->npts_global_gspace[2]) +
            1;
        my_fft_grid->proc2local_y_gs[process][0] =
            imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
        my_fft_grid->proc2local_y_gs[process][1] =
            imin((process + 1) * block_size_y_gs - 1,
                 my_fft_grid->npts_global_gspace[1] - 1) -
            imin(process * block_size_y_gs,
                 my_fft_grid->npts_global_gspace[1]) +
            1;
      }
    } else {
      // With distributed 3D FFTs, we ask the library to perform all FFT steps
      // This data distribution is obtained from the 2D case without data
      // distribution in the second process direction
      int local_z_rs = 0, local_z_start_rs = 0, local_y_gs = 0,
          local_y_start_gs = 0;
      // With ray distribution, we perform only a local 2D FFT to be able
      // to perform the final local FFT of the own rays only
      // So, we need to define the distribution in z-direction ourselves
      if (my_fft_grid->ray_distribution) {
        const int block_size_z_rs = (my_fft_grid->npts_global_gspace[2] +
                                     my_fft_grid->proc_grid[0] - 1) /
                                    my_fft_grid->proc_grid[0];
        const int block_size_y_gs = (my_fft_grid->npts_global_gspace[1] +
                                     my_fft_grid->proc_grid[0] - 1) /
                                    my_fft_grid->proc_grid[0];
        for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
          my_fft_grid->proc2local_z_rs[process][0] = imin(
              process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
          my_fft_grid->proc2local_z_rs[process][1] =
              imin((process + 1) * block_size_z_rs - 1,
                   my_fft_grid->npts_global_gspace[2] - 1) -
              imin(process * block_size_z_rs,
                   my_fft_grid->npts_global_gspace[2]) +
              1;
          my_fft_grid->proc2local_y_gs[process][0] = imin(
              process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
          my_fft_grid->proc2local_y_gs[process][1] =
              imin((process + 1) * block_size_y_gs - 1,
                   my_fft_grid->npts_global_gspace[1] - 1) -
              imin(process * block_size_y_gs,
                   my_fft_grid->npts_global_gspace[1]) +
              1;
        }
      } else {
        // In blocked distribution, we use a distributed 3d FFT
        if (my_fft_grid->use_halfspace) {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes_r2c(
              (const int[3]){my_fft_grid->npts_global[2],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[0]},
              my_fft_grid->sub_comm[0], &local_z_rs, &local_z_start_rs,
              &local_y_gs, &local_y_start_gs);
        } else {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes(
              (const int[3]){my_fft_grid->npts_global[2],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[0]},
              my_fft_grid->sub_comm[0], &local_z_rs, &local_z_start_rs,
              &local_y_gs, &local_y_start_gs);
        }
        my_bounds[0] = local_z_start_rs;
        my_bounds[1] = local_z_rs;
        // Exchange the distribution with the other processes
        cp_mpi_allgather_int((const int *)my_bounds, 2,
                             (int *)my_fft_grid->proc2local_z_rs, 2,
                             my_fft_grid->sub_comm[0]);
        my_bounds[0] = local_y_start_gs;
        my_bounds[1] = local_y_gs;
        // Exchange the distribution with the other processes
        cp_mpi_allgather_int((const int *)my_bounds, 2,
                             (int *)my_fft_grid->proc2local_y_gs, 2,
                             my_fft_grid->sub_comm[0]);
      }
      for (int process = 0; process < my_fft_grid->proc_grid[1]; process++) {
        my_fft_grid->proc2local_y_rs[process][0] = 0;
        my_fft_grid->proc2local_y_rs[process][1] =
            my_fft_grid->npts_global_gspace[1];
        my_fft_grid->proc2local_x_gs[process][0] = 0;
        my_fft_grid->proc2local_x_gs[process][1] =
            my_fft_grid->npts_global_gspace[0];
      }
    }
  } else {
    // Serial case or without distributed FFT
    const int block_size_y_rs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_z_rs =
        (my_fft_grid->npts_global_gspace[2] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    const int block_size_x_gs =
        (my_fft_grid->npts_global_gspace[0] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_y_gs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    for (int process = 0; process < my_fft_grid->proc_grid[1]; process++) {
      my_fft_grid->proc2local_y_rs[process][0] =
          imin(process * block_size_y_rs, my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_y_rs[process][1] =
          imin((process + 1) * block_size_y_rs - 1,
               my_fft_grid->npts_global_gspace[1] - 1) -
          imin(process * block_size_y_rs, my_fft_grid->npts_global_gspace[1]) +
          1;
      my_fft_grid->proc2local_x_gs[process][0] =
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]);
      my_fft_grid->proc2local_x_gs[process][1] =
          imin((process + 1) * block_size_x_gs - 1,
               my_fft_grid->npts_global_gspace[0] - 1) -
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]) +
          1;
    }
    for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
      my_fft_grid->proc2local_z_rs[process][0] =
          imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
      my_fft_grid->proc2local_z_rs[process][1] =
          imin((process + 1) * block_size_z_rs - 1,
               my_fft_grid->npts_global_gspace[2] - 1) -
          imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]) +
          1;
      my_fft_grid->proc2local_y_gs[process][0] =
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_y_gs[process][1] =
          imin((process + 1) * block_size_y_gs - 1,
               my_fft_grid->npts_global_gspace[1] - 1) -
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]) +
          1;
    }
  }
  // Then, we collect the mappings of the ranges in each representation RS/MS/GS
  for (int process = 0; process < number_of_processes; process++) {
    int proc_coords[2];
    cp_mpi_cart_coords(my_fft_grid->comm, process, 2, proc_coords);
    // Compile the distribution in real space
    my_fft_grid->proc2local_rs[process][0][0] = 0;
    my_fft_grid->proc2local_rs[process][0][1] = my_fft_grid->npts_global[0];
    my_fft_grid->proc2local_rs[process][1][0] =
        my_fft_grid->proc2local_y_rs[proc_coords[1]][0];
    my_fft_grid->proc2local_rs[process][1][1] =
        my_fft_grid->proc2local_y_rs[proc_coords[1]][1];
    my_fft_grid->proc2local_rs[process][2][0] =
        my_fft_grid->proc2local_z_rs[proc_coords[0]][0];
    my_fft_grid->proc2local_rs[process][2][1] =
        my_fft_grid->proc2local_z_rs[proc_coords[0]][1];
    // Compile the distribution in reciprocal space
    my_fft_grid->proc2local_gs[process][0][0] =
        my_fft_grid->proc2local_x_gs[proc_coords[1]][0];
    my_fft_grid->proc2local_gs[process][0][1] =
        my_fft_grid->proc2local_x_gs[proc_coords[1]][1];
    my_fft_grid->proc2local_gs[process][1][0] =
        my_fft_grid->proc2local_y_gs[proc_coords[0]][0];
    my_fft_grid->proc2local_gs[process][1][1] =
        my_fft_grid->proc2local_y_gs[proc_coords[0]][1];
    my_fft_grid->proc2local_gs[process][2][0] = 0;
    my_fft_grid->proc2local_gs[process][2][1] = my_fft_grid->npts_global[2];
    // Compile the distribution in mixed space
    my_fft_grid->proc2local_ms[process][0][0] =
        my_fft_grid->proc2local_gs[process][0][0];
    my_fft_grid->proc2local_ms[process][0][1] =
        my_fft_grid->proc2local_gs[process][0][1];
    my_fft_grid->proc2local_ms[process][1][0] = 0;
    my_fft_grid->proc2local_ms[process][1][1] = my_fft_grid->npts_global[1];
    my_fft_grid->proc2local_ms[process][2][0] =
        my_fft_grid->proc2local_rs[process][2][0];
    my_fft_grid->proc2local_ms[process][2][1] =
        my_fft_grid->proc2local_rs[process][2][1];
  }
  // Finally, we determine the buffer size
  buffer_size =
      imax(buffer_size, my_fft_grid->npts_global_gspace[0] *
                            my_fft_grid->proc2local_rs[my_process][1][1] *
                            my_fft_grid->proc2local_rs[my_process][2][1]);
  buffer_size =
      imax(buffer_size, my_fft_grid->proc2local_ms[my_process][0][1] *
                            my_fft_grid->npts_global_gspace[1] *
                            my_fft_grid->proc2local_ms[my_process][2][1]);
  buffer_size =
      imax(buffer_size, my_fft_grid->proc2local_gs[my_process][0][1] *
                            my_fft_grid->proc2local_gs[my_process][1][1] *
                            my_fft_grid->npts_global_gspace[2]);
  my_fft_grid->buffer_size = buffer_size;

  if (false && my_process == 0) {
    printf("Proc2local RS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_rs[process][0][0],
             my_fft_grid->proc2local_rs[process][0][0] +
                 my_fft_grid->proc2local_rs[process][0][1] - 1,
             my_fft_grid->proc2local_rs[process][1][0],
             my_fft_grid->proc2local_rs[process][1][0] +
                 my_fft_grid->proc2local_rs[process][1][1] - 1,
             my_fft_grid->proc2local_rs[process][2][0],
             my_fft_grid->proc2local_rs[process][2][0] +
                 my_fft_grid->proc2local_rs[process][2][1] - 1);
    }
    printf("\n");
    printf("Proc2local MS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_ms[process][0][0],
             my_fft_grid->proc2local_ms[process][0][0] +
                 my_fft_grid->proc2local_ms[process][0][1] - 1,
             my_fft_grid->proc2local_ms[process][1][0],
             my_fft_grid->proc2local_ms[process][1][0] +
                 my_fft_grid->proc2local_ms[process][1][1] - 1,
             my_fft_grid->proc2local_ms[process][2][0],
             my_fft_grid->proc2local_ms[process][2][0] +
                 my_fft_grid->proc2local_ms[process][2][1] - 1);
    }
    printf("\n");
    printf("Proc2local GS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_gs[process][0][0],
             my_fft_grid->proc2local_gs[process][0][0] +
                 my_fft_grid->proc2local_gs[process][0][1] - 1,
             my_fft_grid->proc2local_gs[process][1][0],
             my_fft_grid->proc2local_gs[process][1][0] +
                 my_fft_grid->proc2local_gs[process][1][1] - 1,
             my_fft_grid->proc2local_gs[process][2][0],
             my_fft_grid->proc2local_gs[process][2][0] +
                 my_fft_grid->proc2local_gs[process][2][1] - 1);
    }
    printf("\n");
  }
}

void grid_create_fft_grid_layout(fft_grid_layout **fft_grid,
                                 const cp_mpi_comm_t comm,
                                 const int npts_global[3],
                                 const double dh_inv[3][3],
                                 const bool use_halfspace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_create_grid_layout");
  const int handle = fft_start_timer(routine_name);
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
  if (npts_global[2] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
    // Swap dimension if the large process dimension is not on the large global
    // dimension
    if ((npts_global[2] - npts_global[1]) *
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
    my_fft_grid->npts_global_gspace[0] = npts_global[0] / 2 + 1;
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

  const int(*bounds_gs)[2] = my_fft_grid->proc2local_gs[my_process];

  int number_of_positive_gs_points =
      bounds_gs[0][1] * bounds_gs[1][1] * bounds_gs[2][1];
  int number_of_negative_gs_points = 0;
  if (use_halfspace) {
    // If this process carries data from the first element or independently the
    // half index in x-direction, we need to subtract the number of related
    // elements
    int number_of_x_elements = bounds_gs[0][1];
    if (bounds_gs[0][0] == 0)
      number_of_x_elements--;
    if (npts_global[0] % 2 == 0 &&
        bounds_gs[0][0] + bounds_gs[0][1] - 1 == npts_global[0] / 2)
      number_of_x_elements--;
    number_of_negative_gs_points =
        number_of_x_elements * bounds_gs[1][1] * bounds_gs[2][1];
  }
  my_fft_grid->number_of_positive_gs_points = number_of_positive_gs_points;
  my_fft_grid->number_of_negative_gs_points = number_of_negative_gs_points;

  my_fft_grid->npts_gs_local =
      number_of_positive_gs_points + number_of_negative_gs_points;
  my_fft_grid->buffer_size =
      imax(my_fft_grid->buffer_size, my_fft_grid->npts_gs_local);

  my_fft_grid->xy_to_process = NULL;
  my_fft_grid->ray_to_xy = NULL;
  my_fft_grid->xy_to_ray = NULL;
  my_fft_grid->rays_per_process = NULL;
  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
  if (use_halfspace) {
    my_fft_grid->index_to_cart = NULL;
    my_fft_grid->index_to_cart_pos =
        calloc(number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart_neg =
        calloc(number_of_negative_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray = NULL;
    my_fft_grid->index_to_ray_pos =
        calloc(number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray_neg =
        calloc(number_of_negative_gs_points, sizeof(int[2]));
  } else {
    my_fft_grid->index_to_cart =
        calloc(number_of_positive_gs_points, sizeof(int));
    my_fft_grid->index_to_cart_pos = NULL;
    my_fft_grid->index_to_cart_neg = NULL;
    my_fft_grid->index_to_ray =
        calloc(number_of_positive_gs_points, sizeof(int));
    my_fft_grid->index_to_ray_pos = NULL;
    my_fft_grid->index_to_ray_neg = NULL;
  }
  my_fft_grid->index_to_ray = NULL;
  my_fft_grid->index_to_ray_neg = NULL;
  my_fft_grid->index_to_ray_pos = NULL;
  const int local_size_y = bounds_gs[1][1];
  const int local_size_z = bounds_gs[2][1];
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, bounds_gs, local_size_y, local_size_z,                 \
               number_of_positive_gs_points)
  for (int index = 0; index < number_of_positive_gs_points; index++) {
    my_fft_grid->index_to_g[index][0] =
        bounds_gs[0][0] + index / local_size_y / local_size_z;
    assert(my_fft_grid->index_to_g[index][0] <
               my_fft_grid->npts_global_gspace[0] &&
           my_fft_grid->index_to_g[index][0] >= 0);
    my_fft_grid->index_to_g[index][1] =
        bounds_gs[1][0] + (index / local_size_z) % local_size_y;
    my_fft_grid->index_to_g[index][2] = bounds_gs[2][0] + index % local_size_z;
  }
  if (use_halfspace) {
    int negative_index = number_of_positive_gs_points;
    for (int index = 0; index < number_of_positive_gs_points; index++) {
      const int index_x = my_fft_grid->index_to_g[index][0];
      if (index_x == 0 ||
          (npts_global[0] % 2 == 0 && index_x == npts_global[0] / 2))
        continue;
      my_fft_grid->index_to_g[negative_index][0] = npts_global[0] - index_x;
      my_fft_grid->index_to_g[negative_index][1] =
          (npts_global[1] - my_fft_grid->index_to_g[index][1]) % npts_global[1];
      my_fft_grid->index_to_g[negative_index][2] =
          (npts_global[2] - my_fft_grid->index_to_g[index][2]) % npts_global[2];
      negative_index++;
    }
    assert(negative_index == my_fft_grid->npts_gs_local);
  }

  my_fft_grid->local_index_to_ref_grid =
      calloc(my_fft_grid->npts_gs_local, sizeof(int));
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->local_index_to_ref_grid[index] = index;
  }

  sort_g_vectors(my_fft_grid);

  if (use_halfspace) {
    int positive_index = 0;
    int negative_index = 0;
    if (fft_lib_use_mpi() && my_fft_grid->proc_grid[0] > 1 &&
        my_fft_grid->proc_grid[1] == 1) {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
          my_fft_grid->index_to_cart_pos[positive_index][0] = index;
          my_fft_grid->index_to_cart_pos[positive_index][1] =
              ((index_g[1] - bounds_gs[1][0]) * bounds_gs[2][1] +
               (index_g[2] - bounds_gs[2][0])) *
                  bounds_gs[0][1] +
              (index_g[0] - bounds_gs[0][0]);
          printf("%i (+) %i %i %i\n", my_process, index, positive_index,
                 my_fft_grid->index_to_cart_pos[positive_index][1]);
          positive_index++;
        } else {
          my_fft_grid->index_to_cart_neg[negative_index][0] = index;
          my_fft_grid->index_to_cart_neg[negative_index][1] =
              (((npts_global[1] - index_g[1]) % npts_global[1] -
                bounds_gs[1][0]) *
                   bounds_gs[2][1] +
               ((npts_global[2] - index_g[2]) % npts_global[2] -
                bounds_gs[2][0])) *
                  bounds_gs[0][1] +
              ((npts_global[0] - index_g[0]) % npts_global[0] -
               bounds_gs[0][0]);
          printf("%i (-) %i %i %i\n", my_process, index, negative_index,
                 my_fft_grid->index_to_cart_neg[negative_index][1]);
          negative_index++;
        }
      }
    } else {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
          my_fft_grid->index_to_cart_pos[positive_index][0] = index;
          my_fft_grid->index_to_cart_pos[positive_index][1] =
              ((index_g[0] - bounds_gs[0][0]) * bounds_gs[1][1] +
               (index_g[1] - bounds_gs[1][0])) *
                  bounds_gs[2][1] +
              (index_g[2] - bounds_gs[2][0]);
          printf("%i (+) %i %i %i\n", my_process, index, positive_index,
                 my_fft_grid->index_to_cart_pos[positive_index][1]);
          positive_index++;
        } else {
          my_fft_grid->index_to_cart_neg[negative_index][0] = index;
          my_fft_grid->index_to_cart_neg[negative_index][1] =
              (((npts_global[0] - index_g[0]) % npts_global[0] -
                bounds_gs[0][0]) *
                   bounds_gs[1][1] +
               ((npts_global[1] - index_g[1]) % npts_global[1] -
                bounds_gs[1][0])) *
                  bounds_gs[2][1] +
              ((npts_global[2] - index_g[2]) % npts_global[2] -
               bounds_gs[2][0]);
          printf("%i (-) %i %i %i\n", my_process, index, negative_index,
                 my_fft_grid->index_to_cart_neg[negative_index][1]);
          negative_index++;
        }
      }
    }
    assert(positive_index == my_fft_grid->number_of_positive_gs_points);
    assert(negative_index == my_fft_grid->number_of_negative_gs_points);
    fflush(stdout);
    cp_mpi_barrier(comm);
  } else {
    if (fft_lib_use_mpi() && my_fft_grid->proc_grid[0] > 1 &&
        my_fft_grid->proc_grid[1] == 1) {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        my_fft_grid->index_to_cart[index] =
            ((index_g[1] - bounds_gs[1][0]) * bounds_gs[2][1] +
             (index_g[2] - bounds_gs[2][0])) *
                bounds_gs[0][1] +
            (index_g[0] - bounds_gs[0][0]);
      }
    } else {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        my_fft_grid->index_to_cart[index] =
            ((index_g[0] - bounds_gs[0][0]) * bounds_gs[1][1] +
             (index_g[1] - bounds_gs[1][0])) *
                bounds_gs[2][1] +
            (index_g[2] - bounds_gs[2][0]);
      }
    }
  }

  my_fft_grid->redistribution = calloc(1, sizeof(fft_redistribution_t));
  prepare_redistribution(
      my_fft_grid->redistribution, my_fft_grid->npts_global_gspace,
      my_fft_grid->proc2local_x_gs, my_fft_grid->proc2local_y_rs,
      my_fft_grid->proc2local_y_gs, my_fft_grid->proc2local_z_rs,
      my_fft_grid->sub_comm);

  *fft_grid = my_fft_grid;

  fft_stop_timer(handle);
}

void grid_create_fft_grid_layout_from_reference(
    fft_grid_layout **fft_grid, const int npts_global[3],
    const fft_grid_layout *fft_grid_ref) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_create_grid_layout_from_ref");
  const int handle = fft_start_timer(routine_name);
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
  // const int my_process = cp_mpi_comm_rank(fft_grid_ref->comm);

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

  if (npts_global[2] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
    // Swap dimension if the large process dimension is not on the large global
    // dimension
    if ((npts_global[2] - npts_global[1]) *
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
    my_fft_grid->npts_global_gspace[0] = npts_global[0] / 2 + 1;
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

  // Assign the (xy)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->xy_to_process =
      malloc(my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int));
  memset(my_fft_grid->xy_to_process, -1,
         my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int));
  // Count the number of rays on each process
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  int total_number_of_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, fft_grid_ref, npts_global, number_of_processes,        \
               stdout) reduction(+ : total_number_of_rays)
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_x = fft_grid_ref->proc2local_gs[process][0][0];
         index_x <= fft_grid_ref->proc2local_gs[process][0][0] +
                        fft_grid_ref->proc2local_gs[process][0][1] - 1;
         index_x++) {
      // The right half of the indices is shifted
      const int index_x_shifted = convert_c_index_to_shifted_index(
          index_x, fft_grid_ref->npts_global[0]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (!is_on_grid(index_x_shifted, npts_global[0]))
        continue;
      const int index_x_new =
          convert_shifted_index_to_c_index(index_x_shifted, npts_global[0]);
      for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
           index_y <= fft_grid_ref->proc2local_gs[process][1][0] +
                          fft_grid_ref->proc2local_gs[process][1][1] - 1;
           index_y++) {
        // The right half of the indices is shifted
        const int index_y_shifted = convert_c_index_to_shifted_index(
            index_y, fft_grid_ref->npts_global[1]);
        // Same check for z-coordinate
        if (!is_on_grid(index_y_shifted, npts_global[1]))
          continue;
        const int index_y_new =
            convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
        const int xy_index =
            index_x_new * my_fft_grid->npts_global_gspace[1] + index_y_new;
        assert(my_fft_grid->xy_to_process[xy_index] < 0);
        my_fft_grid->xy_to_process[xy_index] = process;
        my_fft_grid->rays_per_process[process]++;
        total_number_of_rays++;
      }
    }
  }
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
      my_fft_grid->npts_gs_local++;
      my_fft_grid->number_of_positive_gs_points += shifted_indices[0] >= 0;
      my_fft_grid->number_of_negative_gs_points += shifted_indices[0] < 0;
    }
  }
  assert(my_fft_grid->npts_gs_local ==
         my_fft_grid->number_of_positive_gs_points +
             my_fft_grid->number_of_negative_gs_points);
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
  my_fft_grid->ray_to_xy = malloc(total_number_of_rays * sizeof(int[2]));
  my_fft_grid->xy_to_ray =
      malloc(my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int[2]));
  memset(my_fft_grid->ray_to_xy, -1, total_number_of_rays * sizeof(int[2]));
  for (int index_x = 0; index_x < fft_grid_ref->npts_global_gspace[0];
       index_x++) {
    const int index_x_shifted =
        convert_c_index_to_shifted_index(index_x, fft_grid_ref->npts_global[0]);
    if (!is_on_grid(index_x_shifted, npts_global[0]))
      continue;
    const int index_x_new =
        convert_shifted_index_to_c_index(index_x_shifted, npts_global[0]);
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
              ->xy_to_process[index_x_new * my_fft_grid->npts_global_gspace[1] +
                              index_y_new];
      const int current_ray_index = ray_index_per_process[current_process];
      const int current_ray_offset = ray_offsets[current_process];
      my_fft_grid->ray_to_xy[current_ray_offset + current_ray_index][0] =
          index_x_new;
      my_fft_grid->ray_to_xy[current_ray_offset + current_ray_index][1] =
          index_y_new;
      my_fft_grid->xy_to_ray[index_x_new * my_fft_grid->npts_global_gspace[1] +
                             index_y_new] = current_ray_index;
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
      assert(my_fft_grid->ray_to_xy[current_ray_offset + ray_index][0] >= 0 &&
             my_fft_grid->ray_to_xy[current_ray_offset + ray_index][1] >= 0 &&
             "The ray has to be assigned to a valid yz index!");
      assert(my_fft_grid->ray_to_xy[current_ray_offset + ray_index][0] <
                 my_fft_grid->npts_global_gspace[0] &&
             my_fft_grid->ray_to_xy[current_ray_offset + ray_index][1] <
                 my_fft_grid->npts_global_gspace[1] &&
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

  own_index = 0;
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

  if (my_fft_grid->use_halfspace) {
    my_fft_grid->index_to_ray = NULL;
    my_fft_grid->index_to_ray_pos =
        calloc(my_fft_grid->number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray_neg =
        calloc(my_fft_grid->number_of_negative_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart = NULL;
    my_fft_grid->index_to_cart_pos =
        calloc(my_fft_grid->number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart_neg =
        calloc(my_fft_grid->number_of_negative_gs_points, sizeof(int[2]));
  } else {
    my_fft_grid->index_to_ray = calloc(my_fft_grid->npts_gs_local, sizeof(int));
    my_fft_grid->index_to_ray_pos = NULL;
    my_fft_grid->index_to_ray_neg = NULL;
    my_fft_grid->index_to_cart =
        calloc(my_fft_grid->npts_gs_local, sizeof(int));
    my_fft_grid->index_to_cart_pos = NULL;
    my_fft_grid->index_to_cart_neg = NULL;
  }
  my_fft_grid->index_to_cart = NULL;
  my_fft_grid->index_to_cart_neg = NULL;
  my_fft_grid->index_to_cart_pos = NULL;
  if (my_fft_grid->use_halfspace) {
    int positive_index = 0;
    int negative_index = 0;
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      int *index_g = my_fft_grid->index_to_g[index];
      if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
        my_fft_grid->index_to_ray_pos[positive_index][0] = index;
        my_fft_grid->index_to_ray_pos[positive_index][1] =
            my_fft_grid->xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                npts_global[2] +
            index_g[2];
        positive_index++;
      } else {
        my_fft_grid->index_to_ray_neg[negative_index][0] = index;
        my_fft_grid->index_to_ray_neg[negative_index][1] =
            my_fft_grid
                    ->xy_to_ray[(npts_global[0] - index_g[0]) * npts_global[1] +
                                (npts_global[1] - index_g[1]) %
                                    npts_global[1]] *
                npts_global[2] +
            (npts_global[2] - index_g[2]) % npts_global[2];
        negative_index++;
      }
    }
  } else {
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      int *index_g = my_fft_grid->index_to_g[index];
      my_fft_grid->index_to_ray[index] =
          my_fft_grid->xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
              npts_global[2] +
          index_g[2];
    }
  }

  my_fft_grid->redistribution = calloc(1, sizeof(fft_redistribution_t));
  prepare_redistribution(
      my_fft_grid->redistribution, my_fft_grid->npts_global_gspace,
      my_fft_grid->proc2local_x_gs, my_fft_grid->proc2local_y_rs,
      my_fft_grid->proc2local_y_gs, my_fft_grid->proc2local_z_rs,
      my_fft_grid->sub_comm);

  *fft_grid = my_fft_grid;
  fft_stop_timer(handle);
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
                layout->proc2local_rs[process][0][0] +
                    layout->proc2local_rs[process][0][1] - 1,
                layout->proc2local_rs[process][1][0],
                layout->proc2local_rs[process][1][0] +
                    layout->proc2local_rs[process][1][1] - 1,
                layout->proc2local_rs[process][2][0],
                layout->proc2local_rs[process][2][0] +
                    layout->proc2local_rs[process][2][1] - 1);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions MS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_ms[process][0][0],
                layout->proc2local_ms[process][0][0] +
                    layout->proc2local_ms[process][0][1] - 1,
                layout->proc2local_ms[process][1][0],
                layout->proc2local_ms[process][1][0] +
                    layout->proc2local_ms[process][1][1] - 1,
                layout->proc2local_ms[process][2][0],
                layout->proc2local_ms[process][2][0] +
                    layout->proc2local_ms[process][2][1] - 1);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions GS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_gs[process][0][0],
                layout->proc2local_gs[process][0][0] +
                    layout->proc2local_gs[process][0][1] - 1,
                layout->proc2local_gs[process][1][0],
                layout->proc2local_gs[process][1][0] +
                    layout->proc2local_gs[process][1][1] - 1,
                layout->proc2local_gs[process][2][0],
                layout->proc2local_gs[process][2][0] +
                    layout->proc2local_gs[process][2][1] - 1);
      }
      if (layout->ray_distribution) {
        for (int index_x = 0; index_x < layout->npts_global_gspace[0];
             index_x++) {
          for (int index_y = 0; index_y < layout->npts_global_gspace[0];
               index_y++) {
            fprintf(
                stdout, "xy-pair %i %i is on process %i\n", index_x, index_y,
                layout->xy_to_process[index_x * layout->npts_global_gspace[1] +
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->ray_distribution) {
    fft_3d_fw_ray(grid_rs, true, grid_gs, grid_layout->npts_gs_local,
                  grid_layout->npts_global, grid_layout->index_to_ray,
                  grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                  grid_layout->proc2local_x_gs, grid_layout->rays_per_process,
                  grid_layout->ray_to_xy, grid_layout->redistribution,
                  grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_fw_blocked(grid_rs, true, grid_gs, grid_layout->index_to_cart,
                      grid_layout->npts_gs_local, grid_layout->npts_global,
                      grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                      grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
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

  ensure_buffer_size(grid_layout->buffer_size);

  fft_3d_fw_blocked(grid_rs, true, grid_gs, NULL, 0, grid_layout->npts_global,
                    grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                    grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                    grid_layout->proc2local_y_gs, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    fft_3d_fw_r2c_blocked(
        grid_rs, grid_gs, NULL, NULL, 0, 0, grid_layout->npts_global,
        grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_fw_blocked((const double complex *)grid_rs, true, grid_gs, NULL, 0,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
  }
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      fft_3d_fw_r2c_ray(
          grid_rs, grid_gs, grid_layout->index_to_ray_pos,
          grid_layout->index_to_ray_neg,
          grid_layout->number_of_positive_gs_points,
          grid_layout->number_of_negative_gs_points, grid_layout->npts_global,
          grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
          grid_layout->rays_per_process, grid_layout->ray_to_xy,
          grid_layout->redistribution, grid_layout->comm,
          grid_layout->sub_comm);
    } else {
      fft_3d_fw_r2c_blocked(
          grid_rs, grid_gs, grid_layout->index_to_cart_pos,
          grid_layout->index_to_cart_neg,
          grid_layout->number_of_positive_gs_points,
          grid_layout->number_of_negative_gs_points, grid_layout->npts_global,
          grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
          grid_layout->redistribution, grid_layout->comm,
          grid_layout->sub_comm);
    }
  } else {
    if (grid_layout->ray_distribution) {
      fft_3d_fw_ray((const double complex *)grid_rs, false, grid_gs,
                    grid_layout->npts_gs_local, grid_layout->npts_global,
                    grid_layout->index_to_ray, grid_layout->proc2local_rs,
                    grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
                    grid_layout->rays_per_process, grid_layout->ray_to_xy,
                    grid_layout->redistribution, grid_layout->comm,
                    grid_layout->sub_comm);
    } else {
      fft_3d_fw_blocked((const double complex *)grid_rs, false, grid_gs,
                        grid_layout->index_to_cart, grid_layout->npts_gs_local,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                        grid_layout->proc2local_x_gs,
                        grid_layout->proc2local_y_gs,
                        grid_layout->redistribution, grid_layout->comm,
                        grid_layout->sub_comm);
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->ray_distribution) {
    fft_3d_bw_ray(
        grid_gs, grid_layout->index_to_ray, grid_layout->npts_gs_local, grid_rs,
        true, grid_layout->npts_global, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
        grid_layout->rays_per_process, grid_layout->ray_to_xy,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_bw_blocked(
        grid_gs, grid_layout->index_to_cart, grid_layout->npts_gs_local,
        grid_rs, true, grid_layout->npts_global, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  }
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

  ensure_buffer_size(grid_layout->buffer_size);

  fft_3d_bw_blocked(grid_gs, NULL, 0, grid_rs, true, grid_layout->npts_global,
                    grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                    grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                    grid_layout->proc2local_y_gs, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      fft_3d_bw_c2r_ray(
          grid_gs, grid_layout->index_to_ray_pos,
          grid_layout->number_of_positive_gs_points, grid_rs,
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_x_gs, grid_layout->rays_per_process,
          grid_layout->ray_to_xy, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
    } else {
      fft_3d_bw_c2r_blocked(
          grid_gs, grid_layout->index_to_cart_pos,
          grid_layout->number_of_positive_gs_points, grid_rs,
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
          grid_layout->proc2local_y_gs, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
    }
  } else {
    if (grid_layout->ray_distribution) {
      fft_3d_bw_ray(grid_gs, grid_layout->index_to_ray,
                    grid_layout->npts_gs_local, (double complex *)grid_rs,
                    false, grid_layout->npts_global, grid_layout->proc2local_rs,
                    grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
                    grid_layout->rays_per_process, grid_layout->ray_to_xy,
                    grid_layout->redistribution, grid_layout->comm,
                    grid_layout->sub_comm);
    } else {
      fft_3d_bw_blocked(
          grid_gs, grid_layout->index_to_cart, grid_layout->npts_gs_local,
          (double complex *)grid_rs, false, grid_layout->npts_global,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
          grid_layout->proc2local_y_gs, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
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

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    fft_3d_bw_c2r_blocked(
        grid_gs, NULL, 0, grid_rs, grid_layout->npts_global,
        grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_bw_blocked(grid_gs, NULL, 0, (double complex *)grid_rs, false,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
  }
}

// EOF
