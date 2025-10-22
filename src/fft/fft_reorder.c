/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_reorder.h"
#include "../mpiwrap/cp_mpi.h"
#include "fft_timer.h"
#include "fft_utils.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int npts_global_gspace_2,
    const int (*proc2local)[3][2], const int (*proc2local_transposed)[3][2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_distr_z_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[2] == npts_global[2]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[1] == npts_global[1]);
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));

  // Reorder the input data to enable MPI_alltoall
  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {proc_coord[0], process};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_send_size_2 = proc2local_transposed[rank][2][1] -
                                    proc2local_transposed[rank][2][0] + 1;
    const int current_send_count =
        my_sizes[0] * my_sizes[1] * current_send_size_2;
    send_counts[process] = current_send_count;
    const int current_recv_count =
        my_sizes_transposed[0] *
        (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
        my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
    double complex *send_buffer = transposed + send_displacements[process];
    double complex *grid_ptr =
        grid + proc2local_transposed[rank][2][0] * my_sizes[0] * my_sizes[1];
    const int xz_size = my_sizes[0] * current_send_size_2;
    transpose_local_complex(grid_ptr, send_buffer, my_sizes[1], xz_size,
                            my_sizes[1], xz_size);
  }
  assert(send_offset == my_sizes[0] * my_sizes[1] * npts_global_gspace_2);
  assert(recv_offset == product3(my_sizes_transposed));
  memcpy(grid, transposed,
         my_sizes[0] * my_sizes[1] * npts_global_gspace_2 *
             sizeof(double complex));

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[1]);

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int npts_global_gspace_2,
    const int (*proc2local)[3][2], const int (*proc2local_transposed)[3][2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_z_dist_y_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[2] == npts_global[2]);
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {proc_coord[0], process};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_send_count = my_sizes[0] *
                                   (proc2local_transposed[rank][1][1] -
                                    proc2local_transposed[rank][1][0] + 1) *
                                   my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_size_2 =
        proc2local[rank][2][1] - proc2local[rank][2][0] + 1;
    const int current_recv_count =
        my_sizes_transposed[0] * my_sizes_transposed[1] * current_recv_size_2;
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == my_sizes_transposed[0] * my_sizes_transposed[1] *
                            npts_global_gspace_2);

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[1]);

  memcpy(grid, transposed,
         my_sizes_transposed[0] * my_sizes_transposed[1] *
             npts_global_gspace_2 * sizeof(double complex));

  for (int process = 0; process < dims[1]; process++) {
    const int proc_coords[] = {proc_coord[0], process};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_recv_size_2 =
        proc2local[rank][2][1] - proc2local[rank][2][0] + 1;
    double complex *transposed_ptr = transposed + proc2local[rank][2][0] *
                                                      my_sizes_transposed[0] *
                                                      my_sizes_transposed[1];
    double complex *recv_buffer = grid + recv_displacements[process];
    const int xz_size = my_sizes_transposed[0] * current_recv_size_2;
    transpose_local_complex(recv_buffer, transposed_ptr, xz_size,
                            my_sizes_transposed[1], xz_size,
                            my_sizes_transposed[1]);
  }

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (y_D,z,x_D)->(x,y_D,z_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_x_dist_y_bt_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[0] == npts_global[0]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));

  memset(transposed, 0, product3(my_sizes) * sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int send_size_1 = proc2local_transposed[rank][1][1] -
                            proc2local_transposed[rank][1][0] + 1;
    const int current_send_count = my_sizes[0] * my_sizes[2] * send_size_1;
    send_counts[process] = current_send_count;
    const int current_recv_count =
        (proc2local[rank][0][1] - proc2local[rank][0][0] + 1) *
        my_sizes_transposed[1] * my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
    double complex *send_buffer = transposed + send_displacements[process];
    double complex *grid_ptr =
        grid + proc2local_transposed[rank][1][0] * my_sizes[0];
// Copy the data to the send buffer and exchange the last two indices
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, send_buffer, grid_ptr, send_size_1)
    for (int index_z = 0; index_z < my_sizes[2]; index_z++) {
      for (int index_y = 0; index_y < send_size_1; index_y++) {
        for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
          send_buffer[(index_x * send_size_1 + index_y) * my_sizes[2] +
                      index_z] =
              grid_ptr[(index_z * my_sizes[1] + index_y) * my_sizes[0] +
                       index_x];
        }
      }
    }
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  memcpy(grid, transposed, product3(my_sizes) * sizeof(double complex));

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[0]);

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (x,y_D,z_D)->(y_D,z,x_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_bt_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[0] == npts_global[0]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[1] == npts_global[1]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_recv_count =
        my_sizes_transposed[0] * my_sizes_transposed[2] *
        (proc2local[rank][1][1] - proc2local[rank][1][0] + 1);
    recv_counts[process] = current_recv_count;
    const int current_send_count = (proc2local_transposed[rank][0][1] -
                                    proc2local_transposed[rank][0][0] + 1) *
                                   my_sizes[1] * my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[0]);
  memcpy(grid, transposed,
         product3(my_sizes_transposed) * sizeof(double complex));

  for (int process = 0; process < dims[0]; process++) {
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int recv_size_1 = proc2local[rank][1][1] - proc2local[rank][1][0] + 1;
    double complex *transposed_ptr =
        transposed + proc2local[rank][1][0] * my_sizes_transposed[0];
    double complex *recv_buffer = grid + recv_displacements[process];
// Copy the data to the send buffer and exchange the last two indices
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes_transposed, recv_buffer, transposed_ptr, recv_size_1)
    for (int index_y = 0; index_y < recv_size_1; index_y++) {
      for (int index_z = 0; index_z < my_sizes_transposed[2]; index_z++) {
        for (int index_x = 0; index_x < my_sizes_transposed[0]; index_x++) {
          transposed_ptr[(index_z * my_sizes_transposed[1] + index_y) *
                             my_sizes_transposed[0] +
                         index_x] =
              recv_buffer[(index_x * recv_size_1 + index_y) *
                              my_sizes_transposed[2] +
                          index_z];
        }
      }
    }
  }

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_x_dist_y_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[0] == npts_global[0]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));

  memset(transposed, 0, product3(my_sizes) * sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_send_size_1 = proc2local_transposed[rank][1][1] -
                                    proc2local_transposed[rank][1][0] + 1;
    const int current_send_count =
        my_sizes[0] * current_send_size_1 * my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_size_0 =
        proc2local[rank][0][1] - proc2local[rank][0][0] + 1;
    const int current_recv_count =
        current_recv_size_0 * my_sizes_transposed[1] * my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
    double complex *grid_ptr =
        grid + proc2local_transposed[rank][1][0] * my_sizes[2] * my_sizes[0];
    double complex *send_buffer = transposed + send_displacements[process];
    const int yz_size = current_send_size_1 * my_sizes[2];
    transpose_local_complex(grid_ptr, send_buffer, my_sizes[0], yz_size,
                            my_sizes[0], yz_size);
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  memcpy(grid, transposed, product3(my_sizes) * sizeof(double complex));

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[0]);

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int my_process = cp_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  cp_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[0] == npts_global[0]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[1] == npts_global[1]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int current_send_count = (proc2local_transposed[rank][0][1] -
                                    proc2local_transposed[rank][0][0] + 1) *
                                   my_sizes[1] * my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int recv_size_1 = proc2local[rank][1][1] - proc2local[rank][1][0] + 1;
    const int current_recv_count =
        my_sizes_transposed[0] * recv_size_1 * my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  cp_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                  transposed, recv_counts, recv_displacements,
                                  sub_comm[0]);
  memcpy(grid, transposed,
         product3(my_sizes_transposed) * sizeof(double complex));

  for (int process = 0; process < dims[0]; process++) {
    const int proc_coords[] = {process, proc_coord[1]};
    const int rank = cp_mpi_cart_rank(comm, proc_coords);
    const int recv_size_1 = proc2local[rank][1][1] - proc2local[rank][1][0] + 1;
    double complex *transp = transposed + proc2local[rank][1][0] *
                                              my_sizes_transposed[2] *
                                              my_sizes_transposed[0];
    double complex *received_data = grid + recv_displacements[process];
    const int yz_size = recv_size_1 * my_sizes_transposed[2];
    transpose_local_complex(received_data, transp, yz_size,
                            my_sizes_transposed[0], yz_size,
                            my_sizes_transposed[0]);
  }

  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution (x_D, z_D, y) -> (x, yz_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_yz_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_x_dist_yz_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1] - my_bounds[0][0] + 1,
                           my_bounds[1][1] - my_bounds[1][0] + 1,
                           my_bounds[2][1] - my_bounds[2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[0] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                 send_request = cp_mpi_get_request_null();
  const int(*my_rays)[2] = ray_to_yz;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];

  memset(transposed, 0,
         my_number_of_rays * npts_global[0] * sizeof(double complex));

  int number_of_received_elements = 0;
  // Copy and transpose the local data
  int number_of_local_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_bounds, my_number_of_rays, my_rays)                              \
    reduction(+ : number_of_local_rays_to_recv)
  for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
    const int index_z = my_rays[yz_ray][1];

    // Check whether we carry that ray before the transposition
    if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
      number_of_local_rays_to_recv++;
    }
  }
// Copy and transpose the local data
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, my_number_of_rays, grid, my_rays, transposed,  \
               number_of_local_rays_to_recv)
  for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
    int number_of_copied_rays = 0;
    for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
      const int index_y = my_rays[yz_ray][0];
      const int index_z = my_rays[yz_ray][1];

      // Check whether we carry that ray before the transposition
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        // Copy the data
        transposed[index_x * my_number_of_rays + yz_ray] =
            grid[(index_y - my_bounds[1][0]) * my_sizes[0] * my_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_sizes[0] + index_x -
                 my_bounds[0][0]];
        number_of_copied_rays++;
      }
    }
    assert(number_of_local_rays_to_recv == number_of_copied_rays);
  }
  number_of_received_elements +=
      (my_bounds[0][1] - my_bounds[0][0] + 1) * number_of_local_rays_to_recv;

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    const int(*proc2local_recv)[2] = proc2local[recv_process];

    // Determine the number of rays to receive from the given process
    int number_of_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_recv)                        \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_z = my_rays[ray][1];
      if (index_z >= proc2local_recv[2][0] &&
          index_z <= proc2local_recv[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv *
        (proc2local_recv[0][1] - proc2local_recv[0][0] + 1);
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(recv_buffer, number_of_elements_to_recv,
                                recv_process, 1, comm);

    // Determine the number of rays to send to the given process
    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_yz;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_z = send_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[0];
    // Pack the send buffer
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, number_of_rays_to_send, send_buffer, grid,     \
               send_rays, number_of_rays_send)
    for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_send; ray++) {
        const int index_y = send_rays[ray][0];
        const int index_z = send_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          send_buffer[index_x * number_of_rays_to_send + ray_position] =
              grid[index_y * my_sizes[0] * my_sizes[2] +
                   (index_z - my_bounds[2][0]) * my_sizes[0] + index_x];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    send_request = cp_mpi_isend_double_complex(send_buffer, number_of_elements_to_send,
                                send_process, 1, comm);

    // Wait for the receive process and copy the data
    cp_mpi_wait(&recv_request);

    // Unpack the received data
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, recv_buffer, transposed, proc2local_recv,        \
               number_of_rays_to_recv, my_rays)
    for (int index_x = 0;
         index_x < proc2local_recv[0][1] - proc2local_recv[0][0] + 1;
         index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < my_number_of_rays; ray++) {
        const int index_z = my_rays[ray][1];
        if (index_z >= proc2local_recv[2][0] &&
            index_z <= proc2local_recv[2][1]) {
          transposed[(index_x + proc2local_recv[0][0]) * my_number_of_rays +
                     ray] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }
    assert(number_of_elements_to_recv ==
           (proc2local_recv[0][1] - proc2local_recv[0][0] + 1) *
               number_of_rays_to_recv);
    number_of_received_elements +=
        (proc2local_recv[0][1] - proc2local_recv[0][0] + 1) *
        number_of_rays_to_recv;

    // Wait for the send request
    cp_mpi_wait(&send_request);
  }
  assert(number_of_received_elements == npts_global[0] * my_number_of_rays);

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (x, zy_D) -> (x_D, z_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_yz_and_distribute_x_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local_transposed)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_yz_dist_x_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1] - my_bounds[dir][0] + 1;
  assert(my_transposed_sizes[1] == npts_global[1]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[0], product3(my_transposed_sizes));
  const int my_number_of_rays = number_of_rays[my_process];

  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  double complex *send_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                 send_request = cp_mpi_get_request_null();

  memset(transposed, 0, product3(my_transposed_sizes) * sizeof(double complex));

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  const int(*my_rays)[2] = ray_to_yz;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
    const int index_y = my_rays[yz_ray][0];
    const int index_z = my_rays[yz_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_z < my_bounds[2][0] || index_z > my_bounds[2][1])
      continue;

    // Copy the data
    for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
      transposed[(index_y - my_bounds[1][0]) * my_transposed_sizes[0] *
                     my_transposed_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_transposed_sizes[0] +
                 (index_x - my_bounds[0][0])] =
          grid[index_x * my_number_of_rays + yz_ray];
    }
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int number_of_rays_to_recv = 0;
    const int(*recv_rays)[2] = ray_to_yz;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_z = recv_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(recv_buffer,
                                my_transposed_sizes[0] * number_of_rays_to_recv,
                                recv_process, 1, comm);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_z = my_rays[ray][1];
      if (index_z >= proc2local_send[2][0] &&
          index_z <= proc2local_send[2][1]) {
        number_of_rays_to_send++;
      }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_rays_to_send, send_buffer, grid, my_rays,    \
               proc2local_send, my_number_of_rays)
    for (int index_x = proc2local_send[0][0]; index_x <= proc2local_send[0][1];
         index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < my_number_of_rays; ray++) {
        const int index_z = my_rays[ray][1];
        if (index_z >= proc2local_send[2][0] &&
            index_z <= proc2local_send[2][1]) {
          send_buffer[(index_x - proc2local_send[0][0]) *
                          number_of_rays_to_send +
                      ray_position] = grid[index_x * my_number_of_rays + ray];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer,
        number_of_rays_to_send *
            (proc2local_send[0][1] - proc2local_send[0][0] + 1),
        send_process, 1, comm);

    // Wait for the receive process and copy the data
    cp_mpi_wait(&recv_request);

#pragma omp parallel for default(none) shared(                                 \
        my_transposed_sizes, my_bounds, number_of_rays_to_recv, recv_buffer,   \
            transposed, number_of_rays_recv, recv_rays, number_of_rays)
    for (int index_x = 0; index_x < my_transposed_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_recv; ray++) {
        const int index_y = recv_rays[ray][0];
        const int index_z = recv_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          transposed[index_y * my_transposed_sizes[0] * my_transposed_sizes[2] +
                     (index_z - my_bounds[2][0]) * my_transposed_sizes[0] +
                     index_x] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    cp_mpi_wait(&send_request);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution (x_D, z_D, y) -> (y_D, z, x_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_yz_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int *number_of_rays, const int (*ray_to_yz)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_x_dist_yz_rt_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1] - my_bounds[0][0] + 1,
                           my_bounds[1][1] - my_bounds[1][0] + 1,
                           my_bounds[2][1] - my_bounds[2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[0] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                 send_request = cp_mpi_get_request_null();

  memset(transposed, 0,
         my_number_of_rays * npts_global[0] * sizeof(double complex));

// Copy and transpose the local data
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, my_number_of_rays, grid, ray_to_yz,            \
               transposed, my_ray_offset, npts_global) collapse(2)
  for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
    for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
      const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

      // Check whether we carry that ray after the transposition
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        // Copy the data
        transposed[index_x * my_number_of_rays + yz_ray] =
            grid[(index_z - my_bounds[2][0]) * my_sizes[0] * my_sizes[1] +
                 (index_y - my_bounds[1][0]) * my_sizes[0] + index_x -
                 my_bounds[0][0]];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    const int(*proc2local_recv)[2] = proc2local[recv_process];

    int number_of_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_ray_offset, number_of_rays, ray_to_yz,        \
               proc2local_recv) reduction(+ : number_of_rays_to_recv)
    for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
         ray++) {
      const int index_z = ray_to_yz[ray][1];
      if (index_z >= proc2local_recv[2][0] &&
          index_z <= proc2local_recv[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv *
        (proc2local_recv[0][1] - proc2local_recv[0][0] + 1);
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(recv_buffer, number_of_elements_to_recv,
                                recv_process, 1, comm);

    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_yz;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_z = send_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[0];
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, number_of_rays_to_send, send_buffer, grid,     \
               send_rays, number_of_rays_send)
    for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_send; ray++) {
        const int index_y = send_rays[ray][0];
        const int index_z = send_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          send_buffer[index_x * number_of_rays_to_send + ray_position] =
              grid[index_x + (index_y - my_bounds[1][0]) * my_sizes[0] +
                   (index_z - my_bounds[2][0]) * my_sizes[0] * my_sizes[1]];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    send_request = cp_mpi_isend_double_complex(send_buffer, number_of_elements_to_send,
                                send_process, 1, comm);

    // Wait for the receive process and copy the data
    cp_mpi_wait(&recv_request);

#pragma omp parallel for default(none) shared(                                 \
        my_number_of_rays, npts_global, recv_buffer, transposed,               \
            proc2local_recv, number_of_rays_to_recv, ray_to_yz, my_ray_offset)
    for (int index_x = 0;
         index_x < proc2local_recv[0][1] - proc2local_recv[0][0] + 1;
         index_x++) {
      int ray_position = 0;
      for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
           ray++) {
        const int index_z = ray_to_yz[ray][1];
        if (index_z >= proc2local_recv[2][0] &&
            index_z <= proc2local_recv[2][1]) {
          transposed[(index_x + proc2local_recv[0][0]) * my_number_of_rays +
                     (ray - my_ray_offset)] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    cp_mpi_wait(&send_request);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (x, zy_D) -> (x_D, z_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_yz_and_distribute_x_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local_transposed)[3][2],
    const int *number_of_rays, const int (*ray_to_yz)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_yz_dist_x_rt_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1] - my_bounds[dir][0] + 1;
  assert(my_transposed_sizes[1] == npts_global[1]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[0], product3(my_transposed_sizes));
  const int my_number_of_rays = number_of_rays[my_process];

  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  double complex *send_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                 send_request = cp_mpi_get_request_null();

  memset(transposed, 0, product3(my_transposed_sizes) * sizeof(double complex));

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  const int(*my_rays)[2] = ray_to_yz;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
    const int index_y = my_rays[yz_ray][0];
    const int index_z = my_rays[yz_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_z < my_bounds[2][0] || index_z > my_bounds[2][1])
      continue;

    // Copy the data
    for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
      transposed[(index_z - my_bounds[2][0]) * my_transposed_sizes[0] *
                     my_transposed_sizes[1] +
                 (index_y - my_bounds[1][0]) * my_transposed_sizes[0] +
                 (index_x - my_bounds[0][0])] =
          grid[index_x * my_number_of_rays + yz_ray];
    }
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int number_of_rays_to_recv = 0;
    const int(*recv_rays)[2] = ray_to_yz;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_z = recv_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(recv_buffer,
                                my_transposed_sizes[0] * number_of_rays_to_recv,
                                recv_process, 1, comm);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_z = my_rays[ray][1];
      if (index_z >= proc2local_send[2][0] &&
          index_z <= proc2local_send[2][1]) {
        number_of_rays_to_send++;
      }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_rays_to_send, send_buffer, grid, my_rays,    \
               proc2local_send, my_number_of_rays)
    for (int index_x = proc2local_send[0][0]; index_x <= proc2local_send[0][1];
         index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < my_number_of_rays; ray++) {
        const int index_z = my_rays[ray][1];
        if (index_z >= proc2local_send[2][0] &&
            index_z <= proc2local_send[2][1]) {
          send_buffer[(index_x - proc2local_send[0][0]) *
                          number_of_rays_to_send +
                      ray_position] = grid[index_x * my_number_of_rays + ray];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer,
        number_of_rays_to_send *
            (proc2local_send[0][1] - proc2local_send[0][0] + 1),
        send_process, 1, comm);

    // Wait for the receive process and copy the data
    cp_mpi_wait(&recv_request);

#pragma omp parallel for default(none) shared(                                 \
        my_transposed_sizes, my_bounds, number_of_rays_to_recv, recv_buffer,   \
            transposed, number_of_rays_recv, recv_rays, number_of_rays)
    for (int index_x = 0; index_x < my_transposed_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_recv; ray++) {
        const int index_y = recv_rays[ray][0];
        const int index_z = recv_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          transposed[(index_z - my_bounds[2][0]) * my_transposed_sizes[0] *
                         my_transposed_sizes[1] +
                     (index_y - my_bounds[1][0]) * my_transposed_sizes[0] +
                     index_x] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    cp_mpi_wait(&send_request);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

// EOF
