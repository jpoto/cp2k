/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_redistribution.h"
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
 * \brief Cleanup the redistribution type
 * \author Frederick Stein
 ******************************************************************************/
void cleanup_redistribution(fft_redistribution_t *redistribution) {
  free(redistribution->displacements_xy_x);
  free(redistribution->displacements_xy_y);
  free(redistribution->counts_xy_x);
  free(redistribution->counts_xy_y);
  free(redistribution->displacements_yzt_y);
  free(redistribution->displacements_yzt_z);
  free(redistribution->counts_yzt_y);
  free(redistribution->counts_yzt_z);
  free(redistribution->displacements_yz_y);
  free(redistribution->displacements_yz_z);
  free(redistribution->counts_yz_y);
  free(redistribution->counts_yz_z);
}

/*******************************************************************************
 * \brief Prepare the redistribution steps
 * \author Frederick Stein
 ******************************************************************************/
void prepare_redistribution(fft_redistribution_t *redistribution,
                            const int npts_global_gspace[3],
                            const int (*proc2local_x_gs)[2],
                            const int (*proc2local_y_rs)[2],
                            const int (*proc2local_y_gs)[2],
                            const int (*proc2local_z_rs)[2],
                            const cp_mpi_comm_t sub_comm[2]) {

  assert(redistribution != NULL);

  cleanup_redistribution(redistribution);

  const int process_grid[2] = {cp_mpi_comm_size(sub_comm[0]),
                               cp_mpi_comm_size(sub_comm[1])};
  const int process_coords[2] = {cp_mpi_comm_rank(sub_comm[0]),
                                 cp_mpi_comm_rank(sub_comm[1])};
  redistribution->my_size_x_gs = proc2local_x_gs[process_coords[1]][1];
  redistribution->my_size_y_rs = proc2local_y_rs[process_coords[1]][1];
  redistribution->my_size_y_gs = proc2local_y_gs[process_coords[0]][1];
  redistribution->my_size_z_rs = proc2local_z_rs[process_coords[0]][1];

  // Copy the general information on the data
  memcpy(redistribution->npts_global_gspace, npts_global_gspace,
         3 * sizeof(int));

  // Setup the redistribution between x and y being local
  redistribution->displacements_xy_x = calloc(process_grid[1], sizeof(int));
  redistribution->displacements_xy_y = calloc(process_grid[1], sizeof(int));
  redistribution->counts_xy_x = calloc(process_grid[1], sizeof(int));
  redistribution->counts_xy_y = calloc(process_grid[1], sizeof(int));
  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < process_grid[1]; process++) {
    // Setup arrays
    redistribution->displacements_xy_x[process] = send_offset;
    redistribution->displacements_xy_y[process] = recv_offset;
    const int current_send_count = proc2local_x_gs[process][1] *
                                   redistribution->my_size_y_rs *
                                   redistribution->my_size_z_rs;
    redistribution->counts_xy_x[process] = current_send_count;
    const int current_recv_count = redistribution->my_size_x_gs *
                                   proc2local_y_rs[process][1] *
                                   redistribution->my_size_z_rs;
    redistribution->counts_xy_y[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == npts_global_gspace[0] * redistribution->my_size_y_rs *
                            redistribution->my_size_z_rs);
  assert(recv_offset == redistribution->my_size_x_gs * npts_global_gspace[1] *
                            redistribution->my_size_z_rs);

  // Next, yz with transposition
  redistribution->displacements_yzt_y = calloc(process_grid[0], sizeof(int));
  redistribution->displacements_yzt_z = calloc(process_grid[0], sizeof(int));
  redistribution->counts_yzt_y = calloc(process_grid[0], sizeof(int));
  redistribution->counts_yzt_z = calloc(process_grid[0], sizeof(int));

  send_offset = 0;
  recv_offset = 0;
  for (int process = 0; process < process_grid[0]; process++) {
    // Setup arrays
    redistribution->displacements_yzt_y[process] = send_offset;
    redistribution->displacements_yzt_z[process] = recv_offset;
    const int current_send_count = redistribution->my_size_x_gs *
                                   proc2local_y_gs[process][1] *
                                   redistribution->my_size_z_rs;
    redistribution->counts_yzt_y[process] = current_send_count;
    const int current_recv_count = redistribution->my_size_x_gs *
                                   redistribution->my_size_y_gs *
                                   proc2local_z_rs[process][1];
    redistribution->counts_yzt_z[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == redistribution->my_size_x_gs * npts_global_gspace[1] *
                            redistribution->my_size_z_rs);
  assert(recv_offset == redistribution->my_size_x_gs *
                            redistribution->my_size_y_gs *
                            npts_global_gspace[2]);

  // yz redistribution non-transposed
  redistribution->displacements_yz_y = calloc(process_grid[0], sizeof(int));
  redistribution->displacements_yz_z = calloc(process_grid[0], sizeof(int));
  redistribution->counts_yz_y = calloc(process_grid[0], sizeof(int));
  redistribution->counts_yz_z = calloc(process_grid[0], sizeof(int));

  send_offset = 0;
  recv_offset = 0;
  for (int process = 0; process < process_grid[0]; process++) {
    // Setup arrays
    redistribution->displacements_yz_y[process] = send_offset;
    redistribution->displacements_yz_z[process] = recv_offset;
    const int current_send_count = redistribution->my_size_x_gs *
                                   proc2local_y_gs[process][1] *
                                   redistribution->my_size_z_rs;
    redistribution->counts_yz_y[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count = redistribution->my_size_x_gs *
                                   redistribution->my_size_y_gs *
                                   proc2local_z_rs[process][1];
    redistribution->counts_yz_z[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == redistribution->my_size_x_gs * npts_global_gspace[1] *
                            redistribution->my_size_z_rs);
  assert(recv_offset == redistribution->my_size_x_gs *
                            redistribution->my_size_y_gs *
                            npts_global_gspace[2]);
}

/*******************************************************************************
 * \brief Performs a transposition of (y_d,z_D,x)->(y,z_D,x_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_x_ms)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  // Reorder the input data to enable MPI_alltoall
  const int number_of_yz_pairs =
      redistribution->my_size_y_rs * redistribution->my_size_z_rs;
  for (int process = 0; process < number_of_processes; process++) {
    const int current_send_size_0 = proc2local_x_ms[process][1];
    double complex *send_buffer =
        transposed + redistribution->displacements_xy_x[process];
    double complex *grid_ptr = grid + proc2local_x_ms[process][0];
    for (int index_yz = 0; index_yz < number_of_yz_pairs; index_yz++) {
      memcpy(send_buffer + index_yz * current_send_size_0,
             grid_ptr + index_yz * redistribution->npts_global_gspace[0],
             current_send_size_0 * sizeof(double complex));
    }
  }
  memcpy(grid, transposed,
         redistribution->npts_global_gspace[0] * number_of_yz_pairs *
             sizeof(double complex));

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_alltoall");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_xy_x,
                                  redistribution->displacements_xy_x,
                                  transposed, redistribution->counts_xy_y,
                                  redistribution->displacements_xy_y, comm);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (y,z_d,x_d) -> (y_d,z_d,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_x_ms)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_x_dist_y_b");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  const int number_of_yz_pairs =
      redistribution->my_size_y_rs * redistribution->my_size_z_rs;

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_alltoall");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_xy_y,
                                  redistribution->displacements_xy_y,
                                  transposed, redistribution->counts_xy_x,
                                  redistribution->displacements_xy_x, comm);
  fft_stop_timer(handle2);

  memcpy(grid, transposed,
         redistribution->npts_global_gspace[0] * number_of_yz_pairs *
             sizeof(double complex));

  for (int process = 0; process < number_of_processes; process++) {
    const int current_recv_size_0 = proc2local_x_ms[process][1];
    double complex *transposed_ptr = transposed + proc2local_x_ms[process][0];
    double complex *recv_buffer =
        grid + redistribution->displacements_xy_x[process];
    for (int index_yz = 0; index_yz < number_of_yz_pairs; index_yz++) {
      memcpy(transposed_ptr + index_yz * redistribution->npts_global_gspace[0],
             recv_buffer + index_yz * current_recv_size_0,
             current_recv_size_0 * sizeof(double complex));
    }
  }

  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (x_d,y,z_D)->(z,x_D,y_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_y_gs)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_z_dist_y_bt");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  memset(transposed, 0,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  for (int process = 0; process < number_of_processes; process++) {
    // Setup arrays
    const int send_size_1 = proc2local_y_gs[process][1];
    double complex *send_buffer =
        transposed + redistribution->displacements_yzt_y[process];
    double complex *grid_ptr =
        grid + proc2local_y_gs[process][0] * redistribution->my_size_z_rs;
    // Use an explicit loop because we need all values in x-direction but not
    // all in y-direction
    for (int index_x = 0; index_x < redistribution->my_size_x_gs; index_x++) {
      transpose_local_complex(
          grid_ptr + index_x * redistribution->npts_global_gspace[1] *
                         redistribution->my_size_z_rs,
          send_buffer + index_x * send_size_1, redistribution->my_size_z_rs,
          send_size_1, redistribution->my_size_z_rs,
          send_size_1 * redistribution->my_size_x_gs);
    }
  }

  memcpy(grid, transposed,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_alltoall");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_yzt_y,
                                  redistribution->displacements_yzt_y,
                                  transposed, redistribution->counts_yzt_z,
                                  redistribution->displacements_yzt_z, comm);
  fft_stop_timer(handle2);

  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (z,x_d,y_d)->(x_d,y,z_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_y_gs)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_z_bt");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_alltoall");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_yzt_z,
                                  redistribution->displacements_yzt_z,
                                  transposed, redistribution->counts_yzt_y,
                                  redistribution->displacements_yzt_y, comm);
  fft_stop_timer(handle2);

  memcpy(grid, transposed,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  for (int process = 0; process < number_of_processes; process++) {
    const int recv_size_1 = proc2local_y_gs[process][1];
    double complex *transposed_ptr =
        transposed + proc2local_y_gs[process][0] * redistribution->my_size_z_rs;
    double complex *recv_buffer =
        grid + redistribution->displacements_yzt_y[process];
    for (int index_x = 0; index_x < redistribution->my_size_x_gs; index_x++) {
      transpose_local_complex(
          recv_buffer + index_x * recv_size_1,
          transposed_ptr + index_x * redistribution->npts_global_gspace[1] *
                               redistribution->my_size_z_rs,
          recv_size_1, redistribution->my_size_z_rs,
          redistribution->my_size_x_gs * recv_size_1,
          redistribution->my_size_z_rs);
    }
  }

  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution of (z_d,x_d,y)->(z,x_d,y_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_y_gs)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_z_dist_y_b");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  memset(transposed, 0,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  for (int process = 0; process < number_of_processes; process++) {
    double complex *grid_ptr = grid + proc2local_y_gs[process][0];
    double complex *send_buffer =
        transposed + redistribution->displacements_yz_y[process];
    for (int index_xz = 0;
         index_xz < redistribution->my_size_x_gs * redistribution->my_size_z_rs;
         index_xz++) {
      memcpy(send_buffer + index_xz * proc2local_y_gs[process][1],
             grid_ptr + index_xz * redistribution->npts_global_gspace[1],
             proc2local_y_gs[process][1] * sizeof(double complex));
    }
  }

  memcpy(grid, transposed,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b_alltoall");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_yz_y,
                                  redistribution->displacements_yz_y,
                                  transposed, redistribution->counts_yz_z,
                                  redistribution->displacements_yz_z, comm);
  fft_stop_timer(handle2);

  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution of (z,x_d,y_d)->(z_d,x_d,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const int (*proc2local_y_gs)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_z_b");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);

  // Use collective MPI communication
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_y_dist_x_b");
  const int handle2 = fft_start_timer(routine_name);
  cp_mpi_alltoallv_double_complex(grid, redistribution->counts_yz_z,
                                  redistribution->displacements_yz_z,
                                  transposed, redistribution->counts_yz_y,
                                  redistribution->displacements_yz_y, comm);
  fft_stop_timer(handle2);

  memcpy(grid, transposed,
         redistribution->my_size_x_gs * redistribution->npts_global_gspace[1] *
             redistribution->my_size_z_rs * sizeof(double complex));

  const int number_of_xz_pairs =
      redistribution->my_size_x_gs * redistribution->my_size_z_rs;
  for (int process = 0; process < number_of_processes; process++) {
    const int recv_size_1 = proc2local_y_gs[process][1];
    double complex *transp = transposed + proc2local_y_gs[process][0];
    double complex *received_data =
        grid + redistribution->displacements_yz_y[process];
    for (int index_xz = 0; index_xz < number_of_xz_pairs; index_xz++) {
      memcpy(transp + index_xz * redistribution->npts_global_gspace[1],
             received_data + index_xz * recv_size_1,
             recv_size_1 * sizeof(double complex));
    }
  }

  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution (z_D, x_D, y) -> (z, xy_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_xy_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_xy)[2],
                                     const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_z_dist_xy_r");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1], my_bounds[1][1], my_bounds[2][1]};
  assert(my_sizes[1] == npts_global[1]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[2] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                   send_request = cp_mpi_get_request_null();
  const int(*my_rays)[2] = ray_to_xy;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];

  memset(transposed, 0,
         my_number_of_rays * npts_global[2] * sizeof(double complex));

  int number_of_received_elements = 0;
  // Copy and transpose the local data
  int number_of_local_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_bounds, my_number_of_rays, my_rays)                              \
    reduction(+ : number_of_local_rays_to_recv)
  for (int xy_ray = 0; xy_ray < my_number_of_rays; xy_ray++) {
    const int index_x = my_rays[xy_ray][0];

    // Check whether we carry that ray before the transposition
    if (index_x >= my_bounds[0][0] &&
        index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
      number_of_local_rays_to_recv++;
    }
  }
  // Copy and transpose the local data
  int number_of_copied_rays = 0;
  for (int xy_ray = 0; xy_ray < my_number_of_rays; xy_ray++) {
    const int index_x = my_rays[xy_ray][0];
    const int index_y = my_rays[xy_ray][1];

    // Check whether we carry that ray before the transposition
    if (index_x >= my_bounds[0][0] &&
        index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
      // Copy the data
      for (int index_z = my_bounds[2][0];
           index_z <= my_bounds[2][0] + my_bounds[2][1] - 1; index_z++) {
        transposed[xy_ray * npts_global[2] + index_z] =
            grid[((index_z - my_bounds[2][0]) * my_sizes[0] +
                  (index_x - my_bounds[0][0])) *
                     my_sizes[1] +
                 index_y - my_bounds[1][0]];
      }
      number_of_copied_rays++;
    }
  }
  assert(number_of_local_rays_to_recv == number_of_copied_rays);
  number_of_received_elements += my_bounds[2][1] * number_of_local_rays_to_recv;

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
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_recv[0][0] &&
          index_x <= proc2local_recv[0][0] + proc2local_recv[0][1] - 1) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv * proc2local_recv[2][1];
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(
        recv_buffer, number_of_elements_to_recv, recv_process, 1, comm);

    // Determine the number of rays to send to the given process
    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_xy;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_x = send_rays[ray][0];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[2];
    // Pack the send buffer
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
    int ray_position = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_x = send_rays[ray][0];
      const int index_y = send_rays[ray][1];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        for (int index_z = 0; index_z < my_sizes[2]; index_z++) {
          send_buffer[ray_position * my_sizes[2] + index_z] =
              grid[(index_z * my_sizes[0] + (index_x - my_bounds[0][0])) *
                       my_sizes[1] +
                   index_y];
        }
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_send);

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer, number_of_elements_to_send, send_process, 1, comm);

    // Wait for the receive process and copy the data
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_recv");
    const int handle2 = fft_start_timer(routine_name);
    cp_mpi_wait(&recv_request);
    fft_stop_timer(handle2);

    // Unpack the received data
    ray_position = 0;
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_recv[0][0] &&
          index_x <= proc2local_recv[0][0] + proc2local_recv[0][1] - 1) {
        memcpy(transposed + ray * npts_global[2] + proc2local_recv[2][0],
               recv_buffer + ray_position * proc2local_recv[2][1],
               proc2local_recv[2][1] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_recv);
    assert(number_of_elements_to_recv ==
           proc2local_recv[2][1] * number_of_rays_to_recv);
    number_of_received_elements +=
        proc2local_recv[2][1] * number_of_rays_to_recv;

    // Wait for the send request
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_send");
    const int handle3 = fft_start_timer(routine_name);
    cp_mpi_wait(&send_request);
    fft_stop_timer(handle3);
  }
  assert(number_of_received_elements == npts_global[2] * my_number_of_rays);

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (z, xy_D) -> (z_D, x_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_xy_and_distribute_z_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local_transposed)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_xy)[2],
                                     const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1];
  assert(my_transposed_sizes[1] == npts_global[1]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[2], product3(my_transposed_sizes));
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
  const int(*my_rays)[2] = ray_to_xy;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int xy_ray = 0; xy_ray < my_number_of_rays; xy_ray++) {
    const int index_x = my_rays[xy_ray][0];
    const int index_y = my_rays[xy_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_x < my_bounds[0][0] ||
        index_x > my_bounds[0][0] + my_bounds[0][1] - 1)
      continue;

    // Copy the data
    for (int index_z = my_bounds[2][0];
         index_z <= my_bounds[2][0] + my_bounds[2][1] - 1; index_z++) {
      transposed[((index_z - my_bounds[2][0]) * my_transposed_sizes[0] +
                  (index_x - my_bounds[0][0])) *
                     my_transposed_sizes[1] +
                 (index_y - my_bounds[1][0])] =
          grid[xy_ray * npts_global[2] + index_z];
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
    const int(*recv_rays)[2] = ray_to_xy;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_x = recv_rays[ray][0];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(
        recv_buffer, my_transposed_sizes[2] * number_of_rays_to_recv,
        recv_process, 1, comm);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_send[0][0] &&
          index_x <= proc2local_send[0][0] + proc2local_send[0][1] - 1) {
        number_of_rays_to_send++;
      }
    }
    int ray_position = 0;
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_send[0][0] &&
          index_x <= proc2local_send[0][0] + proc2local_send[0][1] - 1) {
        memcpy(send_buffer + ray_position * proc2local_send[2][1],
               grid + ray * npts_global[2] + proc2local_send[2][0],
               proc2local_send[2][1] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_send);

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer, number_of_rays_to_send * proc2local_send[2][1],
        send_process, 1, comm);

    // Wait for the receive process and copy the data
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_recv");
    const int handle2 = fft_start_timer(routine_name);
    cp_mpi_wait(&recv_request);
    fft_stop_timer(handle2);

    ray_position = 0;
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_x = recv_rays[ray][0];
      const int index_y = recv_rays[ray][1];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        for (int index_z = 0; index_z < my_transposed_sizes[2]; index_z++) {
          transposed[(index_z * my_transposed_sizes[0] +
                      (index_x - my_bounds[0][0])) *
                         my_transposed_sizes[1] +
                     index_y] =
              recv_buffer[ray_position * my_transposed_sizes[2] + index_z];
        }
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_recv);

    // Wait for the send request
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_send");
    const int handle3 = fft_start_timer(routine_name);
    cp_mpi_wait(&send_request);
    fft_stop_timer(handle3);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a redistribution (x_D, z_D, y) -> (y_D, z, x_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_xy_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int *number_of_rays, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_z_dist_xy_rt");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1], my_bounds[1][1], my_bounds[2][1]};
  assert(my_sizes[1] == npts_global[1]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[2] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  cp_mpi_request_t recv_request = cp_mpi_get_request_null(),
                   send_request = cp_mpi_get_request_null();

  memset(transposed, 0,
         my_number_of_rays * npts_global[2] * sizeof(double complex));

  // Copy and transpose the local data
  for (int xy_ray = 0; xy_ray < my_number_of_rays; xy_ray++) {
    const int index_x = ray_to_xy[my_ray_offset + xy_ray][0];
    const int index_y = ray_to_xy[my_ray_offset + xy_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_x >= my_bounds[0][0] &&
        index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
      // Copy the data
      memcpy(transposed + xy_ray * npts_global[2] + my_bounds[2][0],
             grid + ((index_x - my_bounds[0][0]) * my_sizes[1] + index_y -
                     my_bounds[1][0]) *
                        my_sizes[2],
             my_bounds[2][1] * sizeof(double complex));
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
    shared(my_number_of_rays, my_ray_offset, number_of_rays, ray_to_xy,        \
               proc2local_recv) reduction(+ : number_of_rays_to_recv)
    for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
         ray++) {
      const int index_x = ray_to_xy[ray][0];
      if (index_x >= proc2local_recv[0][0] &&
          index_x <= proc2local_recv[0][0] + proc2local_recv[0][1] - 1) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv * proc2local_recv[2][1];
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(
        recv_buffer, number_of_elements_to_recv, recv_process, 1, comm);

    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_xy;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_x = send_rays[ray][0];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[2];
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
    int ray_position = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_x = send_rays[ray][0];
      const int index_y = send_rays[ray][1];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        memcpy(send_buffer + ray_position * my_bounds[2][1],
               grid + ((index_x - my_bounds[0][0]) * my_sizes[1] + index_y -
                       my_bounds[1][0]) *
                          my_sizes[2],
               my_sizes[2] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_send);

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer, number_of_elements_to_send, send_process, 1, comm);

    // Wait for the receive process and copy the data
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_recv");
    const int handle2 = fft_start_timer(routine_name);
    cp_mpi_wait(&recv_request);
    fft_stop_timer(handle2);

    ray_position = 0;
    for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
         ray++) {
      const int index_x = ray_to_xy[ray][0];
      if (index_x >= proc2local_recv[0][0] &&
          index_x <= proc2local_recv[0][0] + proc2local_recv[0][1] - 1) {
        memcpy(transposed + (ray - my_ray_offset) * npts_global[2] +
                   proc2local_recv[2][0],
               recv_buffer + ray_position * proc2local_recv[2][1],
               proc2local_recv[2][1] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_recv);

    // Wait for the send request
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_send");
    const int handle3 = fft_start_timer(routine_name);
    cp_mpi_wait(&send_request);
    fft_stop_timer(handle3);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a transposition of (x, zy_D) -> (x_D, z_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_xy_and_distribute_z_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local_transposed)[3][2],
    const int *number_of_rays, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_rt");
  const int handle = fft_start_timer(routine_name);
  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1];
  assert(my_transposed_sizes[1] == npts_global[1]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[2], product3(my_transposed_sizes));
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
  const int(*my_rays)[2] = ray_to_xy;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int xy_ray = 0; xy_ray < my_number_of_rays; xy_ray++) {
    const int index_x = my_rays[xy_ray][0];
    const int index_y = my_rays[xy_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_x < my_bounds[0][0] ||
        index_x > my_bounds[0][0] + my_bounds[0][1] - 1)
      continue;

    // Copy the data
    memcpy(transposed + ((index_x - my_bounds[0][0]) * my_transposed_sizes[1] +
                         index_y) *
                            my_transposed_sizes[2],
           grid + xy_ray * npts_global[2] + my_bounds[2][0],
           my_bounds[2][1] * sizeof(double complex));
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int number_of_rays_to_recv = 0;
    const int(*recv_rays)[2] = ray_to_xy;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_x = recv_rays[ray][0];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    recv_request = cp_mpi_irecv_double_complex(
        recv_buffer, my_transposed_sizes[2] * number_of_rays_to_recv,
        recv_process, 1, comm);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_send[0][0] &&
          index_x <= proc2local_send[0][0] + proc2local_send[0][1] - 1) {
        number_of_rays_to_send++;
      }
    }
    int ray_position = 0;
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_x = my_rays[ray][0];
      if (index_x >= proc2local_send[0][0] &&
          index_x <= proc2local_send[0][0] + proc2local_send[0][1] - 1) {
        memcpy(send_buffer + ray_position * proc2local_send[2][1],
               grid + ray * npts_global[2] + proc2local_send[2][0],
               proc2local_send[2][1] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_send);

    // Post send request
    send_request = cp_mpi_isend_double_complex(
        send_buffer, number_of_rays_to_send * proc2local_send[2][1],
        send_process, 1, comm);

    // Wait for the receive process and copy the data
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_recv");
    const int handle2 = fft_start_timer(routine_name);
    cp_mpi_wait(&recv_request);
    fft_stop_timer(handle2);

    ray_position = 0;
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_x = recv_rays[ray][0];
      const int index_y = recv_rays[ray][1];
      if (index_x >= my_bounds[0][0] &&
          index_x <= my_bounds[0][0] + my_bounds[0][1] - 1) {
        memcpy(transposed +
                   ((index_x - my_bounds[0][0]) * my_transposed_sizes[1] +
                    (index_y - my_bounds[1][0])) *
                       my_transposed_sizes[2],
               recv_buffer + ray_position * my_bounds[2][1],
               my_transposed_sizes[2] * sizeof(double complex));
        ray_position++;
      }
    }
    assert(ray_position == number_of_rays_to_recv);

    // Wait for the send request
    memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
    snprintf(routine_name, FFT_MAX_STRING_LENGTH, "coll_xy_dist_z_r_wait_send");
    const int handle3 = fft_start_timer(routine_name);
    cp_mpi_wait(&send_request);
    fft_stop_timer(handle3);
  }

  free(recv_buffer);
  free(send_buffer);
  fft_stop_timer(handle);
}

// EOF
