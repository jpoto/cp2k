/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_driver.h"
#include "fft_lib.h"
#include "fft_redistribution.h"
#include "fft_timer.h"
#include "fft_utils.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_blocked(
    const double complex *restrict grid_rs, const bool is_complex,
    double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int npts_gs_local, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const int (*proc2local_x_gs)[2],
    const int (*proc2local_y_gs)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

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
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT
      if (is_complex) {
        memcpy(grid_buffer_1, grid_rs,
               product3(fft_sizes_rs) * sizeof(double complex));
      } else {
        const double *grid_rs_double = (const double *)grid_rs;
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
      }

      fft_1d_fw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                      false, grid_buffer_1, grid_buffer_2);

      // Pack buffers
      collect_y_and_distribute_x_blocked_pack(grid_buffer_2, grid_buffer_1,
                                              redistribution, proc2local_x_gs);

      // Communicate buffers
      collect_y_and_distribute_x_blocked_comm(grid_buffer_1, grid_buffer_2,
                                              redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      false, grid_buffer_2, grid_buffer_1);

      // Pack the buffer
      collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);
    }

    // Exchange data
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_g != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    if (is_complex) {
      memcpy(grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    fft_2d_fw_local((const int[2]){npts_global[0], npts_global[1]},
                    fft_sizes_rs[2], true, false, grid_buffer_2, grid_buffer_1);

    // Perform second transpose
    collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                            redistribution, proc2local_y_gs);
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_g != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
    }
  } else {
    if (is_complex) {
      memcpy(grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    if (index_to_g != NULL) {
      fft_3d_fw_local(npts_global, grid_buffer_1, grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_3d_fw_local(npts_global, grid_buffer_1, grid_gs);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_r2c_blocked(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    const int (*index_to_g)[3], const int npts_gs_local,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const int (*proc2local_x_gs)[2],
    const int (*proc2local_y_gs)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

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
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT
      memcpy((double *)grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      fft_1d_fw_local_r2c(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, (double *)grid_buffer_1, grid_buffer_2);

      // Pack buffer
      collect_y_and_distribute_x_blocked_pack(grid_buffer_2, grid_buffer_1,
                                              redistribution, proc2local_x_gs);

      // Communicate buffer
      collect_y_and_distribute_x_blocked_comm(grid_buffer_1, grid_buffer_2,
                                              redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      false, grid_buffer_2, grid_buffer_1);

      // Pack the buffer
      collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);
    }

    // Exchange data
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_g != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    memcpy((double *)grid_buffer_1, grid_rs,
           product3(fft_sizes_rs) * sizeof(double));

    // Perform the first FFT
    fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * fft_sizes_rs[2], true,
                        false, (double *)grid_buffer_1, grid_buffer_2);
    fft_1d_fw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_rs[2],
                    true, false, grid_buffer_2, grid_buffer_1);

    // Perform second transpose
    collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                            redistribution, proc2local_y_gs);
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_g != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
    }
  } else {
    memcpy((double *)grid_buffer_2, grid_rs,
           product3(fft_sizes_rs) * sizeof(double));

    // first FFT (x,y,z) -> (x,y,z)
    fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * npts_global[2], true,
                        true, (double *)grid_buffer_2, grid_buffer_1);
    // second FFT (x,y,z) -> (x,y,z)
    fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global_gspace[0], false, false, grid_buffer_1,
                    grid_buffer_2);

    if (index_to_g != NULL) {
      fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                      npts_global_gspace[0], false, false, grid_buffer_1,
                      grid_buffer_2);
      for (int index = 0; index < npts_gs_local; index++) {
        const int *index_g = index_to_g[index];
        grid_gs[index] =
            grid_buffer_2[((index_g[0] - proc2local_gs[my_process][0][0]) *
                               fft_sizes_gs[1] +
                           (index_g[1] - proc2local_gs[my_process][1][0])) *
                              fft_sizes_gs[2] +
                          (index_g[2] - proc2local_gs[my_process][2][0])];
      }
    } else {
      fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                      npts_global_gspace[0], false, false, grid_buffer_1,
                      grid_gs);
    }
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_blocked(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double complex *restrict grid_rs,
    const bool is_complex, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const int (*proc2local_x_gs)[2],
    const int (*proc2local_y_gs)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (index_to_g != NULL) {
    memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[((index[0] - proc2local_gs[my_process][0][0]) *
                         fft_sizes_gs[1] +
                     index[1] - proc2local_gs[my_process][1][0]) *
                        fft_sizes_gs[2] +
                    index[2] - proc2local_gs[my_process][2][0]] = grid_gs[i];
    }
  } else {
    memcpy(grid_buffer_1, grid_gs,
           product3(fft_sizes_gs) * sizeof(double complex));
  }

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT in z-direction
    fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second redistribution and transpose
    // (z,x_d,y_d) -> (x_d,y,z_d)
    collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    if (fft_sizes_rs[2] > 0) {
      collect_y_and_distribute_z_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

      // Perform the second FFT and one transposition (z_D,x_D,y)->(y,z_D,x_D)
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      false, grid_buffer_2, grid_buffer_1);

      // Collect data in z-direction and distribute y-direction
      collect_x_and_distribute_y_blocked_comm(grid_buffer_1, grid_buffer_2,
                                              redistribution, sub_comm[1]);

      // Collect data in z-direction and distribute y-direction
      collect_x_and_distribute_y_blocked_unpack(
          grid_buffer_2, grid_buffer_1, redistribution, proc2local_x_gs);

      // Perform the third FFT and one transposition (y_D,z_D,x)->(x,y_D,z_D)

      if (is_complex) {
        fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_1, grid_rs);
      } else {
        fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_1, grid_buffer_2);
        double *grid_rs_double = (double *)grid_rs;
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_2[i]);
      }
    }
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
    fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Collect data in y-direction and distribute x-direction
    // (z,x,y_d)->(z_d,x,y)
    collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);
    collect_y_and_distribute_z_blocked_unpack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);

    // Perform the second FFT and one transposition (z_d,x,y)->(x,y,z_d)

    if (is_complex) {
      fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                      fft_sizes_ms[2], true, false, grid_buffer_2, grid_rs);
    } else {
      fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                      fft_sizes_ms[2], true, false, grid_buffer_2,
                      grid_buffer_1);
      double *grid_rs_double = (double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }
  } else {
    if (is_complex) {
      fft_3d_bw_local(npts_global, grid_buffer_1, grid_rs);
    } else {
      fft_3d_bw_local(npts_global, grid_buffer_1, grid_buffer_2);
      double *grid_rs_double = (double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_2[i]);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_c2r_blocked(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double *restrict grid_rs,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const int (*proc2local_x_gs)[2],
    const int (*proc2local_y_gs)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (index_to_g != NULL) {
    memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[((index[0] - proc2local_gs[my_process][0][0]) *
                         fft_sizes_gs[1] +
                     index[1] - proc2local_gs[my_process][1][0]) *
                        fft_sizes_gs[2] +
                    index[2] - proc2local_gs[my_process][2][0]] = grid_gs[i];
    }
  } else {
    memcpy(grid_buffer_1, grid_gs,
           product3(fft_sizes_gs) * sizeof(double complex));
  }

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT in z-direction (x_d,y_d,z)->(z,x_d,y_d)
    fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second redistribution and transpose
    // (z,x_d,y_d) -> (x_d,y,z_d)
    collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    if (fft_sizes_rs[2] > 0) {
      collect_y_and_distribute_z_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      false, grid_buffer_2, grid_buffer_1);

      // Collect data in z-direction and distribute y-direction
      collect_x_and_distribute_y_blocked_comm(grid_buffer_1, grid_buffer_2,
                                              redistribution, sub_comm[1]);
      collect_x_and_distribute_y_blocked_unpack(
          grid_buffer_2, grid_buffer_1, redistribution, proc2local_x_gs);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      fft_1d_bw_local_c2r(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, grid_buffer_1, grid_rs);
    }
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
    fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Collect data in y-direction and distribute x-direction (z,x,y_d) ->
    // (z_d,x,y)
    collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);
    collect_y_and_distribute_z_blocked_unpack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);

    // Perform the second FFT and one transposition (z_d,x,y)->(y,z_d,x)
    fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                    false, grid_buffer_2, grid_buffer_1);
    // Perform the second FFT and one transposition (y,z_d,x) -> (x,y,z_d)
    fft_1d_bw_local_c2r(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_1, grid_rs);
  } else {
    // Perform the first two FFTs
    fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global_gspace[0], false, false, grid_buffer_1,
                    grid_buffer_2);
    // And the R2C FFT separately to get rid of additional transposition steps
    fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * npts_global[2], true,
                        true, grid_buffer_2, grid_rs);
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_ray(
    const double complex *restrict grid_rs, const bool is_complex,
    double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int *xy_to_ray, const int npts_gs_local, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_x_gs)[2], const int *rays_per_process,
    const int (*ray_to_xy)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];
  const int(*my_ray_to_xy)[2] = ray_to_xy;
  for (int process = 0; process < my_process; process++) {
    my_ray_to_xy += rays_per_process[process];
  }

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
    if (is_complex) {
      memcpy(grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_1d_fw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                    false, grid_buffer_2, grid_buffer_1);

    // Perform redistribution (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack(grid_buffer_1, grid_buffer_2,
                                            redistribution, proc2local_x_gs);

    // Perform communication (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_xy,
                                    comm);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_g, xy_to_ray, grid_gs, npts_global,         \
               grid_buffer_2)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
    }
  } else if (proc_grid[0] > 1) {
    if (is_complex) {
      memcpy(grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    // Now, we only need a local 2D FFT (x,y,z_d) -> (z_d,x,y)
    fft_2d_fw_local((const int[2]){npts_global[0], npts_global[1]},
                    fft_sizes_ms[2], true, false, grid_buffer_1, grid_buffer_2);

    // Perform second transpose (z_d,x,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_xy,
                                    comm);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_g, xy_to_ray, grid_gs, npts_global,         \
               grid_buffer_2)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
    }
  } else {
    if (is_complex) {
      memcpy(grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    fft_3d_fw_local(npts_global, grid_buffer_2, grid_buffer_1);
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_g, grid_gs, npts_global, grid_buffer_1)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_1[(index_g[0] * npts_global[1] + index_g[1]) *
                            npts_global[2] +
                        index_g[2]];
    }
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_r2c_ray(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    const int (*index_to_g)[3], const int *xy_to_ray, const int npts_gs_local,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_x_gs)[2], const int *rays_per_process,
    const int (*ray_to_xy)[2], const fft_redistribution_t *redistribution,
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];
  const int(*my_ray_to_xy)[2] = ray_to_xy;
  for (int process = 0; process < my_process; process++) {
    my_ray_to_xy += rays_per_process[process];
  }

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
    memcpy((double *)grid_buffer_2, grid_rs,
           product3(fft_sizes_rs) * sizeof(double));

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_1d_fw_local_r2c(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, (double *)grid_buffer_2, grid_buffer_1);

    // Pack buffer for (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack(grid_buffer_1, grid_buffer_2,
                                            redistribution, proc2local_x_gs);

    // Perform communication for (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray(grid_buffer_2, grid_buffer_1,
                                    npts_global_gspace, proc2local_ms,
                                    rays_per_process, ray_to_xy, comm);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, xy_to_ray,         \
               grid_buffer_2)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
    }
  } else if (proc_grid[0] > 1) {
    memcpy((double *)grid_buffer_2, grid_rs,
           product3(fft_sizes_rs) * sizeof(double));

    // The first two FFTs can be performed locally
    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * fft_sizes_ms[2], true,
                        false, (double *)grid_buffer_2, grid_buffer_1);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_1d_fw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
                    true, false, grid_buffer_1, grid_buffer_2);

    // but we need to redistribute to rays (z_d,x,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray(grid_buffer_2, grid_buffer_1,
                                    npts_global_gspace, proc2local_ms,
                                    rays_per_process, ray_to_xy, comm);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, xy_to_ray, npts_global, index_to_g, grid_gs,         \
               grid_buffer_2)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
    }
  } else {
    memcpy((double *)grid_buffer_1, grid_rs,
           product3(fft_sizes_rs) * sizeof(double));

    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * npts_global[2], true,
                        true, (double *)grid_buffer_1, grid_buffer_2);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global_gspace[0], false, false, grid_buffer_2,
                    grid_buffer_1);

// Copy to the ray format
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, grid_buffer_1)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      grid_gs[index] =
          grid_buffer_1[(index_g[0] * npts_global[1] + index_g[1]) *
                            npts_global[2] +
                        index_g[2]];
    }
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_ray(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int *xy_to_ray, const int number_of_local_gpts,
    double complex *restrict grid_rs, const bool is_complex,
    const int npts_global[3], const int (*proc2local_rs)[3][2],
    const int (*proc2local_ms)[3][2], const int (*proc2local_x_gs)[2],
    const int *rays_per_process, const int (*ray_to_xy)[2],
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

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

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_xy,
                                    comm);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    collect_x_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[1]);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    collect_x_and_distribute_y_blocked_unpack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_x_gs);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    if (is_complex) {
      fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                      false, grid_buffer_2, grid_rs);
    } else {
      fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                      false, grid_buffer_2, grid_buffer_1);

      double *grid_rs_double = (double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }
  } else if (proc_grid[0] > 1) {

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    const int(*my_ray_to_xy)[2] = ray_to_xy;
    for (int process = 0; process < my_process; process++)
      my_ray_to_xy += rays_per_process[process];
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose (z,xy_d) -> (z_d,x,y)
    collect_xy_and_distribute_z_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                    proc2local_ms, rays_per_process, ray_to_xy,
                                    comm);

    // Perform the second FFT
    if (is_complex) {
      fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                      fft_sizes_rs[2], true, false, grid_buffer_1, grid_rs);
    } else {
      fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                      fft_sizes_rs[2], true, false, grid_buffer_1,
                      grid_buffer_2);

      double *grid_rs_double = (double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_2[i]);
    }
  } else {

    // Ray distribution
    memset(grid_buffer_2, 0, product3(npts_global) * sizeof(double complex));
    const int(*my_ray_to_xy)[2] = ray_to_xy;
    for (int process = 0; process < my_process; process++)
      my_ray_to_xy += rays_per_process[process];
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_2[(index[0] * npts_global[1] + index[1]) * npts_global[2] +
                    index[2]] = grid_gs[i];
    }

    if (is_complex) {
      fft_3d_bw_local(npts_global, grid_buffer_2, grid_rs);
    } else {
      fft_3d_bw_local(npts_global, grid_buffer_2, grid_buffer_1);

      double *grid_rs_double = (double *)grid_rs;
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_c2r_ray(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int *xy_to_ray, const int number_of_local_gpts,
    double *restrict grid_rs, const int npts_global[3],
    const int npts_global_gspace[3], const int (*proc2local_rs)[3][2],
    const int (*proc2local_ms)[3][2], const int (*proc2local_x_gs)[2],
    const int *rays_per_process, const int (*ray_to_xy)[2],
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

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

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray(grid_buffer_2, grid_buffer_1,
                                    npts_global_gspace, proc2local_ms,
                                    rays_per_process, ray_to_xy, comm);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                    false, grid_buffer_1, grid_buffer_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    collect_x_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[1]);
    // Unpack the buffer
    collect_x_and_distribute_y_blocked_unpack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_x_gs);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    fft_1d_bw_local_c2r(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_2, grid_rs);
  } else if (proc_grid[0] > 1) {

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose (z,xy_d) -> (z_d,x,y)
    collect_xy_and_distribute_z_ray(grid_buffer_2, grid_buffer_1,
                                    npts_global_gspace, proc2local_ms,
                                    rays_per_process, ray_to_xy, comm);

    // second FFT (z_d,x,y) -> (y,z_d,x)
    fft_1d_bw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
                    true, false, grid_buffer_1, grid_buffer_2);
    // third FFT (y,z_d,x) -> (x,y,z_d)
    fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * fft_sizes_ms[2], true,
                        false, grid_buffer_2, grid_rs);
  } else {

    memset(grid_buffer_2, 0,
           product3(npts_global_gspace) * sizeof(double complex));
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_2[(index[0] * npts_global[1] + index[1]) * npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    // second FFT (z_d,x,y) -> (y,z_d,x)
    fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                    npts_global_gspace[0], false, false, grid_buffer_2,
                    grid_buffer_1);
    // third FFT (y,z_d,x) -> (x,y,z_d)
    fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * npts_global[2], true,
                        true, grid_buffer_1, grid_rs);
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

// EOF
