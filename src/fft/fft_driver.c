/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_driver.h"
#include "fft_lib.h"
#include "fft_reorder.h"
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

// EOF
