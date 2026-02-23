/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
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
void fft_3d_fw_blocked(
    const double complex *restrict grid_rs, const bool is_complex,
    double complex *restrict grid_gs, const int *index_to_cart,
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

  const int(*my_bounds_rs)[2] = proc2local_rs[my_process];
  const int(*my_bounds_ms)[2] = proc2local_ms[my_process];
  const int(*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);
  const int number_of_points_to_scale =
      index_to_cart != NULL ? npts_gs_local : product3(fft_sizes_gs);
  const int stride_size = 1;

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT
      if (fft_lib_use_mpi()) {
        if (is_complex) {
          transpose_local_complex_block(grid_rs, grid_buffer_2, fft_sizes_rs[1],
                                        fft_sizes_rs[0], fft_sizes_rs[2],
                                        fft_sizes_rs[1], fft_sizes_rs[2],
                                        fft_sizes_rs[0], fft_sizes_rs[2]);
        } else {
          const double *grid_rs_double = (const double *)grid_rs;
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
          transpose_local_complex_block(
              grid_buffer_1, grid_buffer_2, fft_sizes_rs[1], fft_sizes_rs[0],
              fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2],
              fft_sizes_rs[0], fft_sizes_rs[2]);
        }

        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x,y_d,z_d) -> (y_d,x,z_d)
        // 2D FFT (y_d,x,z_d) -> (x_d,y,z_d)
        fft_2d_fw_distributed((const int[2]){npts_global[1], npts_global[0]},
                              fft_sizes_rs[2], sub_comm[1], grid_buffer_2,
                              grid_buffer_1);

        // Perform second redistribution and transpose
        // (x_d,y,z_d) -> (z,x_d,y_d)
        collect_z_and_distribute_y_blocked_transpose_pack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);
      } else {
        if (is_complex) {
          memcpy(grid_buffer_1, grid_rs,
                 product3(fft_sizes_rs) * sizeof(double complex));
        } else {
          const double *grid_rs_double = (const double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_buffer_1, grid_rs_double)
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
        }

        fft_1d_fw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_1, grid_buffer_2);

        // Pack buffers
        collect_y_and_distribute_x_blocked_pack(
            grid_buffer_2, grid_buffer_1, redistribution, proc2local_x_gs);

        // Communicate buffers
        collect_y_and_distribute_x_blocked_comm(grid_buffer_1, grid_buffer_2,
                                                redistribution, sub_comm[1]);

        // Perform the second FFT
        fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                        false, grid_buffer_2, grid_buffer_1);

        // Pack the buffer
        collect_z_and_distribute_y_blocked_pack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);
      }
    }

    // Exchange data
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_cart != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_cart, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_2[index_to_cart[index]];
      }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
      zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
              &stride_size);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    if (fft_lib_use_mpi()) {
      // Perform the distributed 3D FFT in one shot
      // First, we need to change to the correct layout (x,y,z_d) -> (z_d,y,x)
      if (is_complex) {
        transpose_xyz2zyx(grid_rs, grid_buffer_2, fft_sizes_rs[0],
                          fft_sizes_rs[1], fft_sizes_rs[2], fft_sizes_rs[1],
                          fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[0]);
      } else {
        const double *grid_rs_double = (const double *)grid_rs;
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
        transpose_xyz2zyx(grid_buffer_1, grid_buffer_2, fft_sizes_rs[0],
                          fft_sizes_rs[1], fft_sizes_rs[2], fft_sizes_rs[1],
                          fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[0]);
      }
      // 3D FFT (z_D,y,x) -> (y_D,z,x)
      fft_3d_fw_distributed(
          (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
          sub_comm[0], grid_buffer_2, grid_buffer_1);
      // Transpose the data (y_D,z,x) -> (x,y_D,z)
      if (index_to_cart != NULL) {
        #pragma omp parallel for default(none) shared(npts_gs_local, index_to_cart, grid_buffer_1, grid_gs, scaling_factor)
        for (int index = 0; index < npts_gs_local; index++) {
          grid_gs[index] = scaling_factor*grid_buffer_1[index_to_cart[index]];
        }
      } else {
        transpose_local_complex(grid_buffer_1, grid_gs, fft_sizes_gs[0],
                                fft_sizes_gs[1] * fft_sizes_gs[2],
                                fft_sizes_gs[0],
                                fft_sizes_gs[1] * fft_sizes_gs[2]);
      }
    } else {
      if (is_complex) {
        memcpy(grid_buffer_2, grid_rs,
               product3(fft_sizes_rs) * sizeof(double complex));
      } else {
        const double *grid_rs_double = (const double *)grid_rs;
        #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_buffer_2, grid_rs_double)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
      }

      fft_2d_fw_local((const int[2]){npts_global[0], npts_global[1]},
                      fft_sizes_rs[2], true, false, grid_buffer_2,
                      grid_buffer_1);

      // Perform second transpose
      collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);
      collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                              redistribution, sub_comm[0]);

      // Perform the third FFT
      if (index_to_cart != NULL) {
        fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                        false, grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_cart, grid_gs, grid_buffer_2, scaling_factor)
        for (int index = 0; index < npts_gs_local; index++) {
          grid_gs[index] = scaling_factor *
                           grid_buffer_2[index_to_cart[index]];
        }
      } else {
        fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                        false, grid_buffer_1, grid_gs);
        zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
                &stride_size);
      }
    }
  } else {
    if (is_complex) {
      memcpy(grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
        #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_buffer_1, grid_rs_double)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_1[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    if (index_to_cart != NULL) {
      fft_3d_fw_local(npts_global, grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, index_to_cart, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_2[index_to_cart[index]];
      }
    } else {
      fft_3d_fw_local(npts_global, grid_buffer_1, grid_gs);
      zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
              &stride_size);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_blocked(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    const int (*index_to_cart_pos)[2], const int (*index_to_cart_neg)[2],
    const int number_of_positive_points, const int number_of_negative_points,
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

  const int(*my_bounds_rs)[2] = proc2local_rs[my_process];
  const int(*my_bounds_ms)[2] = proc2local_ms[my_process];
  const int(*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);
  const int number_of_points_to_scale =
      index_to_cart_pos != NULL ? number_of_negative_points+number_of_positive_points : product3(fft_sizes_gs);
  const int stride_size = 1;

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT
      if (fft_lib_use_mpi()) {
        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x,y_d,z_d) -> (y_d,x,z_d)
        // Padd the z-direction as required by FFTW
        transpose_local_double_block(
            grid_rs, (double *)grid_buffer_2, fft_sizes_rs[1], npts_global[0],
            fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2],
            2 * npts_global_gspace[0], fft_sizes_rs[2]);
        fft_2d_fw_distributed_r2c(
            (const int[2]){npts_global[1], npts_global[0]}, fft_sizes_rs[2],
            sub_comm[1], (double *)grid_buffer_2, grid_buffer_1);

        // Perform second redistribution and transpose
        // (x_d,y,z_d) -> (z,x_d,y_d)
        collect_z_and_distribute_y_blocked_transpose_pack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);
      } else {
        memcpy((double *)grid_buffer_1, grid_rs,
               product3(fft_sizes_rs) * sizeof(double));

        fft_1d_fw_local_r2c(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                            true, false, (double *)grid_buffer_1,
                            grid_buffer_2);

        // Pack buffer
        collect_y_and_distribute_x_blocked_pack(
            grid_buffer_2, grid_buffer_1, redistribution, proc2local_x_gs);

        // Communicate buffer
        collect_y_and_distribute_x_blocked_comm(grid_buffer_1, grid_buffer_2,
                                                redistribution, sub_comm[1]);

        // Perform the second FFT
        fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                        false, grid_buffer_2, grid_buffer_1);

        // Pack the buffer
        collect_z_and_distribute_y_blocked_pack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);
      }
    }

    // Exchange data
    collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                            redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor *
            grid_buffer_2[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor *
            conj(grid_buffer_2[index_to_cart_neg[index][1]]);
        }
    } else {
      fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_gs);
      zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
              &stride_size);
    }
  } else if (proc_grid[0] > 1) {
    assert(fft_sizes_rs[1] == npts_global[1]);
    if (fft_lib_use_mpi()) {

      // We need to reorder the data because the data is padded for the
      // distributed case (x,y,z_d) -> (z_d,y,x)
      transpose_xyz2zyx_double(grid_rs, (double *)grid_buffer_2, npts_global[0],
                               npts_global[1], fft_sizes_rs[2], fft_sizes_rs[1],
                               fft_sizes_rs[2], fft_sizes_rs[1],
                               2 * npts_global_gspace[0]);
      // Perform the distributed 3D FFT in one shot (z_d,y,x)->(y_D,z,x)
      // Returns transposed layout (z_d,y,x) -> (y_d,z,x)
      fft_3d_fw_distributed_r2c(
          (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
          sub_comm[0], (double *)grid_buffer_2, grid_buffer_1);

      // Exchange the first two dimensions to arrive at the correct layout
      // Transpose the data (y_D,z,x) -> (x,y_D,z)
      if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor *
            grid_buffer_1[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor *
            conj(grid_buffer_1[index_to_cart_neg[index][1]]);
        }
      } else {
        transpose_local_complex(grid_buffer_1, grid_gs, fft_sizes_gs[0],
                                fft_sizes_gs[1] * fft_sizes_gs[2],
                                fft_sizes_gs[0],
                                fft_sizes_gs[1] * fft_sizes_gs[2]);
      }
    } else {
      if (fft_lib_has_guru_interface()) {
        memcpy((double *)grid_buffer_2, grid_rs,
               product3(fft_sizes_rs) * sizeof(double));

        // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
        const fft_iodim dims[2] = {
            {.n = npts_global[1], .is = fft_sizes_ms[2], .os = 1},
            {.n = npts_global[0],
             .is = npts_global[1] * fft_sizes_ms[2],
             .os = npts_global[1]}};
        const fft_iodim howmany_dim = {.n = fft_sizes_ms[2],
                                       .is = 1,
                                       .os = (npts_global[0] / 2 + 1) *
                                             npts_global[1]};
        fft_fw_guru_r2c(2, dims, 1, &howmany_dim, omp_get_max_threads(),
                        (double *)grid_buffer_2, grid_buffer_1);
      } else {
        memcpy((double *)grid_buffer_1, grid_rs,
               product3(fft_sizes_rs) * sizeof(double));

        // Perform the first FFT
        fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * fft_sizes_rs[2],
                            true, false, (double *)grid_buffer_1,
                            grid_buffer_2);
        fft_1d_fw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_rs[2],
                        true, false, grid_buffer_2, grid_buffer_1);
      }

      // Perform second transpose
      collect_z_and_distribute_y_blocked_pack(grid_buffer_1, grid_buffer_2,
                                              redistribution, proc2local_y_gs);
      collect_z_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                              redistribution, sub_comm[0]);

      // Perform the third FFT
      if (index_to_cart_neg != NULL) {
        fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                        false, grid_buffer_1, grid_buffer_2);


#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor *
            grid_buffer_2[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor *
            conj(grid_buffer_2[index_to_cart_neg[index][1]]);
        }
      } else {
        fft_1d_fw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                        false, grid_buffer_1, grid_gs);
        zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
                &stride_size);
      }
    }
  } else {
    if (fft_lib_has_guru_interface()) {
      memcpy((double *)grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      const fft_iodim dims[3] = {
          {.n = npts_global[2], .is = 1, .os = 1},
          {.n = npts_global[1], .is = npts_global[2], .os = npts_global[2]},
          {.n = npts_global[0],
           .is = npts_global[1] * npts_global[2],
           .os = npts_global[1] * npts_global[2]}};
      fft_fw_guru_r2c(3, dims, 0, NULL, omp_get_max_threads(),
                      (double *)grid_buffer_1, grid_buffer_2);

      if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor *
            grid_buffer_2[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_buffer_2, scaling_factor, stderr)
      for (int index = 0; index < number_of_negative_points; index++) {
        fprintf(stderr, "%i %i %i\n", index, index_to_cart_neg[index][0], index_to_cart_neg[index][1]);
        fflush(stderr);
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor *
            conj(grid_buffer_2[index_to_cart_neg[index][1]]);
        }
      } else {
        memcpy(grid_gs, grid_buffer_2,
               product3(npts_global_gspace) * sizeof(double complex));
        zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
                &stride_size);
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

      if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
        fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                        npts_global_gspace[0], false, false, grid_buffer_1,
                        grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor *
            grid_buffer_2[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor *
            conj(grid_buffer_2[index_to_cart_neg[index][1]]);
        }
      } else {
        fft_2d_fw_local((const int[2]){npts_global[1], npts_global[2]},
                        npts_global_gspace[0], false, false, grid_buffer_1,
                        grid_gs);
        zdscal_(&number_of_points_to_scale, &scaling_factor, grid_gs,
                &stride_size);
      }
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked(
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
    //#pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, grid_buffer_1, grid_gs, proc2local_gs, my_process, fft_sizes_gs, npts_global, stdout)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      for (int dir = 0; dir < 3; dir++) {
      assert (index[dir] < npts_global[dir]);
      assert (index[dir] >= 0);
    }
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
      if (fft_lib_use_mpi()) {
        collect_y_and_distribute_z_blocked_transpose_unpack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x_d,y,z_d) -> (y_d,x,z_d)
        fft_2d_bw_distributed((const int[2]){npts_global[1], npts_global[0]},
                              fft_sizes_rs[2], sub_comm[1], grid_buffer_2,
                              grid_buffer_1);
        if (is_complex) {
          transpose_local_complex_block(grid_buffer_1, grid_rs, fft_sizes_rs[0],
                                        fft_sizes_rs[1], fft_sizes_rs[2],
                                        fft_sizes_rs[0], fft_sizes_rs[2],
                                        fft_sizes_rs[1], fft_sizes_rs[2]);
        } else {
          transpose_local_complex_block(
              grid_buffer_1, grid_buffer_2, fft_sizes_rs[0], fft_sizes_rs[1],
              fft_sizes_rs[2], fft_sizes_rs[0], fft_sizes_rs[2],
              fft_sizes_rs[1], fft_sizes_rs[2]);

          double *grid_rs_double = (double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_rs_double[i] = creal(grid_buffer_2[i]);
        }
      } else {
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
          fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, grid_buffer_1, grid_rs);
        } else {
          fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, grid_buffer_1, grid_buffer_2);
          double *grid_rs_double = (double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_rs_double[i] = creal(grid_buffer_2[i]);
        }
      }
    }
  } else if (proc_grid[0] > 1) {
    if (fft_lib_use_mpi()) {
      // Exchange the first two dimensions (x,y_D,z) -> (y_D,z,x)
      transpose_local_complex(
          grid_buffer_1, grid_buffer_2, fft_sizes_gs[1] * fft_sizes_gs[2],
          fft_sizes_gs[0], fft_sizes_gs[1] * fft_sizes_gs[2], fft_sizes_gs[0]);
      // 3D FFT (y_d,z,x) -> (z_d,y,x)
      fft_3d_bw_distributed(
          (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
          sub_comm[0], grid_buffer_2, grid_buffer_1);
      // Transpose back (z_d,y,x) -> (x,y,z_d)

      if (is_complex) {
        transpose_xyz2zyx(grid_buffer_1, grid_rs, fft_sizes_rs[2],
                          fft_sizes_rs[1], fft_sizes_rs[0], fft_sizes_rs[1],
                          fft_sizes_rs[0], fft_sizes_rs[1], fft_sizes_rs[2]);
      } else {
        transpose_xyz2zyx(grid_buffer_1, grid_buffer_2, fft_sizes_rs[2],
                          fft_sizes_rs[1], fft_sizes_rs[0], fft_sizes_rs[1],
                          fft_sizes_rs[0], fft_sizes_rs[1], fft_sizes_rs[2]);
        double *grid_rs_double = (double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_2[i]);
      }
    } else {
      // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
      fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction
      // (z,x,y_d)->(z_d,x,y)
      collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                              redistribution, sub_comm[0]);
      collect_y_and_distribute_z_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

      // Perform the second FFT and one transposition (z_d,x,y)->(x,y,z_d)

      if (is_complex) {
        fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                        fft_sizes_ms[2], true, false, grid_buffer_2, grid_rs);
      } else {
        fft_2d_bw_local((const int[2]){npts_global[0], npts_global[1]},
                        fft_sizes_ms[2], true, false, grid_buffer_2,
                        grid_buffer_1);
        double *grid_rs_double = (double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_1[i]);
      }
    }
  } else {
    if (is_complex) {
      fft_3d_bw_local(npts_global, grid_buffer_1, grid_rs);
    } else {
      fft_3d_bw_local(npts_global, grid_buffer_1, grid_buffer_2);
      double *grid_rs_double = (double *)grid_rs;
          #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
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
void fft_3d_bw_c2r_blocked(
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
          #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, fft_sizes_gs, proc2local_gs, my_process, grid_gs, grid_buffer_1, npts_global)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
        if (index[0] <= npts_global[0]/2) {
      grid_buffer_1[((index[0] - proc2local_gs[my_process][0][0]) *
                         fft_sizes_gs[1] +
                     index[1] - proc2local_gs[my_process][1][0]) *
                        fft_sizes_gs[2] +
                    index[2] - proc2local_gs[my_process][2][0]] = grid_gs[i];
        }
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
      if (fft_lib_use_mpi()) {
        collect_y_and_distribute_z_blocked_transpose_unpack(
            grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x_d,y,z_d) -> (y_d,x,z_d)
        fft_2d_bw_distributed_c2r(
            (const int[2]){npts_global[1], npts_global[0]}, fft_sizes_rs[2],
            sub_comm[1], grid_buffer_2, (double *)grid_buffer_1);
        transpose_local_double_block(
            (const double *)grid_buffer_1, grid_rs, npts_global[0],
            fft_sizes_rs[1], fft_sizes_rs[2], 2 * npts_global_gspace[0],
            fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2]);
      } else {
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
    }
  } else if (proc_grid[0] > 1) {
    if (fft_lib_use_mpi()) {
      // Exchange the first two dimensions (x,y_d,z) -> (y_d,z,x)
      transpose_local_complex(
          grid_buffer_1, grid_buffer_2, fft_sizes_gs[1] * fft_sizes_gs[2],
          fft_sizes_gs[0], fft_sizes_gs[1] * fft_sizes_gs[2], fft_sizes_gs[0]);
      // Distributed FFT (y_d,z,x) -> (z_d,y,x)
      fft_3d_bw_distributed_c2r(
          (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
          sub_comm[0], grid_buffer_2, (double *)grid_buffer_1);
      // Swap indices back (z_d,y,x) -> (x,y,z_d)
      transpose_xyz2zyx_double((const double *)grid_buffer_1, grid_rs,
                               fft_sizes_rs[2], fft_sizes_rs[1], npts_global[0],
                               fft_sizes_rs[1], 2 * npts_global_gspace[0],
                               fft_sizes_rs[1], fft_sizes_rs[2]);
    } else {
      // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
      fft_1d_bw_local(npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1], true,
                      false, grid_buffer_1, grid_buffer_2);

      // Collect data in y-direction and distribute x-direction (z,x,y_d) ->
      // (z_d,x,y)
      collect_y_and_distribute_z_blocked_comm(grid_buffer_2, grid_buffer_1,
                                              redistribution, sub_comm[0]);
      collect_y_and_distribute_z_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_y_gs);

      if (fft_lib_has_guru_interface()) {
        // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
        const fft_iodim dims[2] = {
            {.n = npts_global[1], .is = 1, .os = fft_sizes_ms[2]},
            {.n = npts_global[0],
             .is = npts_global[1],
             .os = npts_global[1] * fft_sizes_ms[2]}};
        const fft_iodim howmany_dim = {.n = fft_sizes_ms[2],
                                       .is = npts_global_gspace[0] *
                                             npts_global_gspace[1],
                                       .os = 1};
        fft_bw_guru_c2r(2, dims, 1, &howmany_dim, omp_get_max_threads(),
                        grid_buffer_2, (double *)grid_buffer_1);
        // Copy back afterwards to prevent OOB accesses
        memcpy(grid_rs, grid_buffer_1, product3(fft_sizes_rs) * sizeof(double));
      } else {
        // Perform the second FFT and one transposition (z_d,x,y)->(y,z_d,x)
        fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                        false, grid_buffer_2, grid_buffer_1);
        // Perform the second FFT and one transposition (y,z_d,x) -> (x,y,z_d)
        fft_1d_bw_local_c2r(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                            true, false, grid_buffer_1, grid_rs);
      }
    }
  } else {
    if (fft_lib_has_guru_interface()) {
      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      const fft_iodim dims[3] = {
          {.n = npts_global[2], .is = 1, .os = 1},
          {.n = npts_global[1], .is = npts_global[2], .os = npts_global[2]},
          {.n = npts_global[0],
           .is = npts_global[1] * npts_global[2],
           .os = npts_global[1] * npts_global[2]}};
      fft_bw_guru_c2r(3, dims, 0, NULL, omp_get_max_threads(), grid_buffer_1,
                      (double *)grid_buffer_2);
      // We need to copy afterwards to prevent OOB access within the Guru
      // interface
      memcpy(grid_rs, (double *)grid_buffer_2,
             product3(fft_sizes_rs) * sizeof(double));
    } else {
      // Perform the first two FFTs
      fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                      npts_global_gspace[0], false, false, grid_buffer_1,
                      grid_buffer_2);
      // And the R2C FFT separately to get rid of additional transposition steps
      fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * npts_global[2], true,
                          true, grid_buffer_2, grid_rs);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_ray(const double complex *restrict grid_rs,
                   const bool is_complex, double complex *restrict grid_gs,
                   const int npts_gs_local, const int npts_global[3], const int *index_to_ray, 
                   const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2],
                   const int (*proc2local_x_gs)[2], const int *rays_per_process,
                   const int (*ray_to_xy)[2],
                   const fft_redistribution_t *redistribution,
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

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      if (fft_sizes_rs[2] > 0) {
        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x,y_d,z_d) -> (y_d,x,z_d)
        if (is_complex) {
          transpose_local_complex_block(grid_rs, grid_buffer_1, fft_sizes_rs[1],
                                        fft_sizes_rs[0], fft_sizes_rs[2],
                                        fft_sizes_rs[1], fft_sizes_rs[2],
                                        fft_sizes_rs[0], fft_sizes_rs[2]);
        } else {
          const double *grid_rs_double = (const double *)grid_rs;
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
          transpose_local_complex_block(
              grid_buffer_2, grid_buffer_1, fft_sizes_rs[1], fft_sizes_rs[0],
              fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2],
              fft_sizes_rs[0], fft_sizes_rs[2]);
        }
        // (y_d,x,z_d) -> (x_d,y,z_d)
        fft_2d_fw_distributed((const int[2]){npts_global[1], npts_global[0]},
                              fft_sizes_rs[2], sub_comm[1], grid_buffer_1,
                              grid_buffer_2);
      }

      // Perform second redistribution and transpose
      // (x_d,y,z_d) -> (z,xy_d)
      collect_z_and_distribute_xy_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_xy, comm);
    } else {
      if (is_complex) {
        memcpy(grid_buffer_2, grid_rs,
               product3(fft_sizes_rs) * sizeof(double complex));
      } else {
        const double *grid_rs_double = (const double *)grid_rs;
        #pragma omp parallel for default(none) shared(fft_sizes_rs,grid_buffer_2,grid_rs_double  )
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
                                      proc2local_ms, rays_per_process,
                                      ray_to_xy, comm);
    }

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, grid_gs, grid_buffer_2, scaling_factor, index_to_ray)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_2[index_to_ray[index]];
    }
  } else if (proc_grid[0] > 1) {
    if (is_complex) {
      memcpy(grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_buffer_1, grid_rs_double)
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

                    printf("Start copying from buffer to gs grid\n");
                    fflush(stdout);
                    cp_mpi_barrier(comm);
                    #pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, grid_gs, grid_buffer_2, scaling_factor, index_to_ray, my_process)
    for (int index = 0; index < npts_gs_local; index++) {
      printf("%i Copy %i to %i: (%f %f)\n", my_process, index, index_to_ray[index], creal(grid_buffer_2[index_to_ray[index]]), cimag(grid_buffer_2[index_to_ray[index]]));
      grid_gs[index] = scaling_factor * grid_buffer_2[index_to_ray[index]];
    }
                    printf("%i Done copying from buffer to gs grid\n", my_process);
                    fflush(stdout);
                    cp_mpi_barrier(comm);
  } else {
    if (is_complex) {
      memcpy(grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const double *grid_rs_double = (const double *)grid_rs;
      #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_buffer_2, grid_rs_double)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_buffer_2[i] = CMPLX(grid_rs_double[i], 0.0);
    }

    fft_3d_fw_local(npts_global, grid_buffer_2, grid_buffer_1);
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, grid_gs, grid_buffer_1, scaling_factor, index_to_ray)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_1[index_to_ray[index]];
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_ray(
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

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_lib_use_mpi()) {
      if (fft_sizes_rs[2] > 0) {
        // Perform the first two FFTs in x- and y-direction
        // Transpose to prepare distributed FFT (x,y_d,z_d) -> (y_d,x,z_d)
        transpose_local_double_block(
            grid_rs, (double *)grid_buffer_1, fft_sizes_rs[1], npts_global[0],
            fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2],
            2 * npts_global_gspace[0], fft_sizes_rs[2]);
        // FFT (y_d,x,z_d) -> (x_d,y,z_d)
        fft_2d_fw_distributed_r2c(
            (const int[2]){npts_global[1], npts_global[0]}, fft_sizes_rs[2],
            sub_comm[1], (double *)grid_buffer_1, grid_buffer_2);
      }

      // Perform second redistribution and transpose
      // (x_d,y,z_d) -> (z,xy_d)
      collect_z_and_distribute_xy_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global_gspace, proc2local_ms,
          rays_per_process, ray_to_xy, comm);

      // Perform the third FFT (z,xy_d) -> (xy_d,z)
      fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                      grid_buffer_1, grid_buffer_2);
    } else {
      memcpy((double *)grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
      fft_1d_fw_local_r2c(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, (double *)grid_buffer_2, grid_buffer_1);

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
    }

#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, xy_to_ray,         \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
        if (convert_c_index_to_shifted_index(index_g[0], npts_global[0]) >= 0) {
      grid_gs[index] =
          scaling_factor *
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
        }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, xy_to_ray,         \
               grid_buffer_2, scaling_factor, my_process)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
        if (convert_c_index_to_shifted_index(index_g[0], npts_global[0]) < 0) {
      grid_gs[index] =
          scaling_factor *
          conj(grid_buffer_2[xy_to_ray[(npts_global[0]-index_g[0])%npts_global[0] * npts_global[1] + (npts_global[1]-index_g[1])%npts_global[1]] *
                            npts_global[2] +(npts_global[2]-
                        index_g[2])%npts_global[2]]);
        }
    }
  } else if (proc_grid[0] > 1) {
    if (fft_lib_has_guru_interface()) {
      memcpy((double *)grid_buffer_1, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      fft_iodim dims[2] = {
          {.n = npts_global[1], .is = fft_sizes_ms[2], .os = 1},
          {.n = npts_global[0],
           .is = npts_global[1] * fft_sizes_ms[2],
           .os = npts_global[1]}};
      fft_iodim howmany_dim = {.n = fft_sizes_ms[2],
                               .is = 1,
                               .os = npts_global_gspace[0] *
                                     npts_global_gspace[1]};
      fft_fw_guru_r2c(2, dims, 1, &howmany_dim, omp_get_max_threads(),
                      (double *)grid_buffer_1, grid_buffer_2);
    } else {
      memcpy((double *)grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      // The first two FFTs can be performed locally
      // FFTs (x,y,z_d) -> (y,z_d,x)
      fft_1d_fw_local_r2c(npts_global[0], npts_global[1] * fft_sizes_ms[2],
                          true, false, (double *)grid_buffer_2, grid_buffer_1);
      // second FFT (y,z_d,x) -> (z_d,x,y)
      fft_1d_fw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
                      true, false, grid_buffer_1, grid_buffer_2);
    }

    // but we need to redistribute to rays (z_d,x,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray(grid_buffer_2, grid_buffer_1,
                                    npts_global_gspace, proc2local_ms,
                                    rays_per_process, ray_to_xy, comm);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_1d_fw_local(npts_global[2], number_of_local_xy_rays, false, false,
                    grid_buffer_1, grid_buffer_2);
                    
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, xy_to_ray, npts_global, index_to_g, grid_gs,         \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
        if (convert_c_index_to_shifted_index(index_g[0], npts_global[0]) >= 0) {
      grid_gs[index] =
          scaling_factor *
          grid_buffer_2[xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                            npts_global[2] +
                        index_g[2]];
                      }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, xy_to_ray, npts_global, index_to_g, grid_gs,         \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
        if (convert_c_index_to_shifted_index(index_g[0], npts_global[0]) < 0) {
      grid_gs[index] =
          scaling_factor *
          conj(grid_buffer_2[xy_to_ray[(npts_global[0]-index_g[0])%npts_global[0]* npts_global[1] + (npts_global[1]-index_g[1])%npts_global[1]] *
                            npts_global[2] +(npts_global[2]-
                        index_g[2])%npts_global[2]]);
                      }
    }
  } else {
    if (fft_lib_has_guru_interface()) {
      memcpy((double *)grid_buffer_2, grid_rs,
             product3(fft_sizes_rs) * sizeof(double));

      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      const fft_iodim dims[3] = {
          {.n = npts_global[2], .is = 1, .os = 1},
          {.n = npts_global[1], .is = npts_global[2], .os = npts_global[2]},
          {.n = npts_global[0],
           .is = npts_global[1] * npts_global[2],
           .os = npts_global[1] * npts_global[2]}};
      fft_fw_guru_r2c(3, dims, 0, NULL, omp_get_max_threads(),
                      (double *)grid_buffer_2, grid_buffer_1);
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
    }

// Copy to the ray format
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, grid_buffer_1,     \
               scaling_factor)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      const int index_x_shifted = convert_c_index_to_shifted_index(index_g[0], npts_global[0]);
        if (index_x_shifted >= 0) {
      grid_gs[index] =
          scaling_factor *
          grid_buffer_1[(index_g[0] * npts_global[1] + index_g[1]) *
                            npts_global[2] +
                        index_g[2]];
        }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, npts_global, index_to_g, grid_gs, grid_buffer_1,     \
               scaling_factor)
    for (int index = 0; index < npts_gs_local; index++) {
      const int *index_g = index_to_g[index];
      const int index_x_shifted = convert_c_index_to_shifted_index(index_g[0], npts_global[0]);
        if (index_x_shifted < 0) {
      grid_gs[index] =
          scaling_factor *
          conj(grid_buffer_1[((npts_global[0]-index_g[0]) * npts_global[1] +(npts_global[1]- index_g[1])%npts_global[1]) *
                            npts_global[2] + (npts_global[2]-
                        index_g[2])%npts_global[2]]);
        }
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_ray(const double complex *restrict grid_gs,
                   const int (*index_to_g)[3], const int *xy_to_ray,
                   const int number_of_local_gpts,
                   double complex *restrict grid_rs, const bool is_complex,
                   const int npts_global[3], const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2],
                   const int (*proc2local_x_gs)[2], const int *rays_per_process,
                   const int (*ray_to_xy)[2],
                   const fft_redistribution_t *redistribution,
                   const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
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
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, grid_buffer_1, xy_to_ray, npts_global, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in z-direction
      fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                      grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (z,xy_d) -> (x_d,y,z_d)
      collect_xy_and_distribute_z_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global, proc2local_ms,
          rays_per_process, ray_to_xy, comm);

      if (fft_sizes_rs[2] > 0) {
        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x_d,y,z_d) -> (y_d,x,z_d)
        fft_2d_bw_distributed((const int[2]){npts_global[1], npts_global[0]},
                              fft_sizes_rs[2], sub_comm[1], grid_buffer_1,
                              grid_buffer_2);

        if (is_complex) {
          transpose_local_complex_block(grid_buffer_2, grid_rs, fft_sizes_rs[0],
                                        fft_sizes_rs[1], fft_sizes_rs[2],
                                        fft_sizes_rs[0], fft_sizes_rs[2],
                                        fft_sizes_rs[1], fft_sizes_rs[2]);
        } else {
          transpose_local_complex_block(
              grid_buffer_2, grid_buffer_1, fft_sizes_rs[0], fft_sizes_rs[1],
              fft_sizes_rs[2], fft_sizes_rs[0], fft_sizes_rs[2],
              fft_sizes_rs[1], fft_sizes_rs[2]);

          double *grid_rs_double = (double *)grid_rs;
           #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
          for (int i = 0; i < product3(fft_sizes_rs); i++)
            grid_rs_double[i] = creal(grid_buffer_1[i]);
        }
      }
    } else {
      // Perform the first FFT (xy_d,z) -> (z,xy_d)
      fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                      grid_buffer_1, grid_buffer_2);

      // Perform transpose (z,xy_d) -> (z_d,x_d,y)
      collect_xy_and_distribute_z_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                      proc2local_ms, rays_per_process,
                                      ray_to_xy, comm);

      // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
      fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true,
                      false, grid_buffer_1, grid_buffer_2);

      // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
      collect_x_and_distribute_y_blocked_comm(grid_buffer_2, grid_buffer_1,
                                              redistribution, sub_comm[1]);

      // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
      collect_x_and_distribute_y_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_x_gs);

      // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
      if (is_complex) {
        fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_2, grid_rs);
      } else {
        fft_1d_bw_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true,
                        false, grid_buffer_2, grid_buffer_1);

        double *grid_rs_double = (double *)grid_rs;
           #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_1[i]);
      }
    }
  } else if (proc_grid[0] > 1) {

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    const int(*my_ray_to_xy)[2] = ray_to_xy;
    for (int process = 0; process < my_process; process++)
      my_ray_to_xy += rays_per_process[process];
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, grid_buffer_1, xy_to_ray, npts_global, grid_gs)
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
           #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_2[i]);
    }
  } else {

    // Ray distribution
    memset(grid_buffer_2, 0, product3(npts_global) * sizeof(double complex));
    const int(*my_ray_to_xy)[2] = ray_to_xy;
    for (int process = 0; process < my_process; process++)
      my_ray_to_xy += rays_per_process[process];
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, grid_buffer_2, xy_to_ray, npts_global, grid_gs)
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
           #pragma omp parallel for default(none) shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
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
void fft_3d_bw_c2r_ray(
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
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, xy_to_ray, grid_buffer_1, npts_global, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      if (index[0] > npts_global[0]/2) continue;
      grid_buffer_1[xy_to_ray[index[0] * npts_global[1] + index[1]] *
                        npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    if (fft_lib_use_mpi()) {
      // Perform the first FFT in x-direction
      fft_1d_bw_local(npts_global[2], number_of_local_xy_rays, false, false,
                      grid_buffer_1, grid_buffer_2);

      // Perform second redistribution and transpose
      // (z,xy_d) -> (x_d,y,z_d)
      collect_xy_and_distribute_z_ray_transpose(
          grid_buffer_2, grid_buffer_1, npts_global_gspace, proc2local_ms,
          rays_per_process, ray_to_xy, comm);

      if (fft_sizes_rs[2] > 0) {
        // Perform the first two FFTs in x- and y-direction
        // transpose the last two indices (is cheaper)
        // (x_d,y,z_d) -> (y_d,x,z_d)
        fft_2d_bw_distributed_c2r(
            (const int[2]){npts_global[1], npts_global[0]}, fft_sizes_rs[2],
            sub_comm[1], grid_buffer_1, (double *)grid_buffer_2);
        transpose_local_double_block(
            (const double *)grid_buffer_2, grid_rs, npts_global[0],
            fft_sizes_rs[1], fft_sizes_rs[2], 2 * npts_global_gspace[0],
            fft_sizes_rs[2], fft_sizes_rs[1], fft_sizes_rs[2]);
      }
    } else {
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
      collect_x_and_distribute_y_blocked_unpack(
          grid_buffer_1, grid_buffer_2, redistribution, proc2local_x_gs);

      // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
      fft_1d_bw_local_c2r(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                          true, false, grid_buffer_2, grid_rs);
    }
  } else if (proc_grid[0] > 1) {

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, xy_to_ray, grid_buffer_1, npts_global, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      if (index[0] > npts_global[0]/2) continue;
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

    if (fft_lib_has_guru_interface()) {
      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      fft_iodim dims[2] = {
          {.n = npts_global[1], .is = 1, .os = fft_sizes_ms[2]},
          {.n = npts_global[0],
           .is = npts_global[1],
           .os = npts_global[1] * fft_sizes_ms[2]}};
      fft_iodim howmany_dim = {.n = fft_sizes_ms[2],
                               .is = npts_global_gspace[0] *
                                     npts_global_gspace[1],
                               .os = 1};
      // two subsequent FFTs (z_d,x,y) -> (x,y,z_d)
      fft_bw_guru_c2r(2, dims, 1, &howmany_dim, omp_get_max_threads(),
                      grid_buffer_1, (double *)grid_buffer_2);
      // We need to copy here to prevent OOB accesses in the Guru interface
      memcpy(grid_rs, (double *)grid_buffer_2,
             product3(fft_sizes_rs) * sizeof(double));
    } else {
      // second FFT (z_d,x,y) -> (y,z_d,x)
      fft_1d_bw_local(npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
                      true, false, grid_buffer_1, grid_buffer_2);
      // third FFT (y,z_d,x) -> (x,y,z_d)
      fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * fft_sizes_ms[2],
                          true, false, grid_buffer_2, grid_rs);
    }
  } else {

    memset(grid_buffer_2, 0,
           product3(npts_global_gspace) * sizeof(double complex));
           #pragma omp parallel for default(none) shared(number_of_local_gpts, index_to_g, grid_buffer_2, npts_global, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      const int *index = index_to_g[i];
      if (index[0] > npts_global[0]/2) continue;
      grid_buffer_2[(index[0] * npts_global[1] + index[1]) * npts_global[2] +
                    index[2]] = grid_gs[i];
    }
    if (fft_lib_has_guru_interface()) {
      // Use the guru interface to merge both 1D FFTs into a single 2D FFT)
      fft_iodim dims[3] = {
          {.n = npts_global[2], .is = 1, .os = 1},
          {.n = npts_global[1], .is = npts_global[2], .os = npts_global[2]},
          {.n = npts_global[0],
           .is = npts_global[1] * npts_global[2],
           .os = npts_global[1] * npts_global[2]}};
      fft_bw_guru_c2r(3, dims, 0, NULL, omp_get_max_threads(), grid_buffer_2,
                      grid_rs);
    } else {
      // second FFT (z_d,x,y) -> (y,z_d,x)
      fft_2d_bw_local((const int[2]){npts_global[1], npts_global[2]},
                      npts_global_gspace[0], false, false, grid_buffer_2,
                      grid_buffer_1);
      // third FFT (y,z_d,x) -> (x,y,z_d)
      fft_1d_bw_local_c2r(npts_global[0], npts_global[1] * npts_global[2], true,
                          true, grid_buffer_1, grid_rs);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

// EOF
