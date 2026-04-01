/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../mpiwrap/cp_mpi.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_timer.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Performance test for the FFT code.
 * \author Frederick Stein
 ******************************************************************************/
static void run_test_c2c(const int fft_size[3], const int number_of_runs) {
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *grid_layout = NULL;
  grid_create_fft_grid_layout(&grid_layout, cp_mpi_get_comm_world(), fft_size,
                              dh_inv, false, -1.0);

  const int(*my_bound_rs)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  const int(*my_bound_gs)[2] =
      grid_layout->proc2local_gs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  double complex *grid_rs;
  fft_allocate_complex(
      my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1], &grid_rs);
  double complex *grid_gs;
  fft_allocate_complex(
      my_bound_gs[0][1] * my_bound_gs[1][1] * my_bound_gs[2][1], &grid_gs);

  memset(grid_rs, 0,
         my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1] *
             sizeof(double complex));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_3d_fw_with_layout_to_cart(grid_rs, grid_gs, grid_layout);
  fft_3d_bw_with_layout_from_cart(grid_gs, grid_rs, grid_layout);
  cp_mpi_barrier(cp_mpi_get_comm_world());
  double end = omp_get_wtime();

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf(
        "Planning time for FW and BW C2C (cart) FFTs of size %i %i %i : %f\n",
        fft_size[0], fft_size[1], fft_size[2], end - begin);
    fflush(stdout);
  }

  double min_time = -1.0;
  double max_time = -1.0;
  double sum_time = 0.0;
  double sum_time_squared = 0.0;
  for (int run = 0; run < number_of_runs; run++) {
    cp_mpi_barrier(cp_mpi_get_comm_world());
    begin = omp_get_wtime();
    fft_3d_fw_with_layout_to_cart(grid_rs, grid_gs, grid_layout);
    fft_3d_bw_with_layout_from_cart(grid_gs, grid_rs, grid_layout);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  fft_free_complex(grid_rs);
  fft_free_complex(grid_gs);
  grid_free_fft_grid_layout(grid_layout);

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Time for %i FW and BW C2C (cart) FFTs of size %i %i %i : min %f, "
           "max %f, avg %f (stdev %f)\n",
           number_of_runs, fft_size[0], fft_size[1], fft_size[2], min_time,
           max_time, sum_time / number_of_runs,
           sqrt((sum_time_squared - sum_time * sum_time / number_of_runs) /
                (number_of_runs - 1)));
  }
}

/*******************************************************************************
 * \brief Performance test for the FFT code.
 * \author Frederick Stein
 ******************************************************************************/
static void run_test_r2c(const int fft_size[3], const int number_of_runs,
                         const bool use_halfspace) {
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *grid_layout = NULL;
  grid_create_fft_grid_layout(&grid_layout, cp_mpi_get_comm_world(), fft_size,
                              dh_inv, use_halfspace, -1.0);

  const int(*my_bound_rs)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  const int(*my_bound_gs)[2] =
      grid_layout->proc2local_gs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  double *grid_rs;
  fft_allocate_double(my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1],
                      &grid_rs);
  double complex *grid_gs;
  fft_allocate_complex(
      my_bound_gs[0][1] * my_bound_gs[1][1] * my_bound_gs[2][1], &grid_gs);

  memset(grid_rs, 0,
         my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1] *
             sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_3d_fw_r2c_with_layout_to_cart(grid_rs, grid_gs, grid_layout);
  fft_3d_bw_c2r_with_layout_from_cart(grid_gs, grid_rs, grid_layout);
  cp_mpi_barrier(cp_mpi_get_comm_world());
  double end = omp_get_wtime();

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Planning time for FW and BW %s FFTs of size %i %i %i : %f\n",
           use_halfspace ? "R2C/C2R" : "C2C", fft_size[0], fft_size[1],
           fft_size[2], end - begin);
    fflush(stdout);
  }

  double min_time = -1.0;
  double max_time = -1.0;
  double sum_time = 0.0;
  double sum_time_squared = 0.0;
  for (int run = 0; run < number_of_runs; run++) {
    cp_mpi_barrier(cp_mpi_get_comm_world());
    begin = omp_get_wtime();
    fft_3d_fw_r2c_with_layout_to_cart(grid_rs, grid_gs, grid_layout);
    fft_3d_bw_c2r_with_layout_from_cart(grid_gs, grid_rs, grid_layout);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }
  fft_free_double(grid_rs);
  fft_free_complex(grid_gs);
  grid_free_fft_grid_layout(grid_layout);

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Time for %i FW and BW %s FFTs of size %i %i %i : min %f, max %f, "
           "avg %f (stdev %f)\n",
           number_of_runs, use_halfspace ? "R2C/C2R" : "C2C", fft_size[0],
           fft_size[1], fft_size[2], min_time, max_time,
           sum_time / number_of_runs,
           sqrt((sum_time_squared - sum_time * sum_time / number_of_runs) /
                (number_of_runs - 1)));
  }
}

/*******************************************************************************
 * \brief Performance test for the FFT code.
 * \author Frederick Stein
 ******************************************************************************/
static void run_test_ray_c2c(const int fft_size[3], const int number_of_runs) {
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *grid_layout = NULL;
  grid_create_fft_grid_layout(&grid_layout, cp_mpi_get_comm_world(), fft_size,
                              dh_inv, false, -1.0);
  fft_grid_layout *grid_layout_ray = NULL;
  grid_create_fft_grid_layout_from_reference(&grid_layout_ray, fft_size, -1.0,
                                             grid_layout);

  const int(*my_bound_rs)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  double complex *grid_rs;
  fft_allocate_complex(
      my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1], &grid_rs);
  double complex *grid_gs;
  fft_allocate_complex(grid_layout_ray->npts_gs_local, &grid_gs);

  memset(grid_rs, 0,
         my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1] *
             sizeof(double complex));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_3d_fw_with_layout(grid_rs, grid_gs, grid_layout_ray);
  fft_3d_bw_with_layout_from_cart(grid_gs, grid_rs, grid_layout_ray);
  cp_mpi_barrier(cp_mpi_get_comm_world());
  double end = omp_get_wtime();

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Planning time for FW and BW C2C (cart) FFTs (ray) of size %i %i %i "
           ": %f\n",
           fft_size[0], fft_size[1], fft_size[2], end - begin);
    fflush(stdout);
  }

  double min_time = -1.0;
  double max_time = -1.0;
  double sum_time = 0.0;
  double sum_time_squared = 0.0;
  for (int run = 0; run < number_of_runs; run++) {
    cp_mpi_barrier(cp_mpi_get_comm_world());
    begin = omp_get_wtime();
    fft_3d_fw_with_layout(grid_rs, grid_gs, grid_layout_ray);
    fft_3d_bw_with_layout_from_cart(grid_gs, grid_rs, grid_layout_ray);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  fft_free_complex(grid_rs);
  fft_free_complex(grid_gs);
  grid_free_fft_grid_layout(grid_layout_ray);
  grid_free_fft_grid_layout(grid_layout);

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Time for %i FW and BW C2C (cart) FFTs (ray) of size %i %i %i : min "
           "%f, "
           "max %f, avg %f (stdev %f)\n",
           number_of_runs, fft_size[0], fft_size[1], fft_size[2], min_time,
           max_time, sum_time / number_of_runs,
           sqrt((sum_time_squared - sum_time * sum_time / number_of_runs) /
                (number_of_runs - 1)));
  }
}

/*******************************************************************************
 * \brief Performance test for the FFT code.
 * \author Frederick Stein
 ******************************************************************************/
static void run_test_ray_r2c(const int fft_size[3], const int number_of_runs,
                             const bool use_halfspace) {
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *grid_layout = NULL;
  grid_create_fft_grid_layout(&grid_layout, cp_mpi_get_comm_world(), fft_size,
                              dh_inv, use_halfspace, -1.0);
  fft_grid_layout *grid_layout_ray = NULL;
  grid_create_fft_grid_layout_from_reference(&grid_layout_ray, fft_size, -1.0,
                                             grid_layout);

  const int(*my_bound_rs)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  double *grid_rs;
  fft_allocate_double(my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1],
                      &grid_rs);
  double complex *grid_gs;
  fft_allocate_complex(grid_layout_ray->npts_gs_local, &grid_gs);

  memset(grid_rs, 0,
         my_bound_rs[0][1] * my_bound_rs[1][1] * my_bound_rs[2][1] *
             sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_3d_fw_r2c_with_layout(grid_rs, grid_gs, grid_layout_ray);
  fft_3d_bw_c2r_with_layout(grid_gs, grid_rs, grid_layout_ray);
  cp_mpi_barrier(cp_mpi_get_comm_world());
  double end = omp_get_wtime();

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Planning time for FW and BW %s FFTs (ray) of size %i %i %i : %f\n",
           use_halfspace ? "R2C/C2R" : "C2C", fft_size[0], fft_size[1],
           fft_size[2], end - begin);
    fflush(stdout);
  }

  double min_time = -1.0;
  double max_time = -1.0;
  double sum_time = 0.0;
  double sum_time_squared = 0.0;
  for (int run = 0; run < number_of_runs; run++) {
    cp_mpi_barrier(cp_mpi_get_comm_world());
    begin = omp_get_wtime();
    fft_3d_fw_r2c_with_layout(grid_rs, grid_gs, grid_layout_ray);
    fft_3d_bw_c2r_with_layout(grid_gs, grid_rs, grid_layout_ray);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  fft_free_double(grid_rs);
  fft_free_complex(grid_gs);
  grid_free_fft_grid_layout(grid_layout_ray);
  grid_free_fft_grid_layout(grid_layout);

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Time for %i FW and BW %s FFTs (ray) of size %i %i %i : min %f, max "
           "%f, "
           "avg %f (stdev %f)\n",
           number_of_runs, use_halfspace ? "R2C/C2R" : "C2C", fft_size[0],
           fft_size[1], fft_size[2], min_time, max_time,
           sum_time / number_of_runs,
           sqrt((sum_time_squared - sum_time * sum_time / number_of_runs) /
                (number_of_runs - 1)));
  }
}

void run_tests(const bool debug, const int backend, const int planning_mode,
               const bool use_mpi, const bool use_guru,
               const double threshold) {
  fft_finalize_timer();
  fft_finalize_lib(NULL);
  fft_init_timer(debug);
  fft_init_lib(backend, planning_mode, use_mpi, use_guru, NULL);

  // These are approximate grid sizes of the finest grid level for the
  // standard benchmark systems in benchmarks/QS
  run_test_c2c((const int[3]){100, 100, 100}, 10);
  run_test_c2c((const int[3]){125, 125, 125}, 10);
  run_test_c2c((const int[3]){160, 160, 160}, 10);
  run_test_c2c((const int[3]){200, 200, 200}, 10);
  // run_test_c2c((const int[3]){256, 256, 256}, 10);
  //  run_test_c2c((const int[3]){315, 315, 315}, 10);
  //  run_test_c2c((const int[3]){400, 400, 400}, 10);
  //  run_test_c2c((const int[3]){500, 500, 500}, 10);
  //  run_test_c2c((const int[3]){630, 630, 630}, 10);
  //   QS_low_scaling_GW
  // run_test_c2c((const int[3]){600, 180, 120}, 10);

  // Repeat using the half-space formalism (R2C/C2R FFTs)
  run_test_r2c((const int[3]){100, 100, 100}, 10, false);
  run_test_r2c((const int[3]){125, 125, 125}, 10, false);
  run_test_r2c((const int[3]){160, 160, 160}, 10, false);
  run_test_r2c((const int[3]){200, 200, 200}, 10, false);
  // run_test_r2c((const int[3]){256, 256, 256}, 10, false);
  //  run_test_r2c((const int[3]){315, 315, 315}, 10, false);
  //  run_test_r2c((const int[3]){400, 400, 400}, 10, false);
  //   run_test_r2c((const int[3]){500, 500, 500}, 10, false);
  //   run_test_r2c((const int[3]){630, 630, 630}, 10, false);
  //    QS_low_scaling_GW
  // run_test_r2c((const int[3]){600, 180, 120}, 10, false);

  // Repeat using the half-space formalism (R2C/C2R FFTs)
  run_test_r2c((const int[3]){100, 100, 100}, 10, true);
  run_test_r2c((const int[3]){125, 125, 125}, 10, true);
  run_test_r2c((const int[3]){160, 160, 160}, 10, true);
  run_test_r2c((const int[3]){200, 200, 200}, 10, true);
  // run_test_r2c((const int[3]){256, 256, 256}, 10, true);
  //  run_test_r2c((const int[3]){315, 315, 315}, 10, true);
  //  run_test_r2c((const int[3]){400, 400, 400}, 10, true);
  //   run_test_r2c((const int[3]){500, 500, 500}, 10, true);
  //   run_test_r2c((const int[3]){630, 630, 630}, 10, true);
  //    QS_low_scaling_GW
  // run_test_r2c((const int[3]){600, 180, 120}, 10, true);

  // Continue with the ray distribution

  // These are approximate grid sizes of the finest grid level for the
  // standard benchmark systems in benchmarks/QS
  run_test_ray_c2c((const int[3]){100, 100, 100}, 10);
  run_test_ray_c2c((const int[3]){125, 125, 125}, 10);
  run_test_ray_c2c((const int[3]){160, 160, 160}, 10);
  run_test_ray_c2c((const int[3]){200, 200, 200}, 10);
  //  run_test_ray_c2c((const int[3]){256, 256, 256}, 10);
  //   run_test_ray_c2c((const int[3]){315, 315, 315}, 10);
  //   run_test_ray_c2c((const int[3]){400, 400, 400}, 10);
  //   run_test_ray_c2c((const int[3]){500, 500, 500}, 10);
  //   run_test_ray_c2c((const int[3]){630, 630, 630}, 10);
  //    QS_low_scaling_GW
  //  run_test_ray_c2c((const int[3]){600, 180, 120}, 10);

  // Repeat using the half-space formalism (R2C/C2R FFTs)
  run_test_ray_r2c((const int[3]){100, 100, 100}, 10, false);
  run_test_ray_r2c((const int[3]){125, 125, 125}, 10, false);
  run_test_ray_r2c((const int[3]){160, 160, 160}, 10, false);
  run_test_ray_r2c((const int[3]){200, 200, 200}, 10, false);
  //  run_test_ray_r2c((const int[3]){256, 256, 256}, 10, false);
  //   run_test_ray_r2c((const int[3]){315, 315, 315}, 10, false);
  //   run_test_ray_r2c((const int[3]){400, 400, 400}, 10, false);
  //    run_test_ray_r2c((const int[3]){500, 500, 500}, 10, false);
  //    run_test_ray_r2c((const int[3]){630, 630, 630}, 10, false);
  //     QS_low_scaling_GW
  //  run_test_ray_r2c((const int[3]){600, 180, 120}, 10, false);

  // Repeat using the half-space formalism (R2C/C2R FFTs)
  run_test_ray_r2c((const int[3]){100, 100, 100}, 10, true);
  run_test_ray_r2c((const int[3]){125, 125, 125}, 10, true);
  run_test_ray_r2c((const int[3]){160, 160, 160}, 10, true);
  run_test_ray_r2c((const int[3]){200, 200, 200}, 10, true);
  //  run_test_ray_r2c((const int[3]){256, 256, 256}, 10, true);
  //   run_test_ray_r2c((const int[3]){315, 315, 315}, 10, true);
  //   run_test_ray_r2c((const int[3]){400, 400, 400}, 10, true);
  //    run_test_ray_r2c((const int[3]){500, 500, 500}, 10, true);
  //    run_test_ray_r2c((const int[3]){630, 630, 630}, 10, true);
  //     QS_low_scaling_GW
  //  run_test_ray_r2c((const int[3]){600, 180, 120}, 10, true);

  fft_print_timing_report(threshold);
}

int main(int argc, char *argv[]) {
  cp_mpi_init(&argc, &argv);

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Number of processes: %i\n",
           cp_mpi_comm_size(cp_mpi_get_comm_world()));
    printf("Number of threads per process: %i\n", omp_get_max_threads());
    fflush(stdout);
  }

  const bool debug = false;
  const int backend = FFT_LIB_FFTW;
  const int planning_mode = FFT_MEASURE;
  const double threshold = 0.01;

  // Test with Guru and MPI backend turned on
  run_tests(debug, backend, planning_mode, true, true, threshold);

  // Test without distributed FFTW but with Guru interface
  if (fft_lib_use_mpi()) {
    run_tests(debug, backend, planning_mode, false, true, threshold);
  }

  if (fft_lib_has_guru_interface()) {
    // Now, test with MPI but without Guru interface
    run_tests(debug, backend, planning_mode, true, false, threshold);

    // Test without MPI and without Guru interface
    if (fft_lib_use_mpi()) {
      run_tests(debug, backend, planning_mode, false, false, threshold);
    }
  }

  fft_finalize_lib(NULL);
  fft_finalize_timer();

  cp_mpi_finalize();

  return 0;
}

// EOF
