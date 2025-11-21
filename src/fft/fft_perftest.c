/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../mpiwrap/cp_mpi.h"
#include "fft_grid.h"
#include "fft_grid_layout.h"
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
                              dh_inv, false);

  fft_complex_rs_grid grid_rs;
  grid_create_complex_rs_grid(&grid_rs, grid_layout);
  fft_complex_cart_gs_grid grid_gs;
  grid_create_complex_cart_gs_grid(&grid_gs, grid_layout);

  const int(*my_bound)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  memset(grid_rs.data, 0,
         my_bound[0][1] * my_bound[1][1] * my_bound[2][1] * sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_fw_to_cart(&grid_rs, &grid_gs);
  fft_bw_from_cart(&grid_gs, &grid_rs);
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
    fft_fw_to_cart(&grid_rs, &grid_gs);
    fft_bw_from_cart(&grid_gs, &grid_rs);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  grid_free_complex_rs_grid(&grid_rs);
  grid_free_complex_cart_gs_grid(&grid_gs);
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
                              dh_inv, use_halfspace);

  fft_real_rs_grid grid_rs;
  grid_create_real_rs_grid(&grid_rs, grid_layout);
  fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, grid_layout);

  const int(*my_bound)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  memset(grid_rs.data, 0,
         my_bound[0][1] * my_bound[1][1] * my_bound[2][1] * sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_fw_r2c(&grid_rs, &grid_gs);
  fft_bw_c2r(&grid_gs, &grid_rs);
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
    fft_fw_r2c(&grid_rs, &grid_gs);
    fft_bw_c2r(&grid_gs, &grid_rs);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  grid_free_real_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
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
                              dh_inv, false);
  fft_grid_layout *grid_layout_ray = NULL;
  grid_create_fft_grid_layout_from_reference(&grid_layout_ray, fft_size,
                                             grid_layout);

  fft_complex_rs_grid grid_rs;
  grid_create_complex_rs_grid(&grid_rs, grid_layout_ray);
  fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, grid_layout_ray);

  const int(*my_bound)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  memset(grid_rs.data, 0,
         my_bound[0][1] * my_bound[1][1] * my_bound[2][1] * sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_fw(&grid_rs, &grid_gs);
  fft_bw(&grid_gs, &grid_rs);
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
    fft_fw(&grid_rs, &grid_gs);
    fft_bw(&grid_gs, &grid_rs);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  grid_free_complex_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
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
                              dh_inv, use_halfspace);
  fft_grid_layout *grid_layout_ray = NULL;
  grid_create_fft_grid_layout_from_reference(&grid_layout_ray, fft_size,
                                             grid_layout);

  fft_real_rs_grid grid_rs;
  grid_create_real_rs_grid(&grid_rs, grid_layout_ray);
  fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, grid_layout_ray);

  const int(*my_bound)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(cp_mpi_get_comm_world())];
  memset(grid_rs.data, 0,
         my_bound[0][1] * my_bound[1][1] * my_bound[2][1] * sizeof(double));
  cp_mpi_barrier(cp_mpi_get_comm_world());

  double begin = omp_get_wtime();
  fft_fw_r2c(&grid_rs, &grid_gs);
  fft_bw_c2r(&grid_gs, &grid_rs);
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
    fft_fw_r2c(&grid_rs, &grid_gs);
    fft_bw_c2r(&grid_gs, &grid_rs);
    cp_mpi_barrier(cp_mpi_get_comm_world());
    end = omp_get_wtime();
    const double current_time = end - begin;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
    sum_time_squared += current_time * current_time;
  }

  grid_free_real_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
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

  run_tests(false, FFT_LIB_FFTW, FFT_ESTIMATE, true, true, 0.01);

  // Test also the reference backend and without distributed FFTs from the
  // library
  if (fft_lib_use_mpi()) {
    run_tests(false, FFT_LIB_FFTW, FFT_ESTIMATE, false, true, 0.01);
  }

  if (fft_lib_has_guru_interface()) {
    run_tests(false, FFT_LIB_FFTW, FFT_ESTIMATE, true, false, 0.01);

    // Test also the reference backend and without distributed FFTs from the
    // library
    if (fft_lib_use_mpi()) {
      run_tests(false, FFT_LIB_FFTW, FFT_ESTIMATE, false, false, 0.01);
    }
  }

  fft_finalize_lib(NULL);
  fft_finalize_timer();

  cp_mpi_finalize();

  return 0;
}

// EOF
