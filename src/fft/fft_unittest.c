/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../mpiwrap/cp_mpi.h"
#include "../offload/offload_library.h"
#include "fft_grid_test.h"
#include "fft_lib.h"
#include "fft_lib_test.h"
#include "fft_redistribution_test.h"
#include "fft_timer.h"

int run_unittests(const bool debug, const int backend, const int planning_mode,
              const bool use_mpi, const bool use_guru, const double threshold) {
  int errors = 0;
  fft_finalize_timer();
  fft_finalize_lib(NULL);

  fflush(stdout);
  cp_mpi_barrier(cp_mpi_get_comm_world());

  fft_init_timer(debug);
  fft_init_acc_lib();
  fft_init_lib(backend, planning_mode, use_mpi, use_guru, NULL);

  errors += fft_test_local();
  errors += fft_test_distributed();
  errors += fft_test_transpose();
  errors += fft_test_transpose_parallel();
  errors += fft_test_3d();
  fft_print_timing_report(threshold);

  fft_finalize_lib(NULL);
  fft_finalize_acc_lib();

  return errors;
}

int main(int argc, char *argv[]) {
  cp_mpi_init(&argc, &argv);

  offload_set_chosen_device(0);

  const bool debug = false;
  const int backend = FFT_LIB_FFTW;
  const int planning_mode = FFT_MEASURE;

  if (cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0) {
    printf("Number of MPI ranks: %i\n", cp_mpi_comm_size(cp_mpi_get_comm_world()));
    int number_of_threads = 1;
    #pragma omp parallel default(none) shared(number_of_threads)
    {
    #pragma omp single
    {
    number_of_threads = omp_get_num_threads();
    }
  }
    printf("Number of OpenMP threads: %i\n", number_of_threads);
    fflush(stdout);
  }
  cp_mpi_barrier(cp_mpi_get_comm_world());

  int errors = run_unittests(debug, backend, planning_mode, true, true, 0.01);

  // Test also the reference backend and without distributed FFTs from the
  // library
  if (fft_lib_use_mpi()) {
    errors += run_unittests(debug, backend, planning_mode, false, true, 0.01);
  }

  if (fft_lib_has_guru_interface()) {

    int errors = run_unittests(debug, backend, planning_mode, true, false, 0.01);

    // Test also the reference backend and without distributed FFTs from the
    // library
    if (fft_lib_use_mpi()) {
      errors += run_unittests(debug, backend, planning_mode, false, false, 0.01);
    }
  }

  fft_finalize_lib(NULL);
  fft_finalize_timer();

  if (errors == 0) {
    printf("\nAll tests have passed :-)\n");
  } else {
    printf("\nFound %i errors :-(\n", errors);
  }
  fflush(stdout);
  cp_mpi_barrier(cp_mpi_get_comm_world());

  cp_mpi_finalize();

  return errors;
}

// EOF
