/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mpiwrap/cp_mpi.h"
#include "../offload/offload_library.h"
#include "fft_grid_test.h"
#include "fft_lib.h"
#include "fft_lib_test.h"
#include "fft_reorder_test.h"
#include "fft_timer.h"

int main(int argc, char *argv[]) {
  cp_mpi_init(&argc, &argv);

  offload_set_chosen_device(0);
  fft_init_timer(true);
  fft_init_lib(FFT_LIB_DEFAULT, FFT_MEASURE, true, NULL);

  int errors = 0;

  errors += fft_test_local();
  errors += fft_test_distributed();
  errors += fft_test_transpose();
  errors += fft_test_transpose_parallel();
  errors += fft_test_3d();
  errors += fft_test_add_copy();
  fft_print_timing_report();

  // Test also the reference backend and without distributed FFTs from the
  // library
  if (true) {
    fft_finalize_timer();
    fft_finalize_lib(NULL);
    fft_init_timer(true);
    fft_init_lib(FFT_LIB_REF, FFT_MEASURE, false, NULL);
    errors += fft_test_local();
    errors += fft_test_distributed();
    errors += fft_test_transpose();
    errors += fft_test_transpose_parallel();
    errors += fft_test_3d();
    errors += fft_test_add_copy();
    fft_print_timing_report();
  }

  fft_finalize_lib(NULL);
  fft_finalize_timer();

  if (errors == 0) {
    printf("\nAll tests have passed :-)\n");
  } else {
    printf("\nFound %i errors :-(\n", errors);
  }

  cp_mpi_finalize();

  return errors;
}

// EOF
