/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib.h"
#include "fft_timer.h"

#include <assert.h>

// Keep in accordance to fft_api.F
const int FFT_LIBRARY_BACKEND_DEFAULT = 1;
const int FFT_LIBRARY_BACKEND_FFTW = 2;
const int FFT_LIBRARY_BACKEND_REFERENCE = 3;
const int FFT_LIBRARY_FFTW_MODE_ESTIMATE = 11;
const int FFT_LIBRARY_FFTW_MODE_MEASURE = 12;
const int FFT_LIBRARY_FFTW_MODE_PATIENT = 13;
const int FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE = 14;

void fft_library_init_F(const int backend_F, const int fftw_plan,
                        const bool use_fftw_mpi, const char *wisdom_file) {
  fft_lib backend;
  switch (backend_F) {
  case FFT_LIBRARY_BACKEND_DEFAULT:
    backend = FFT_LIB_DEFAULT;
    break;
  case FFT_LIBRARY_BACKEND_FFTW:
    backend = FFT_LIB_FFTW;
    break;
  case FFT_LIBRARY_BACKEND_REFERENCE:
    backend = FFT_LIB_REF;
    break;
  default:
    assert(false && "Unknown FFT library backend!");
    backend = FFT_LIB_DEFAULT;
  }
  fftw_plan_type plan_type;
  switch (fftw_plan) {
  case FFT_LIBRARY_FFTW_MODE_ESTIMATE:
    plan_type = FFT_ESTIMATE;
    break;
  case FFT_LIBRARY_FFTW_MODE_MEASURE:
    plan_type = FFT_MEASURE;
    break;
  case FFT_LIBRARY_FFTW_MODE_PATIENT:
    plan_type = FFT_PATIENT;
    break;
  case FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE:
    plan_type = FFT_EXHAUSTIVE;
    break;
  default:
    assert(false && "Unknown FFT library backend!");
    plan_type = FFT_ESTIMATE;
  }
  fft_init_lib(backend, plan_type, use_fftw_mpi, wisdom_file);
  fft_init_timer(false);
}

void fft_library_finalize_F(const char *wisdom_file) {
  fft_finalize_lib(wisdom_file);
}

void fft_print_timing_report_F() { fft_print_timing_report(); }

void fft_1d_fw_local_F(const int fft_size, const int number_of_ffts,
                       const bool transpose_rs, const bool transpose_gs,
                       double complex *grid_in, double complex *grid_out) {
  fft_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in,
                  grid_out);
}

void fft_1d_bw_local_F(const int fft_size, const int number_of_ffts,
                       const bool transpose_rs, const bool transpose_gs,
                       double complex *grid_in, double complex *grid_out) {
  fft_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in,
                  grid_out);
}

// EOF
