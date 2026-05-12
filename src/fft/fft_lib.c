/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib.h"
#include "../common/cp_data_dir.h"
#include "fft_lib_fftw.h"
#include "fft_timer.h"
#include "fpga/fft_fpga.h"
#include "gpu/fft_gpu.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
fft_lib fft_lib_choice = FFT_LIB_GPU;
#elif defined(__FFTW3)
fft_lib fft_lib_choice = FFT_LIB_FFTW;
#else
#error "The FFT backend needs at least the FFTW3 backend."
#endif
bool fft_lib_initialized = false;

double complex *buffer_1 = NULL;
double complex *buffer_2 = NULL;
int buffer_size = -1;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi, const bool use_guru_interface,
                  const char *wisdom_file) {
  if (fft_lib_initialized) {
    return;
  }
  fft_lib_initialized = true;
  fft_lib_choice = lib;
  fft_fftw_init_lib(fftw_planning_flag, use_fft_mpi, use_guru_interface,
                    wisdom_file);
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    printf("Using FFTW library.\n");
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_allocate_complex(1, &buffer_1);
  fft_allocate_complex(1, &buffer_2);
  buffer_size = 1;
}

/*******************************************************************************
 * \brief Initialize the accelerated FFT library (FPGA+GPU).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_acc_lib() {
#if defined(__FFT_FPGA)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
#error                                                                         \
    "OFFLOAD and FPGA cannot be configured concurrently! Recompile with -D__NO_OFFLOAD_FFT."
  CPABORT("OFFLOAD and FPGA cannot be configured concurrently! Recompile with "
          "-D__NO_OFFLOAD_FFT.")
#endif
  const int stat = fft_fpga_initialize()
      assert(stat == 0 && "Initialization of FPGA failed!");
#endif
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
  fft_gpu_init();
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib(const char *wisdom_file) {
  fft_fftw_finalize_lib(wisdom_file);
  if (buffer_1 != NULL)
    fft_free_complex(buffer_1);
  if (buffer_2 != NULL)
    fft_free_complex(buffer_2);
  buffer_1 = NULL;
  buffer_2 = NULL;
  buffer_size = -1;
  fft_lib_initialized = false;
}

/*******************************************************************************
 * \brief Finalize the accelerated FFT library.
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_acc_lib() {
#if defined(__FFT_FPGA)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
#error                                                                         \
    "OFFLOAD and FPGA cannot be configured concurrently! Recompile with -D__NO_OFFLOAD_FFT."
  CPABORT("OFFLOAD and FPGA cannot be configured concurrently! Recompile with "
          "-D__NO_OFFLOAD_FFT.")
#endif
  fft_fpga_final_();
#endif
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
  fft_gpu_finalize();
#endif
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
int fft_lib_backend_in_use() { return fft_lib_choice; }

/*******************************************************************************
 * \brief Whether a compound MPI implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi() {
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    return fft_fftw_lib_use_mpi();
  case FFT_LIB_GPU:
    return false;
  default:
    assert(0 && "Unknown FFT library.");
    return false;
  }
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_has_guru_interface() {
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    return fft_fftw_lib_has_guru_interface();
  case FFT_LIB_GPU:
    return false;
  default:
    assert(0 && "Unknown FFT library.");
    return false;
  }
}

/*******************************************************************************
 * \brief Whether compound operations (FFT+copy) are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_has_compound_operations() {
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    return false;
  case FFT_LIB_GPU:
    return true;
  default:
    assert(0 && "Unknown FFT library.");
    return false;
  }
}

/*******************************************************************************
 * \brief Ensure that buffers have a required size (in units of complex numbers)
 * \author Frederick Stein
 ******************************************************************************/
void ensure_buffer_size(const int size) {
  if (buffer_size < size) {
    fft_free_complex(buffer_1);
    fft_free_complex(buffer_2);
    buffer_1 = NULL;
    buffer_2 = NULL;
    fft_allocate_complex(size, &buffer_1);
    fft_allocate_complex(size, &buffer_2);
    buffer_size = size;
  }
}

/*******************************************************************************
 * \brief Get the first internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_1() { return buffer_1; }

/*******************************************************************************
 * \brief Get the second internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_2() { return buffer_2; }

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer) {
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_allocate_double(length, buffer);
    break;
  case FFT_LIB_GPU:
    fft_gpu_allocate_double(length, buffer);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer) {
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_allocate_complex(length, buffer);
    break;
  case FFT_LIB_GPU:
    fft_gpu_allocate_complex(length, buffer);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_free_double(buffer);
  } else if (fft_lib_choice == FFT_LIB_GPU) {
    fft_gpu_free_double(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_free_complex(buffer);
  } else if (fft_lib_choice == FFT_LIB_GPU) {
    fft_gpu_free_complex(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_fw_c2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_gpu_f((const double *)grid_in, (double *)grid_out, 1, fft_size,
              number_of_ffts, transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_fw_r2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_r2c_gpu_f((const double *)grid_in, (double *)grid_out, 1, fft_size,
                  number_of_ffts, transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_bw_c2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_gpu_f((const double *)grid_in, (double *)grid_out, -1, fft_size,
              number_of_ffts, transpose_gs, transpose_rs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_bw_c2r_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_r2c_gpu_f((const double *)grid_in, (double *)grid_out, -1, fft_size,
                  number_of_ffts, transpose_gs, transpose_rs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_c2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_gpu_ff((const double *)grid_in, (double *)grid_out, 1, fft_size,
               number_of_ffts, transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_r2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_r2c_gpu_ff((const double *)grid_in, (double *)grid_out, 1, fft_size,
                   number_of_ffts, transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2c_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_gpu_ff((const double *)grid_in, (double *)grid_out, -1, fft_size,
               number_of_ffts, transpose_gs, transpose_rs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2r_local");
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_GPU:
    fft_r2c_gpu_ff((const double *)grid_in, (double *)grid_out, -1, fft_size,
                   number_of_ffts, transpose_gs, transpose_rs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_c2c_local");
  const int handle = fft_start_timer(routine_name);
#if defined(__FFT_FPGA)
  if (fft_fpga_check_bitstream_(get_data_dir(), fft_size)) {
    const int number_of_elements = product3(fft_size);
#if (__FFT_FPGA_SP && __FFT_FPGA)
    float complex *grid_sp = calloc(number_of_elements, sizeof(float complex));
    for (int i = 0; i < number_of_elements; i++)
      grid_sp[i] = (float complex)grid_in[i];
    fft_fpga_fft3d_sp_(1, fft_size, grid_sp);
    for (int i = 0; i < number_of_elements; i++)
      grid_out[i] = (double complex)grid_sp[i];
#else
    memcpy(grid_out, grid_in, number_of_elements * sizeof(double complex));
    fft_fpga_fft3d_dp_(1, fft_size, grid_out);
#endif
  } else {
#endif
    switch (fft_lib_choice) {
    case FFT_LIB_GPU:
      fft_gpu_fff((const double *)grid_in, (double *)grid_out, +1, fft_size);
      break;
    case FFT_LIB_FFTW:
      fft_fftw_3d_fw_local(fft_size, grid_in, grid_out);
      break;
    default:
      assert(0 && "Unknown FFT library.");
    }
#if defined(__FFT_FPGA)
  }
#endif
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local_r2c(const int fft_size[3], double *grid_in,
                         double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_local");
  const int handle = fft_start_timer(routine_name);
#if defined(__FFT_FPGA)
  if (fft_fpga_check_bitstream_(get_data_dir(), fft_size)) {
    const int number_of_elements = product3(fft_size);
#if (__FFT_FPGA_SP && __FFT_FPGA)
    float complex *grid_sp = calloc(number_of_elements, sizeof(float complex));
    for (int i = 0; i < number_of_elements; i++)
      grid_sp[i] = CMPLXF((float)grid_in[i], 0.0f);
    fft_fpga_fft3d_sp_(1, fft_size, grid_sp);
    for (int i = 0; i < number_of_elements; i++)
      grid_out[i] = (double complex)grid_sp[i];
#else
    for (int i = 0; i < number_of_elements; i++)
      grid_out[i] = CMPLX(grid_in[i], 0.0);
    fft_fpga_fft3d_dp_(1, fft_size, grid_out);
#endif
  } else {
#endif
    switch (fft_lib_choice) {
    case FFT_LIB_GPU:
      fft_r2c_gpu_fff((const double *)grid_in, (double *)grid_out, +1,
                      fft_size);
      break;
    case FFT_LIB_FFTW:
      fft_fftw_3d_fw_local_r2c(fft_size, grid_in, grid_out);
      break;
    default:
      assert(0 && "Unknown FFT library.");
    }
#if defined(__FFT_FPGA)
  }
#endif
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2c_local");
  const int handle = fft_start_timer(routine_name);
#if defined(__FFT_FPGA)
  if (fft_fpga_check_bitstream_(get_data_dir(), fft_size)) {
    const int number_of_elements = product3(fft_size);
#if (__FFT_FPGA_SP && __FFT_FPGA)
    float complex *grid_sp = calloc(number_of_elements, sizeof(float complex));
    for (int i = 0; i < number_of_elements; i++)
      grid_sp[i] = (float complex)grid_in[i];
    fft_fpga_fft3d_sp_(-1, fft_size, grid_sp);
    for (int i = 0; i < number_of_elements; i++)
      grid_out[i] = (double complex)grid_sp[i];
#else
    memcpy(grid_out, grid_in, number_of_elements * sizeof(double complex));
    fft_fpga_fft3d_dp_(-1, fft_size, grid_out);
#endif
  } else {
#endif
    switch (fft_lib_choice) {
    case FFT_LIB_GPU:
      fft_gpu_fff((const double *)grid_in, (double *)grid_out, -1, fft_size);
      break;
    case FFT_LIB_FFTW:
      fft_fftw_3d_bw_local(fft_size, grid_in, grid_out);
      break;
    default:
      assert(0 && "Unknown FFT library.");
    }
#if defined(__FFT_FPGA)
  }
#endif
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local_c2r(const int fft_size[3], double complex *grid_in,
                         double *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_local");
  const int handle = fft_start_timer(routine_name);
#if defined(__FFT_FPGA)
  if (fft_fpga_check_bitstream_(get_data_dir(), fft_size)) {
    const int number_of_elements =
        (fft_size[0] / 2 + 1) * fft_size[1] * fft_size[2];
#if (__FFT_FPGA_SP && __FFT_FPGA)
    float complex *grid_sp = calloc(number_of_elements, sizeof(float complex));
    for (int i = 0; i < number_of_elements; i++)
      grid_sp[i] = (float complex)grid_in[i];
    fft_fpga_fft3d_sp_(-1, fft_size, grid_sp);
    for (int i = 0; i < number_of_elements; i++)
      grid_out[i] = (double)crealf(grid_sp[i]);
#else
    memcpy((double complex *)grid_out, grid_in,
           number_of_elements * sizeof(double));
    fft_fpga_fft3d_dp_(-1, fft_size, grid_out);
#endif
  } else {
#endif
    switch (fft_lib_choice) {
    case FFT_LIB_GPU:
      fft_r2c_gpu_fff((const double *)grid_in, (double *)grid_out, -1,
                      fft_size);
      break;
    case FFT_LIB_FFTW:
      fft_fftw_3d_bw_local_c2r(fft_size, grid_in, grid_out);
      break;
    default:
      assert(0 && "Unknown FFT library.");
    }
#if defined(__FFT_FPGA)
  }
#endif
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a local C2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw_guru(int rank, const fft_iodim *dims, int howmany_rank,
                 const fft_iodim *howmany_dims, const int number_of_threads,
                 double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_fw_c2c_%i_%i", rank,
           howmany_rank);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_fw_guru(rank, dims, howmany_rank, howmany_dims, number_of_threads,
                     grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a local forward R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw_guru_r2c(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_fw_r2c_%i_%i", rank,
           howmany_rank);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_fw_guru_r2c(rank, dims, howmany_rank, howmany_dims,
                         number_of_threads, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a local backwards C2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw_guru(int rank, const fft_iodim *dims, int howmany_rank,
                 const fft_iodim *howmany_dims, const int number_of_threads,
                 double complex *grid_in, double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_bw_c2c_%i_%i", rank,
           howmany_rank);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_bw_guru(rank, dims, howmany_rank, howmany_dims, number_of_threads,
                     grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a local backwards R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw_guru_c2r(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double complex *grid_in, double *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_bw_c2r_%i_%i", rank,
           howmany_rank);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_bw_guru_c2r(rank, dims, howmany_rank, howmany_dims,
                         number_of_threads, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const cp_mpi_comm_t comm, int *local_n0,
                             int *local_n0_start, int *local_n1,
                             int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_2d_distributed_sizes(npts_global, number_of_ffts, comm,
                                         local_n0, local_n0_start, local_n1,
                                         local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes_r2c(const int npts_global[2],
                                 const int number_of_ffts,
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_2d_distributed_sizes_r2c(npts_global, number_of_ffts, comm,
                                             local_n0, local_n0_start, local_n1,
                                             local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const cp_mpi_comm_t comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_3d_distributed_sizes(
        npts_global, comm, local_n2, local_n2_start, local_n1, local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes_r2c(const int npts_global[3],
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_3d_distributed_sizes_r2c(
        npts_global, comm, local_n0, local_n0_start, local_n1, local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed(const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_c2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_distributed(npts_global, number_of_ffts, comm, grid_in,
                               grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed_r2c(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_r2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_distributed_r2c(npts_global, number_of_ffts, comm, grid_in,
                                   grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed(const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_distributed(npts_global, number_of_ffts, comm, grid_in,
                               grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed_c2r(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2r_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_distributed_c2r(npts_global, number_of_ffts, comm, grid_in,
                                   grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed(const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_c2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed_r2c(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_distributed_r2c(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed(const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2c_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed_c2r(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_distr");
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_distributed_c2r(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

// EOF
