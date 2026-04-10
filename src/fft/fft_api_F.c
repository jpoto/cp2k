/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_timer.h"

#include <assert.h>
#include <string.h>

// Keep in accordance to fft_api.F
const int FFT_LIBRARY_BACKEND_DEFAULT = 1;
const int FFT_LIBRARY_BACKEND_FFTW = 2;
const int FFT_LIBRARY_BACKEND_REFERENCE = 3;
const int FFT_LIBRARY_FFTW_MODE_ESTIMATE = 11;
const int FFT_LIBRARY_FFTW_MODE_MEASURE = 12;
const int FFT_LIBRARY_FFTW_MODE_PATIENT = 13;
const int FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE = 14;

void fft_library_init_F(const int backend_F, const int fftw_plan,
                        const bool use_fftw_mpi, const bool use_guru_interface,
                        const char *wisdom_file) {
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
  fft_init_lib(backend, plan_type, use_fftw_mpi, use_guru_interface,
               wisdom_file);
  fft_init_timer(false);
}

void fft_library_finalize_F(const char *wisdom_file) {
  fft_finalize_lib(wisdom_file);
}

void fft_print_timing_report_F(const double threshold) {
  fft_print_timing_report(threshold);
}

int fft_backend_in_use_F() { return fft_lib_backend_in_use(); }

bool fft_use_mpi_F() { return fft_lib_use_mpi(); }

bool fft_has_guru_interface_F() { return fft_lib_has_guru_interface(); }

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

void fft_2d_fw_local_F(const int fft_size[2], const int number_of_ffts,
                       const bool transpose_rs, const bool transpose_gs,
                       double complex *grid_in, double complex *grid_out) {
  fft_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in,
                  grid_out);
}

void fft_2d_bw_local_F(const int fft_size[2], const int number_of_ffts,
                       const bool transpose_rs, const bool transpose_gs,
                       double complex *grid_in, double complex *grid_out) {
  fft_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid_in,
                  grid_out);
}

void fft_3d_fw_local_F(const int fft_size[3], double complex *grid_in,
                       double complex *grid_out) {
  fft_3d_fw_local(fft_size, grid_in, grid_out);
}

void fft_3d_bw_local_F(const int fft_size[3], double complex *grid_in,
                       double complex *grid_out) {
  fft_3d_bw_local(fft_size, grid_in, grid_out);
}

void fft_1d_fw_local_inplace_F(const int fft_size, const int number_of_ffts,
                               const bool transpose_rs, const bool transpose_gs,
                               double complex *grid) {
  fft_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid,
                  grid);
}

void fft_1d_bw_local_inplace_F(const int fft_size, const int number_of_ffts,
                               const bool transpose_rs, const bool transpose_gs,
                               double complex *grid) {
  fft_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid,
                  grid);
}

void fft_2d_fw_local_inplace_F(const int fft_size[2], const int number_of_ffts,
                               const bool transpose_rs, const bool transpose_gs,
                               double complex *grid) {
  fft_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid,
                  grid);
}

void fft_2d_bw_local_inplace_F(const int fft_size[2], const int number_of_ffts,
                               const bool transpose_rs, const bool transpose_gs,
                               double complex *grid) {
  fft_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs, grid,
                  grid);
}

void fft_3d_fw_local_inplace_F(const int fft_size[3], double complex *grid) {
  fft_3d_fw_local(fft_size, grid, grid);
}

void fft_3d_bw_local_inplace_F(const int fft_size[3], double complex *grid) {
  fft_3d_bw_local(fft_size, grid, grid);
}

/*******************************************************************************
 * \brief Create a FFT grid.
 * \note If a grid layout was created using use_halfspace, only real-valued
 *grids in real-space can be created from this layout \author Frederick Stein
 ******************************************************************************/
void fft_create_grid_F(fft_grid_layout **fft_grid, const int comm_F,
                       const int npts_global[3], const double dh_inv[3][3],
                       const bool use_halfspace, const double cutoff, const int *pgrid_guess) {
  grid_create_fft_grid_layout(
      fft_grid, cp_mpi_comm_f2c(comm_F),
      (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
      (const double[3][3]){{dh_inv[2][2], dh_inv[2][1], dh_inv[2][0]},
                           {dh_inv[1][2], dh_inv[1][1], dh_inv[1][0]},
                           {dh_inv[0][2], dh_inv[0][1], dh_inv[0][0]}},
      cutoff, use_halfspace, pgrid_guess);
}

/*******************************************************************************
 * \brief Create a FFT grid using a reference grid to interact with this grid.
 * \note The reference grid has had to be created using
 *grid_create_fft_grid_layout \author Frederick Stein
 ******************************************************************************/
void fft_create_grid_from_reference_F(fft_grid_layout **fft_grid,
                                      const int npts_global[3], const double cutoff,
                                      const fft_grid_layout *fft_grid_ref) {
  assert(fft_grid_ref != NULL);
  grid_create_fft_grid_layout_from_reference(
      fft_grid, (const int[3]){npts_global[2], npts_global[1], npts_global[0]}, cutoff,
      fft_grid_ref);
}

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void fft_retain_grid_F(fft_grid_layout *fft_grid) {
  grid_retain_fft_grid_layout(fft_grid);
}

/*******************************************************************************
 * \brief Frees a FFT grid.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_grid_F(fft_grid_layout *fft_grid) {
  grid_free_fft_grid_layout(fft_grid);
}

/*******************************************************************************
 * \brief Get the global number of points in each direction.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_npts_global_F(const fft_grid_layout *fft_grid, int *npts_global) {
  assert(fft_grid != NULL);
  memcpy(npts_global, fft_grid->npts_global, 3*sizeof(int));
}

/*******************************************************************************
 * \brief Get the (2D)-cartesian communicator.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_comm_F(const fft_grid_layout *fft_grid, int *comm_handle) {
  assert(fft_grid != NULL);
  *comm_handle = cp_mpi_comm_c2f(fft_grid->comm);
}

// EOF
