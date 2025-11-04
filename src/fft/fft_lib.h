/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_LIB_H
#define FFT_LIB_H

#include "../mpiwrap/cp_mpi.h"
#include "fft_lib_fftw.h"

#include <complex.h>
#include <stdbool.h>

typedef enum { FFT_LIB_REF, FFT_LIB_FFTW } fft_lib;

#if defined(__FFTW3)
static const fft_lib FFT_LIB_DEFAULT = FFT_LIB_FFTW;
#else
static const fft_lib FFT_LIB_DEFAULT = FFT_LIB_REF;
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi, const char *wisdom_file);

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib(const char *wisdom_file);

/*******************************************************************************
 * \brief Whether compound MPI implementations are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi();

/*******************************************************************************
 * \brief Ensure that buffers have a required size (in units of complex numbers)
 * \author Frederick Stein
 ******************************************************************************/
void ensure_buffer_size(const int size);

/*******************************************************************************
 * \brief Get the first internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_1();

/*******************************************************************************
 * \brief Get the second internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_2();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local_r2c(const int fft_size[3], double *grid_in,
                         double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local_c2r(const int fft_size[3], double complex *grid_in,
                         double *grid_out);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const cp_mpi_comm_t comm, int *local_n0,
                             int *local_n0_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes_r2c(const int npts_global[2],
                                 const int number_of_ffts,
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const cp_mpi_comm_t comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes_r2c(const int npts_global[3],
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed(const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed_r2c(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed(const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed_c2r(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed(const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed_r2c(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed(const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed_c2r(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out);

#endif /* FFT_LIB_H */

// EOF
