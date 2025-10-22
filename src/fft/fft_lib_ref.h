/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_LIB_REF_H
#define FFT_LIB_REF_H

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_init_lib();

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_finalize_lib();

/*******************************************************************************
 * \brief Whether a compound MPI implementation of FFT is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_ref_lib_use_mpi();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_free_double(double *buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs, const int fft_size,
                         const int number_of_ffts, const bool transpose_rs,
                         const bool transpose_gs);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size, const int number_of_ffts,
                             const bool transpose_rs, const bool transpose_gs);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs, const int fft_size,
                         const int number_of_ffts, const bool transpose_rs,
                         const bool transpose_gs);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size,
                             const int number_of_ffts, const bool transpose_rs,
                             const bool transpose_gs);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs,
                         const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size[2], const int number_of_ffts,
                             const bool transpose_rs, const bool transpose_gs);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs,
                         const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size[2],
                             const int number_of_ffts, const bool transpose_rs,
                             const bool transpose_gs);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs,
                         const int fft_size[3]);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size[3]);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs,
                         const int fft_size[3]);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size[3]);

#endif /* FFT_LIB_REF_H */

// EOF
