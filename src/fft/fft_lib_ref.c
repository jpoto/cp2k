/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib_ref.h"
#include "fft_lib_ref_low.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_init_lib() {
  // Nothing to be done
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_finalize_lib() {
  // Nothing to be done
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation of FFT is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_ref_lib_use_mpi() { return false; }

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_allocate_double(const int length, double **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = (double *)malloc(length * sizeof(double));
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_allocate_complex(const int length, double complex **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = (double complex *)malloc(length * sizeof(double complex));
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_free_double(double *buffer) { free(buffer); }

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_free_complex(double complex *buffer) { free(buffer); }

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs, const int fft_size,
                         const int number_of_ffts, const bool transpose_rs,
                         const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_1d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                              number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_1d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                              number_of_ffts, 1, 1, fft_size);
    }
  } else {
    if (transpose_gs) {
      fft_ref_1d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                              number_of_ffts, fft_size, 1);
    } else {
      fft_ref_1d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1, 1,
                              fft_size, fft_size);
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size, const int number_of_ffts,
                             const bool transpose_rs, const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_1d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                                  number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_1d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                                  number_of_ffts, 1, 1, fft_size / 2 + 1);
    }
  } else {
    if (transpose_gs) {
      fft_ref_1d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                                  number_of_ffts, fft_size, 1);
    } else {
      fft_ref_1d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                                  1, fft_size, fft_size / 2 + 1);
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs, const int fft_size,
                         const int number_of_ffts, const bool transpose_rs,
                         const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_1d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                              number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_1d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                              number_of_ffts, fft_size, 1);
    }
  } else {
    if (transpose_gs) {
      fft_ref_1d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                              number_of_ffts, 1, 1, fft_size);
    } else {
      fft_ref_1d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1, 1,
                              fft_size, fft_size);
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size,
                             const int number_of_ffts, const bool transpose_rs,
                             const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_1d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                                  number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_1d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                                  number_of_ffts, fft_size / 2 + 1, 1);
    }
  } else {
    if (transpose_gs) {
      fft_ref_1d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                                  number_of_ffts, 1, 1, fft_size);
    } else {
      fft_ref_1d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                                  1, fft_size / 2 + 1, fft_size);
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs,
                         const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_2d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                              number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_2d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                              number_of_ffts, 1, 1, fft_size[0] * fft_size[1]);
    }
  } else {
    if (transpose_gs) {
      fft_ref_2d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                              number_of_ffts, fft_size[0] * fft_size[1], 1);
    } else {
      fft_ref_2d_fw_local_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1, 1,
                              fft_size[0] * fft_size[1],
                              fft_size[0] * fft_size[1]);
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size[2], const int number_of_ffts,
                             const bool transpose_rs, const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_2d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                                  number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_2d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts,
                                  number_of_ffts, 1, 1,
                                  fft_size[0] * (fft_size[1] / 2 + 1));
    }
  } else {
    if (transpose_gs) {
      fft_ref_2d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                                  number_of_ffts, fft_size[0] * fft_size[1], 1);
    } else {
      fft_ref_2d_fw_local_r2c_low(grid_rs, grid_gs, fft_size, number_of_ffts, 1,
                                  1, fft_size[0] * fft_size[1],
                                  fft_size[0] * (fft_size[1] / 2 + 1));
    }
  }
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs,
                         const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_2d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                              number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_2d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                              number_of_ffts, fft_size[0] * fft_size[1], 1);
    }
  } else {
    if (transpose_gs) {
      fft_ref_2d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                              number_of_ffts, 1, 1, fft_size[0] * fft_size[1]);
    } else {
      fft_ref_2d_bw_local_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1, 1,
                              fft_size[0] * fft_size[1],
                              fft_size[0] * fft_size[1]);
    }
  }
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size[2],
                             const int number_of_ffts, const bool transpose_rs,
                             const bool transpose_gs) {
  if (transpose_rs) {
    if (transpose_gs) {
      fft_ref_2d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                                  number_of_ffts, number_of_ffts, 1, 1);
    } else {
      fft_ref_2d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                                  number_of_ffts,
                                  fft_size[0] * (fft_size[1] / 2 + 1), 1);
    }
  } else {
    if (transpose_gs) {
      fft_ref_2d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts,
                                  number_of_ffts, 1, 1,
                                  fft_size[0] * fft_size[1]);
    } else {
      fft_ref_2d_bw_local_c2r_low(grid_gs, grid_rs, fft_size, number_of_ffts, 1,
                                  1, fft_size[0] * (fft_size[1] / 2 + 1),
                                  fft_size[0] * fft_size[1]);
    }
  }
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local(double complex *restrict grid_rs,
                         double complex *restrict grid_gs,
                         const int fft_size[3]) {

  fft_ref_3d_fw_local_low(grid_rs, grid_gs, fft_size);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c(double *restrict grid_rs,
                             double complex *restrict grid_gs,
                             const int fft_size[3]) {

  fft_ref_3d_fw_local_r2c_low(grid_rs, grid_gs, fft_size);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local(double complex *restrict grid_gs,
                         double complex *restrict grid_rs,
                         const int fft_size[3]) {

  fft_ref_3d_bw_local_low(grid_gs, grid_rs, fft_size);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r(double complex *restrict grid_gs,
                             double *restrict grid_rs, const int fft_size[3]) {

  fft_ref_3d_bw_local_c2r_low(grid_gs, grid_rs, fft_size);
}

// EOF
