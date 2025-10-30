/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <complex.h>
#include <string.h>

/*******************************************************************************
 * \brief Returns the smaller of two given integer (missing from the C standard)
 * \author Ole Schuett
 ******************************************************************************/
static inline int imin(int x, int y) { return (x < y ? x : y); }

/*******************************************************************************
 * \brief Returns the larger of two given integer (missing from the C standard)
 * \author Ole Schuett
 ******************************************************************************/
static inline int imax(int x, int y) { return (x > y ? x : y); }

/*******************************************************************************
 * \brief Returns the smaller of two given integer (missing from the C standard)
 * \author Frederick Stein
 ******************************************************************************/
static inline int dmin(double x, double y) { return (x < y ? x : y); }

/*******************************************************************************
 * \brief Returns the larger of two given integer (missing from the C standard)
 * \author Frederick Stein
 ******************************************************************************/
static inline int dmax(double x, double y) { return (x > y ? x : y); }

/*******************************************************************************
 * \brief Equivalent of Fortran's MODULO which always returns a positive number.
 *        https://gcc.gnu.org/onlinedocs/gfortran/MODULO.html
 * \author Ole Schuett
 ******************************************************************************/
static inline int modulo(int a, int m) { return ((a % m + m) % m); }

/*******************************************************************************
 * \brief Calculates the product of three numbers.
 * \author Frederick Stein
 ******************************************************************************/
static inline int product3(const int array3[3]) {
  return array3[0] * array3[1] * array3[2];
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_local_complex(
    double complex *restrict grid, double complex *restrict grid_transposed,
    const int number_of_columns_grid, const int number_of_rows_grid,
    const int total_number_of_columns_grid,
    const int total_number_of_columns_transposed) {

  const int number_of_double_complex_per_cache_line = 4;

#pragma omp parallel for default(none) shared(                                 \
        grid, grid_transposed, number_of_columns_grid, number_of_rows_grid,    \
            total_number_of_columns_grid, total_number_of_columns_transposed,  \
            number_of_double_complex_per_cache_line) collapse(2)
  for (int column_index_start = 0; column_index_start < number_of_columns_grid;
       column_index_start += number_of_double_complex_per_cache_line) {
    for (int row_index_start = 0; row_index_start < number_of_rows_grid;
         row_index_start += number_of_double_complex_per_cache_line) {
      for (int column_index = column_index_start;
           column_index <
           imin(column_index_start + number_of_double_complex_per_cache_line,
                number_of_columns_grid);
           column_index++) {
        for (int row_index = row_index_start;
             row_index <
             imin(row_index_start + number_of_double_complex_per_cache_line,
                  number_of_rows_grid);
             row_index++) {
          grid_transposed[column_index * total_number_of_columns_transposed +
                          row_index] =
              grid[row_index * total_number_of_columns_grid + column_index];
        }
      }
    }
  }
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_local_double(
    const double *restrict grid, double *restrict grid_transposed,
    const int number_of_columns_grid, const int number_of_rows_grid,
    const int total_number_of_columns_grid,
    const int total_number_of_columns_transposed) {

  const int number_of_doubles_per_cache_line = 8;

#pragma omp parallel for default(none) shared(                                 \
        grid, grid_transposed, number_of_columns_grid, number_of_rows_grid,    \
            total_number_of_columns_grid, total_number_of_columns_transposed,  \
            number_of_doubles_per_cache_line) collapse(2)
  for (int column_index_start = 0; column_index_start < number_of_columns_grid;
       column_index_start += number_of_doubles_per_cache_line) {
    for (int row_index_start = 0; row_index_start < number_of_rows_grid;
         row_index_start += number_of_doubles_per_cache_line) {
      for (int column_index = column_index_start;
           column_index <
           imin(column_index_start + number_of_doubles_per_cache_line,
                number_of_columns_grid);
           column_index++) {
        for (int row_index = row_index_start;
             row_index <
             imin(row_index_start + number_of_doubles_per_cache_line,
                  number_of_rows_grid);
             row_index++) {
          grid_transposed[column_index * total_number_of_columns_transposed +
                          row_index] =
              grid[row_index * total_number_of_columns_grid + column_index];
        }
      }
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks.
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_local_complex_block(
    double complex *restrict grid, double complex *restrict grid_transposed,
    const int number_of_columns_grid, const int number_of_rows_grid,
    const int block_size, const int total_number_of_columns_grid,
    const int total_block_size_grid,
    const int total_number_of_columns_transposed,
    const int total_block_size_transp) {
  if (block_size == 1) {
    transpose_local_complex(grid, grid_transposed, number_of_columns_grid,
                            number_of_rows_grid, number_of_columns_grid,
                            number_of_rows_grid);
  } else {
#pragma omp parallel for default(none)                                         \
    shared(grid, grid_transposed, number_of_columns_grid, number_of_rows_grid, \
               block_size, total_block_size_grid,                              \
               total_number_of_columns_transposed,                             \
               total_number_of_columns_grid, total_block_size_transp)          \
    collapse(2)
    for (int column_index = 0; column_index < number_of_columns_grid;
         column_index++) {
      for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
        memcpy(&grid_transposed[(column_index *
                                     total_number_of_columns_transposed +
                                 row_index) *
                                total_block_size_grid],
               &grid[(row_index * total_number_of_columns_grid + column_index) *
                     total_block_size_transp],
               block_size * sizeof(double complex));
      }
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks. (x,y,z) -> (y,z,x)
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_local_double_block(
    const double *restrict grid, double *restrict grid_transposed,
    const int number_of_columns_grid, const int number_of_rows_grid,
    const int block_size, const int total_number_of_columns_grid,
    const int total_block_size_grid,
    const int total_number_of_columns_transposed,
    const int total_block_size_transposed) {
  if (block_size == 1) {
    transpose_local_double(
        grid, grid_transposed, number_of_columns_grid, number_of_rows_grid,
        total_number_of_columns_grid * total_block_size_grid,
        total_number_of_columns_transposed * total_block_size_transposed);
  } else {
#pragma omp parallel for default(none)                                         \
    shared(grid, grid_transposed, number_of_columns_grid, number_of_rows_grid, \
               block_size, total_number_of_columns_transposed,                 \
               total_number_of_columns_grid, total_block_size_transposed,      \
               total_block_size_grid) collapse(2)
    for (int column_index = 0; column_index < number_of_columns_grid;
         column_index++) {
      for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
        memcpy(&grid_transposed[(column_index *
                                     total_number_of_columns_transposed +
                                 row_index) *
                                total_block_size_transposed],
               &grid[(row_index * total_number_of_columns_grid + column_index) *
                     total_block_size_grid],
               block_size * sizeof(double));
      }
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks. (x,y,z) -> (z,y,x)
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_xyz2zyx(const double complex *restrict grid,
                                     double complex *restrict grid_transposed,
                                     const int npts_0, const int npts_1,
                                     const int npts_2, const int local_dim_1,
                                     const int local_dim_2,
                                     const int local_dim_transposed_0,
                                     const int local_dim_transposed_2) {
  for (int index_x = 0; index_x < npts_0; index_x++) {
    for (int index_y = 0; index_y < npts_1; index_y++) {
      for (int index_z = 0; index_z < npts_2; index_z++) {
        grid_transposed[(index_z * local_dim_transposed_0 + index_y) *
                            local_dim_transposed_2 +
                        index_x] =
            grid[(index_x * local_dim_1 + index_y) * local_dim_2 + index_z];
      }
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks. (x,y,z) -> (z,y,x)
 * \author Frederick Stein
 ******************************************************************************/
static inline void transpose_xyz2zyx_double(
    const double *restrict grid, double *restrict grid_transposed,
    const int npts_0, const int npts_1, const int npts_2, const int local_dim_1,
    const int local_dim_2, const int local_dim_transposed_0,
    const int local_dim_transposed_2) {
  for (int index_x = 0; index_x < npts_0; index_x++) {
    for (int index_y = 0; index_y < npts_1; index_y++) {
      for (int index_z = 0; index_z < npts_2; index_z++) {
        grid_transposed[(index_z * local_dim_transposed_0 + index_y) *
                            local_dim_transposed_2 +
                        index_x] =
            grid[(index_x * local_dim_1 + index_y) * local_dim_2 + index_z];
      }
    }
  }
}

void dscal_(const int *n, const double *da, double *dx, const int *incx);

void zdscal_(const int *n, const double *da, double complex *za,
             const int *incx);

void dcopy_(const int *n, const double *dx, const int *incx, double *dy,
            const int *incy);

void zcopy_(const int *n, const double complex *zx, const int *incx,
            double complex *zy, const int *incy);

#endif /* FFT_UTILS_H */

// EOF
