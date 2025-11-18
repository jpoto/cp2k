/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_GRID_LAYOUT_H
#define FFT_GRID_LAYOUT_H

#include "../mpiwrap/cp_mpi.h"
#include "fft_lib.h"

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Container to represent fft grids.
 * \author Frederick Stein
 ******************************************************************************/
typedef struct {
  // ID for comparison and referencing grids
  int grid_id;
  // ID of the reference grid
  int ref_grid_id;
  // Reference counter
  int ref_counter;
  // Global number of points
  int npts_global[3];
  // Grid spacing in reciprocal space
  double h_inv[3][3];
  // Whether to use only one half of g-space
  // If used, then complex-valued real-space grids are not possible anymore
  bool use_halfspace;
  int npts_global_gspace[3];
  // Number of local points in g-space (relevant with ray-distribution)
  int npts_gs_local;
  bool ray_distribution;
  int (*ray_to_xy)[2];
  int *xy_to_process;
  int *rays_per_process;
  int my_number_of_rays;
  // maps of index in g-space to g-space vectors
  int (*index_to_g)[3];
  int *local_index_to_ref_grid;
  // New communicator
  cp_mpi_comm_t comm;
  cp_mpi_comm_t sub_comm[2];
  int proc_grid[2];
  int periodic[2];
  int proc_coords[2];
  // distributions for each FFT step (real space/mixed-space 1 (rs), mixed space
  // 1/mixed space 2 (ms), mixed-space 2/g-space (gs)) first index is for the
  // process, the second for the coordinate, the third for start (0) / end(1)
  // proc2local_rs is also used for the distribution of the data in realspace
  // (that's why it is called "rs") proc2local_gs is also used for the
  // distribution of the data in reciprocal (g)-space (that's why it is called
  // "gs") in blocked mode (usually the finest grid)
  int (*proc2local_rs)[3][2]; // Order: (x, y, z)
  int (*proc2local_ms)[3][2]; // Order: (z, x, y)
  int (*proc2local_gs)[3][2]; // Order: (y, z, x)
  // Buffers for FFTs
  int buffer_size;
  // buffers for different purposes
} fft_grid_layout;

/*******************************************************************************
 * \brief Frees a FFT grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_fft_grid_layout(fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Create a FFT grid.
 * \note If a grid layout was created using use_halfspace, only real-valued
 *grids in real-space can be created from this layout \author Frederick Stein
 ******************************************************************************/
void grid_create_fft_grid_layout(fft_grid_layout **fft_grid,
                                 const cp_mpi_comm_t comm,
                                 const int npts_global[3],
                                 const double dh_inv[3][3],
                                 const bool use_halfspace);

/*******************************************************************************
 * \brief Print some information on a grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_print_grid_layout_info(const fft_grid_layout *layout,
                                 bool print_distribution);

/*******************************************************************************
 * \brief Create a FFT grid using a reference grid to interact with this grid.
 * \note The reference grid has had to be created using
 *grid_create_fft_grid_layout \author Frederick Stein
 ******************************************************************************/
void grid_create_fft_grid_layout_from_reference(
    fft_grid_layout **fft_grid, const int npts_global[3],
    const fft_grid_layout *fft_grid_ref);

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void grid_retain_fft_grid_layout(fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Convert between C indices (0...n-1) and shifted indices (-n/2...n/2).
 * \author Frederick Stein
 ******************************************************************************/
inline int convert_c_index_to_shifted_index(const int c_index, const int npts) {
  return (c_index > npts / 2 ? c_index - npts : c_index);
}

/*******************************************************************************
 * \brief Convert between shifted indices (-n/2...n/2) and C indices (0...n-1).
 * \author Frederick Stein
 ******************************************************************************/
inline int convert_shifted_index_to_c_index(const int shifted_index,
                                            const int npts) {
  return (shifted_index < 0 ? npts + shifted_index : shifted_index);
}

/*******************************************************************************
 * \brief Check whether a shifted index is on the grid.
 * \author Frederick Stein
 ******************************************************************************/
inline bool is_on_grid(const int shifted_index, const int npts) {
  return (shifted_index >= -(npts - 1) / 2 && shifted_index <= npts / 2);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT and sorts the data in g-space.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex-valued data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout(const double complex *restrict grid_rs,
                           double complex *restrict grid_gs,
                           const fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT and sorts the data in g-space.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex-valued data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout(const double *restrict grid_rs,
                               double complex *restrict grid_gs,
                               const fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param fft_grid FFT grid object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout(const double complex *restrict grid_gs,
                           double complex *restrict grid_rs,
                           const fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param fft_grid FFT grid object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout(const double complex *restrict grid_gs,
                               double *restrict grid_rs,
                               const fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT and sorts the data in g-space.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex-valued data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout_to_cart(const double complex *restrict grid_rs,
                                   double complex *restrict grid_gs,
                                   const fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT and sorts the data in g-space.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex-valued data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout_to_cart(const double *restrict grid_rs,
                                       double complex *restrict grid_gs,
                                       const fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param fft_grid FFT grid object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout_from_cart(const double complex *restrict grid_gs,
                                     double complex *restrict grid_rs,
                                     const fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param fft_grid FFT grid object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout_from_cart(const double complex *restrict grid_gs,
                                         double *restrict grid_rs,
                                         const fft_grid_layout *fft_grid);

#endif

// EOF
