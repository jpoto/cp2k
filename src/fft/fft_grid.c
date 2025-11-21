/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_timer.h"
#include "fft_utils.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Add one grid to another one in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void grid_add_to_fine_grid(const fft_complex_gs_grid *coarse_grid,
                           const fft_complex_gs_grid *fine_grid) {
  assert(coarse_grid != NULL);
  assert(fine_grid != NULL);
  for (int index = 0; index < coarse_grid->fft_grid_layout->npts_gs_local;
       index++) {
    const int ref_index =
        coarse_grid->fft_grid_layout->local_index_to_ref_grid[index];
    for (int dir = 0; dir < 3; dir++)
      assert(convert_c_index_to_shifted_index(
                 coarse_grid->fft_grid_layout->index_to_g[index][dir],
                 coarse_grid->fft_grid_layout->npts_global[dir]) ==
             convert_c_index_to_shifted_index(
                 fine_grid->fft_grid_layout->index_to_g[ref_index][dir],
                 fine_grid->fft_grid_layout->npts_global[dir]));
    fine_grid->data[ref_index] += coarse_grid->data[index];
  }
}

/*******************************************************************************
 * \brief Copy fine grid to coarse grid in reciprocal space
 * \author Frederick Stein
 ******************************************************************************/
void grid_copy_to_coarse_grid(const fft_complex_gs_grid *fine_grid,
                              const fft_complex_gs_grid *coarse_grid) {
  assert(fine_grid != NULL);
  assert(coarse_grid != NULL);
  for (int index = 0; index < coarse_grid->fft_grid_layout->npts_gs_local;
       index++) {
    const int ref_index =
        coarse_grid->fft_grid_layout->local_index_to_ref_grid[index];
    for (int dir = 0; dir < 3; dir++) {
      assert(convert_c_index_to_shifted_index(
                 coarse_grid->fft_grid_layout->index_to_g[index][dir],
                 coarse_grid->fft_grid_layout->npts_global[dir]) ==
             convert_c_index_to_shifted_index(
                 fine_grid->fft_grid_layout->index_to_g[ref_index][dir],
                 fine_grid->fft_grid_layout->npts_global[dir]));
    }
    coarse_grid->data[index] = fine_grid->data[ref_index];
  }
}

/*******************************************************************************
 * \brief Create a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_real_rs_grid(fft_real_rs_grid *grid,
                              fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  assert(grid_layout->ref_counter > 0);
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  const int(*my_bounds)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(grid_layout->comm)];
  int number_of_elements = 1;
  for (int dir = 0; dir < 3; dir++) {
    number_of_elements *= imax(0, my_bounds[dir][1]);
  }
  grid->data = NULL;
  fft_allocate_double(number_of_elements, &grid->data);
}

/*******************************************************************************
 * \brief Create a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_rs_grid(fft_complex_rs_grid *grid,
                                 fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  assert(grid_layout->ref_counter > 0);
  assert(!grid_layout->use_halfspace &&
         "Complex RS grid require the whole g-space!");
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  const int(*my_bounds)[2] =
      grid_layout->proc2local_rs[cp_mpi_comm_rank(grid_layout->comm)];
  int number_of_elements = 1;
  for (int dir = 0; dir < 3; dir++) {
    number_of_elements *= imax(0, my_bounds[dir][1]);
  }
  grid->data = NULL;
  fft_allocate_complex(number_of_elements, &grid->data);
}

/*******************************************************************************
 * \brief Create a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_gs_grid(fft_complex_gs_grid *grid,
                                 fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  grid->data = NULL;
  fft_allocate_complex(grid_layout->npts_gs_local, &grid->data);
}

/*******************************************************************************
 * \brief Create a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_cart_gs_grid(fft_complex_cart_gs_grid *grid,
                                      fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  assert(grid_layout->ref_counter > 0);
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  const int(*my_bounds)[2] =
      grid_layout->proc2local_gs[cp_mpi_comm_rank(grid_layout->comm)];
  int number_of_elements = 1;
  for (int dir = 0; dir < 3; dir++) {
    number_of_elements *= imax(0, my_bounds[dir][1]);
  }
  grid->data = NULL;
  fft_allocate_complex(number_of_elements, &grid->data);
}

/*******************************************************************************
 * \brief Frees a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_rs_grid(fft_complex_rs_grid *grid) {
  if (grid != NULL) {
    fft_free_complex(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Frees a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_real_rs_grid(fft_real_rs_grid *grid) {
  if (grid != NULL) {
    fft_free_double(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Frees a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_gs_grid(fft_complex_gs_grid *grid) {
  if (grid != NULL) {
    fft_free_complex(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Frees a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_cart_gs_grid(fft_complex_cart_gs_grid *grid) {
  if (grid != NULL) {
    fft_free_complex(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw(const fft_complex_rs_grid *grid_rs,
            const fft_complex_gs_grid *grid_gs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_fw_c2c_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_gs->fft_grid_layout;
  fft_3d_fw_with_layout(grid_rs->data, grid_gs->data, grid_layout);
  const double scale = 1.0 / (((double)product3(grid_layout->npts_global)));
  const int incx = 1;
  zdscal_(&grid_layout->npts_gs_local, &scale, grid_gs->data, &incx);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw_to_cart(const fft_complex_rs_grid *grid_rs,
                    const fft_complex_cart_gs_grid *grid_gs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_fw_c2c_to_cart_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_gs->fft_grid_layout;
  fft_3d_fw_with_layout_to_cart(grid_rs->data, grid_gs->data, grid_layout);
  const double scale = 1.0 / (((double)product3(grid_layout->npts_global)));
  const int incx = 1;
  zdscal_(&grid_layout->npts_gs_local, &scale, grid_gs->data, &incx);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw_r2c(const fft_real_rs_grid *grid_rs,
                const fft_complex_gs_grid *grid_gs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_fw_r2c_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_gs->fft_grid_layout;
  fft_3d_fw_r2c_with_layout(grid_rs->data, grid_gs->data, grid_layout);
  const double scale =
      1.0 / (((double)grid_layout->npts_global[0]) *
             grid_layout->npts_global[1] * grid_layout->npts_global[2]);
  const int incx = 1;
  zdscal_(&grid_layout->npts_gs_local, &scale, grid_gs->data, &incx);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fw_r2c_to_cart(const fft_real_rs_grid *grid_rs,
                        const fft_complex_cart_gs_grid *grid_gs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_fw_r2c_cart_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_gs->fft_grid_layout;
  fft_3d_fw_r2c_with_layout_to_cart(grid_rs->data, grid_gs->data, grid_layout);
  const double scale =
      1.0 / (((double)grid_layout->npts_global[0]) *
             grid_layout->npts_global[1] * grid_layout->npts_global[2]);
  const int incx = 1;
  zdscal_(&grid_layout->npts_gs_local, &scale, grid_gs->data, &incx);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw(const fft_complex_gs_grid *grid_gs,
            const fft_complex_rs_grid *grid_rs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_bw_c2c_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  fft_3d_bw_with_layout(grid_gs->data, grid_rs->data, grid_layout);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw_from_cart(const fft_complex_cart_gs_grid *grid_gs,
                      const fft_complex_rs_grid *grid_rs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_bw_c2c_cart_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  fft_3d_bw_with_layout_from_cart(grid_gs->data, grid_rs->data, grid_layout);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw_c2r(const fft_complex_gs_grid *grid_gs,
                const fft_real_rs_grid *grid_rs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_bw_r2c_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  fft_3d_bw_c2r_with_layout(grid_gs->data, grid_rs->data, grid_layout);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_bw_c2r_from_cart(const fft_complex_cart_gs_grid *grid_gs,
                          const fft_real_rs_grid *grid_rs) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_bw_r2c_cart_%i_%i_%i_%i",
           cp_mpi_comm_size(grid_rs->fft_grid_layout->comm),
           grid_rs->fft_grid_layout->npts_global[0],
           grid_rs->fft_grid_layout->npts_global[1],
           grid_rs->fft_grid_layout->npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  fft_3d_bw_c2r_with_layout_from_cart(grid_gs->data, grid_rs->data,
                                      grid_layout);
  fft_stop_timer(handle);
}

// EOF
