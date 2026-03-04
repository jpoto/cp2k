/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_REDISTRIBUTION_H
#define FFT_REDISTRIBUTION_H

#include "../mpiwrap/cp_mpi.h"

#include <complex.h>

typedef struct {
  // Some general information on the redistribution step
  int npts_global_gspace[3];
  int process_grid[2];
  int my_size_x_gs;
  int my_size_y_rs;
  int my_size_y_gs;
  int my_size_z_rs;
  // Displacements and counts for the different redistribution steps
  // x<->y
  int *displacements_xy_x;
  int *counts_xy_x;
  int *displacements_xy_y;
  int *counts_xy_y;
  // y<->z (non-transposed)
  int *displacements_yz_y;
  int *counts_yz_y;
  int *displacements_yz_z;
  int *counts_yz_z;
  // y<->z (ray-based)
  int *displacements_yz_ray_y;
  int *counts_yz_ray_y;
  int *xy_to_proc_ray_y;
  int *displacements_yz_ray_z;
  int *counts_yz_ray_z;
  int (*xy_to_proc_ray_z)[3];
} fft_redistribution_t;

void prepare_redistribution(fft_redistribution_t *redistribution,
                            const int npts_global_gspace[3],
                            const int (*proc2local_ms)[3][2],
                            const int (*proc2local_x_gs)[2],
                            const int (*proc2local_y_rs)[2],
                            const int (*proc2local_y_gs)[2],
                            const int (*proc2local_z_rs)[2],
                            const int *rays_per_process,
                   const int (*ray_to_xy)[2],
                            const cp_mpi_comm_t comm,
                            const cp_mpi_comm_t sub_comm[2]);

void cleanup_redistribution(fft_redistribution_t *redistribution);

/*******************************************************************************
 * \brief Performs the packing of (y_d,z_D,x)->(y,z_D,x_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked_pack(
    double complex *restrict grid, double complex *restrict grid_packed,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_x_ms)[2]);

/*******************************************************************************
 * \brief Performs the communication (y_d,z_D,x)->(y,z_D,x_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked_comm(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm);

/*******************************************************************************
 * \brief Performs the communication of the transposition of (y,z_d,x_d) ->
 *(y_d,z_d,x). \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked_comm(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm);

/*******************************************************************************
 * \brief Performs the unpacking of the transposition of (y,z_d,x_d) ->
 *(y_d,z_d,x). \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked_unpack(
    double complex *restrict grid_packed, double complex *restrict grid,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_x_ms)[2]);

/*******************************************************************************
 * \brief Performs the packing to a transposition of the kind
 *(x_d,y,z_D)->(z,x_D,y_D). \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked_transpose_pack(
    double complex *restrict grid, double complex *restrict grid_packed,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_y_gs)[2]);

/*******************************************************************************
 * \brief Performs the unpacking to transposition of the kind
 *(z,x_d,y_d)->(x_d,y,z_d). \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked_transpose_unpack(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_y_gs)[2]);

/*******************************************************************************
 * \brief Performs the packing to a redistribution of (z_d,x_d,y)->(z,x_d,y_d).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked_pack(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_y_gs)[2]);

/*******************************************************************************
 * \brief Performs the communication to a redistribution of
 *(z_d,x_d,y)->(z,x_d,y_d). \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked_comm(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm);

/*******************************************************************************
 * \brief Performs a redistribution of (z,x_d,y_d)->(z_d,x_d,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked_comm(
    double complex *restrict grid, double complex *restrict transposed,
    const fft_redistribution_t *redistribution, const cp_mpi_comm_t comm);

/*******************************************************************************
 * \brief Performs a redistribution of (z,x_d,y_d)->(z_d,x_d,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked_unpack(
    double complex *restrict grid_packed, double complex *restrict grid,
    const fft_redistribution_t *redistribution,
    const int (*proc2local_y_gs)[2]);

void collect_z_and_distribute_xy_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_xy)[2],
    const fft_redistribution_t *redistribution,
                                     const cp_mpi_comm_t comm);

void collect_xy_and_distribute_z_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local_transposed)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_xy)[2],
                                     const cp_mpi_comm_t comm);

void collect_z_and_distribute_xy_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int *number_of_rays, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm);

void collect_xy_and_distribute_z_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local_transposed)[3][2],
    const int *number_of_rays, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm);

#endif /* FFT_REORDER_H */

// EOF
