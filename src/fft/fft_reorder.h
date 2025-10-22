/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_REORDER_H
#define FFT_REORDER_H

#include "../mpiwrap/cp_mpi.h"

#include <complex.h>

void collect_y_and_distribute_z_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int npts_global_gspace_2,
    const int (*proc2local)[3][2], const int (*proc2local_transposed)[3][2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]);

void collect_z_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int npts_global_gspace_2,
    const int (*proc2local)[3][2], const int (*proc2local_transposed)[3][2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]);

void collect_x_and_distribute_y_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

void collect_y_and_distribute_x_blocked_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

void collect_x_and_distribute_y_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

void collect_y_and_distribute_x_blocked(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

void collect_x_and_distribute_yz_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const cp_mpi_comm_t comm);

void collect_yz_and_distribute_x_ray(double complex *restrict grid,
                                     double complex *restrict transposed,
                                     const int npts_global[3],
                                     const int (*proc2local_transposed)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const cp_mpi_comm_t comm);

void collect_x_and_distribute_yz_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int *number_of_rays, const int (*ray_to_yz)[2],
    const cp_mpi_comm_t comm);

void collect_yz_and_distribute_x_ray_transpose(
    double complex *restrict grid, double complex *restrict transposed,
    const int npts_global[3], const int (*proc2local_transposed)[3][2],
    const int *number_of_rays, const int (*ray_to_yz)[2],
    const cp_mpi_comm_t comm);

#endif /* FFT_REORDER_H */

// EOF
