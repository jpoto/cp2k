/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_DRIVER_H
#define FFT_DRIVER_H

#include "../mpiwrap/cp_mpi.h"

#include <complex.h>

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked_low(
    const double complex *restrict grid_rs, const bool is_complex,
    double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int npts_gs_local, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_blocked_low(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    const int (*index_to_g)[3], const int npts_gs_local,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked_low(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double complex *restrict grid_rs,
    const bool is_complex, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_blocked_low(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double *restrict grid_rs,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int (*proc2local_gs)[3][2], const cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_ray_low(const double complex *restrict grid_rs, bool is_complex,
                       double complex *restrict grid_gs,
                       const int (*index_to_g)[3], const int npts_gs_local,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int *rays_per_process, const int (*ray_to_xy)[2],
                       const cp_mpi_comm_t comm,
                       const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_ray_low(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    const int (*index_to_g)[3], const int npts_gs_local,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int *rays_per_process, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_ray_low(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double complex *restrict grid_rs,
    const bool is_complex, const int npts_global[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int *rays_per_process, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_ray_low(
    const double complex *restrict grid_gs, const int (*index_to_g)[3],
    const int number_of_local_gpts, double *restrict grid_rs,
    const int npts_global[3], const int npts_global_gspace[3],
    const int (*proc2local_rs)[3][2], const int (*proc2local_ms)[3][2],
    const int *rays_per_process, const int (*ray_to_xy)[2],
    const cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]);
#endif

// EOF
