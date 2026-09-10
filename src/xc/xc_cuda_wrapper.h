/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#ifndef XC_CUDA_WRAPPER_H
#define XC_CUDA_WRAPPER_H

#include <xc.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the CUDA wrapper
int xc_cuda_wrapper_init();

// Cleanup the CUDA wrapper
void xc_cuda_wrapper_cleanup();

// Check if CUDA is available
int xc_cuda_wrapper_is_available();

// CUDA memory transfer functions
void libxc_host_to_device_2d(void *buffer, double *host_2d, int ncomp,
                             int npoints);
void libxc_device_to_host_2d(double *host_2d, void *buffer, int ncomp,
                             int npoints);

// Wrapper for GGA operations
int xc_cuda_wrapper_gga_exc_vxc(xc_func_type *func, int np, const double *rho,
                                double *exc, double *vrho, double *vsigma,
                                void *work);

// Wrapper for GGA operations
int xc_cuda_wrapper_gga_exc_vxc(xc_func_type *func, int np, const double *rho,
                                double *exc, double *vrho, double *vsigma,
                                void *work);

// Wrapper for MGGA operations
int xc_cuda_wrapper_mgga_exc_vxc(xc_func_type *func, int np, const double *rho,
                                 const double *sigma, const double *lapl,
                                 const double *tau, double *exc, double *vrho,
                                 double *vsigma, double *vlapl, double *vtau,
                                 void *work);

#ifdef __cplusplus
}
#endif

#endif // XC_CUDA_WRAPPER_H