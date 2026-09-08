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

// Wrapper for GGA operations
int xc_cuda_wrapper_gga_exc_vxc(xc_func_type *func,
                                int np,
                                const double *rho,
                                double *exc, double *vrho, double *vsigma,
                                void *work);

// Wrapper for MGGA operations  
int xc_cuda_wrapper_mgga_exc_vxc(xc_func_type *func,
                                 int np,
                                 const double *rho, const double *sigma,
                                 const double *lapl, const double *tau,
                                 double *exc, double *vrho, double *vsigma,
                                 double *vlapl, double *vtau, void *work);

#ifdef __cplusplus
}
#endif

#endif // XC_CUDA_WRAPPER_H