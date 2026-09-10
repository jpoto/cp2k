/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#include "../offload/offload_buffer.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xc.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// Fortran function declarations
void libxc_host_to_device_2d_(void *buffer, double *host_2d, int *ncomp,
                              int *npoints);
void libxc_device_to_host_2d_(double *host_2d, void *buffer, int *ncomp,
                              int *npoints);

// Forward declaration
int xc_cuda_wrapper_is_available(void);

// CUDA initialization function
static void libxc_cuda_init(void) {
  static int initialized = 0;
  if (!initialized) {
    CUDA_CHECK(cudaFree(0)); // Initialize CUDA runtime
    initialized = 1;
  }
}

// Internal implementation using offload library
static void libxc_host_to_device_2d_impl(double *host_ptr, double *device_ptr,
                                         int ncomp, int npoints);
static void libxc_device_to_host_2d_impl(double *device_ptr, double *host_ptr,
                                         int ncomp, int npoints);

// Error handling and logging
#define LOG_ERROR(msg) fprintf(stderr, "[CUDA Wrapper Error] %s\n", msg)
#define LOG_WARNING(msg) fprintf(stderr, "[CUDA Wrapper Warning] %s\n", msg)
#define LOG_INFO(msg) fprintf(stderr, "[CUDA Wrapper Info] %s\n", msg)

// Memory registration status
typedef enum {
  MEMORY_UNREGISTERED = 0,
  MEMORY_REGISTERED,
  MEMORY_CUDA_ALLOCATED,
  MEMORY_HOST
} memory_status_t;

// Wrapper context to track memory state
typedef struct {
  void *original_ptr;
  void *cuda_ptr;
  size_t size;
  memory_status_t status;
  int registered;
} cuda_memory_context;

// Initialize memory context
static void init_memory_context(cuda_memory_context *ctx, void *ptr,
                                size_t size) {
  if (!ctx)
    return;

  ctx->original_ptr = ptr;
  ctx->cuda_ptr = ptr;
  ctx->size = size;
  ctx->status = MEMORY_UNREGISTERED;
  ctx->registered = 0;
}

// Clean up memory context
static void cleanup_memory_context(cuda_memory_context *ctx) {
  if (!ctx)
    return;

  if (ctx->registered && ctx->status == MEMORY_REGISTERED) {
    cudaError_t err = cudaHostUnregister(ctx->original_ptr);
    if (err != cudaSuccess) {
      LOG_WARNING("Failed to unregister memory");
    }
  }

  if (ctx->status == MEMORY_CUDA_ALLOCATED && ctx->cuda_ptr) {
    cudaFree(ctx->cuda_ptr);
  }

  ctx->original_ptr = NULL;
  ctx->cuda_ptr = NULL;
  ctx->size = 0;
  ctx->status = MEMORY_UNREGISTERED;
  ctx->registered = 0;
}

// Check pointer attributes and determine status
static memory_status_t check_pointer_status(void *ptr) {
  if (!ptr)
    return MEMORY_UNREGISTERED;

  struct cudaPointerAttributes attrs;
  memset(&attrs, 0, sizeof(attrs));

  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    return MEMORY_UNREGISTERED;
  }

  switch (attrs.type) {
  case cudaMemoryTypeHost:
    return MEMORY_HOST;
  case cudaMemoryTypeDevice:
    return MEMORY_CUDA_ALLOCATED;
  case cudaMemoryTypeManaged:
    return MEMORY_REGISTERED;
  case cudaMemoryTypeUnregistered:
  default:
    return MEMORY_UNREGISTERED;
  }
}

// Validate and prepare pointer for libxc CUDA operations
static int validate_and_prepare_pointer(cuda_memory_context *ctx) {
  if (!ctx || !ctx->original_ptr) {
    LOG_ERROR("Null pointer provided");
    return 0;
  }

  // Check current status
  ctx->status = check_pointer_status(ctx->original_ptr);

  switch (ctx->status) {
  case MEMORY_CUDA_ALLOCATED:
  case MEMORY_REGISTERED:
  case MEMORY_HOST:
    // Pointer is already in a valid state for CUDA
    ctx->cuda_ptr = ctx->original_ptr;
    return 1;

  case MEMORY_UNREGISTERED:
    // Try to register the memory with CUDA
    LOG_INFO("Registering unregistered memory with CUDA runtime");
    cudaError_t err =
        cudaHostRegister(ctx->original_ptr, ctx->size, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
      ctx->status = MEMORY_REGISTERED;
      ctx->registered = 1;
      ctx->cuda_ptr = ctx->original_ptr;
      return 1;
    } else {
      LOG_ERROR("Failed to register memory with CUDA");

      // Fallback: Allocate CUDA memory and copy data
      LOG_INFO("Fallback: Allocating CUDA memory and copying data");
      void *cuda_ptr;
      err = cudaMalloc(&cuda_ptr, ctx->size);
      if (err == cudaSuccess) {
        err = cudaMemcpy(cuda_ptr, ctx->original_ptr, ctx->size,
                         cudaMemcpyHostToDevice);
        if (err == cudaSuccess) {
          ctx->status = MEMORY_CUDA_ALLOCATED;
          ctx->cuda_ptr = cuda_ptr;
          return 1;
        } else {
          cudaFree(cuda_ptr);
          LOG_ERROR("Failed to copy data to CUDA memory");
          return 0;
        }
      } else {
        LOG_ERROR("Failed to allocate CUDA memory");
        return 0;
      }
    }
    break;

  default:
    LOG_ERROR("Unknown memory status");
    return 0;
  }
}

// Memory validation and translation context
typedef struct {
  cuda_memory_context rho_ctx;
  cuda_memory_context sigma_ctx;
  cuda_memory_context exc_ctx;
  cuda_memory_context vrho_ctx;
  cuda_memory_context vsigma_ctx;
  int use_cuda;
  int fallback_to_cpu;
} gga_wrapper_context;

// Wrapper for libxc GGA operations with memory validation
int xc_cuda_wrapper_gga_exc_vxc(xc_func_type *func, int np, const double *rho,
                                const double *sigma, double *exc, double *vrho,
                                double *vsigma, void *work) {
  if (!func) {
    LOG_ERROR("Null functional provided");
    return 1;
  }

  gga_wrapper_context ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.use_cuda = xc_cuda_wrapper_is_available();
  ctx.fallback_to_cpu = 0;

  // Initialize memory contexts
  size_t rho_size = np * sizeof(double);
  size_t sigma_size =
      3 * np * sizeof(double); // 3 components for spin-polarized
  size_t exc_size = np * sizeof(double);
  size_t vrho_size = 2 * np * sizeof(double);   // 2 components (alpha, beta)
  size_t vsigma_size = 3 * np * sizeof(double); // 3 components

  init_memory_context(&ctx.rho_ctx, (void *)rho, rho_size);
  init_memory_context(&ctx.exc_ctx, (void *)exc, exc_size);
  init_memory_context(&ctx.vrho_ctx, (void *)vrho, vrho_size);
  init_memory_context(&ctx.vsigma_ctx, (void *)vsigma, vsigma_size);

  // Validate and prepare input pointers
  if (!validate_and_prepare_pointer(&ctx.rho_ctx)) {
    LOG_ERROR("Failed to validate rho pointer");
    ctx.fallback_to_cpu = 1;
  }

  // Validate and prepare output pointers
  if (!validate_and_prepare_pointer(&ctx.exc_ctx) ||
      !validate_and_prepare_pointer(&ctx.vrho_ctx) ||
      !validate_and_prepare_pointer(&ctx.vsigma_ctx)) {
    LOG_ERROR("Failed to validate output pointers");
    ctx.fallback_to_cpu = 1;
  }

  // Handle sigma pointer (may be NULL for LDA)
  if (sigma) {
    init_memory_context(&ctx.sigma_ctx, (void *)sigma, sigma_size);
    if (!validate_and_prepare_pointer(&ctx.sigma_ctx)) {
      LOG_ERROR("Failed to validate sigma pointer");
      ctx.fallback_to_cpu = 1;
    }
  }

  // Use CUDA path if available and memory validation succeeded
  if (ctx.use_cuda && !ctx.fallback_to_cpu) {
    LOG_INFO("Using CUDA path for GGA operations");

    // Set CUDA execution flags
    xc_func_set_dens_threshold(func, 1e-12);

    // Call CUDA Libxc with validated pointers
    xc_gga_exc_vxc(
        func, np, (const double *)ctx.rho_ctx.cuda_ptr,
        (const double *)ctx.sigma_ctx.cuda_ptr, (double *)ctx.exc_ctx.cuda_ptr,
        (double *)ctx.vrho_ctx.cuda_ptr, (double *)ctx.vsigma_ctx.cuda_ptr);
  }

  // Fallback to CPU if CUDA failed or is not available
  if (!ctx.use_cuda || ctx.fallback_to_cpu) {
    LOG_INFO("Using CPU fallback for GGA operations");

    // Reset to host execution
    xc_func_set_dens_threshold(func, 1e-12);

    // Call CPU version
    xc_gga_exc_vxc(func, np, rho, sigma, exc, vrho, vsigma);
  }

  // Clean up memory contexts
  cleanup_memory_context(&ctx.rho_ctx);
  cleanup_memory_context(&ctx.exc_ctx);
  cleanup_memory_context(&ctx.vrho_ctx);
  cleanup_memory_context(&ctx.vsigma_ctx);
  if (sigma) {
    cleanup_memory_context(&ctx.sigma_ctx);
  }

  return ctx.fallback_to_cpu;
}

// Memory validation and translation context for MGGA
typedef struct {
  cuda_memory_context rho_ctx;
  cuda_memory_context sigma_ctx;
  cuda_memory_context lapl_ctx;
  cuda_memory_context tau_ctx;
  cuda_memory_context exc_ctx;
  cuda_memory_context vrho_ctx;
  cuda_memory_context vsigma_ctx;
  cuda_memory_context vlapl_ctx;
  cuda_memory_context vtau_ctx;
  int use_cuda;
  int fallback_to_cpu;
} mgga_wrapper_context;

// Wrapper for libxc MGGA operations with memory validation
int xc_cuda_wrapper_mgga_exc_vxc(xc_func_type *func, int np, const double *rho,
                                 const double *sigma, const double *lapl,
                                 const double *tau, double *exc, double *vrho,
                                 double *vsigma, double *vlapl, double *vtau,
                                 void *work) {
  if (!func) {
    LOG_ERROR("Null functional provided");
    return 1;
  }

  mgga_wrapper_context ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.use_cuda = xc_cuda_wrapper_is_available();
  ctx.fallback_to_cpu = 0;

  // Initialize memory contexts
  size_t rho_size = 2 * np * sizeof(double);   // 2 components (alpha, beta)
  size_t sigma_size = 3 * np * sizeof(double); // 3 components
  size_t lapl_size = 2 * np * sizeof(double);  // 2 components
  size_t tau_size = 2 * np * sizeof(double);   // 2 components
  size_t exc_size = np * sizeof(double);
  size_t vrho_size = 2 * np * sizeof(double);   // 2 components
  size_t vsigma_size = 3 * np * sizeof(double); // 3 components
  size_t vlapl_size = 2 * np * sizeof(double);  // 2 components
  size_t vtau_size = 2 * np * sizeof(double);   // 2 components

  // Initialize and validate input pointers
  init_memory_context(&ctx.rho_ctx, (void *)rho, rho_size);
  if (!validate_and_prepare_pointer(&ctx.rho_ctx)) {
    LOG_ERROR("Failed to validate rho pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.sigma_ctx, (void *)sigma, sigma_size);
  if (!validate_and_prepare_pointer(&ctx.sigma_ctx)) {
    LOG_ERROR("Failed to validate sigma pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.lapl_ctx, (void *)lapl, lapl_size);
  if (!validate_and_prepare_pointer(&ctx.lapl_ctx)) {
    LOG_ERROR("Failed to validate lapl pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.tau_ctx, (void *)tau, tau_size);
  if (!validate_and_prepare_pointer(&ctx.tau_ctx)) {
    LOG_ERROR("Failed to validate tau pointer");
    ctx.fallback_to_cpu = 1;
  }

  // Initialize and validate output pointers
  init_memory_context(&ctx.exc_ctx, (void *)exc, exc_size);
  if (!validate_and_prepare_pointer(&ctx.exc_ctx)) {
    LOG_ERROR("Failed to validate exc pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.vrho_ctx, (void *)vrho, vrho_size);
  if (!validate_and_prepare_pointer(&ctx.vrho_ctx)) {
    LOG_ERROR("Failed to validate vrho pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.vsigma_ctx, (void *)vsigma, vsigma_size);
  if (!validate_and_prepare_pointer(&ctx.vsigma_ctx)) {
    LOG_ERROR("Failed to validate vsigma pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.vlapl_ctx, (void *)vlapl, vlapl_size);
  if (!validate_and_prepare_pointer(&ctx.vlapl_ctx)) {
    LOG_ERROR("Failed to validate vlapl pointer");
    ctx.fallback_to_cpu = 1;
  }

  init_memory_context(&ctx.vtau_ctx, (void *)vtau, vtau_size);
  if (!validate_and_prepare_pointer(&ctx.vtau_ctx)) {
    LOG_ERROR("Failed to validate vtau pointer");
    ctx.fallback_to_cpu = 1;
  }

  // Use CUDA path if available and memory validation succeeded
  if (ctx.use_cuda && !ctx.fallback_to_cpu) {
    LOG_INFO("Using CUDA path for MGGA operations");

    // Set CUDA execution flags
    xc_func_set_dens_threshold(func, 1e-12);

    // Call CUDA Libxc with validated pointers
    int result = xc_mgga_exc_vxc(
        func, np, (const double *)ctx.rho_ctx.cuda_ptr,
        (const double *)ctx.sigma_ctx.cuda_ptr,
        (const double *)ctx.lapl_ctx.cuda_ptr,
        (const double *)ctx.tau_ctx.cuda_ptr, (double *)ctx.exc_ctx.cuda_ptr,
        (double *)ctx.vrho_ctx.cuda_ptr, (double *)ctx.vsigma_ctx.cuda_ptr,
        (double *)ctx.vlapl_ctx.cuda_ptr, (double *)ctx.vtau_ctx.cuda_ptr);

    if (result != 0) {
      LOG_ERROR("CUDA MGGA operation failed, falling back to CPU");
      ctx.fallback_to_cpu = 1;
    }
  }

  // Fallback to CPU if CUDA failed or is not available
  if (!ctx.use_cuda || ctx.fallback_to_cpu) {
    LOG_INFO("Using CPU fallback for MGGA operations");

    // Reset to host execution
    xc_func_set_dens_threshold(func, 1e-12);

    // Call CPU version
    xc_mgga_exc_vxc(func, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl,
                    vtau);
  }

  // Clean up memory contexts
  cleanup_memory_context(&ctx.rho_ctx);
  cleanup_memory_context(&ctx.sigma_ctx);
  cleanup_memory_context(&ctx.lapl_ctx);
  cleanup_memory_context(&ctx.tau_ctx);
  cleanup_memory_context(&ctx.exc_ctx);
  cleanup_memory_context(&ctx.vrho_ctx);
  cleanup_memory_context(&ctx.vsigma_ctx);
  cleanup_memory_context(&ctx.vlapl_ctx);
  cleanup_memory_context(&ctx.vtau_ctx);

  return ctx.fallback_to_cpu;
}

// Initialize CUDA wrapper
int xc_cuda_wrapper_init() {
  LOG_INFO("Initializing CUDA wrapper");

  // Check CUDA availability
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG_WARNING("CUDA error detected during initialization");
    // Don't return error - we can still use CPU fallback
  }

  return 0;
}

// Cleanup CUDA wrapper
void xc_cuda_wrapper_cleanup() {
  LOG_INFO("Cleaning up CUDA wrapper");

  // Synchronize any pending CUDA operations
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    LOG_WARNING("CUDA synchronization failed during cleanup");
  }
}

// Check if CUDA is available and functional
int xc_cuda_wrapper_is_available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    return 0;
  }

  return 1;
}

// Implementation using direct CUDA API for host to device transfer
static void libxc_host_to_device_2d_impl(double *host_ptr, double *device_ptr,
                                         int ncomp, int npoints) {
  int total_size = ncomp * npoints;
  size_t bytes = total_size * sizeof(double);

  // Allocate device memory
  double *d_temp;
  CUDA_CHECK(cudaMalloc((void **)&d_temp, bytes));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_temp, host_ptr, bytes, cudaMemcpyHostToDevice));

  // Return the device pointer
  memcpy(device_ptr, &d_temp, sizeof(double *));

  // Note: The caller is responsible for freeing the device memory
  // This is a simple implementation - in a production environment,
  // you would want to manage device memory more carefully
}

// Implementation using direct CUDA API for device to host transfer
static void libxc_device_to_host_2d_impl(double *device_ptr, double *host_ptr,
                                         int ncomp, int npoints) {
  int total_size = ncomp * npoints;
  size_t bytes = total_size * sizeof(double);

  // Extract the actual device pointer (stored as a pointer in device_ptr)
  double *d_data;
  memcpy(&d_data, device_ptr, sizeof(double *));

  // Copy data from device to host
  CUDA_CHECK(cudaMemcpy(host_ptr, d_data, bytes, cudaMemcpyDeviceToHost));

  // Free the device memory
  CUDA_CHECK(cudaFree(d_data));
}

// Fortran-compatible wrapper functions
void libxc_host_to_device_2d_(void *buffer, double *host_2d, int *ncomp,
                              int *npoints) {
  // Ensure CUDA is initialized
  libxc_cuda_init();

  // Call the implementation
  libxc_host_to_device_2d_impl(host_2d, (double *)buffer, *ncomp, *npoints);
}

void libxc_device_to_host_2d_(double *host_2d, void *buffer, int *ncomp,
                              int *npoints) {
  // Ensure CUDA is initialized
  libxc_cuda_init();

  // Call the implementation
  libxc_device_to_host_2d_impl((double *)buffer, host_2d, *ncomp, *npoints);
}

// C wrapper for Fortran libxc_host_to_device_2d function
void libxc_host_to_device_2d(void *buffer, double *host_2d, int ncomp,
                             int npoints) {
  libxc_host_to_device_2d_impl(host_2d, (double *)buffer, ncomp, npoints);
}

// C wrapper for Fortran libxc_device_to_host_2d function
void libxc_device_to_host_2d(double *host_2d, void *buffer, int ncomp,
                             int npoints) {
  libxc_device_to_host_2d_impl((double *)buffer, host_2d, ncomp, npoints);
}