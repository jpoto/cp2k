#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <xc.h>

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
static void init_memory_context(cuda_memory_context *ctx, void *ptr, size_t size) {
    if (!ctx) return;
    
    ctx->original_ptr = ptr;
    ctx->cuda_ptr = ptr;
    ctx->size = size;
    ctx->status = MEMORY_UNREGISTERED;
    ctx->registered = 0;
}

// Clean up memory context
static void cleanup_memory_context(cuda_memory_context *ctx) {
    if (!ctx) return;
    
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
    if (!ptr) return MEMORY_UNREGISTERED;
    
    cudaPointerAttributes attrs;
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
            cudaError_t err = cudaHostRegister(ctx->original_ptr, ctx->size, 
                                             cudaHostRegisterDefault);
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

// Wrapper for libxc GGA operations
int xc_cuda_wrapper_gga_exc_vxc(xc_func_type *func, 
                                int np, 
                                const double *rho, 
                                double *exc, double *vrho, double *vsigma,
                                void *work) {
    if (!func) {
        LOG_ERROR("Null functional provided");
        return 1;
    }
    
    // For now, use CPU path as a safe fallback
    // TODO: Implement full CUDA path with proper memory management
    LOG_INFO("Using CPU fallback for GGA operations");
    
    int ret = xc_gga_exc_vxc(func, np, rho, exc, vrho, vsigma);
    if (ret != 0) {
        LOG_ERROR("libxc GGA operation failed");
        return ret;
    }
    
    return 0;
}

// Wrapper for libxc MGGA operations
int xc_cuda_wrapper_mgga_exc_vxc(xc_func_type *func, 
                                 int np, 
                                 const double *rho, const double *sigma,
                                 const double *lapl, const double *tau,
                                 double *exc, double *vrho, double *vsigma,
                                 double *vlapl, double *vtau, void *work) {
    if (!func) {
        LOG_ERROR("Null functional provided");
        return 1;
    }
    
    // For now, use CPU path as a safe fallback
    // TODO: Implement full CUDA path with proper memory management
    LOG_INFO("Using CPU fallback for MGGA operations");
    
    int ret = xc_mgga_exc_vxc(func, np, rho, sigma, lapl, tau, 
                              exc, vrho, vsigma, vlapl, vtau);
    if (ret != 0) {
        LOG_ERROR("libxc MGGA operation failed");
        return ret;
    }
    
    return 0;
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