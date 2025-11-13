/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)

#include "../../offload/offload_fft.h"
#include "../../offload/offload_library.h"
#include "fft_gpu_kernels.h"

#include <assert.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/*******************************************************************************
 * \brief Initializes the fft_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_init(void);

/*******************************************************************************
 * \brief Releases resources held by the fft_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_finalize(void);

/*******************************************************************************
 * \brief Checks size of device buffers and re-allocates them if necessary.
 * \author Ole Schuett
 ******************************************************************************/
static void ensure_memory_sizes(const size_t requested_buffer_size,
                                const size_t requested_map_size);

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)

// EOF
