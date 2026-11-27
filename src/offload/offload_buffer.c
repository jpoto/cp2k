/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "offload_buffer.h"
#include "offload_library.h"
#include "offload_mempool.h"
#include "offload_runtime.h"

#include "../base/cp_assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if 1
#define OFFLOAD_BUFFER_MEMPOOL
#endif

/*******************************************************************************
 * \brief Internal routine to deallocate given buffer.
 * \author Hans Pabst
 ******************************************************************************/
static void offload_free_buffer_internal(offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  if (buffer->ready_event != NULL) {
    offload_event_destroy(buffer->ready_event);
    buffer->ready_event = NULL;
  }
#if defined(OFFLOAD_BUFFER_MEMPOOL)
  if (NULL != buffer->host_buffer &&
      buffer->host_buffer != buffer->device_buffer) {
    offload_mempool_host_free(buffer->host_buffer);
  }
  if (NULL != buffer->device_buffer) {
    offload_mempool_device_free(buffer->device_buffer);
  }
#elif defined(__OFFLOAD)
  if (NULL != buffer->host_buffer &&
      buffer->host_buffer != buffer->device_buffer) {
    offloadFreeHost(buffer->host_buffer);
  }
  if (NULL != buffer->device_buffer) {
    offloadFree(buffer->device_buffer);
  }
#else
  free(buffer->host_buffer);
  cp_assert(NULL == buffer->device_buffer);
#endif
}

/*******************************************************************************
 * \brief Allocates a buffer (NULL or valid) with the given number of elements.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
void offload_create_buffer(const int length, offload_buffer **buffer) {
  const size_t requested_size = sizeof(double) * length;

  if (*buffer != NULL) {
    if (0 == requested_size) {
      offload_free_buffer_internal(*buffer);
      (*buffer)->size = 0;
      (*buffer)->host_buffer = NULL;
      (*buffer)->device_buffer = NULL;
      (*buffer)->host_current = 0;
      (*buffer)->device_current = 0;
      (*buffer)->device_resident = 0;
      (*buffer)->ready_event = NULL;
      return;
    } else if (requested_size <= (*buffer)->size) {
      return; // reuse existing buffer
    } else {
      offload_free_buffer_internal(*buffer);
    }
  } else {
    (*buffer) = malloc(sizeof(offload_buffer));
    cp_assert(NULL != *buffer);
  }

  (*buffer)->size = requested_size;
  (*buffer)->host_buffer = NULL;
  (*buffer)->device_buffer = NULL;
  (*buffer)->host_current = 0;
  (*buffer)->device_current = 0;
  (*buffer)->device_resident = 0;
  (*buffer)->ready_event = NULL;
  if (0 == requested_size) {
    return;
  }
#if defined(OFFLOAD_BUFFER_MEMPOOL)
#if !defined(__OFFLOAD_UNIFIED_MEMORY)
  (*buffer)->host_buffer = offload_mempool_host_malloc(requested_size);
#endif
  (*buffer)->device_buffer = offload_mempool_device_malloc(requested_size);
#elif defined(__OFFLOAD)
  offload_activate_chosen_device();
#if !defined(__OFFLOAD_UNIFIED_MEMORY)
  offloadMallocHost((void **)&(*buffer)->host_buffer, requested_size);
#endif
  offloadMalloc((void **)&(*buffer)->device_buffer, requested_size);
#else
  (*buffer)->host_buffer = malloc(requested_size);
  (*buffer)->device_buffer = NULL;
#endif
  if (NULL == (*buffer)->host_buffer) { /* unified memory */
    (*buffer)->host_buffer = (*buffer)->device_buffer;
  }
  // Host is authoritative right after creation (code fills/uses host first).
  (*buffer)->host_current = 1;
  (*buffer)->device_current =
      ((*buffer)->host_buffer == (*buffer)->device_buffer) ? 1 : 0;
}

/*******************************************************************************
 * \brief Deallocate given buffer.
 * \author Ole Schuett
 ******************************************************************************/
void offload_free_buffer(offload_buffer *buffer) {
  if (NULL != buffer) {
    offload_free_buffer_internal(buffer);
    free(buffer);
  }
}

/*******************************************************************************
 * \brief Returns a pointer to the host buffer (Fortran API).
 * \author Ole Schuett
 ******************************************************************************/
double *offload_get_buffer_host_pointer(offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  return buffer->host_buffer;
}

/*******************************************************************************
 * \brief Returns a pointer to the device buffer (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
double *offload_get_buffer_device_pointer(offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  return buffer->device_buffer;
}

/*******************************************************************************
 * \brief Copy data from host to device (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_h2d(offload_buffer *buffer, const int length) {
  cp_assert(NULL != buffer);
#if defined(__OFFLOAD)
  if (buffer->host_buffer != NULL && buffer->device_buffer != NULL) {
    const size_t bytes = sizeof(double) * length;
    offloadMemcpyAsyncHtoD(buffer->device_buffer, buffer->host_buffer, bytes,
                           0);
    buffer->device_current = 1;
  }
#else
  (void)length; // no-op on CPU-only builds: host memory is the same store
#endif
}

/*******************************************************************************
 * \brief Copy data from device to host (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_d2h(offload_buffer *buffer, const int length) {
  cp_assert(NULL != buffer);
#if defined(__OFFLOAD)
  if (buffer->host_buffer != NULL && buffer->device_buffer != NULL) {
    const size_t bytes = sizeof(double) * length;
    offloadMemcpyAsyncDtoH(buffer->host_buffer, buffer->device_buffer, bytes,
                           0);
    buffer->host_current = 1;
  }
#else
  (void)length; // no-op on CPU-only builds: host memory is the same store
#endif
}

/*******************************************************************************
 * \brief Host/device authority flags and lazy sync (grid-residency).
 ******************************************************************************/
int offload_buffer_is_device_current(const offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  return buffer->device_current;
}
void offload_buffer_mark_device_current(offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  buffer->device_current = 1;
  buffer->host_current = 0;
}

void offload_buffer_mark_host_current(offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  buffer->host_current = 1;
  buffer->device_current = 0;
}

int offload_buffer_is_device_resident(const offload_buffer *buffer) {
  cp_assert(NULL != buffer);
  return buffer->device_resident;
}

void offload_buffer_set_device_resident(offload_buffer *buffer,
                                        const int value) {
  cp_assert(NULL != buffer);
  buffer->device_resident = value;
}

void offload_buffer_record_event(offload_buffer *buffer, void *stream) {
  cp_assert(NULL != buffer);
  if (buffer->ready_event == NULL) {
    buffer->ready_event = offload_event_create();
  }
  offload_event_record(buffer->ready_event, stream);
}
void offload_buffer_d2d_streamed(offload_buffer *wait_buffer, void *dst_device,
                                 const void *src_device, size_t bytes) {
  const void *event = (wait_buffer != NULL) ? wait_buffer->ready_event : NULL;
  offload_device_d2d_streamed((void *)event, dst_device, src_device, bytes);
}

void offload_buffer_sync_to_host(offload_buffer *buffer, const int length) {
  if (buffer->device_current && !buffer->host_current) {
    offload_buffer_d2h(buffer, length);
  }
}
/*******************************************************************************
 * \brief Allocate device-only memory (no host mirror).
 *
 * Uses the device mempool (same as offload_buffer) so that repeated
 * allocate/free of device-only buffers (e.g. the pw device mirrors) reuse
 * cached chunks instead of hitting cudaMalloc/cudaFree every time.
 ******************************************************************************/
void *offload_device_malloc(size_t bytes) {
  if (0 == bytes)
    return NULL;
  offload_activate_chosen_device();
#if defined(__OFFLOAD)
  return offload_mempool_device_malloc(bytes);
#else
  return malloc(bytes);
#endif
}

/*******************************************************************************
 * \brief Free device-only memory.
 ******************************************************************************/
void offload_device_free(void *device_memory) {
  if (NULL == device_memory)
    return;
#if defined(__OFFLOAD)
  offload_mempool_device_free(device_memory);
#else
  free(device_memory);
#endif
}

/*******************************************************************************
 * \brief Copy host -> device-only buffer.
 ******************************************************************************/
void offload_device_h2d(void *device_memory, const void *host_memory,
                        size_t bytes) {
  if (NULL == device_memory || NULL == host_memory || 0 == bytes)
    return;
  offload_activate_chosen_device();
#if defined(__OFFLOAD)
  offloadMemcpyHtoD(device_memory, host_memory, bytes);
#else
  memcpy(device_memory, host_memory, bytes);
#endif
}

/*******************************************************************************
 * \brief Copy device-only buffer -> host.
 ******************************************************************************/
void offload_device_d2h(void *host_memory, const void *device_memory,
                        size_t bytes) {
  if (NULL == device_memory || NULL == host_memory || 0 == bytes)
    return;
  offload_activate_chosen_device();
#if defined(__OFFLOAD)
  offloadMemcpyDtoH(host_memory, device_memory, bytes);
#else
  memcpy(host_memory, device_memory, bytes);
#endif
}

/*******************************************************************************
 * \brief Copy device-only buffer -> device-only buffer.
 ******************************************************************************/
void offload_device_d2d(void *dst_device, const void *src_device,
                        size_t bytes) {
  if (NULL == dst_device || NULL == src_device || 0 == bytes)
    return;
  offload_activate_chosen_device();
#if defined(__OFFLOAD)
  offloadMemcpyAsyncDtoD(dst_device, src_device, bytes, 0);
  // The copy runs on the legacy default stream, which is implicitly ordered
  // against the other (blocking) streams. Synchronising just that stream makes
  // the copy complete for the caller without waiting for unrelated kernels on
  // other streams (a whole-device sync used to serialise against those).
  offloadStreamSynchronize((offloadStream_t)0);
#else
  memcpy(dst_device, src_device, bytes);
#endif
}

// EOF
