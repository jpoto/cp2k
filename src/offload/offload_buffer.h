/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef OFFLOAD_BUFFER_H
#define OFFLOAD_BUFFER_H

#include <stddef.h>

/*******************************************************************************
 * \brief Internal representation of a buffer.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  size_t size;
  double *host_buffer;
  double *device_buffer;
  /* Host/device authority flags for the grid-residency work. Default: the host
   * buffer is authoritative (existing behaviour). Unified-memory builds keep
   * both current. Nothing consumes these yet; consumers opt in progressively.
   */
  int host_current;
  int device_current;
  /* Grid-residency opt-in (Phase A). When set, the grid backend may leave a
   * freshly collocated grid in device_buffer (skip the D2H) and may skip the
   * H2D before integrate when device_current is set. The caller promises that
   * the buffer is consumed on the device before any host read. Default: 0
   * (existing host-authoritative behaviour). */
  int device_resident;
  /* Optional event recording when the buffer's device copy was produced on a
   * non-default stream (grid backend). Consumers wait on it before reading. */
  void *ready_event;
} offload_buffer;

/*******************************************************************************
 * \brief Allocate a buffer (NULL or valid) with the given number of elements.
 * \author Ole Schuett
 ******************************************************************************/
void offload_create_buffer(const int length, offload_buffer **buffer);

/*******************************************************************************
 * \brief Deallocate given buffer.
 * \author Ole Schuett
 ******************************************************************************/
void offload_free_buffer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Return a pointer to the host buffer (Fortran API).
 * \author Ole Schuett
 ******************************************************************************/
double *offload_get_buffer_host_pointer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Return a pointer to the device buffer (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
double *offload_get_buffer_device_pointer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Copy data from host to device (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_h2d(offload_buffer *buffer, const int length);

/*******************************************************************************
 * \brief Copy data from device to host (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_d2h(offload_buffer *buffer, const int length);

/*******************************************************************************
 * \brief Host/device authority flags (grid-residency). Additive: default is
 *        host_current=1, device_current=0 (i.e. existing behaviour).
 ******************************************************************************/
int offload_buffer_is_device_current(const offload_buffer *buffer);
void offload_buffer_mark_device_current(offload_buffer *buffer);
void offload_buffer_mark_host_current(offload_buffer *buffer);
int offload_buffer_is_device_resident(const offload_buffer *buffer);
void offload_buffer_set_device_resident(offload_buffer *buffer,
                                        const int value);
/* Record/wait a producer-stream event stored on the buffer. */
void offload_buffer_record_event(offload_buffer *buffer, void *stream);
/*******************************************************************************
 * \brief Ensure host is up to date (D2H only if the device is authoritative).
 ******************************************************************************/
void offload_buffer_sync_to_host(offload_buffer *buffer, const int length);
/*******************************************************************************
 * \brief Ensure device is up to date (H2D only if the host is authoritative).
 ******************************************************************************/

/*******************************************************************************
 * \brief Allocate device-only memory (no host mirror), reusable across calls.
 ******************************************************************************/
void *offload_device_malloc(size_t bytes);

/*******************************************************************************
 * \brief Free device-only memory obtained from offload_device_malloc.
 ******************************************************************************/
void offload_device_free(void *device_memory);

/*******************************************************************************
 * \brief Copy host -> device-only buffer (synchronous).
 ******************************************************************************/
void offload_device_h2d(void *device_memory, const void *host_memory,
                        size_t bytes);

/*******************************************************************************
 * \brief Copy device-only buffer -> host (synchronous).
 ******************************************************************************/
void offload_device_d2h(void *host_memory, const void *device_memory,
                        size_t bytes);

/*******************************************************************************
 * \brief Copy device-only buffer -> device-only buffer (synchronous).
 ******************************************************************************/
void offload_device_d2d(void *dst_device, const void *src_device, size_t bytes);

#endif

// EOF
