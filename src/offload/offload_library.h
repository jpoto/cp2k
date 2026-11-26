/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef OFFLOAD_LIBRARY_H
#define OFFLOAD_LIBRARY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/*******************************************************************************
 * \brief Initalize runtime.
 * \author Rocco Meli
 ******************************************************************************/
void offload_init(void);

/*******************************************************************************
 * \brief Returns the number of available devices.
 * \author Ole Schuett
 ******************************************************************************/
int offload_get_device_count(void);

/*******************************************************************************
 * \brief Selects the chosen device to be used.
 * \author Ole Schuett
 ******************************************************************************/
void offload_set_chosen_device(int device_id);

/*******************************************************************************
 * \brief Blocks until outstanding work on the legacy default stream is done
 *        (cheaper than a whole-device sync; safe under the default-stream
 * model).
 ******************************************************************************/
void offload_default_stream_synchronize(void);

/*******************************************************************************
 * \brief Opaque GPU event helpers for cross-stream ordering. A NULL stream
 *        argument denotes the legacy default stream.
 ******************************************************************************/
void *offload_event_create(void);
void offload_event_destroy(void *event);
void offload_event_record(void *event, void *stream);
void offload_event_wait(void *event, void *stream);
void offload_event_synchronize(void *event);

/*******************************************************************************
 * \brief Returns the chosen device.
 * \author Ole Schuett
 ******************************************************************************/
int offload_get_chosen_device(void);

/*******************************************************************************
 * \brief Activates the device selected via offload_set_chosen_device()
 * \author Ole Schuett
 ******************************************************************************/
void offload_activate_chosen_device(void);

/*******************************************************************************
 * \brief Starts a timing range.
 * \author Ole Schuett
 ******************************************************************************/
void offload_timeset(const char *message);

/*******************************************************************************
 * \brief Ends a timing range.
 * \author Ole Schuett
 ******************************************************************************/
void offload_timestop(void);

/*******************************************************************************
 * \brief Gets free and total device memory.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mem_info(size_t *free, size_t *total);

/*******************************************************************************
 * \brief Allocate pinned memory (or simple malloc when there is no gpu)
 ******************************************************************************/
int offload_host_malloc(void **ptr__, const size_t size__);

/*******************************************************************************
 * \brief free pinned memory (or simple free when there is no gpu)
 ******************************************************************************/
int offload_host_free(void *ptr__);

void offload_device_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif

// EOF
