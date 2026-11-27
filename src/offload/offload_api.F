!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Fortran API for the offload package, which is written in C.
!> \author Ole Schuett
! **************************************************************************************************
MODULE offload_api
   USE ISO_C_BINDING,                   ONLY: &
        C_ASSOCIATED, C_CHAR, C_FUNLOC, C_FUNPTR, C_F_POINTER, C_INT, C_NULL_CHAR, C_NULL_PTR, &
        C_PTR, C_SIZE_T
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE message_passing,                 ONLY: mp_comm_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'offload_api'

   PUBLIC :: offload_init
   PUBLIC :: offload_get_device_count
   PUBLIC :: offload_set_chosen_device, offload_get_chosen_device, offload_activate_chosen_device
   PUBLIC :: offload_timeset, offload_timestop, offload_mem_info
   PUBLIC :: offload_buffer_type, offload_create_buffer, offload_free_buffer, &
             offload_buffer_h2d, offload_buffer_d2h, &
             offload_get_buffer_device_pointer, offload_device_synchronize, &
             offload_default_stream_synchronize, offload_event_create, &
             offload_event_destroy, offload_event_record, offload_event_wait, &
             offload_event_synchronize
   PUBLIC :: offload_malloc_pinned_mem, offload_free_pinned_mem
   PUBLIC :: offload_device_alloc, offload_device_free, offload_device_h2d, &
             offload_device_d2h, offload_device_d2d
   PUBLIC :: offload_mempool_stats_print
   PUBLIC :: offload_mempool_device_bytes
   PUBLIC :: offload_get_buffer_host_pointer
   PUBLIC :: offload_buffer_is_device_current, offload_buffer_is_host_current, &
             offload_buffer_mark_device_current, offload_buffer_mark_host_current, &
             offload_buffer_sync_to_host, offload_buffer_sync_to_device, &
             offload_buffer_is_device_resident, offload_buffer_set_device_resident, &
             offload_buffer_wait_event, offload_buffer_d2d_streamed
   PUBLIC :: offload_grid_resident_enabled

   TYPE offload_buffer_type
      REAL(KIND=dp), DIMENSION(:), CONTIGUOUS, POINTER :: host_buffer => Null()
      TYPE(C_PTR)                          :: c_ptr = C_NULL_PTR
   END TYPE offload_buffer_type

CONTAINS

! **************************************************************************************************
!> \brief allocate pinned memory.
!> \param buffer address of the buffer
!> \param length length of the buffer
!> \return 0
! **************************************************************************************************
   FUNCTION offload_malloc_pinned_mem(buffer, length) RESULT(res)
      TYPE(C_PTR)                                        :: buffer
      INTEGER(C_SIZE_T), VALUE                           :: length
      INTEGER                                            :: res

      INTERFACE
         FUNCTION offload_malloc_pinned_mem_c(buffer, length) &
            BIND(C, name="offload_host_malloc")
            IMPORT C_SIZE_T, C_PTR, C_INT
            TYPE(C_PTR)              :: buffer
            INTEGER(C_SIZE_T), VALUE :: length
            INTEGER(KIND=C_INT)                :: offload_malloc_pinned_mem_c
         END FUNCTION offload_malloc_pinned_mem_c
      END INTERFACE

      res = offload_malloc_pinned_mem_c(buffer, length)
   END FUNCTION offload_malloc_pinned_mem

! **************************************************************************************************
!> \brief free pinned memory
!> \param buffer address of the buffer
!> \return 0
! **************************************************************************************************
   FUNCTION offload_free_pinned_mem(buffer) RESULT(res)
      TYPE(C_PTR), VALUE                                 :: buffer
      INTEGER                                            :: res

      INTERFACE
         FUNCTION offload_free_pinned_mem_c(buffer) &
            BIND(C, name="offload_host_free")
            IMPORT C_PTR, C_INT
            INTEGER(KIND=C_INT)                :: offload_free_pinned_mem_c
            TYPE(C_PTR), VALUE            :: buffer
         END FUNCTION offload_free_pinned_mem_c
      END INTERFACE

      res = offload_free_pinned_mem_c(buffer)
   END FUNCTION offload_free_pinned_mem

! **************************************************************************************************
!> \brief Initialize runtime.
!> \return ...
!> \author Rocco Meli
! **************************************************************************************************
   SUBROUTINE offload_init()
      INTERFACE
         SUBROUTINE offload_init_c() &
            BIND(C, name="offload_init")
         END SUBROUTINE offload_init_c
      END INTERFACE

      CALL offload_init_c()

   END SUBROUTINE offload_init

! **************************************************************************************************
!> \brief Returns the number of available devices.
!> \return ...
!> \author Ole Schuett
! **************************************************************************************************
   FUNCTION offload_get_device_count() RESULT(count)
      INTEGER                                            :: count

      INTERFACE
         FUNCTION offload_get_device_count_c() &
            BIND(C, name="offload_get_device_count")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT)                :: offload_get_device_count_c
         END FUNCTION offload_get_device_count_c
      END INTERFACE

      count = offload_get_device_count_c()

   END FUNCTION offload_get_device_count

! **************************************************************************************************
!> \brief Selects the chosen device to be used.
!> \param device_id ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_set_chosen_device(device_id)
      INTEGER, INTENT(IN)                                :: device_id

      INTERFACE
         SUBROUTINE offload_set_chosen_device_c(device_id) &
            BIND(C, name="offload_set_chosen_device")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                :: device_id
         END SUBROUTINE offload_set_chosen_device_c
      END INTERFACE

      CALL offload_set_chosen_device_c(device_id=device_id)

   END SUBROUTINE offload_set_chosen_device

! **************************************************************************************************
!> \brief Returns the chosen device.
!> \return ...
!> \author Ole Schuett
! **************************************************************************************************
   FUNCTION offload_get_chosen_device() RESULT(device_id)
      INTEGER                                            :: device_id

      INTERFACE
         FUNCTION offload_get_chosen_device_c() &
            BIND(C, name="offload_get_chosen_device")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT)                :: offload_get_chosen_device_c
         END FUNCTION offload_get_chosen_device_c
      END INTERFACE

      device_id = offload_get_chosen_device_c()

      IF (device_id < 0) THEN
         CPABORT("No offload device has been chosen.")
      END IF

   END FUNCTION offload_get_chosen_device

! **************************************************************************************************
!> \brief Activates the device selected via offload_set_chosen_device()
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_activate_chosen_device()

      INTERFACE
         SUBROUTINE offload_activate_chosen_device_c() &
            BIND(C, name="offload_activate_chosen_device")
         END SUBROUTINE offload_activate_chosen_device_c
      END INTERFACE

      CALL offload_activate_chosen_device_c()

   END SUBROUTINE offload_activate_chosen_device

! **************************************************************************************************
!> \brief Starts a timing range.
!> \param routineN ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_timeset(routineN)
      CHARACTER(LEN=*), INTENT(IN)                       :: routineN

      INTERFACE
         SUBROUTINE offload_timeset_c(message) BIND(C, name="offload_timeset")
            IMPORT :: C_CHAR
            CHARACTER(kind=C_CHAR), DIMENSION(*), INTENT(IN) :: message
         END SUBROUTINE offload_timeset_c
      END INTERFACE

      CALL offload_timeset_c(TRIM(routineN)//C_NULL_CHAR)

   END SUBROUTINE offload_timeset

! **************************************************************************************************
!> \brief  Ends a timing range.
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_timestop()

      INTERFACE
         SUBROUTINE offload_timestop_c() BIND(C, name="offload_timestop")
         END SUBROUTINE offload_timestop_c
      END INTERFACE

      CALL offload_timestop_c()

   END SUBROUTINE offload_timestop

! **************************************************************************************************
!> \brief Gets free and total device memory.
!> \param free ...
!> \param total ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_mem_info(free, total)
      INTEGER(KIND=int_8), INTENT(OUT)                   :: free, total

      INTEGER(KIND=C_SIZE_T)                             :: my_free, my_total
      INTERFACE
         SUBROUTINE offload_mem_info_c(free, total) BIND(C, name="offload_mem_info")
            IMPORT :: C_SIZE_T
            INTEGER(KIND=C_SIZE_T)                   :: free, total
         END SUBROUTINE offload_mem_info_c
      END INTERFACE

      CALL offload_mem_info_c(my_free, my_total)

      ! On 32-bit architectures this converts from int_4 to int_8.
      free = my_free
      total = my_total

   END SUBROUTINE offload_mem_info

! **************************************************************************************************
!> \brief Allocates a buffer of given length, ie. number of elements.
!> \param length ...
!> \param buffer ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_create_buffer(length, buffer)
      INTEGER, INTENT(IN)                                :: length
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer

      CHARACTER(LEN=*), PARAMETER :: routineN = 'offload_create_buffer'

      INTEGER                                            :: handle
      TYPE(C_PTR)                                        :: host_buffer_c
      INTERFACE
         SUBROUTINE offload_create_buffer_c(length, buffer) &
            BIND(C, name="offload_create_buffer")
            IMPORT :: C_PTR, C_INT
            INTEGER(KIND=C_INT), VALUE                :: length
            TYPE(C_PTR)                               :: buffer
         END SUBROUTINE offload_create_buffer_c
      END INTERFACE
      INTERFACE
         FUNCTION offload_get_buffer_host_pointer_c(buffer) &
            BIND(C, name="offload_get_buffer_host_pointer")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
            TYPE(C_PTR)                               :: offload_get_buffer_host_pointer_c
         END FUNCTION offload_get_buffer_host_pointer_c
      END INTERFACE

      CALL timeset(routineN, handle)

      IF (ASSOCIATED(buffer%host_buffer)) THEN
         IF (SIZE(buffer%host_buffer) == 0) DEALLOCATE (buffer%host_buffer)
         IF (ASSOCIATED(buffer%host_buffer)) NULLIFY (buffer%host_buffer)
      END IF

      CALL offload_create_buffer_c(length=length, buffer=buffer%c_ptr)
      CPASSERT(C_ASSOCIATED(buffer%c_ptr))

      IF (length == 0) THEN
         ! While C_F_POINTER usually accepts a NULL pointer it's not standard compliant.
         ALLOCATE (buffer%host_buffer(0))
      ELSE
         host_buffer_c = offload_get_buffer_host_pointer_c(buffer%c_ptr)
         CPASSERT(C_ASSOCIATED(host_buffer_c))
         CALL C_F_POINTER(host_buffer_c, buffer%host_buffer, shape=[length])
      END IF

      CALL timestop(handle)
   END SUBROUTINE offload_create_buffer

! **************************************************************************************************
!> \brief Deallocates given buffer.
!> \param buffer ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_free_buffer(buffer)
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer

      CHARACTER(LEN=*), PARAMETER :: routineN = 'offload_free_buffer'

      INTEGER                                            :: handle
      INTERFACE
         SUBROUTINE offload_free_buffer_c(buffer) &
            BIND(C, name="offload_free_buffer")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
         END SUBROUTINE offload_free_buffer_c
      END INTERFACE

      CALL timeset(routineN, handle)

      IF (C_ASSOCIATED(buffer%c_ptr)) THEN

         CALL offload_free_buffer_c(buffer%c_ptr)

         buffer%c_ptr = C_NULL_PTR

         IF (ASSOCIATED(buffer%host_buffer)) THEN
         IF (SIZE(buffer%host_buffer) == 0) THEN
            DEALLOCATE (buffer%host_buffer)
         ELSE
            NULLIFY (buffer%host_buffer)
         END IF
         END IF
      END IF

      CALL timestop(handle)
   END SUBROUTINE offload_free_buffer

! **************************************************************************************************
!> \brief Print allocation statistics.
!> \param mpi_comm ...
!> \param output_unit ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE offload_mempool_stats_print(mpi_comm, output_unit)
      TYPE(mp_comm_type), INTENT(IN)                     :: mpi_comm
      INTEGER, INTENT(IN)                                :: output_unit

      INTERFACE
         SUBROUTINE offload_mempool_stats_print_c(mpi_comm, print_func, output_unit) &
            BIND(C, name="offload_mempool_stats_print")
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(KIND=C_INT), VALUE          :: mpi_comm
            TYPE(C_FUNPTR), VALUE               :: print_func
            INTEGER(KIND=C_INT), VALUE          :: output_unit
         END SUBROUTINE offload_mempool_stats_print_c
      END INTERFACE

      ! Since Fortran units groups can't be used from C, we pass a function pointer instead.
      CALL offload_mempool_stats_print_c(mpi_comm=mpi_comm%get_handle(), &
                                         print_func=C_FUNLOC(print_func), &
                                         output_unit=output_unit)

   END SUBROUTINE offload_mempool_stats_print

! **************************************************************************************************
!> \brief Callback to write to a Fortran output unit (called by C-side).
!> \param msg to be printed.
!> \param msglen number of characters excluding the terminating character.
!> \param output_unit used for output.
!> \author Hans Pabst
! **************************************************************************************************
   SUBROUTINE print_func(msg, msglen, output_unit) BIND(C, name="offload_api_print_func")
      CHARACTER(KIND=C_CHAR), INTENT(IN)                 :: msg(*)
      INTEGER(KIND=C_INT), INTENT(IN), VALUE             :: msglen, output_unit

      IF (output_unit <= 0) RETURN ! Omit to print the message.
      WRITE (output_unit, FMT="(100A)", ADVANCE="NO") msg(1:msglen)
   END SUBROUTINE print_func

   ! GPU data transfer functions
! **************************************************************************************************
!> \brief Copy host data to device.
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_h2d(buffer, length)
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer
      INTEGER, INTENT(IN)                                :: length

      INTERFACE
         SUBROUTINE offload_buffer_h2d_c(buffer, length) &
            BIND(C, name="offload_buffer_h2d")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                            :: buffer
            INTEGER(KIND=C_INT), VALUE                    :: length
         END SUBROUTINE offload_buffer_h2d_c
      END INTERFACE

      CALL offload_buffer_h2d_c(buffer%c_ptr, length)
   END SUBROUTINE offload_buffer_h2d

! **************************************************************************************************
!> \brief Copy device data to host.
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_d2h(buffer, length)
      TYPE(offload_buffer_type), INTENT(INOUT)           :: buffer
      INTEGER, INTENT(IN)                                :: length

      INTERFACE
         SUBROUTINE offload_buffer_d2h_c(buffer, length) &
            BIND(C, name="offload_buffer_d2h")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                            :: buffer
            INTEGER(KIND=C_INT), VALUE                    :: length
         END SUBROUTINE offload_buffer_d2h_c
      END INTERFACE

      CALL offload_buffer_d2h_c(buffer%c_ptr, length)
   END SUBROUTINE offload_buffer_d2h

! **************************************************************************************************
!> \brief Return a C pointer to the buffer's underlying device memory.
!> \param buffer ...
!> \return ...
!> \author Samal Bibek
! **************************************************************************************************
   FUNCTION offload_get_buffer_device_pointer(buffer) RESULT(device_ptr)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      TYPE(C_PTR)                                        :: device_ptr

      INTERFACE
         FUNCTION offload_get_buffer_device_pointer_c(buffer) &
            BIND(C, name="offload_get_buffer_device_pointer")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
            TYPE(C_PTR)                               :: offload_get_buffer_device_pointer_c
         END FUNCTION offload_get_buffer_device_pointer_c
      END INTERFACE

      device_ptr = offload_get_buffer_device_pointer_c(buffer%c_ptr)
   END FUNCTION offload_get_buffer_device_pointer

! **************************************************************************************************
!> \brief Host/device authority flags and lazy sync for an offload_buffer.
!>!        (grid-residency; additive, host_current=1 on creation)
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   FUNCTION offload_buffer_is_device_current(buffer) RESULT(res)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      LOGICAL                                            :: res

      INTERFACE
         FUNCTION offload_buffer_is_device_current_c(buffer) RESULT(r) &
            BIND(C, name="offload_buffer_is_device_current")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT)                       :: r
         END FUNCTION offload_buffer_is_device_current_c
      END INTERFACE

      res = (offload_buffer_is_device_current_c(buffer%c_ptr) /= 0)
   END FUNCTION offload_buffer_is_device_current

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   FUNCTION offload_buffer_is_host_current(buffer) RESULT(res)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      LOGICAL                                            :: res

      INTERFACE
         FUNCTION offload_buffer_is_host_current_c(buffer) RESULT(r) &
            BIND(C, name="offload_buffer_is_host_current")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT)                       :: r
         END FUNCTION offload_buffer_is_host_current_c
      END INTERFACE

      res = (offload_buffer_is_host_current_c(buffer%c_ptr) /= 0)
   END FUNCTION offload_buffer_is_host_current

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_mark_device_current(buffer)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer

      INTERFACE
         SUBROUTINE offload_buffer_mark_device_current_c(buffer) &
            BIND(C, name="offload_buffer_mark_device_current")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
         END SUBROUTINE offload_buffer_mark_device_current_c
      END INTERFACE

      CALL offload_buffer_mark_device_current_c(buffer%c_ptr)
   END SUBROUTINE offload_buffer_mark_device_current

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_mark_host_current(buffer)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer

      INTERFACE
         SUBROUTINE offload_buffer_mark_host_current_c(buffer) &
            BIND(C, name="offload_buffer_mark_host_current")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
         END SUBROUTINE offload_buffer_mark_host_current_c
      END INTERFACE

      CALL offload_buffer_mark_host_current_c(buffer%c_ptr)
   END SUBROUTINE offload_buffer_mark_host_current

! **************************************************************************************************
!> \brief Whether the grid backend may keep a buffer device-resident (Phase A).
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   FUNCTION offload_buffer_is_device_resident(buffer) RESULT(res)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      LOGICAL                                            :: res

      INTERFACE
         FUNCTION offload_buffer_is_device_resident_c(buffer) RESULT(r) &
            BIND(C, name="offload_buffer_is_device_resident")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT)                       :: r
         END FUNCTION offload_buffer_is_device_resident_c
      END INTERFACE

      res = (offload_buffer_is_device_resident_c(buffer%c_ptr) /= 0)
   END FUNCTION offload_buffer_is_device_resident

! **************************************************************************************************
!> \brief Enable/disable device-resident grid handling for a buffer.
!> \param buffer ...
!> \param value ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_set_device_resident(buffer, value)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      LOGICAL, INTENT(IN)                                :: value

      INTERFACE
         SUBROUTINE offload_buffer_set_device_resident_c(buffer, value) &
            BIND(C, name="offload_buffer_set_device_resident")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT), VALUE                :: value
         END SUBROUTINE offload_buffer_set_device_resident_c
      END INTERFACE

      CALL offload_buffer_set_device_resident_c(buffer%c_ptr, MERGE(1_C_INT, 0_C_INT, value))
   END SUBROUTINE offload_buffer_set_device_resident

! **************************************************************************************************
!> \brief Wait for the producer-stream event recorded on a buffer (no-op if none).
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_wait_event(buffer)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer

      INTERFACE
         SUBROUTINE offload_buffer_wait_event_c(buffer) &
            BIND(C, name="offload_buffer_wait_event")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
         END SUBROUTINE offload_buffer_wait_event_c
      END INTERFACE

      CALL offload_buffer_wait_event_c(buffer%c_ptr)
   END SUBROUTINE offload_buffer_wait_event

! **************************************************************************************************
!> \brief D2D copy ordered after the buffer's producer event (dedicated copy stream).
!> \param buffer ...
!> \param dst_device ...
!> \param src_device ...
!> \param bytes ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_d2d_streamed(buffer, dst_device, src_device, bytes)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      TYPE(C_PTR), INTENT(IN)                            :: dst_device, src_device
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: bytes

      INTERFACE
         SUBROUTINE offload_buffer_d2d_streamed_c(buffer, dst, src, bytes) &
            BIND(C, name="offload_buffer_d2d_streamed")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), VALUE                        :: buffer, dst, src
            INTEGER(KIND=C_SIZE_T), VALUE             :: bytes
         END SUBROUTINE offload_buffer_d2d_streamed_c
      END INTERFACE

      CALL offload_buffer_d2d_streamed_c(buffer%c_ptr, dst_device, src_device, bytes)
   END SUBROUTINE offload_buffer_d2d_streamed

! **************************************************************************************************
!> \brief Runtime toggle for the device-resident grid round-trip (Phase A).
!>        Default off; requires an accelerator build.
!> \return ...
! **************************************************************************************************
   FUNCTION offload_grid_resident_enabled() RESULT(res)
      LOGICAL                                            :: res
#if defined(__OFFLOAD)
      CHARACTER(LEN=32)                                  :: val
      INTEGER                                            :: status
      LOGICAL, SAVE                                      :: initialized = .FALSE.
      LOGICAL, SAVE                                      :: enabled = .FALSE.
      IF (.NOT. initialized) THEN
         CALL GET_ENVIRONMENT_VARIABLE("CP2K_GRID_DEVICE_RESIDENT", val, STATUS=status)
         IF (status == 0) THEN
            enabled = (val(1:1) == "1" .OR. val(1:1) == "T" .OR. val(1:1) == "t")
         END IF
         initialized = .TRUE.
      END IF
      res = enabled
#else
      res = .FALSE.
#endif
   END FUNCTION offload_grid_resident_enabled

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_sync_to_host(buffer, length)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      INTEGER, INTENT(IN)                                :: length

      INTERFACE
         SUBROUTINE offload_buffer_sync_to_host_c(buffer, length) &
            BIND(C, name="offload_buffer_sync_to_host")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT), VALUE                :: length
         END SUBROUTINE offload_buffer_sync_to_host_c
      END INTERFACE

      CALL offload_buffer_sync_to_host_c(buffer%c_ptr, length)
   END SUBROUTINE offload_buffer_sync_to_host

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \param length ...
! **************************************************************************************************
   SUBROUTINE offload_buffer_sync_to_device(buffer, length)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      INTEGER, INTENT(IN)                                :: length

      INTERFACE
         SUBROUTINE offload_buffer_sync_to_device_c(buffer, length) &
            BIND(C, name="offload_buffer_sync_to_device")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE                        :: buffer
            INTEGER(KIND=C_INT), VALUE                :: length
         END SUBROUTINE offload_buffer_sync_to_device_c
      END INTERFACE

      CALL offload_buffer_sync_to_device_c(buffer%c_ptr, length)
   END SUBROUTINE offload_buffer_sync_to_device

! **************************************************************************************************
!> \brief Blocks until all outstanding device work has completed.
!> \author Samal Bibek
! **************************************************************************************************
   SUBROUTINE offload_device_synchronize()
      INTERFACE
         SUBROUTINE offload_device_synchronize_c() &
            BIND(C, name="offload_device_synchronize")
         END SUBROUTINE offload_device_synchronize_c
      END INTERFACE

      CALL offload_device_synchronize_c()
   END SUBROUTINE offload_device_synchronize

! **************************************************************************************************
!> \brief Blocks until outstanding work on the legacy default stream is done.
! **************************************************************************************************
   SUBROUTINE offload_default_stream_synchronize()
      INTERFACE
         SUBROUTINE offload_default_stream_synchronize_c() &
            BIND(C, name="offload_default_stream_synchronize")
         END SUBROUTINE offload_default_stream_synchronize_c
      END INTERFACE

      CALL offload_default_stream_synchronize_c()
   END SUBROUTINE offload_default_stream_synchronize

! **************************************************************************************************
!> \brief Creates an opaque GPU event (cross-stream ordering helper).
!> \return ...
! **************************************************************************************************
   FUNCTION offload_event_create() RESULT(event)
      TYPE(C_PTR)                                        :: event

      INTERFACE
         FUNCTION offload_event_create_c() RESULT(e) BIND(C, name="offload_event_create")
            IMPORT :: C_PTR
            TYPE(C_PTR)                               :: e
         END FUNCTION offload_event_create_c
      END INTERFACE

      event = offload_event_create_c()
   END FUNCTION offload_event_create

! **************************************************************************************************
!> \brief Destroys an opaque GPU event.
!> \param event ...
! **************************************************************************************************
   SUBROUTINE offload_event_destroy(event)
      TYPE(C_PTR), INTENT(IN)                            :: event

      INTERFACE
         SUBROUTINE offload_event_destroy_c(event) BIND(C, name="offload_event_destroy")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: event
         END SUBROUTINE offload_event_destroy_c
      END INTERFACE

      CALL offload_event_destroy_c(event)
   END SUBROUTINE offload_event_destroy

! **************************************************************************************************
!> \brief Records an event on a stream (C_NULL_PTR = legacy default stream).
!> \param event ...
!> \param stream ...
! **************************************************************************************************
   SUBROUTINE offload_event_record(event, stream)
      TYPE(C_PTR), INTENT(IN)                            :: event
      TYPE(C_PTR), INTENT(IN), OPTIONAL                  :: stream

      TYPE(C_PTR)                                        :: my_stream
      INTERFACE
         SUBROUTINE offload_event_record_c(event, stream) BIND(C, name="offload_event_record")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: event, stream
         END SUBROUTINE offload_event_record_c
      END INTERFACE

      my_stream = C_NULL_PTR
      IF (PRESENT(stream)) my_stream = stream
      CALL offload_event_record_c(event, my_stream)
   END SUBROUTINE offload_event_record

! **************************************************************************************************
!> \brief Makes a stream wait for an event.
!> \param event ...
!> \param stream ...
! **************************************************************************************************
   SUBROUTINE offload_event_wait(event, stream)
      TYPE(C_PTR), INTENT(IN)                            :: event
      TYPE(C_PTR), INTENT(IN), OPTIONAL                  :: stream

      TYPE(C_PTR)                                        :: my_stream
      INTERFACE
         SUBROUTINE offload_event_wait_c(event, stream) BIND(C, name="offload_event_wait")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: event, stream
         END SUBROUTINE offload_event_wait_c
      END INTERFACE

      my_stream = C_NULL_PTR
      IF (PRESENT(stream)) my_stream = stream
      CALL offload_event_wait_c(event, my_stream)
   END SUBROUTINE offload_event_wait

! **************************************************************************************************
!> \brief Blocks the host until an event has completed.
!> \param event ...
! **************************************************************************************************
   SUBROUTINE offload_event_synchronize(event)
      TYPE(C_PTR), INTENT(IN)                            :: event

      INTERFACE
         SUBROUTINE offload_event_synchronize_c(event) BIND(C, name="offload_event_synchronize")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: event
         END SUBROUTINE offload_event_synchronize_c
      END INTERFACE

      CALL offload_event_synchronize_c(event)
   END SUBROUTINE offload_event_synchronize
! **************************************************************************************************
!> \brief Return a C pointer to the buffer's host memory.
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   FUNCTION offload_get_buffer_host_pointer(buffer) RESULT(host_ptr)
      TYPE(offload_buffer_type), INTENT(IN)              :: buffer
      TYPE(C_PTR)                                        :: host_ptr

      INTERFACE
         FUNCTION offload_get_buffer_host_pointer_c(buffer) &
            BIND(C, name="offload_get_buffer_host_pointer")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                        :: buffer
            TYPE(C_PTR)                               :: offload_get_buffer_host_pointer_c
         END FUNCTION offload_get_buffer_host_pointer_c
      END INTERFACE

      host_ptr = offload_get_buffer_host_pointer_c(buffer%c_ptr)
   END FUNCTION offload_get_buffer_host_pointer
! **************************************************************************************************
!> \brief Allocates device-only memory (no host mirror), reusable across calls.
!> \param bytes ...
!> \return ...
! **************************************************************************************************
   FUNCTION offload_device_alloc(bytes) RESULT(device_ptr)
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: bytes
      TYPE(C_PTR)                                        :: device_ptr

      INTERFACE
         FUNCTION offload_device_alloc_c(bytes) RESULT(p) BIND(C, name="offload_device_malloc")
            IMPORT :: C_PTR, C_SIZE_T
            INTEGER(KIND=C_SIZE_T), VALUE              :: bytes
            TYPE(C_PTR)                                :: p
         END FUNCTION offload_device_alloc_c
      END INTERFACE

      device_ptr = offload_device_alloc_c(bytes)
   END FUNCTION offload_device_alloc

! **************************************************************************************************
!> \brief Frees device-only memory obtained from offload_device_alloc.
!> \param device_ptr ...
! **************************************************************************************************
   SUBROUTINE offload_device_free(device_ptr)
      TYPE(C_PTR), INTENT(IN)                            :: device_ptr

      INTERFACE
         SUBROUTINE offload_device_free_c(device_ptr) BIND(C, name="offload_device_free")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                         :: device_ptr
         END SUBROUTINE offload_device_free_c
      END INTERFACE

      CALL offload_device_free_c(device_ptr)
   END SUBROUTINE offload_device_free

! **************************************************************************************************
!> \brief Copies host memory to a device-only buffer (synchronous).
!> \param device_ptr ...
!> \param host_ptr ...
!> \param bytes ...
! **************************************************************************************************
   SUBROUTINE offload_device_h2d(device_ptr, host_ptr, bytes)
      TYPE(C_PTR), INTENT(IN)                            :: device_ptr, host_ptr
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: bytes

      INTERFACE
         SUBROUTINE offload_device_h2d_c(dev, host, bytes) BIND(C, name="offload_device_h2d")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), VALUE                         :: dev, host
            INTEGER(KIND=C_SIZE_T), VALUE              :: bytes
         END SUBROUTINE offload_device_h2d_c
      END INTERFACE

      CALL offload_device_h2d_c(device_ptr, host_ptr, bytes)
   END SUBROUTINE offload_device_h2d

! **************************************************************************************************
!> \brief Copies a device-only buffer into host memory (synchronous).
!> \param host_ptr ...
!> \param device_ptr ...
!> \param bytes ...
! **************************************************************************************************
   SUBROUTINE offload_device_d2h(host_ptr, device_ptr, bytes)
      TYPE(C_PTR), INTENT(IN)                            :: host_ptr, device_ptr
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: bytes

      INTERFACE
         SUBROUTINE offload_device_d2h_c(host, dev, bytes) BIND(C, name="offload_device_d2h")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), VALUE                         :: host, dev
            INTEGER(KIND=C_SIZE_T), VALUE              :: bytes
         END SUBROUTINE offload_device_d2h_c
      END INTERFACE

      CALL offload_device_d2h_c(host_ptr, device_ptr, bytes)
   END SUBROUTINE offload_device_d2h

! **************************************************************************************************
!> \brief Copies device-only buffer to device-only buffer (synchronous).
!> \param dst_device_ptr ...
!> \param src_device_ptr ...
!> \param bytes ...
! **************************************************************************************************
   SUBROUTINE offload_device_d2d(dst_device_ptr, src_device_ptr, bytes)
      TYPE(C_PTR), INTENT(IN)                            :: dst_device_ptr, src_device_ptr
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: bytes

      INTERFACE
         SUBROUTINE offload_device_d2d_c(dst, src, bytes) BIND(C, name="offload_device_d2d")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), VALUE                         :: dst, src
            INTEGER(KIND=C_SIZE_T), VALUE              :: bytes
         END SUBROUTINE offload_device_d2d_c
      END INTERFACE

      CALL offload_device_d2d_c(dst_device_ptr, src_device_ptr, bytes)
   END SUBROUTINE offload_device_d2d

! **************************************************************************************************
!> \brief Total device bytes currently held by the offload mempool (device_size).
!>        Used by the residency/leak unit test.
!> \return ...
! **************************************************************************************************
   FUNCTION offload_mempool_device_bytes() RESULT(res)
      INTEGER(KIND=int_8)                                :: res

      INTERFACE
         FUNCTION offload_mempool_device_bytes_c() RESULT(r) BIND(C, name="offload_mempool_device_bytes")
            IMPORT :: int_8
            INTEGER(KIND=int_8)                         :: r
         END FUNCTION offload_mempool_device_bytes_c
      END INTERFACE

      res = offload_mempool_device_bytes_c()
   END FUNCTION offload_mempool_device_bytes

END MODULE offload_api
