#!-------------------------------------------------------------------------------------------------!
#!   CP2K: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

# Add the CUDA wrapper source files
set(XC_SOURCES xc_libxc_wrap.F xc_libxc.F xc_cuda_wrapper.c)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${LIBXC_INCLUDE_DIRS})

# Add CUDA support if available
if(CUDA_FOUND)
  enable_language(CUDA)
  list(APPEND XC_SOURCES xc_cuda_wrapper.c)

  # CUDA specific flags
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_80 -O3)

  # Link CUDA libraries
  find_package(CUDA REQUIRED)
  list(APPEND XC_LINK_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARIES})
endif()

# Create the XC library
add_library(xc_module ${XC_SOURCES})

target_link_libraries(xc_module ${LIBXC_LIBRARIES} ${XC_LINK_LIBRARIES} offload)

# Install targets
install(
  TARGETS xc_module
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES xc_cuda_wrapper.h DESTINATION include)
