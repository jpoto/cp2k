/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib_fftw.h"
#include "fft_timer.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FFTW3)
#if defined(__OFFLOAD_CUDA) && !defined(__NO_OFFLOAD_PW)
#include <cufftw.h>
#else
#include <fftw3.h>
#if defined(__parallel) && defined(__FFTW3_MPI)
#include <fftw3-mpi.h>
#define __USE_FFTW3_MPI
#endif
#endif

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
typedef struct {
  // The key contains
  // 0: rank, transposition (see below)
  // 1: associated Fortran communicator handle (to store it as an integer)
  // 2: direction (forward/backward)
  // 3, 4, 5: FFT sizes (or FFT sizes and number of FFTs)
  int key[6];
  fftw_plan *plan;
} cache_entry;

// We need to reserve more space because of the different combinations
// (local/distributed, C2C/R2C) This works to run all tests
#define FFTW_CACHE_SIZE 128
static cache_entry cache[FFTW_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static bool is_initialized = false;

static int fftw_planning_mode = -1;
static bool use_fftw_mpi = false;
static bool has_guru_interface = true;

// These constants encode transposition and MPI usage into the key to cache the
// plans
// Modulo 4 encodes the rank (1, 2, 3)
// 4 == 2^2
#define FFTW_TRANSPOSE_RS 4
// 8 == 2^3
#define FFTW_TRANSPOSE_GS 8
// 16 == 2^4
#define FFTW_R2C 16
// 32 == 2^5
#define FFTW_INPLACE 32

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static fftw_plan *lookup_plan_from_cache(const int key[6]) {
  assert(is_initialized);
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3] &&
        x[4] == key[4] && x[5] == key[5]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static void add_plan_to_cache(const int key[6], fftw_plan *plan) {
  const int i = cache_oldest_entry;
  cache_oldest_entry = (cache_oldest_entry + 1) % FFTW_CACHE_SIZE;
  if (cache[i].plan != NULL) {
    fprintf(stderr,
            "Storage to cache FFTW plans is full. Delete an old plan...\n");
    fftw_destroy_plan(*cache[i].plan);
    free(cache[i].plan);
  }
  cache[i].key[0] = key[0];
  cache[i].key[1] = key[1];
  cache[i].key[2] = key[2];
  cache[i].key[3] = key[3];
  cache[i].key[4] = key[4];
  cache[i].key[5] = key[5];
  cache[i].plan = plan;
}
#endif

#if defined(__FFTW3)
bool is_guru_interface_available() {
  fftw_iodim dims[1];
  dims[0].n = 1;
  dims[0].is = 1;
  dims[0].os = 1;
  fftw_iodim howmany_dims[2];
  howmany_dims[0].n = 1;
  howmany_dims[0].is = 1;
  howmany_dims[0].os = 1;
  howmany_dims[1].n = 1;
  howmany_dims[1].is = 1;
  howmany_dims[1].os = 1;
  double complex buffer1;
  fftw_plan plan =
      fftw_plan_guru_dft(1, dims, 2, howmany_dims, &buffer1, &buffer1,
                         FFTW_FORWARD, fftw_planning_mode);

  if (plan != NULL) {
    fftw_destroy_plan(plan);
    return true;
  } else {
    return false;
  }
}
#endif

#if defined(__USE_FFTW3_MPI)
bool fft_fftw_test_mpi_backend() {
  const int nthreads = omp_get_max_threads();
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const fft_size[3] = {2, 2, 2};
  fftw_plan_with_nthreads(nthreads);
  const int block_size_0 =
      (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const int block_size_1 =
      (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t local_n1, local_1_start;
  const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
  const int buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
      &local_n1, &local_1_start);
  double complex *buffer_1 = fftw_alloc_complex(buffer_size);
  double complex *buffer_2 = fftw_alloc_complex(buffer_size);
  plan = malloc(sizeof(fftw_plan));
  *plan = fftw_mpi_plan_many_dft(3, n, 1, block_size_0, block_size_1, buffer_1,
                                 buffer_2, comm, direction,
                                 fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
  fftw_free(buffer_1);
  fftw_free(buffer_2);
  if (plan != NULL) {
    fftw_destroy_plan(plan);
    return true;
  } else {
    return false;
  }
}
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_init_lib(const fftw_plan_type fftw_planning_flag,
                       const bool use_fft_mpi, const char *wisdom_file) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (is_initialized) {
    return;
  }
  memset(cache, 0, sizeof(cache_entry) * FFTW_CACHE_SIZE);
  cache_oldest_entry = 0;

  is_initialized = true;
  // We need a threaded library!
  fftw_init_threads();

  fftw_planning_mode = fftw_planning_flag;
  switch (fftw_planning_flag) {
  case FFT_ESTIMATE:
    fftw_planning_mode = FFTW_ESTIMATE;
    break;
  case FFT_MEASURE:
    fftw_planning_mode = FFTW_MEASURE;
    break;
  case FFT_PATIENT:
    fftw_planning_mode = FFTW_PATIENT;
    break;
  case FFT_EXHAUSTIVE:
    fftw_planning_mode = FFTW_EXHAUSTIVE;
    break;
  default:
    assert(0 && "Unknown FFTW planning flag.");
  }

  // TODO:
  // This is only necessary if these routines are called with non-aligned memory
  // (use fftw_alignment_of to check for that)
#if defined(__FFTW3_UNALIGNED)
  fftw_planning_mode += FFTW_UNALIGNED
#endif

      has_guru_interface = is_guru_interface_available();

#if defined(__USE_FFTW3_MPI)
  use_fftw_mpi = use_fft_mpi;
  fftw_mpi_init();
  if (use_fft_mpi) {
    use_fft_mpi = fft_fftw_test_mpi_backend();
    fprintf(stderr,
            "Creation of a MPI plan failed! MPI features are turned off!");
  }
#else
  (void)use_fft_mpi;
  use_fftw_mpi = false;
#endif
  // Export wisdom after intializing the library to ensure correct threading
  // etc.
  if (wisdom_file != NULL) {
    const int error = fftw_import_wisdom_from_filename(wisdom_file);
    if (error != 0 && cp_mpi_comm_rank(cp_mpi_get_comm_world()))
      fprintf(stderr,
              "Importing wisdom failed! Maybe the file does not exist.");
  }
  if (use_fftw_mpi)
    printf("Using FFTW MPI\n");
#else
  (void)fftw_planning_flag;
  (void)use_fft_mpi;
  (void)wisdom_file;
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_finalize_lib(const char *wisdom_file) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (!is_initialized) {
    return;
  }
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    if (cache[i].plan != NULL) {
      fftw_destroy_plan(*cache[i].plan);
      free(cache[i].plan);
    }
  }
  // Export wisdom before finalizing the library to ensure storing the correct
  // threading etc.
  if (wisdom_file != NULL) {
    const int error = fftw_export_wisdom_to_filename(wisdom_file);
    if (error != 0 && cp_mpi_comm_rank(cp_mpi_get_comm_world()))
      fprintf(stderr,
              "Exporting wisdom failed! Maybe writing access is missing.");
  }
  is_initialized = false;
  fftw_planning_mode = -1;
#if defined(__USE_FFTW3_MPI)
  fftw_mpi_cleanup();
#else
  fftw_cleanup();
#endif
#else
  (void)wisdom_file;
#endif
}

/*******************************************************************************
 * \brief Whether a distributed FFT implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_fftw_lib_use_mpi() {
#if defined(__USE_FFTW3_MPI)
  return use_fftw_mpi;
#else
  return false;
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_double(const int length, double **buffer) {
#if defined(__FFTW3)
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_real(length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_complex(const int length, double complex **buffer) {
#if defined(__FFTW3)
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_complex(length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_double(double *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_complex(double complex *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

#if defined(__FFTW3)
/*******************************************************************************
 * \brief Create plan of a local C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_1d_plan(const int direction, const int fft_size,
                                   const int number_of_ffts,
                                   const bool transpose_rs,
                                   const bool transpose_gs,
                                   double complex *grid_out,
                                   const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_%cw_c2c_Plocal_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size, number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {1 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs +
                          FFTW_INPLACE * inplace,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      direction,
                      fft_size,
                      number_of_ffts,
                      0};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 1;
    const int n[] = {fft_size};
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    const int idist = transpose_rs ? 1 : fft_size;
    const int odist = transpose_gs ? 1 : fft_size;
    const int istride = transpose_rs ? number_of_ffts : 1;
    const int ostride = transpose_gs ? number_of_ffts : 1;
    double complex *buffer_1 = fftw_alloc_complex(fft_size * number_of_ffts);
    double complex *buffer_2 = inplace ? buffer_1 : grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, inembed, istride,
                                 idist, buffer_2, onembed, ostride, odist,
                                 FFTW_FORWARD, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, onembed, ostride,
                                 odist, buffer_2, inembed, istride, idist,
                                 FFTW_BACKWARD, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}
/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_1d_plan_r2c(const int direction, const int fft_size,
                                       const int number_of_ffts,
                                       const bool transpose_rs,
                                       const bool transpose_gs,
                                       double complex *grid_out,
                                       const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_%s_Plocal_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size,
           number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {1 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs + FFTW_R2C +
                          FFTW_INPLACE * inplace,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      direction,
                      fft_size,
                      number_of_ffts,
                      0};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 1;
    const int n[] = {fft_size};
    const int howmany = number_of_ffts;
    const int *inembed = NULL;
    const int *onembed = NULL;
    const int idist = transpose_rs ? 1 : fft_size;
    const int odist = transpose_gs ? 1 : fft_size / 2 + 1;
    const int istride = transpose_rs ? number_of_ffts : 1;
    const int ostride = transpose_gs ? number_of_ffts : 1;
    double *buffer_1 = fftw_alloc_real(2 * (fft_size / 2 + 1) * number_of_ffts);
    double complex *buffer_2 = inplace ? (double complex *)buffer_1 : grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft_r2c(rank, n, howmany, buffer_1, inembed,
                                     istride, idist, buffer_2, onembed, ostride,
                                     odist, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft_c2r(rank, n, howmany, buffer_2, onembed,
                                     ostride, odist, buffer_1, inembed, istride,
                                     idist, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_2d_plan(const int direction, const int fft_size[2],
                                   const int number_of_ffts,
                                   const bool transpose_rs,
                                   const bool transpose_gs,
                                   double complex *grid_out,
                                   const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_%cw_c2c_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size[0], fft_size[1],
           number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {2 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs +
                          FFTW_INPLACE * inplace,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      direction,
                      fft_size[0],
                      fft_size[1],
                      number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 2;
    const int *n = fft_size;
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    const int idist = transpose_rs ? 1 : fft_size[0] * fft_size[1];
    const int odist = transpose_gs ? 1 : fft_size[0] * fft_size[1];
    const int istride = transpose_rs ? number_of_ffts : 1;
    const int ostride = transpose_gs ? number_of_ffts : 1;
    double complex *buffer_1 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * number_of_ffts);
    double complex *buffer_2 = inplace ? buffer_1 : grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, inembed, istride,
                                 idist, buffer_2, onembed, ostride, odist,
                                 FFTW_FORWARD, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, onembed, ostride,
                                 odist, buffer_2, inembed, istride, idist,
                                 FFTW_BACKWARD, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *
fft_fftw_create_2d_plan_r2c(const int direction, const int fft_size[2],
                            const int number_of_ffts, const bool transpose_rs,
                            const bool transpose_gs, double complex *grid_out,
                            const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size[0],
           fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {2 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs + FFTW_R2C +
                          FFTW_INPLACE * inplace,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      direction,
                      fft_size[0],
                      fft_size[1],
                      number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    // We need the guru interface here because cuts the last dimension in half
    // whereas we want the first dimension
    const int rank = 2;
    const int *n = fft_size;
    const int howmany = number_of_ffts;
    const int *inembed = NULL; // = fft_size;
    const int *onembed = NULL; // = {fft_size[0],fft_size[1]/2+1};
    const int idist = transpose_rs ? 1 : fft_size[0] * fft_size[1];
    const int odist = transpose_gs ? 1 : fft_size[0] * (fft_size[1] / 2 + 1);
    const int istride = transpose_rs ? number_of_ffts : 1;
    const int ostride = transpose_gs ? number_of_ffts : 1;
    double *double_buffer = fftw_alloc_real(
        2 * fft_size[0] * (fft_size[1] / 2 + 1) * number_of_ffts);
    double complex *complex_buffer =
        inplace ? (double complex *)double_buffer : grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft_r2c(rank, n, howmany, double_buffer, inembed,
                                     istride, idist, complex_buffer, onembed,
                                     ostride, odist, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft_c2r(rank, n, howmany, complex_buffer, onembed,
                                     ostride, odist, double_buffer, inembed,
                                     istride, idist, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(double_buffer);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_3d_plan(const int direction, const int fft_size[3],
                                   double complex *grid_out,
                                   const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_%cw_c2c_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size[0], fft_size[1],
           fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {3 + FFTW_INPLACE * inplace,
                      direction,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      fft_size[0],
                      fft_size[1],
                      fft_size[2]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    double complex *buffer_1 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * fft_size[2]);
    double complex *buffer_2 = inplace ? buffer_1 : grid_out;
    plan = malloc(sizeof(fftw_plan));
    *plan = fftw_plan_dft_3d(fft_size[0], fft_size[1], fft_size[2], buffer_1,
                             buffer_2, direction, fftw_planning_mode);
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_3d_plan_r2c(const int direction,
                                       const int fft_size[3],
                                       double complex *grid_out,
                                       const bool inplace) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size[0],
           fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {3 + FFTW_R2C + FFTW_INPLACE * inplace,
                      cp_mpi_comm_c2f(cp_mpi_get_comm_self()),
                      direction,
                      fft_size[0],
                      fft_size[1],
                      fft_size[2]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    double *double_buffer =
        fftw_alloc_real(2 * fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1));
    double complex *complex_buffer =
        inplace ? (double complex *)double_buffer : grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_dft_r2c_3d(fft_size[0], fft_size[1], fft_size[2],
                                   double_buffer, complex_buffer,
                                   fftw_planning_mode);
    } else {
      *plan = fftw_plan_dft_c2r_3d(fft_size[0], fft_size[1], fft_size[2],
                                   complex_buffer, double_buffer,
                                   fftw_planning_mode);
    }
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
    fftw_free(double_buffer);
  }
  fft_stop_timer(handle);
  return plan;
}

#if defined(__USE_FFTW3_MPI)
/*******************************************************************************
 * \brief Create plan of a distributed C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_2d_plan(const int direction,
                                               const int fft_size[2],
                                               const int number_of_ffts,
                                               const cp_mpi_comm_t comm,
                                               double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_%cw_c2c_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', cp_mpi_comm_size(comm),
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {2,           cp_mpi_comm_c2f(comm), direction,
                      fft_size[0], fft_size[1],           number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    if (number_of_ffts == 0)
      return plan;
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[2] = {fft_size[0], fft_size[1]};
    const ptrdiff_t howmany = number_of_ffts;
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        2, n, howmany, block_size_0, block_size_1, comm, &local_n0,
        &local_0_start, &local_n1, &local_1_start);
    double complex *buffer_1 = fftw_alloc_complex(buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    fflush(stderr);
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft(
          2, n, howmany, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft(
          2, n, howmany, block_size_1, block_size_0, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    fftw_free(buffer_1);
    add_plan_to_cache(key, plan);
  }
  fft_stop_timer(handle);
  return plan;
}
/*******************************************************************************
 * \brief Create plan of a distributed R2C/C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_2d_plan_r2c(const int direction,
                                                   const int fft_size[2],
                                                   const int number_of_ffts,
                                                   const cp_mpi_comm_t comm,
                                                   double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {2 + FFTW_R2C, cp_mpi_comm_c2f(comm), direction,
                      fft_size[0],  fft_size[1],           number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    if (number_of_ffts == 0)
      return plan;
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] / 2 + 1 + cp_mpi_comm_size(comm) - 1) /
        cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[2] = {fft_size[0], fft_size[1]};
    const ptrdiff_t howmany = number_of_ffts;
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        2, (const ptrdiff_t[2]){fft_size[0], fft_size[1] / 2 + 1}, howmany,
        block_size_0, block_size_1, comm, &local_n0, &local_0_start, &local_n1,
        &local_1_start);
    double *real_buffer = fftw_alloc_real(2 * buffer_size);
    double complex *complex_buffer = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft_r2c(
          2, n, howmany, block_size_0, block_size_1, real_buffer,
          complex_buffer, comm, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft_c2r(
          2, n, howmany, block_size_1, block_size_0, complex_buffer,
          real_buffer, comm, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    fftw_free(real_buffer);
    add_plan_to_cache(key, plan);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a distributed C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_3d_plan(const int direction,
                                               const int fft_size[3],
                                               const cp_mpi_comm_t comm,
                                               double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {3,           cp_mpi_comm_c2f(comm), direction,
                      fft_size[0], fft_size[1],           fft_size[2]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        3, n, 1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
        &local_n1, &local_1_start);
    double complex *buffer_1 = fftw_alloc_complex(buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_1, block_size_0, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a distributed R2C/C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_3d_plan_r2c(const int direction,
                                                   const int fft_size[3],
                                                   const cp_mpi_comm_t comm,
                                                   double complex *grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  const int key[6] = {3 + FFTW_R2C, cp_mpi_comm_c2f(comm), direction,
                      fft_size[2],  fft_size[1],           fft_size[0]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        3, (const ptrdiff_t[3]){fft_size[0], fft_size[1], fft_size[2] / 2 + 1},
        1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
        &local_n1, &local_1_start);
    double *buffer_1 = fftw_alloc_real(2 * buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft_r2c(
          3, n, 1, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft_c2r(
          3, n, 1, block_size_1, block_size_0, buffer_2, buffer_1, comm,
          fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle);
  return plan;
}
#endif
#endif

/*******************************************************************************
 * \brief Performs a local forward C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, grid_in == grid_out);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan_r2c(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, (double complex *)grid_in == grid_out);
  assert(plan != NULL);
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, grid_in == grid_out);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2R 1D FFT
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double complex *grid_in, double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan_r2c(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      (double complex *)grid_out, grid_in == (double complex *)grid_out);
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward C2C 2D FFT
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, grid_in == grid_out);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan_r2c(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, (double complex *)grid_in == grid_out);
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      grid_out, grid_in == grid_out);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double complex *grid_in, double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan_r2c(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      (double complex *)grid_out, grid_in == (double complex *)grid_out);
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (false && has_guru_interface &&
#if defined(__FFTW3_UNALIGNED)
          (fftw_planning_mode == FFTW_ESTIMATE + FFTW_UNALIGNED) ||
#else
      (fftw_planning_mode == FFTW_ESTIMATE) &&
#endif
      (fft_size[0] >= 256 || fft_size[1] >= 256 || fft_size[2] >= 256 ||
       omp_get_max_threads() > 1)) {
    fft_fftw_1d_fw_local(fft_size[2], fft_size[0] * fft_size[1], false, true,
                         grid_in, grid_out);
    fft_fftw_1d_fw_local(fft_size[1], fft_size[0] * fft_size[2], false, true,
                         grid_out, grid_in);
    fft_fftw_1d_fw_local(fft_size[0], fft_size[1] * fft_size[2], false, true,
                         grid_in, grid_out);
  } else {
    fftw_plan *plan = fft_fftw_create_3d_plan(FFTW_FORWARD, fft_size, grid_out,
                                              grid_in == grid_out);
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local_r2c(const int fft_size[3], double *grid_in,
                              double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_3d_plan_r2c(
      FFTW_FORWARD, fft_size, grid_out, (double complex *)grid_in == grid_out);
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (false && has_guru_interface) {
    fft_fftw_1d_bw_local(fft_size[0], fft_size[1] * fft_size[2], false, true,
                         grid_in, grid_out);
    fft_fftw_1d_bw_local(fft_size[1], fft_size[0] * fft_size[2], false, true,
                         grid_out, grid_in);
    fft_fftw_1d_bw_local(fft_size[2], fft_size[0] * fft_size[1], false, true,
                         grid_in, grid_out);
  } else {
    fftw_plan *plan = fft_fftw_create_3d_plan(FFTW_BACKWARD, fft_size, grid_out,
                                              grid_in == grid_out);
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local_c2r(const int fft_size[3], double complex *grid_in,
                              double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_3d_plan_r2c(
      FFTW_BACKWARD, fft_size, (double complex *)grid_out,
      grid_in == (double complex *)grid_out);
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed C2C 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes(const int npts_global[2],
                                  const int number_of_ffts,
                                  const cp_mpi_comm_t comm, int *local_n0,
                                  int *local_n0_start, int *local_n1,
                                  int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || number_of_ffts <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[2] = {npts_global[0], npts_global[1]};
  const ptrdiff_t howmany = number_of_ffts;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  ptrdiff_t my_local_n0, my_local_n0_start, my_local_n1, my_local_n1_start;
  const ptrdiff_t buffer_size = fftw_mpi_local_size_many_transposed(
      2, n, howmany, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return buffer_size;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed R2C/C2R 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes_r2c(const int npts_global[2],
                                      const int number_of_ffts,
                                      const cp_mpi_comm_t comm, int *local_n0,
                                      int *local_n0_start, int *local_n1,
                                      int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || number_of_ffts <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[2] = {npts_global[0], npts_global[1] / 2 + 1};
  const ptrdiff_t howmany = number_of_ffts;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] / 2 + 1 + cp_mpi_comm_size(comm) - 1) /
      cp_mpi_comm_size(comm);
  ptrdiff_t my_local_n0, my_local_n0_start, my_local_n1, my_local_n1_start;
  const ptrdiff_t buffer_size = fftw_mpi_local_size_many_transposed(
      2, n, howmany, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return buffer_size;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed C2C 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes(const int npts_global[3],
                                  const cp_mpi_comm_t comm, int *local_n0,
                                  int *local_n0_start, int *local_n1,
                                  int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || npts_global[2] <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[3] = {npts_global[0], npts_global[1], npts_global[2]};
  ptrdiff_t my_local_n0, my_local_n0_start;
  ptrdiff_t my_local_n1, my_local_n1_start;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t my_buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return my_buffer_size;
#else
  (void)npts_global;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed R2C/C2R 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes_r2c(const int npts_global[3],
                                      const cp_mpi_comm_t comm, int *local_n0,
                                      int *local_n0_start, int *local_n1,
                                      int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || npts_global[2] <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[3] = {npts_global[0], npts_global[1], npts_global[2]};
  ptrdiff_t my_local_n0, my_local_n0_start;
  ptrdiff_t my_local_n1, my_local_n1_start;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t my_buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return my_buffer_size;
#else
  (void)npts_global;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Performs a distributed forward C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan(
      FFTW_FORWARD, npts_global, number_of_ffts, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed forward R2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed_r2c(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm, double *grid_in,
                                    double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan_r2c(
      FFTW_FORWARD, npts_global, number_of_ffts, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed backwards C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan(
      FFTW_BACKWARD, npts_global, number_of_ffts, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed backwards C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_distributed_c2r(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double complex *grid_in, double *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan_r2c(
      FFTW_BACKWARD, npts_global, number_of_ffts, comm,
      (double complex *)grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed forwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed(const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan(
      FFTW_FORWARD, npts_global, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed forward R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed_r2c(const int npts_global[3],
                                    const cp_mpi_comm_t comm, double *grid_in,
                                    double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan_r2c(
      FFTW_FORWARD, npts_global, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed backwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed(const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan(
      FFTW_BACKWARD, npts_global, comm, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed backwards C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed_c2r(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double complex *grid_in, double *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan_r2c(
      FFTW_BACKWARD, npts_global, comm, (double complex *)grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

// EOF
