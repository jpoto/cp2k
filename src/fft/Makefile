.PHONY : all clean

all: fft_unittest.x fft_perftest.x

clean:
	rm -fv *.o */*.o *.x ../offload/*.o

CFLAGS := -fopenmp -no-pie -g -O3 -march=native -Wall -Wextra -Wno-vla-parameter -D__FFTW3 -I${FFTW3_ROOT}/include -L${FFTW3_ROOT}/lib -I${OPENBLAS_ROOT}/include -L${OPENBLAS_ROOT}/lib
NVFLAGS := -g -O3 -lineinfo -arch sm_70 -Wno-deprecated-gpu-targets -Xcompiler "$(CFLAGS)" -D__OFFLOAD
LIBS := -lopenblas

MPICC_PATH := $(shell which mpicc)
ifeq ($(MPICC_PATH),)
  LIBS += -lfftw3 -lfftw3_omp -lm
else
  CC = mpicc
  CFLAGS += -D__FFTW3_MPI -D__parallel
  LIBS += -lfftw3_mpi -lfftw3 -lfftw3_omp -lm
endif

ALL_HEADERS := $(shell find . -name "*.h") $(shell find ../offload/ -name "*.h") $(shell find ../mpiwrap/ -name "*.h")
ALL_OBJECTS := ../offload/offload_mempool.o \
        ../offload/offload_buffer.o \
        ../offload/offload_library.o \
        ../mpiwrap/cp_mpi.o \
        gpu/fft_gpu.o \
        fft_timer.o \
        fft_lib_fftw.o \
        fft_lib.o \
        fft_redistribution.o \
        fft_driver.o \
        fft_grid_layout.o \
        fft_lib_test.o \
        fft_redistribution_test.o \
        fft_grid_test.o

# Enable Cuda when nvcc compiler is present.
NVCC := $(shell which nvcc)
ifneq ($(NVCC),)
LIBS += -lcudart -lcuda -lcublas -lcufftw -lcufft -L${CUDA_PATH}/lib64
CFLAGS += -I${CUDA_PATH}/include -D__OFFLOAD_CUDA

%.o: %.cu $(ALL_HEADERS)
	cd $(dir $<); $(NVCC) -c $(NVFLAGS) $(notdir $<)
endif

%.o: %.c $(ALL_HEADERS)
	cd $(dir $<); $(CC) -c -std=c11 $(CFLAGS) $(notdir $<)

fft_unittest.x: fft_unittest.c $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

fft_perftest.x: fft_perftest.c $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#EOF
