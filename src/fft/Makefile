.PHONY : all clean

all: fft_unittest.x fft_perftest.x

clean:
	rm -fv *.o */*.o *.x ../offload/*.o

CFLAGS := -fopenmp -g -O3 -march=native -Wall -Wextra -Wno-vla-parameter
NVFLAGS := -g -O3 -lineinfo -arch sm_70 -Wno-deprecated-gpu-targets -Xcompiler "$(CFLAGS)" -D__OFFLOAD_CUDA
LIBS := -lm -lblas

ALL_HEADERS := $(shell find . -name "*.h") $(shell find ../offload/ -name "*.h")
ALL_OBJECTS := ../offload/offload_buffer.o \
        ../offload/offload_library.o \
        fft_perftest.o \
        fft_redistribution.o \
        fft_redistribution_test.o \
        fft_grid_test.o \
        fft_grid.o \
        fft_grid_layout.o \
        fft_lib.o \
        fft_lib_fftw.o \
        fft_lib_test.o \
        fft_timer.o \
        ../mpiwrap/cp_mpi.o

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

fft_unittest.x: fft_unittest.o $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

fft_perftest.x: fft_perftest.o $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#EOF
