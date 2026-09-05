#!/bin/bash

# Source CP2K environment
source /workspace/install/cp2k_env

# Set OpenMP threads
export OMP_NUM_THREADS=1

# Run with GPU profiling using nsight systems
# Note: This will create a .qdstrm file for analysis
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=libxc_gpu_profile \
    /workspace/install/bin/cp2k.psmp -i H2O-hybrid-b3lyp_libxc_gpu.inp -o H2O-hybrid-b3lyp_libxc_gpu.out

/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i libxc_gpu_profile.qdstrm -o libxc_gpu_profile.nsys-rep
echo "Report generated"

# Generate text summaries of the profile for analysis
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    libxc_gpu_profile.nsys-rep  | tee libxc_gpu_run.dat
echo "ANALYZED"
