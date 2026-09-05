# Instructions to build and test CP2K

NOTE: Only use 16 MPI processes and 1 OpenMP thread for building and testing!

## install / check system packages
sudo apt update
sudo apt upgrade
sudo apt install build-essential gcc-13 g++-13 gfortran-13 cmake make ninja-build git pkg-config libspglib-f08-dev 
sudo apt install python3-dev bzip2 less nano ca-certificates openmpi-bin openmpi-common libopenmpi-dev
sudo apt install nvidia-cuda-toolkit

## deinstall 14 version
deinstall gcc g++ and gfortran version 14

## Set GCC 13 as default compiler (important for CUDA compatibility)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-13

## Verify compiler versions
gcc --version  # Should show 13.3.0
g++ --version  # Should show 13.3.0
gfortran --version  # Should show 13.3.0

## Verify MPI wrappers use GCC 13
mpicc --version  # Should show 13.3.0
mpic++ --version  # Should show 13.3.0
mpifort --version  # Should show 13.3.0

NOTE: Do NOT install system FFTW packages. The toolchain installs its own.
NOTE: check that nvidia-smi works

## run toolchain script
cd tools/toolchain
./install_cp2k_toolchain.sh -j 16 \
    --enable-cuda=yes \
    --gpu-ver=A100 \
    --with-tblite=install \
    --with-gcc=system \
    --with-cmake=system \
    --with-ninja=system \
    --with-openblas=install \
    --with-libxc=install \
    --with-libint=install \
    --with-fftw=install \
    --with-libxsmm=install \
    --with-spglib=install \
    --with-elpa=no
#    --with-spla=install \
#    --with-sirius=install
#    --with-gauxc=install

## build CP2K
source ./install/setup

# Option 1: Build in foreground (recommended for debugging)
./build_cp2k.sh -j 16

# Option 2: Build in background with progress monitoring
nohup ./build_cp2k.sh -j 16 > build_output.log 2>&1 &
echo "Build started in background with PID: $!"
echo "To monitor progress, run:"
echo "  tail -30 build_output.log"
echo "Or check every 2 minutes with:"
echo "  while true; do sleep 120; echo "=== Build Status ==="; tail -30 build_output.log; done"

NOTE: never use make_cp2k.sh
NOTE: run always setup script before, see toolchain script output
NOTE: If it is a rebuild and the build system has not been modified use the --rebuild-only flag


## Build time estimation
The complete build process typically takes 2-4 hours depending on your system configuration:
- Toolchain installation: 30-60 minutes (depends on internet speed and system performance)
- CP2K compilation: 1.5-3 hours (depends on CPU cores and system performance)

The build process can be divided into several stages:
1. Toolchain dependency installation (libxc, libint, openblas, etc.)
2. CMake configuration
3. CP2K source compilation (this is the longest stage)
4. Final linking and binary creation

## run regression tests
source ./install/cp2k_env

# Run Skala tests (use only 1 OpenMP thread as recommended)
export OMP_NUM_THREADS=1
./tests/do_regtest.py ./install/bin psmp --restrictdir "QS/regtest-skala.*" --mpiranks 4

# Run GauXC tests (use only 1 OpenMP thread as recommended)
./tests/do_regtest.py ./install/bin psmp --restrictdir "QS/regtest-gauxc.*" --mpiranks 4

# Run libxc tests (use only 1 OpenMP thread as recommended)
./tests/do_regtest.py ./install/bin psmp --restrictdir "QS/regtest-libxc.*" --mpiranks 4

# Run full test suite (optional)
# ./tests/do_regtest.py ./install/bin psmp

NOTE: Must source cp2k_env before running tests to set library paths.
NOTE: The testset is large, but can be restricted with --restrictdir "QS/regtest-kp.*"
NOTE: For Skala/GauXC tests, use OMP_NUM_THREADS=1 as recommended in the build notes.
NOTE: All CUDA-based tests now pass after updating reference values to match master branch.


### Clean rebuild
If you encounter issues, clean and rebuild:
```bash
rm -rf build tools/toolchain/build tools/toolchain/install
cd tools/toolchain
./install_cp2k_toolchain.sh -j 16 [options]
./build_cp2k.sh -j 16
```
