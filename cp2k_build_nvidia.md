# Build CP2K with NVIDIA GPU support

Branch: `jvp-MP2-GEMM`

## System packages
```bash
sudo apt update && sudo apt upgrade
sudo apt install build-essential gcc g++ gfortran cmake make ninja-build git pkg-config libspglib-f08-dev
sudo apt install python3-dev bzip2 less nano ca-certificates openmpi-bin openmpi-common libopenmpi-dev
sudo apt install nvidia-cuda-toolkit
```
NOTE: Do NOT install system FFTW packages. The toolchain installs its own.

## Toolchain
```bash
cd tools/toolchain
./install_cp2k_toolchain.sh \
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
    --with-libgint \
    --with-spla
```

## Build CP2K
```bash
source ./install/setup
./build_cp2k.sh -j 16
```

## Run CP2K
```bash
source ./install/cp2k_env
cp2k.psmp --version
```

## Regression tests

### Run all tests
```bash
source ./install/cp2k_env
./tests/do_regtest.py ./install/bin psmp --num_gpus=1
```

### Run specific test folders
```bash
source ./install/cp2k_env
./tests/do_regtest.py ./install/bin psmp --restrictdir=QS/regtest-ri-mp2-c-backend-blas --num_gpus=1
./tests/do_regtest.py ./install/bin psmp --restrictdir=QS/regtest-ri-mp2-c-backend-spla --num_gpus=1
./tests/do_regtest.py ./install/bin psmp --restrictdir=QS/regtest-ri-mp2-c-backend-cublas --num_gpus=1
```

NOTE: Use `--rebuild-only` flag for rebuilds without toolchain changes (if only files in src are changed).