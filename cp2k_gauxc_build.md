# CP2K with GauXC Integration - Build Instructions

## Overview

This document provides step-by-step instructions for building CP2K with GauXC support, including the OneDFT deep learning functional via the SKALA model.

## Prerequisites

- Linux system (tested on Ubuntu)
- Git
- CMake (>= 3.16)
- GCC/G++/GFortran (>= 9.0 recommended)
- OpenMPI
- Python (>= 3.8)
- At least 16GB RAM and 50GB disk space

## Step 1: Clone CP2K Repository

```bash
cd /workspace
git clone https://github.com/cp2k/cp2k.git .
```

## Step 2: Install Toolchain with GauXC

```bash
cd /workspace/tools/toolchain
# Install with GauXC and all dependencies
./install_cp2k_toolchain.sh \
  --with-gauxc=install \
  --with-openmpi=system \
  --with-gcc=system \
  --with-cmake=system \
  --with-ninja=system \
  --with-openblas=install \
  --with-tblite=install \
  --with-libxc=install \
  --with-libint=install \
  --with-fftw=install \
  --with-elpa=install \
  --with-libxsmm=install \
  --with-scalapack=install \
  --with-cosma=install \
  --with-spla=install \
  --with-spglib=install \
  --with-hdf5=install \
  --with-libvdwxc=install \
  --with-sirius=install \
  --with-libvori=install \
  --with-pugixml=install \
  --with-libtorch=install
```

**Note:** This step may take several hours depending on your system resources.

## Step 3: Build CP2K with GauXC Support

```bash
cd /workspace/tools/toolchain
# Build CP2K with GauXC enabled
./build_cp2k.sh -j 10 --prefix /workspace/install
```

**Build options:**
- `-j 10`: Use 10 parallel jobs (adjust based on your CPU cores)
- `--prefix /workspace/install`: Install to this directory
- The build will automatically detect and use GauXC from the toolchain

## Step 4: Verify Installation

```bash
# Source the CP2K environment
source /workspace/install/cp2k_env

# Check CP2K version and GauXC support
/workspace/install/bin/cp2k.psmp --version
```

**Expected output should include:**
```
cp2kflags: ... gauxc gauxc_mpi gauxc_onedft gauxc_host ...
-D __GAUXC
```

## Step 5: Run GauXC Tests

```bash
cd /workspace/tests
# Run GauXC regression tests
python3 do_regtest.py /workspace/install/bin psmp --restrictdir "QS/regtest-gauxc"
```

**Alternative with wrapper script (if library paths are needed):**
```bash
cd /workspace
python3 tests/do_regtest.py /workspace/install/bin psmp --restrictdir "QS/regtest-gauxc" --mpiexec "/tmp/run_cp2k.sh" --workbasedir /tmp/cp2k_test
```

## Step 6: Run Example Calculations

### Example 1: Basic GauXC Calculation

```bash
cd /workspace/tests/QS/regtest-gauxc
source /workspace/install/cp2k_env
/workspace/install/bin/cp2k.psmp 1H2_GAUXC_PBE.inp
```

### Example 2: OneDFT with SKALA

```bash
cd /workspace/tests/QS/regtest-gauxc
source /workspace/install/cp2k_env
/workspace/install/bin/cp2k.psmp 1H2_ONEDFT_PBE.inp
```

## Troubleshooting

### Library Path Issues

If you get "library not found" errors, ensure the environment is properly sourced:

```bash
source /workspace/install/cp2k_env
# Or manually set library paths:
export LD_LIBRARY_PATH=/workspace/install/lib:/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/lib:${LD_LIBRARY_PATH}
```

### Build Stalls or Fails

1. **Reduce parallel jobs:** Use `-j 4` instead of `-j 10`
2. **Check system resources:** Ensure you have enough memory and disk space
3. **Clean build:** Remove `/workspace/build` and try again
4. **Check logs:** Look at `/workspace/build/cmake.log` for errors

### GauXC-Specific Issues

1. **SKALA model not found:** Ensure `/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/share/gauxc/onedft_models/skala-1.1.fun` exists
2. **MPI issues:** Ensure OpenMPI is properly installed and configured
3. **Library conflicts:** Check that no other GauXC installations are in your path

## Key Files and Directories

### GauXC Installation
```
/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/
  ├── lib/
  │   ├── libgauxc.a          # Main GauXC library
  │   ├── libintegratorxx.a    # Integrator library
  │   └── libexchcxx.a        # Exchange-correlation library
  ├── include/
  │   ├── gauxc/              # GauXC headers
  │   └── exchcxx/            # Exchange-correlation headers
  └── share/
      └── gauxc/
          └── onedft_models/
              └── skala-1.1.fun  # SKALA model file
```

### CP2K Installation
```
/workspace/install/
  ├── bin/
  │   ├── cp2k.psmp           # Main CP2K binary with GauXC
  │   └── cp2k.popt -> cp2k.psmp  # Symbolic link
  ├── lib/
  │   └── libcp2k.so*        # CP2K shared libraries
  └── include/                # CP2K headers
```

### Test Files
```
/workspace/tests/QS/regtest-gauxc/
  ├── 1H2_GAUXC_PBE.inp           # Basic GauXC test
  ├── 1H2_ONEDFT_PBE.inp          # OneDFT test
  ├── H2_GAPW_GAUXC_PBE.inp       # GAPW with GauXC
  ├── H2O_GAPW_XC_GAUXC_PBE.inp   # Water with GauXC
  └── TEST_FILES.toml            # Test configuration
```

## Environment Variables

The build script automatically sets up these variables in `/workspace/install/cp2k_env`:

```bash
# GauXC-specific variables
export GAUXC_ROOT="/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes"
export GAUXC_SKALA_MODEL="/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/share/gauxc/onedft_models/skala-1.1.fun"
export GAUXC_VER="1.1-skala-cp2k-fixes"

# Library paths
export LD_LIBRARY_PATH="/workspace/install/lib:/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/lib:${LD_LIBRARY_PATH}"

# Compiler flags
export CP_DFLAGS="-D__GAUXC -DGAUXC_HAS_MPI"
export CP_CFLAGS="-I'/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/include' -I'/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/include/gauxc/modules'"
export CP_LDFLAGS="-L'/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/lib' -Wl,-rpath,'/workspace/tools/toolchain/install/gauxc-1.1-skala-cp2k-fixes/lib'"
export CP_LIBS="-lgauxc -lintegratorxx -lexchcxx"
```

## Performance Notes

- GauXC calculations are computationally intensive
- Expect 1-5 minutes per test case on a modern workstation
- Memory usage typically 1-4GB per MPI process
- OneDFT calculations with SKALA model require additional memory for neural network evaluation

## References

- CP2K: https://www.cp2k.org/
- GauXC: https://github.com/wavefunction91/GauXC
- SKALA: https://huggingface.co/microsoft/skala-1.1

## Support

For issues with this build:
1. Check the CP2K documentation: https://manual.cp2k.org/
2. Consult the GauXC documentation
3. Report issues to the CP2K GitHub repository

## Build Summary

✅ **GauXC Installation**: Complete
✅ **CP2K Build**: Complete with GauXC support
✅ **Regression Tests**: All GauXC tests passing
✅ **Functionality**: OneDFT and SKALA model working

The build is ready for production use with advanced machine learning exchange-correlation functionals.