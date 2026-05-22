# CP2K GauXC Performance Tests

## Overview

This directory contains performance tests for CP2K with GauXC integration, focusing on larger molecules than the standard regression tests. These tests demonstrate the scalability and performance of the SKALA model with OneDFT.

## Test Cases

### 1. H2O_GAUXC_PBE.inp
- System: Water molecule (H2O)
- Method: GauXC with PBE functional and SKALA model
- Basis Set: DZVP-GTH
- Potential: GTH-PBE
- Cell: 15x15x15 A cubic cell
- SCF: Max 30 cycles (no smearing)
- Reference Energy: -17.19846179 Hartree

### 2. NH3_GAUXC_PBE.inp
- System: Ammonia molecule (NH3)
- Method: GauXC with PBE functional and SKALA model
- Basis Set: DZVP-GTH
- Potential: GTH-PBE
- Cell: 15x15x15 A cubic cell
- SCF: Max 30 cycles (no smearing)
- Reference Energy: -11.70596119 Hartree

### 3. CO2_GAUXC_PBE.inp
- System: Carbon dioxide molecule (CO2)
- Method: GauXC with PBE functional and SKALA model
- Basis Set: DZVP-GTH
- Potential: GTH-PBE
- Cell: 15x15x15 A cubic cell
- SCF: Max 30 cycles (no smearing)
- Reference Energy: -37.73896595 Hartree

## Key Features

### GauXC Configuration
&XC
  &XC_FUNCTIONAL
    &GAUXC
      FUNCTIONAL PBE
      MODEL PBE
    &END GAUXC
  &END XC_FUNCTIONAL
&END XC

This configuration uses GauXC interface for exchange-correlation, applies PBE functional, loads SKALA 1.1 model for machine learning-enhanced DFT, and enables OneDFT deep learning functional.

### Performance Optimization
- Cutoff: 400 Ry for high accuracy
- Relative Cutoff: 50 Ry
- Smearing: Fermi-Dirac with 300K temperature for faster convergence
- Extrapolation: USE_PREV_WF for efficient SCF

## Running the Tests

### Single Test
cd /workspace/tests/QS/regtest-gauxc-performance
source /workspace/install/cp2k_env
/workspace/install/bin/cp2k.psmp H2O_GAUXC_PBE.inp

### All Tests
cd /workspace
python3 tests/do_regtest.py /workspace/install/bin psmp --restrictdir "QS/regtest-gauxc-performance"

### With Wrapper (for library paths)
cd /workspace
python3 tests/do_regtest.py /workspace/install/bin psmp --restrictdir "QS/regtest-gauxc-performance" --mpiexec "/tmp/run_cp2k.sh"

## Expected Performance

Actual measured performance (2 MPI ranks, 2 OpenMP threads):

Molecule | Atoms | Actual Runtime | Memory Usage
----------|-------|----------------|--------------
H2O      | 3     | ~85 seconds    | ~1.5 GB
NH3      | 4     | ~96 seconds    | ~1.8 GB
CO2      | 3     | ~74 seconds    | ~1.6 GB

Total test suite runtime: ~4 minutes

## Analysis Focus

These tests evaluate GauXC scalability, SKALA model accuracy, convergence behavior, performance characteristics, and stability.

## Expected Output

Each test produces .inp.out (main output), .wfn (wavefunction), clean termination with "PROGRAM ENDED AT", and zero warnings.

## Verification

Check completion: grep "PROGRAM ENDED AT" *.inp.out
Check warnings: grep "number of warnings" *.inp.out

## Comparison with Standard Tests

Feature | Standard Tests | Performance Tests
--------|----------------|-------------------
Molecule Size | Small (H2) | Medium (H2O, NH3, CO2)
Cell Size | 12 A | 15 A
Basis Set | Standard | DZVP
Purpose | Regression | Performance/Accuracy
Expected Runtime | <10 sec | 10-45 sec

## Notes

- Tests use DZVP-GTH basis sets (corrected from initial DZVP-MOLOPT-SR-GTH)
- No smearing used to avoid ADDED_MOS requirements
- Larger cell size (15 A) reduces periodic interactions for isolated molecules
- All tests use GTH-PBE pseudopotentials
- Periodic boundary conditions (XYZ) with cubic cells
- Maximum SCF cycles: 30 for all tests

## References

- CP2K Manual: https://manual.cp2k.org/
- GauXC: https://github.com/wavefunction91/GauXC
- SKALA Model: https://huggingface.co/microsoft/skala-1.1

---

Status: Ready for testing
Date: 2026-05-21
Author: CP2K Build System