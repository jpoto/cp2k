# Periodic SKALA Tests

Tests comparing Native Grid vs GauXC Grid performance for SKALA AI functional in periodic boundary
conditions.

## Tests

| Test                          | System | Grid Type         | Expected SCF Steps | Energy [Ha]   |
| ----------------------------- | ------ | ----------------- | ------------------ | ------------- |
| H2_PERIODIC_NATIVE_SKALA.inp  | H2     | Native (CP2K)     | 11-12              | -1.170490688  |
| H2_PERIODIC_GAUXC_SKALA.inp   | H2     | GauXC (GRID FINE) | 11-12              | -1.170732898  |
| H2O_PERIODIC_NATIVE_SKALA.inp | H2O    | Native (CP2K)     | 11-12              | -17.216280822 |
| H2O_PERIODIC_GAUXC_SKALA.inp  | H2O    | GauXC (GRID FINE) | 11-12              | -17.216204891 |

## Test Configuration

- Cell size: 6x6x6 Angstrom
- CUTOFF: 150 Ry
- REL_CUTOFF: 30 Ry
- XC functional: GAUXC with MODEL SKALA
- Periodic reference: T (required for periodic systems)

## Native Grid Settings

```fortran
&GAUXC
  NATIVE_GRID T
  NATIVE_GRID_ATOM_PARTITION SMOOTH
  NATIVE_GRID_DIAGNOSTICS T
&END GAUXC
```

## Performance Summary

| System           | GAUXC_SKALA | NATIVE_SKALA | Ratio |
| ---------------- | ----------- | ------------ | ----- |
| H2O (total time) | 173.2s      | 308.5s       | 1.78x |
| H2O (per step)   | 15.7s       | 25.7s        | 1.64x |

### Bottlenecks - H2O NATIVE_SKALA (308.5s)

| Routine                          | Time [s] | % Total |
| -------------------------------- | -------- | ------- |
| `skala_gpw_backward`             | 210.3    | 68.2%   |
| `torch_model_forward_mol_tensor` | 94.9     | 30.7%   |

### Bottlenecks - H2O GAUXC_SKALA (173.2s)

| Routine                        | Time [s] | % Total |
| ------------------------------ | -------- | ------- |
| `qs_ks_build_kohn_sham_matrix` | 172.6    | 99.7%   |

## Energy Comparison

| System | GAUXC_SKALA [Ha] | NATIVE_SKALA [Ha] | Difference [Ha] |
| ------ | ---------------- | ----------------- | --------------- |
| H2     | -1.170732898     | -1.170490688      | +0.000242       |
| H2O    | -17.216204891    | -17.216280822     | -0.000076       |

The energy differences are small and expected due to different grid implementations.

## Implementation Notes

### Periodic Reference

All periodic GauXC tests require `PERIODIC_REFERENCE T` in the GAUXC section:

```fortran
&XC_FUNCTIONAL
  &GAUXC
    PERIODIC_REFERENCE T
    MODEL SKALA
  &END GAUXC
&END XC_FUNCTIONAL
```

### Native Grid Atom Partition

The NATIVE_SKALA tests use `SMOOTH` atom partition scheme:

```fortran
NATIVE_GRID_ATOM_PARTITION SMOOTH
```

This provides smoother numerical behavior compared to the default partition.

## Running Tests

```bash
source /path/to/cp2k/install/cp2k_env
OMP_NUM_THREADS=1 mpirun --oversubscribe -np 1 cp2k.popt test.inp
```

## Relevant Files

- `src/skala_gpw_functional.F`: SKALA GPW functional implementation with native grid support
- `src/xc/xc_gauxc_functional.F`: GauXC interface with optimized DBCSR operations
