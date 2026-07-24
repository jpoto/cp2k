# Running CP2K Tests

## Environment Setup

Before running any tests, source the CP2K environment:

```bash
source /workspace/install/cp2k_env
```

## Running Regression Tests

### Full MP2 C-Backend Tests

```bash
./tests/do_regtest.py ./install/bin psmp --restrictdir=QS/regtest-ri-mp2-c-backend --restrictdir=QS/regtest-ri-mp2 --num_gpus=1
```

### Single Test Directory

```bash
./tests/do_regtest.py ./install/bin psmp --restrictdir=QS/regtest-ri-mp2-c-backend --num_gpus=1
```

## Test Output

Test results are written to:
- `/workspace/regtesting/TEST-YYYY-MM-DD_HH-MM-SS/`

## Notes

- Tests require `LD_LIBRARY_PATH` to be set (handled by `cp2k_env`)
- GPU tests use `--num_gpus=1` flag
- Each test run creates a new timestamped directory