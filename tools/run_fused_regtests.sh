#!/bin/bash
# ------------------------------------------------------------------------------
# Exercise the pw GPU toggles through the regression harness: for each mode
# (default / fused / grid-resident / both) run the selected regtest dirs and
# compare against the stored reference values. Regtests run with both toggles
# off by default, so this script is where they are actually exercised.
#
# Requires the CP2K runtime environment to be set up already (compilers/MPI and
# an LD_LIBRARY_PATH / RUNPATH that resolves the freshly built libcp2k -- do NOT
# let a stale install/lib shadow it).
#
# Environment overrides (all optional):
#   CP2K_BIN_DIR        directory containing cp2k.<arch>        (build/bin)
#   CP2K_ARCH           binary flavour                          (psmp)
#   CP2K_DATA_DIR       CP2K data directory                     (<root>/data)
#   MPIRANKS            MPI ranks per test                      (2)
#   OMPTHREADS          OpenMP threads per rank                 (1)
#   MPIEXEC             launcher template; {N}=ranks            (mpirun -np {N})
#   REGTEST_TIMEOUT     per-test timeout seconds                (900)
#   FUSED_REG_MODES     space separated modes                   (default fused resident both)
#   FUSED_REGTEST_DIRS  space separated test dirs               (below)
# ------------------------------------------------------------------------------
set -u

CP2K_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CP2K_BIN_DIR=${CP2K_BIN_DIR:-"${CP2K_ROOT}/build/bin"}
CP2K_ARCH=${CP2K_ARCH:-psmp}
CP2K_DATA_DIR=${CP2K_DATA_DIR:-"${CP2K_ROOT}/data"}
MPIRANKS=${MPIRANKS:-2}
OMPTHREADS=${OMPTHREADS:-1}
MPIEXEC=${MPIEXEC:-"mpirun --allow-run-as-root -np {N}"}
REGTEST_TIMEOUT=${REGTEST_TIMEOUT:-900}
# The dedicated dir is NGRIDS 1 (Phase A actually fires); the others add
# force/stress (kp-4, FULLSPACE M5) and plain GPW forces.
FUSED_REGTEST_DIRS=${FUSED_REGTEST_DIRS:-"QS/regtest-gpu-resident QS/regtest-kp-4 QS/regtest-gpw-8"}
FUSED_REG_MODES=${FUSED_REG_MODES:-"default fused resident both"}
read -r -a DIRS <<< "${FUSED_REGTEST_DIRS}"
read -r -a MODES <<< "${FUSED_REG_MODES}"

echo "fused-regtests: bin=${CP2K_BIN_DIR} arch=${CP2K_ARCH} ranks=${MPIRANKS} omp=${OMPTHREADS}"
echo "fused-regtests: dirs=${DIRS[*]}"
echo "fused-regtests: modes=${MODES[*]}"

run_mode() {
  local mode=$1 fused resident rc=0
  case "${mode}" in
    default) fused=0; resident=0 ;;
    fused) fused=1; resident=0 ;;
    resident) fused=0; resident=1 ;;
    both) fused=1; resident=1 ;;
    *) echo "fused-regtests: unknown mode '${mode}'"; return 1 ;;
  esac
  export CP2K_PW_GPU_FUSED=${fused}
  export CP2K_GRID_DEVICE_RESIDENT=${resident}
  for d in "${DIRS[@]}"; do
    echo "--------------------------------------------------------------------------"
    echo "### [${mode}] ${d}  (FUSED=${fused} GRID_DEVICE_RESIDENT=${resident})"
    echo "--------------------------------------------------------------------------"
    python3 "${CP2K_ROOT}/tests/do_regtest.py" "${CP2K_BIN_DIR}" "${CP2K_ARCH}" \
      --restrictdir "${d}" \
      --mpiranks "${MPIRANKS}" --ompthreads "${OMPTHREADS}" \
      --cp2kdatadir "${CP2K_DATA_DIR}" \
      --mpiexec "${MPIEXEC}" --timeout "${REGTEST_TIMEOUT}" || rc=1
  done
  return ${rc}
}

fail=0
for mode in "${MODES[@]}"; do
  run_mode "${mode}" || {
    echo "fused-regtests: mode '${mode}' FAILED"
    fail=1
  }
done

if [[ ${fail} -ne 0 ]]; then
  echo "fused-regtests: FAILED"
  exit 1
fi
echo "fused-regtests: OK (all modes match references)"
