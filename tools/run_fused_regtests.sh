#!/bin/bash
# ------------------------------------------------------------------------------
# Run a small force/stress regression subset of CP2K twice: once with the
# default (host) path and once with the fused pw GPU path enabled
# (CP2K_PW_GPU_FUSED=1). The two runs must match the stored reference values.
#
# This gates the "fused" residency work on real force/stress evaluations.
# Regtests run with CP2K_PW_GPU_FUSED off by default, so this script is the
# place where the toggle is actually exercised in CI.
#
# Requires the CP2K runtime environment to be set up already (compilers/MPI and
# a LD_LIBRARY_PATH / RUNPATH that resolves the freshly built libcp2k -- do NOT
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
#   FUSED_REGTEST_DIRS  space separated test dirs               (below)
#   RUN_BASELINE        also run the default path first (1)     (1)
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
RUN_BASELINE=${RUN_BASELINE:-1}
# k-point (FULLSPACE) OT force+stress exercises M5 during the SCF and then the
# host force/stress path; the GPW dir adds plain GPW forces.
FUSED_REGTEST_DIRS=${FUSED_REGTEST_DIRS:-"QS/regtest-kp-4 QS/regtest-gpw-8"}
read -r -a DIRS <<< "${FUSED_REGTEST_DIRS}"

echo "fused-regtests: bin=${CP2K_BIN_DIR} arch=${CP2K_ARCH} ranks=${MPIRANKS} omp=${OMPTHREADS}"
echo "fused-regtests: dirs=${DIRS[*]}"

run_mode() {
  local fused=$1 label=$2
  export CP2K_PW_GPU_FUSED=$fused
  local rc=0
  for d in "${DIRS[@]}"; do
    echo "--------------------------------------------------------------------------"
    echo "### [${label}] ${d}  (CP2K_PW_GPU_FUSED=${fused})"
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
if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_mode 0 "default" || {
    echo "fused-regtests: DEFAULT path FAILED"
    fail=1
  }
fi
run_mode 1 "fused" || {
  echo "fused-regtests: FUSED path FAILED"
  fail=1
}

if [[ ${fail} -ne 0 ]]; then
  echo "fused-regtests: FAILED"
  exit 1
fi
echo "fused-regtests: OK (default and fused match references)"
