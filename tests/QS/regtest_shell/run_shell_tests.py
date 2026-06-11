#!/usr/bin/env python3
"""
Shell mode regression tests for CP2K.

This script runs CP2K shell mode tests (-s flag) by feeding commands
to cp2k.psmp and checking the returned energies against expected values.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Tuple

WORKSPACE_ROOT = "/workspace"
CP2K_ENV = "/workspace/install/cp2k_env"

TESTS = [
    {
        "name": "shell_dft",
        "commands": "LOAD tests/QS/regtest_shell/shell_dft.inp\nCALC_E\nEXIT",
        "expected": -17.20455006654803,
    },
    {
        "name": "shell_dftb",
        "commands": "LOAD tests/QS/regtest_shell/shell_dftb.inp\nCALC_E\nEXIT",
        "expected": -4.09819803923235,
    },
    {
        "name": "shell_mp2_direct",
        "commands": "LOAD tests/QS/regtest_shell/shell_mp2.inp\nCALC_MP2 DIRECT\nEXIT",
        "expected": -28.14594509228936,
    },
    {
        "name": "shell_mp2_ri",
        "commands": "LOAD tests/QS/regtest_shell/shell_mp2.inp\nCALC_MP2 RI\nEXIT",
        "expected": -28.14594509228936,
    },
    {
        "name": "shell_rpa_ri",
        "commands": "LOAD tests/QS/regtest_shell/shell_rpa.inp\nCALC_RPA RI\nEXIT",
        "expected": -28.14594509228936,
    },
]

TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs CP2K shell mode regression tests.")
    parser.add_argument("--mpiranks", type=int, default=2)
    parser.add_argument("--ompthreads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--binary", default="cp2k.psmp", help="CP2K binary name (default: cp2k.psmp)")
    return parser.parse_args()


def run_shell_test(test: Dict, mpiranks: int, ompthreads: int, binary: str, timeout: int) -> Tuple[bool, float, str]:
    """Run a shell mode test and return (success, energy, output)."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ompthreads)

    mpiexec = f"mpiexec -n {mpiranks} --bind-to none"
    cmd = f"source {CP2K_ENV} && {mpiexec} {binary} -s"

    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", cmd],
            input=test["commands"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_ROOT,
            env=env,
        )
        output = proc.stdout + proc.stderr

        energy = None
        for line in output.split("\n"):
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith("?"):
                try:
                    val = float(line)
                    if val < 0:
                        energy = val
                        break
                except ValueError:
                    continue

        if energy is None:
            return False, 0.0, output

        success = abs(energy - test["expected"]) < TOLERANCE
        return success, energy, output

    except subprocess.TimeoutExpired:
        return False, 0.0, "Timeout"
    except Exception as e:
        return False, 0.0, str(e)


def main() -> int:
    args = parse_args()

    results = []
    total_time = 0.0

    print("Shell Mode Regression Tests")
    print("=" * 60)
    print(f"MPI ranks:      {args.mpiranks}")
    print(f"OpenMP threads: {args.ompthreads}")
    print(f"Binary:         {args.binary}")
    print("=" * 60)

    for test in TESTS:
        print(f"Testing {test['name']}...", end=" ", flush=True)
        start = time.time()
        success, energy, output = run_shell_test(
            test, args.mpiranks, args.ompthreads, args.binary, args.timeout
        )
        duration = time.time() - start
        total_time += duration

        if success:
            print(f"OK ({duration:.2f}s)")
            results.append(("OK", test["name"], duration))
        else:
            print(f"FAILED ({duration:.2f}s)")
            print(f"  Expected: {test['expected']:.11e}")
            print(f"  Got:      {energy:.11e}")
            results.append(("FAILED", test["name"], duration))

    print("=" * 60)
    passed = sum(1 for r in results if r[0] == "OK")
    failed = sum(1 for r in results if r[0] == "FAILED")
    print(f"Passed: {passed}/{len(results)}, Failed: {failed}, Time: {total_time:.1f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())