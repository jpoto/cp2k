# Status

This repository has three related but independent work streams:

1. **Asynchronous DBCSR multiplication** — committed/pushed (`jvp-dbcsr` /
   `jvp-async-api`). Retested on OT/PBE: **no gain** (OT ~11 % slower, ADMM
   neutral) because the multiplies are diffuse/dependency-bound and the cost is
   host-side.
2. **GPU memory / pinned staging** — committed/pushed (`jvp-hfx-gpu-staging`
   CP2K, `jvp-acc-buffer-api` DBCSR). Pinning alone gave no speedup.
3. **pw device-mirror / GPU cache (`jvp-gpu-cache-v2`)** — current work: remove
   the ~20 MB grid/FFT host↔device transfers by keeping pw arrays
   device-resident. The fused device-resident G-space pipeline is implemented
   end-to-end (milestones M1–M6; see `cp2k_gpu_port.md`).

## Repositories / branches

| Repo | Path | Branch | Status |
|---|---|---|---|
| CP2K | `/workspace` | `jvp-gpu-cache-v2` (from `master`) | **current**, fused pipeline M1–M6 done |
| CP2K | `/workspace` | `jvp-hfx-gpu-staging` | pushed (pinned staging) |
| CP2K | `/workspace` | `jvp-dbcsr` | pushed (async) |
| DBCSR | `tools/toolchain/build/dbcsr` | `jvp-acc-buffer-api` (from `develop`) | **current**, pushed (pinned + device-buffer API) |
| DBCSR | `tools/toolchain/build/dbcsr` | `jvp-async-api` | pushed (async) |

Currently checked out: CP2K `jvp-gpu-cache-v2`, DBCSR `jvp-acc-buffer-api`.

Async commits (author/committer: Johann Pototschnig <j.pototschnig@hzdr.de>):

- DBCSR `dff90106` — Add layer A asynchronous multiply API
- CP2K `d599eb1790` — Use asynchronous DBCSR multiplies in ADMM/McWeeny paths
- CP2K `d0f80eb3d3` — Batch independent DBCSR multiplies in OT and LS-SCF
- CP2K `ff36e2795b` — Batch DBCSR multiplies in LS-SCF sqrt and orth McWeeny

GPU-cache commits (`jvp-gpu-cache-v2`, from `ed115b7042` to `3d858214af`,
author/committer Johann; 20 files, ≈ +2.2k/−71 vs `master`):

- CP2K `ed115b7042` — Port pw device-mirror / GPU-cache work onto fft_gpu refactor
- CP2K `684a3c99b0` — Reduce device alloc/sync churn in the pw GPU path
- CP2K `9b1585887a` — Persist cuSOLVERMp eigensolver workspaces across calls
- CP2K `ad8e161c4d` — Fix device-mirror leak in the fused Poisson chain
- CP2K `43b940170a` — Harden pw device-mirror correctness
- CP2K `41bcba2c37` — Add GPU porting guide
- CP2K `aede425350` — Add device residency/leak regression test and fix stale-source reads
- CP2K `7de99519a2` — Unify PW offload guard to `__NO_OFFLOAD_FFT`; apply make_pretty
- CP2K `eb2709101c` — Add CI script to run force/stress regtest subset with fused GPU path
- CP2K `c344a5910a` — Enable device Poisson chain for HALFSPACE (Γ-point)
- CP2K `f1db045140` — Keep Hartree potential device-resident into integrate
- CP2K `986b151232` — offload_buffer: host/device authority flags + lazy sync API
- CP2K `03e96355d0` — Add device full-sum reduction for r3d pw arrays
- CP2K `3d858214af` — Update GPU porting guide to the current master..HEAD diff

The earlier `jvp-gpu-cache` branch (Stages 1–2, `b7f1904417`/`317ad46642`) is
superseded by `jvp-gpu-cache-v2`.

The DBCSR build is installed into `tools/toolchain/install/dbcsr-git-cuda`; the
toolchain `setup` points at it. CP2K is built with `build_cp2k.sh` (per
`cp2k_build_ftorch.md`), which installs to `/workspace/install`:

```bash
cd tools/toolchain && source ./install/setup && ./build_cp2k.sh -j 16
```

Note: `build_cp2k.sh` does **not** forward GPU-profiling; for NVTX attribution a
local line `CMAKE_OPTIONS+=" -DCP2K_WITH_GPU_PROFILING=ON"` was injected into
`tools/toolchain/build_cp2k.sh` (profiling is a plain CMake flag, independent of
the toolchain).

---

# Work stream 1: asynchronous DBCSR multiplication

## Goal

Use asynchronous DBCSR multiplication to overlap accelerator work with other
computation in CP2K.

## Design: Layer A (deferred completion)

A multiplication runs all DBCSR scheduling / accelerator enqueue on the calling
thread. Only the *completion* is deferred: stream synchronization, device→host
merge, product finalization and table cleanup. The caller can then do independent
work before `dbcsr_multiply_wait`.

API (DBCSR `src/dbcsr_api.F`, exported through `src/dbx/cp_dbcsr_api.F`):

- `dbcsr_async_handle_type`
- `dbcsr_multiply_async(..., handle)`, `dbcsr_multiply_wait(handle)`,
  `dbcsr_multiply_pending(handle)`, `dbcsr_multiply_cancel(handle)`
- `dbcsr_async_set_enabled/get_enabled`

Toggle: `CP2K_ASYNC_MM=1` (default off). Read once in `cp_dbcsr_api`.

Properties:

- Up to 8 in-flight products (independent slots, each with its own product
  matrix, multrec objects and thread-distribution state).
- `flop` output unsupported for async multiplies.
- Caller must not touch the product (or release it) before `wait`; guarded by a
  new `async_pending` flag on the DBCSR matrix type (`dbcsr_release`,
  `dbcsr_copy` source, `dbcsr_filter`, `dbcsr_get_block_p`,
  `dbcsr_multiply_generic` entry).
- Pending slots are completed by `dbcsr_multiply_lib_finalize`.

Implementation notes:

- `multiply_cannon` gains `defer_finalize`/`multrec_out`; the per-thread
  `dbcsr_mm_multrec` objects survive until completion
  (`*_finalize_async` + `*_complete` split in accdrv/sched/csr/multrec).
- `dbcsr_multiply_generic` gains `defer_completion`/`async_slot`; the deferred
  tail is finished by `dbcsr_multiply_complete_async_slot`.
- Unsupported configurations (MPI-RMA/3D, G2G, dense product) fall back to a
  synchronous completion and report a non-pending handle.
- Memory reuse: data areas released during the launch window are kept in a
  per-slot retention list and only returned to the mempool after completion
  (each area's accelerator stream is synchronized first).

## Implemented call sites

- `admm_dm_methods:purify_mcweeny` — launch `PS*PSP` async, overlap with the
  idempotency check, then `P = 3*PSP - 2*PS*PSP`.
- `admm_dm_methods:reverse_mcweeny_step` — submit `P*S` and `S*P` together.
- `iterate_matrix:purify_mcweeny_orth` / `_nonorth` — overlap the next-P product
  with the idempotency check.
- `iterate_matrix:matrix_sqrt_Newton_Schulz` — batch `Y(k+1)=Yk*p` and
  `Z(k+1)=p*Zk`.
- `qs_ot:qs_ot_complex_multiply` — submit the two products of each real/imag
  pair together.

All sites keep the original synchronous code when the toggle is off.

## Not done / rejected (async)

- `cp_dbcsr_operations:cp_dbcsr_plus_fm_fm_t` symmetric `vᵀg`/`gᵀv` pair:
  skipped (breaks `retain_sparsity`/`last_k` semantics).
- `qs_ot_p2m_diag` cosine/sinc overlap: implemented, then reverted — consistently
  ~3–6 % slower (32-H₂O OT 37.0 s → 39.2 s); deferred-flight overhead dominates.
- Cross-routine `mo_derivs` deferral: blocked (OT consumes `mo_derivs` immediately).
- ADMM aux/primary XC reorder: blocked by virial-coupled ordering.
- Large ADMM-DM McWeeny inputs (16-H₂O): pre-existing `double free or corruption`
  during HFX (reproduced with async off).

## Async regtests / perf

Regtests (4 ranks, `OMP_NUM_THREADS=1`), `CP2K_ASYNC_MM=0` and `=1`, identical
results: `regtest-admm-3` 8/8, `regtest-admm-dm` 7/7, `regtest-hybrid-1` 4/4,
`regtest-dm-ls-scf-*` 64/64, `regtest-kp-2` 19/19, `regtest-kp-4` 36/36.

No speedup from the implemented batching on the 16/32-H₂O perf runs (batched
sites are cold). **Key insight:** standard SCF/OT is a dependency chain with no
independent grid work behind a single multiply; per-multiply deferral loses to
its overhead. Large wins need concurrency of independent physics terms.

---

# Work stream 2: GPU memory / pinned staging (current)

## Goal

Reduce/avoid host↔device copies for the dense density/Fock buffers used by GPU
XC/exchange (GauXC, libGint), and lay the groundwork for true zero-copy.

## DBCSR API added (uncommitted, `src/dbcsr_api.F`)

Exported through `src/dbx/cp_dbcsr_api.F`; all fall back cleanly / abort when
built without `__DBCSR_ACC`:

- Host-pinned staging:
  - `dbcsr_acc_hostmem_available()`
  - `dbcsr_acc_hostmem_alloc/dealloc_r8_1d`, `..._r8_2d`
- Persistent device buffer (new, foundation for Level 1):
  - `dbcsr_acc_device_buffer_type`
  - `dbcsr_acc_device_buffer_init/finalize`
  - `dbcsr_acc_device_buffer_ptr` (raw device `TYPE(C_PTR)`), `..._size`, `..._sync`
  - `dbcsr_acc_device_buffer_h2d/d2h_r8_1d`, `..._r8_2d`

DBCSR builds and installs; CP2K rebuilds and links (rc=0).

## HFX (libGint) pinned staging

- `src/hfx_energy_potential.F`: when `use_libGint` **and** the accelerator is
  available, the dense `full_density_alpha/beta` and `full_ks_alpha/beta`
  buffers are allocated host-pinned (otherwise plain `ALLOCATE`).
- `src/hfx_types.F`: `full_ks_pinned` flag; `src/hfx_communication.F` frees
  `full_ks_*` with the matching deallocator.
- Gated on libGint: this build has `CP2K_USE_LIBGINT=OFF`, so the CPU libint HFX
  path is unchanged (no GPU consumer → pinning would be pointless).

## GauXC pinned dense P/VXC

- `src/xc/xc_gauxc_interface.F`: `cp_gauxc_xc_type%vxc_scalar/vxc_zeta` are
  pointers allocated via `gauxc_alloc_dense`; temporaries
  (`density_scalar_rks`, `density_zeta_zero`) are pinned too.
- `src/xc/xc_gauxc_functional.F`: `density_scalar/zeta` are pointers, gathered
  by `dbcsr_to_dense(..., pinned=...)`, freed with the matching deallocator.
- Toggle: **`CP2K_GAUXC_PINNED=0|1`** (default on when accelerator available).
- GauXC already copies P/VXC with `cudaMemcpyAsync(..., cudaMemcpyDefault, ...)`
  (`GauXC .../xc_device_stack_data.cxx`), so pinned memory makes these async.

## Findings

- **Pinning alone gives no measurable speedup.** GauXC device benchmark
  (16 H₂O, GAPW/PBE, `LB/INT_EXECUTION_SPACE DEVICE`):
  - energies identical for pinned vs pageable (`CP2K_GAUXC_PINNED=1/0`);
  - timing within noise (min CP2K ≈ 40.8 s pinned vs 40.7 s pageable; means
    43.9 vs 46.0 s but high variance). A single-rep run even had pinned slower
    (42.2 vs 40.1 s).
  - Reason: the O(nao²) copies are small vs GAPW/XC work and dependency-bound,
    and the buffers are allocated/freed **every SCF iteration**
    (`cudaHostAlloc`/`cudaFreeHost` cost cancels any gain).
- Standard PBE/OT/ADMM benchmarks are **unaffected** by these changes (paths are
  gated to libGint/GauXC; DBCSR additions are additive). Energies match; timing
  on this shared node is noisy.

## Perf directories

- `perf/` — GauXC-device benchmarks:
  - `make_perf_inputs.py` generates `gapw_pbe_gauxc_device_16h2o.inp` and
    `..._32h2o.inp` (GAPW/PBE, GauXC device XC, isolated cluster, 10 SCF steps)
    from `tests/perf-dbcsr-src`.
  - `run_gauxc_pinned_bench.sh` compares `CP2K_GAUXC_PINNED=1` vs `0`.
- `perf-baseline/` — fixed-config A/B harness for standard benchmarks:
  - **2 MPI ranks × 16 OpenMP threads**, `OMP_DYNAMIC=false`, REPS=3.
  - `inputs/` = the 8 standard PBE/OT/ADMM/RI-HFX 16/32-H₂O inputs.
  - `run_bench.sh` records `cp2k_total`, `multiply_cannon`,
    `dbcsr_multiply_generic`, energy, wall into `run/<label>/results.csv`.
  - Usage: `LABEL=baseline CP2K_BIN=... ./perf-baseline/run_bench.sh` then
    `LABEL=current ...`, compare results.
  - **Binding pitfall:** do not combine `mpirun --bind-to none` with several OMP
    threads/rank (`OMP_PROC_BIND=close`): ranks land on the same cores → ~15×
    slowdown. Use the MPI runtime's default (disjoint) binding.

## Standard-benchmark sanity (current build, shared node)

Energies identical to references; CP2K totals vary run-to-run on this shared
node (e.g. `pbe_16h2o_force` 5.7–16 s across runs; stored logs/status-table are
inconsistent baselines). Use `perf-baseline/` and compare `cp2k_min` over several
reps, ideally current vs a baseline binary interleaved.

## GPU profiling budget (nsys)

Setup: `perf/sizes/` (generator `make_sizes.py`, runner
`run_profile_series.sh`, results in `perf/sizes/run/`). Series 4/8/16/32 H₂O with
the *same* 9.85 Å cell, methods PBE/OT, PBE0/ADMM, PBE/DIIS (force), plus the
1-H₂O B3LYP/GAPW case from `gpu_profiling/`. Config: current build binary,
**1 MPI rank × OMP_NUM_THREADS=16**, pinned to the idle GPU. `*.qdstrm` deleted
after every run (`*.stats.txt` / `.nsys-rep` kept). Note: 1-rank + nsys inflates
host/API times; production is 2 ranks × 16.

`cp2k` = CP2K self-reported total; `kern`/`mem` = GPU kernel / memop time;
`GPU%`=(kern+mem)/cp2k; grid/fft/dbcsr = kernel-time split (ms).

| method | H₂O | cp2k (s) | kern | mem | GPU% | grid | fft | dbcsr | H2D+D2H |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DIIS  | 4  | 3.27 | 236 | 496 | 22.4 | 127 | 75 | 6 | 492 ms |
| DIIS  | 8  | 3.56 | 377 | 437 | 22.9 | 251 | 75 | 20 | 434 ms |
| DIIS  | 16 | 4.51 | 727 | 440 | 25.9 | 535 | 75 | 76 | 436 ms |
| OT    | 4  | 9.55 | 689 | 1265 | 20.5 | 355 | 212 | 48 | 1242 ms |
| OT    | 8  | 11.74 | 1285 | 1758 | 25.9 | 732 | 262 | 200 | 1724 ms |
| OT    | 16 | 13.41 | 1846 | 1253 | 23.1 | 1155 | 206 | 410 | 1227 ms |
| OT    | 32 | 41.08 | 11168 | 3113 | 34.8 | 9297 | 437 | 1256 | 3043 ms |
| ADMM  | 4  | 10.12 | 1671 | 1059 | 27.0 | 1415 | 179 | 12 | 1051 ms |
| ADMM  | 8  | 14.09 | 3977 | 1074 | 35.8 | 3724 | 156 | 39 | 1064 ms |
| ADMM  | 16 | 33.09 | 13203 | 968 | 42.8 | 12829 | 156 | 150 | 954 ms |
| ADMM  | 32 | ~593* | 49443 | 1095 | ~8.5 | 48812 | 156 | 371 | 1061 ms |
| B3LYP/GAPW | 1 | 2.63 | 29 | 50 | 3.0 | 13 | 4 | 0.1 | 25 ms |

\*32-H₂O ADMM (1 rank + nsys) is dominated by CPU `integrate_four_center_bin`
(511.7 s self ≈ 86 %); GPU only ~8.5 %.

Observations:

- **GPU share grows with size/method:** PBE/DIIS ~22–26 %, PBE/OT 20→35 %,
  ADMM 27→43 % (4→16 H₂O).
- **GPU kernels are grid-dominated** (collocate+integrate = 60–99 % of kernel
  time for PBE/OT and ADMM); FFT secondary; **DBCSR SMM small** (OT 16: 410 ms /
  1846 ms ≈ 22 % of kernels; OT 32: 1.26 s / 11.2 s ≈ 11 %; ADMM ≈ 1 %).
- **Copies matter mainly for OT:** H2D+D2H = 7–15 % of wall (thousands of small
  transfers: 16 H₂O = 20,698 H2D + 13,236 D2H), ~3 % for ADMM/DIIS.
- **Large CUDA API/alloc churn** (biggest hidden host cost): OT 16 `cudaFree`
  6.4 s / 329k calls, `cudaMemcpyAsync` 2.6 s, sync 1.7 s; OT 32
  `cudaStreamSynchronize` 15.4 s / 59.9k, `cudaFree` 14.3 s / 720k calls;
  ADMM sync 12.8 s (16) / 49.2 s (32) waiting on the grid.
- **Large hybrids are CPU-HFX-bound:** 32-H₂O ADMM = 511 s 4-center exchange
  vs 49 s GPU grid.
- Tiny 1-H₂O B3LYP/GAPW is overhead-bound (GPU 3 %; libxc-GPU wrapper ~41 % of
  host time).

## DBCSR localization (where DBCSR optimizations pay off)

DBCSR cost is almost entirely **host-side**, and large only for PBE/OT (and
DIIS) — not for hybrids/ADMM.

| case | cp2k | `dbcsr_multiply_generic` | of wall | DBCSR GPU kernels | GPU share of DBCSR |
|---|--:|--:|--:|--:|--:|
| OT 16 | 13.4 s | 6.05 s | 45 %* | 0.41 s | ~7 % |
| OT 32 | 41.1 s | 11.4 s | 28 %* | 1.26 s | ~11 % |
| DIIS 16 | 4.5 s | 1.15 s | 25 % | 0.08 s | ~7 % |
| ADMM 16 | 33.1 s | ~2.5 s | 8 % | 0.15 s | ~6 % |

\*nsys-inflated; non-profiled 2-rank OT had multiply ≈19 %. Realistic 20–45 %.
Inside `multiply_cannon` (5.2 s at 16, 9.1 s at 32), the GPU SMM kernels are
only 0.41 / 1.26 s → **~90 % of DBCSR time is the host cannon loop** (stack
build/bin/transpose, packing, launches, sync, MPI).

Where it helps:

- **PBE/OT (and DIIS), medium/large** — reduce host per-multiply cost: batch hot
  independent real-OT products (`qs_ot_refine_irac` `C·FT`/`SC·FT`,
  `qs_ot_localize` rotations), lighter packing, fewer/coarser GPU launches, less
  sync, no per-multiply alloc/free. The async/deferred work aimed here but hit
  cold/dependency-bound sites → no gain; the hot OT chain is where it pays.
- **Copies (OT):** device-resident operands + fewer/larger (pinned) transfers to
  cut the 7–15 % H2D/D2H with 10⁴–10⁵ small copies.
- **Multi-rank production:** the cannon loop also contains MPI packing/comm, so
  host-overhead reductions help the 2-rank runs too.

Where it does **not** help:

- **Hybrid/ADMM:** DBCSR kernels ≈1 %, `multiply_cannon` ≈2.5 s/33 s (8 %); the
  wall is GPU grid + CPU 4-center HFX. Only lever is **overlap** (run the DBCSR
  chain concurrently with grid/HFX on a worker thread).
- **GPU SMM kernel micro-optimization** (3–11 % of wall) and **tiny systems**
  (overhead-bound; DBCSR negligible).

---

# Work stream 3: pw device-mirror / GPU cache (`jvp-gpu-cache-v2`)

**Canonical write-up: `cp2k_gpu_port.md`** (unified porting guide + fused-region
plan). Summary below.

## Goal
Remove the ~20 MB real-space/FFT host↔device transfers (the dominant copy cost
from the profiling: ~1.2 s/13.4 s OT16, ~1 s/33 s ADMM16; the ≥4 MB copies are
92 % of transfer time).

## Root cause (from nsys)
The 20/10/5 MB buffers are the **plane-wave FFT arrays**, copied on every
transform by the pw FFT wrappers (now `fft_gpu.c`; originally
`pw_gpu_internal.c`): r3d→c1d does `HtoD(din)`+`DtoH(zout)`, c1d→r3d does
`HtoD(zin)`+`DtoH(dout)`. Data path: grid collocate writes `rs_rho` offload
buffers → `density_rs2pw` (`qs_collocate_density.F:1611`) merges/FFTs into pw
`rho`/`rho_g` → integrate reads `rs_v`. Skipping a single copy only swaps one
D2H for another because SCF interleaves host copies/reductions; hence the
**fused device-only region** approach.

## Fused pipeline (done, M1–M6)
The whole reciprocal chain stays device-resident; only tiny boundary transfers
remain (G=0 16 B, `ehartree` 8 B, plus `v_hartree_rspace` D2H for the host
real-space integrator). Landing:

- Device mirror + residency contract in `pw_types.F` (`ensure_device`,
  `ensure_host`, `sync_host`, `invalidate_device`, `mark_device_current`,
  `release`, capacity, defined assignment that detaches `device_data`).
- Pooled device allocator + H2D/D2H/D2D + `offload_buffer` authority flags
  (`offload_api.F`, `offload_buffer.{c,h}`, `offload_mempool.{c,h}`).
- Device-pointer FFT variants + c1d multiply/axpy/dot + r3d full-sum
  (`fft_gpu.F`, `fft_gpu.c`, `fft_gpu_kernels.{cu,h}`).
- `pw_offload.F` (new, `pw_gpu`-independent glue; single toggle
  `CP2K_PW_GPU_FUSED`, default off).
- **M3** device `rho_tot_gspace` (`qs_ks_methods.F`); **M4/M5** device Poisson
  reciprocal chain for **Γ-point and k-point** (stress still host), influence-fn
  leak fixed; grid→device producer (`rs_pw_interface.F`); Hartree potential kept
  device-resident into integrate.
- Residency/leak unit test (`pw_unittest.F`), force/stress CI gate
  (`tools/run_fused_regtests.sh`), cuSOLVERMp workspace pooling (compiled only
  with `CP2K_USE_CUSOLVER_MP=ON`; not built here).

The code-review findings from the earlier `jvp-gpu-cache` diff (influence-fn
value-copy leak, unguarded `multiply/dot`, missing capacity tracking, static
`dot_dev` leak, prettifier cosmetics) were addressed in the v2 port.

## Phase A: device-resident grid round-trip (implemented, default off)
Adds `offload_buffer::device_resident` + `CP2K_GRID_DEVICE_RESIDENT` (default
off). For local single-grid (`NGRIDS 1`), the GPU grid backend skips the
collocate D2H / integrate H2D, and `transfer_rs2pw`/`transfer_pw2rs` do
device→device copies (`calculate_rho_elec` / `integrate_v_rspace` opt in;
multi-grid and `GRID%VALIDATE` excluded). Single-grid PBE/OT energies match the
host path (~1e-12); multi-grid is unchanged. The integrate side is active too:
`sum_up_and_integrate` ensures the summed `v_xc+v_hartree` potential is
device-current (single grid) and `potential_pw2rs` does a direct device→device
transfer (D2D if device-current, else one direct H2D from the host array), so
integrate skips its H2D. The `v_xc+v_hartree` sum uses a device r3d axpy
(`pw_offload_axpy_r3d`) when both potentials are device-current (future GPU-XC
backends); otherwise host `pw_axpy`. On 4-H₂O nsys, HtoD drops 4801→4699 calls
and total mem-copy time ~30 %. D2D does a device-wide sync, so larger grids are
the real target. The density host copy is refreshed (`rho%ensure_host()`) for
host XC consumers; SKALA GPW (RKS/UKS) matches the host path. See
`cp2k_gpu_port.md` §4.7.

## Open / not pursued
- **Stress/virial device Poisson** (needs device `pw_derive`/gradient).
- **Distributed-grid device FFT** (multi-rank PPC; currently `PW_MODE_LOCAL` /
  `ray_distribution` only).
- **Device-pointer dense LA** (cuSOLVERMp/SLATE directly on device buffers);
  cuSOLVERMp path not built/validated here.
- **nsys before/after counters** not yet captured (nsys absent in dev container).
- **Realspace/grid residency** judged **not worth it**: host consumers
  (XC, CPU integrate, CDFT/Hirshfeld/CNeo, STM/ELF/cube) would each need to
  become device-aware; the flags + r3d sum are the safe foundation.

## Build / profiling
Clean build via `build_cp2k.sh -j 16` (installs `/workspace/install`). For NVTX
attribution, `-DCP2K_WITH_GPU_PROFILING=ON` is a plain CMake flag (a local line
was injected into `tools/toolchain/build_cp2k.sh`); the toolchain is only needed
for compilers/deps.

## Validation (2 ranks, OMP 1)
`QS/regtest-admm-dm` 7/7 default and 7/7 fused, `QS/regtest-kp-2` 19/19 fused
(FULLSPACE → exercises M5), `QS/regtest-gapw-1` 16/16 fused (GAPW M3);
multi-rank (2×16) OT/ADMM fused vs default agree to ~1e-12; `make_pretty` clean.
A/B of mempool+sync-coarsening on fused OT showed ~2 % mean wall improvement
(small sample).

---

# Next steps (data-driven priority)

1. **Reduce GPU allocation/free + sync churn** (pooling / persistent buffers,
   reuse streams): targets the largest non-GPU overhead for PBE/OT
   (hundreds of thousands of `cudaMalloc/cudaFree`, long `cudaStreamSynchronize`).
2. **Grid collocate/integrate** are the dominant GPU kernels for PBE and ADMM —
   tune kernel/batching settings (LWD scheme, CUTLASS, batch size).
3. **Host-side OT multiply cost** (DBCSR cannon loop, 20–45 % of wall) — batch
   hot independent real-OT products; lighter packing/launches/sync; keep operands
   on the device to cut copies.
4. **Overlap for hybrid/ADMM** — coarse background executor so the ADMM/HFX
   DBCSR/CPU block runs while the main thread does grid XC (the multiplies
   themselves are only ~8 %).
5. **Large-hybrid HFX** — 4-center CPU exchange dominates (32-H₂O ADMM 511 s);
   attack via screening/ADMM/GPU exchange (libGint) rather than DBCSR.
6. **Zero-copy (Level 1)** — device gather/scatter + GPU-library device-pointer
   API; only worth it for OT where copies are 7–15 %, and only after 1/3.
7. **GPU-cache follow-ups (`jvp-gpu-cache-v2`)** — the fused pw pipeline is
   done (M1–M6); next are stress/virial device Poisson, distributed-grid device
   FFT, and nsys before/after counters. See `cp2k_gpu_port.md` §6/§9.
8. Add lifetime assertions / DBCSR sync-vs-async equivalence test; keep the
   `perf-baseline/` A/B harness current.

## How to run

Build (per `cp2k_build_ftorch.md`, installs to `/workspace/install`):

```bash
cd /workspace/tools/toolchain
source ./install/setup
./build_cp2k.sh -j 16
```

Standard fixed-config benchmark (uses the installed binary):

```bash
cd /workspace
source /workspace/install/cp2k_env
LABEL=current CP2K_BIN=/workspace/install/bin/cp2k.psmp \
  ./perf-baseline/run_bench.sh
```

GauXC pinned-vs-pageable:

```bash
REPS=1 ./perf/run_gauxc_pinned_bench.sh gapw_pbe_gauxc_device_16h2o.inp
```

Manual runs: `source /workspace/install/cp2k_env`, then
`CP2K_DATA_DIR=/workspace/data`, `OMP_NUM_THREADS=1` (GauXC regtests) or `16`
(perf), `mpirun -np 2`.
