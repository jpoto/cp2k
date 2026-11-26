# Bringing CP2K to the GPU — a systematic guide

This document is the single, canonical write-up that merges the earlier
**GPU-porting guide** and the **fused device-resident G-space / GPU-cache plan**
(`gpu_cache_plan.md`, now folded in here). It describes a repeatable method for
moving CP2K's hot numerical paths onto the GPU, and documents the concrete
implementation on the `jvp-gpu-cache-v2` branch relative to `master`.

It focuses on **keeping data device-resident** rather than only offloading
kernels, because on modern GPUs the kernels are usually not the bottleneck — the
host↔device round-trips, host-side orchestration and per-call
allocation/synchronisation are.

The reference implementation covers the **plane-wave (pw) FFT / density /
Poisson path** (the "pw device-mirror / GPU cache" work) plus the infrastructure
that all device work shares (device allocator, mempool, sync policy, dense-LA
workspace pooling).

History: the original staged plan lived on `jvp-gpu-cache` (Stages 1–2 committed,
Stage 3 in progress). It was superseded by `jvp-gpu-cache-v2`, which ports the
same primitives onto the `fft_gpu` refactor and completes the fused pipeline
through the Poisson chain. This document describes the v2 end state; the plan
sections below are kept as the rationale and as the blueprint for extending the
approach to further regions.

______________________________________________________________________

## 1. Guiding principles

1. **Offload hot kernels first, but plan for residency.** Profiling decides what
   to touch; priority = (GPU time) + (copy time) + (host glue around the kernel).
1. **Keep arrays device-resident across a whole SCF step.** A kernel that reads
   from and writes to device memory is cheap; a D2H followed by a H2D is not.
1. **Treat the fused chain as one device-only region, with host touch points only
   at its boundaries.** Sitting between two device stages, a host copy or host
   reduction re-introduces the round-trip the residency was meant to remove.
1. **Never change the numerical algorithm.** The device path must produce the same
   math as the host path; it is gated by a runtime toggle and always has a host
   fallback.
1. **One explicit residency contract per object.** Flags + helper methods, so
   every producer/consumer knows whose copy is authoritative.
1. **Preserve Fortran value semantics safely.** Value copies (`=`) must not alias
   device buffers (see the defined assignment below).
1. **Make it measurable.** nsys counters (`cudaMalloc/cudaFree`, `cudaMemcpy*`,
   `cudaStreamSynchronize`) before/after, plus fused-vs-default regtests.

______________________________________________________________________

## 2. Architecture (layers)

```
Fortran call sites        pw_methods / pw_poisson_methods / qs_ks_methods / rs_pw_interface
        │  (host fallback unchanged; device fast path behind CP2K_PW_GPU_FUSED)
        ▼
pw_offload (new)          device copy/axpy/multiply/dot/G0/r3d-sum + availability + toggle
        │
        ▼
fft_gpu (Fortran)         device-pointer FFT wrappers, prepare_device, multiply_device
        │  BIND(C)
        ▼
fft_gpu.c / fft_gpu_kernels.cu    device-pointer FFT + elementwise complex kernels (CUDA/HIP)
        │
        ▼
offload_api / offload_buffer (.F/.c/.h)   device alloc/free/H2D/D2H/D2D; mempool
        │
        ▼
offload_runtime.h         thin CUDA/HIP wrappers
```

`pw_offload` is deliberately free of any dependency on the `pw_gpu` Fortran
module: device kernels are reached through C bindings and all memory management
goes through the offload library (`offload_api`). This is what lets SCF/Poisson
modules call device ops without pulling in `pw_gpu`.

### Core abstraction: the pw type device mirror (`pw_types.F`)

Every generated pw type (`pw_<kind>_<space>_type`) carries an optional device
mirror of `array`:

```fortran
${type}$, CONTIGUOUS, POINTER :: array => NULL()
TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
TYPE(C_PTR) :: device_data = C_NULL_PTR     ! device mirror of `array`
INTEGER(KIND=C_SIZE_T) :: device_capacity = 0_C_SIZE_T  ! bytes actually allocated
LOGICAL :: device_valid = .FALSE.           ! device copy is authoritative
LOGICAL :: host_valid   = .TRUE.            ! host copy is authoritative
```

Type-bound methods (the residency contract):

| method                    | effect                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| `ensure_device()`         | allocate/grow mirror; H2D if host authoritative                                            |
| `ensure_host()`           | D2H if device authoritative; **keeps** device valid                                        |
| `sync_host()`             | D2H if needed, then mark device stale (`device_valid=.FALSE.`) — call before mutating host |
| `invalidate_device()`     | host becomes authoritative, device stale — call before writing host                        |
| `mark_device_current()`   | device becomes authoritative, host stale — call after a device write                       |
| `release()`               | free device mirror + array + grid                                                          |
| `ASSIGNMENT(=)` (defined) | copies `array`/`pw_grid` pointers but **detaches** the device mirror                       |

**Invariants:** at least one of `host_valid`/`device_valid` is true;
`device_valid .AND. .NOT. host_valid` means the host copy is stale.
`device_capacity >= SIZE(array)*elem_size` whenever `device_data` is associated.

`offload_buffer` carries the same idea for grid buffers: additive
`host_current`/`device_current` authority flags (default `host_current=1`), with
`mark_*`, `is_*` and lazy `sync_to_host`/`sync_to_device`.

______________________________________________________________________

## 3. The fused device-resident G-space pipeline

### 3.1 Why: profiling root cause

nsys attributes the large host↔device copies to the **plane-wave FFT arrays**
(the ~20/10/5 MB real-space/c1d buffers), copied on every transform. The two hot
wrappers are `pw_gpu_c1dr3d_3d` (≈ 52 %) and `pw_gpu_r3dc1d_3d` (≈ 45 %) of the
large-copy time. The ≥ 4 MB copies are ~92 % of transfer time, which is ~7–15 %
of wall for OT (`16 H₂O: ~1.2 s / 13.4 s`; `32 H₂O: ~3 s / 41 s`) and ~3 % for
ADMM/DIIS.

The earlier **incremental** approach failed to activate the inverse FFT because
the SCF interleaves host copies and host reductions between the device stages.
Skipping one copy only swaps a D2H for another. The fix is to treat the whole
reciprocal chain as one **device-only region**.

### 3.2 Design principle: device-only fused region, host only at boundaries

```
rho(c1d, device)                         <- ensure_device at entry
  -> rho_tot_gspace(c1d, device)         <- device copy + axpy (device-only)
  -> rhog(c1d, device)                   <- device copy
  -> rhog *= influence_fn                <- device multiply
  -> v_hartree_gspace(c1d, device)       <- device copy (or in-place)
  -> ehartree = 0.5 * Re<rho|v>          <- device dot -> 1 scalar (D2H 8 B)
  -> v_hartree_rspace(r3d, device)       <- c1dr3d DEVICE (no H2D/D2H)
  -> sync_host(v_hartree_rspace)         <- D2H at boundary for the grid integrator
```

Only two host touch points remain inside the region, both tiny:

- the G=0 coefficient for `pw_integrate_function` (charge) → 16 B D2H,
- the scalar `ehartree` → 8 B D2H.

No full grid/c1d array crosses the bus inside the region. The only unavoidable
boundary copy is `v_hartree_rspace` → host for the grid integrator (unless the
realspace consumer is itself made device-aware; see §4.6, which concluded broad
realspace residency is not a net win without making all host consumers
device-aware).

### 3.3 Residency rules (avoiding stale-host bugs)

- Producer sets `device_valid` (e.g. the grid→device producer, or after a `_dev`
  FFT).
- Device-only ops keep `host_valid=.FALSE.`.
- Host read/write goes only through routines that first `sync_host` (already the
  case for `pw_zero/scale/axpy/multiply_with/derive` read-modify-write; FFT host
  outputs use `invalidate_device`).
- Boundary policy: `ensure_device` at region entry; `sync_host` at region exit
  and at any real host consumer (real-space integrate, printing, cube I/O).
- Read-only D2H of a device-current source is allowed in `pw_axpy`/`pw_copy`
  (used where a host result is still required downstream).
- `INTENT(IN)` limitation: routines receiving a `pw` as `INTENT(IN)` cannot
  `ensure_device` it. Ensure residency at the pointer-owning producer, or take
  `INTENT(INOUT)`/`TARGET` in the dedicated `pw_offload` entry points.

### 3.4 Milestones and how they landed

The fused pipeline was executed as separate, individually validated steps:

| Milestone | Scope | Landed as |
| --- | --- | --- |
| **M1** | Relocate kernels to the offload path + `pw_offload` glue (fix the `pw_gpu`↔SCF link boundary); no behaviour change | `pw_offload.F`, `fft_gpu_*_dev` wrappers, C launchers |
| **M2** | Device G=0 read (`pw_integrate_function` c1d) + device dot in `pw_integral_ab` (HALFSPACE-guarded) | `pw_offload_g0_c1d`, `pw_offload_dot_c1d` |
| **M3** | Device-only copy/axpy + device build of `rho_tot_gspace` | `qs_ks_methods.F` device `rho_tot_gspace` mirror |
| **M4** | Poisson reciprocal branch on device-only copies (host fallback kept) | `pw_poisson_methods.F` device Poisson chain |
| **M5** | Consumer `c1dr3d` + `sync_host(v_hartree_rspace)`; keep the Hartree potential device-resident into integrate; validate + profile | `qs_ks_methods.F` consumer, HALFSPACE (Γ-point) enablement |
| **M6** | Remove temporary gates / document toggles | single `CP2K_PW_GPU_FUSED` toggle (default **off**), this document |

The commit sequence on `jvp-gpu-cache-v2` tracks these steps (M1 glue → M3
density mirror fix → M5 device Poisson → HALFSPACE enablement → r3d full-sum →
authority flags → CI gate → this guide).

______________________________________________________________________

## 4. What was changed relative to `master`

`git diff --stat master..HEAD` — **~20 files, ≈ +2.2k/−71** at the time of
writing (this document itself changes the count; it and
`tools/run_fused_regtests.sh`, `pw_offload.F` are new).

| file                                  | change | purpose                                                                                                                                                         |
| ------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/offload/offload_api.F`           | +245   | Fortran device alloc/free/h2d/d2h/d2d; offload_buffer authority flags + sync; mempool-bytes getter                                                              |
| `src/offload/offload_buffer.c`        | +125   | C impl; **mempool-backed** device alloc; D2D; device_bytes; host/device authority flags + lazy sync                                                             |
| `src/offload/offload_buffer.h`        | +50    | declarations + `host_current`/`device_current` fields                                                                                                           |
| `src/offload/offload_mempool.c/.h`    | +6/+6  | `offload_mempool_device_bytes()` (residency/leak test)                                                                                                          |
| `src/pw/pw_types.F`                   | +121   | device mirror fields, contract methods, **defined assignment**, capacity                                                                                        |
| `src/pw/pw_offload.F` *(new)*         | +265   | device copy/axpy/multiply/dot/G0/**r3d-sum**, `CP2K_PW_GPU_FUSED`, availability                                                                                 |
| `src/pw/pw_methods.F`                 | +107   | device fast paths (multiply_with/integral_ab/integrate_function + device r3d sum), device FFT selection, invalidation/sync, read-sync of device-current sources |
| `src/pw/pw_poisson_methods.F`         | +86/−  | **M5** device Poisson reciprocal chain (**Γ-point and k-point**; stress still host); leak fix                                                                    |
| `src/pw_env/rs_pw_interface.F`        | +64    | grid→device producer, device FFT, capacity                                                                                                                      |
| `src/qs_ks_methods.F`                 | +42    | **M3** device `rho_tot_gspace` mirror; keep Hartree potential device-resident into integrate                                                                    |
| `src/fft/fft_gpu.F`                   | +178   | device-pointer FFT wrappers (`fft_gpu_*_dev`), `prepare_device_*`, `multiply_device_c1d`                                                                        |
| `src/fft/gpu/fft_gpu.c`               | +175   | `fft_gpu_cfffg_dev`/`sfffc_dev`, c1d multiply/axpy/dot + r3d sum, coarse sync                                                                                   |
| `src/fft/gpu/fft_gpu_kernels.cu`      | +120   | CUDA kernels + launchers (complex multiply/axpy, block-reduced dot + **r3d sum**)                                                                               |
| `src/fft/gpu/fft_gpu_kernels.h`       | +25    | launcher declarations                                                                                                                                           |
| `src/pw/pw_unittest.F`                | +121   | residency/leak regression test (`check_device_mirror`)                                                                                                          |
| `src/fm/cp_fm_cusolver.c`             | +166/− | persistent cuSOLVERMp workspaces (infra; not residency)                                                                                                         |
| `src/CMakeSources.cmake`              | +1     | add `pw/pw_offload.F`                                                                                                                                           |
| `tools/run_fused_regtests.sh` *(new)* | +79    | CI step: force/stress regtests default+fused                                                                                                                    |
| `cp2k_gpu_port.md` *(new)*            | +314   | this unified guide                                                                                                                                              |

### 4.1 Device allocator + mempool (`offload_api`/`offload_buffer`)

New synchronous, device-only primitives (no host mirror):

```fortran
offload_device_alloc(bytes) -> C_PTR          ! offload_mempool_device_malloc (pooled!)
offload_device_free(ptr)
offload_device_h2d(dev, host, bytes)          ! cudaMemcpy HtoD
offload_device_d2h(host, dev, bytes)          ! cudaMemcpy DtoH
offload_device_d2d(dst, src, bytes)           ! async DtoD (stream 0) + device sync
```

Device-only allocation goes through the **device mempool**, so repeated
allocate/free (e.g. the pw mirrors) reuses cached chunks instead of hitting
`cudaMalloc/cudaFree` every call. (The earlier plan's separate
`CP2K_PW_GPU_PRODUCER` toggle is gone; the producer is gated by the single
`CP2K_PW_GPU_FUSED` toggle below.)

### 4.2 Device kernels (`fft_gpu.c`, `fft_gpu_kernels.cu`)

- Device-pointer FFT variants `fft_gpu_cfffg_dev` / `fft_gpu_sfffc_dev`: the real
  or complex transform arrays are already on the device, so there is **no data
  H2D/D2H** — only the `ghatmap` index map is uploaded and the result is
  D2D-copied to the caller's device buffer.
- Elementwise complex kernels on device-resident `c1d` arrays: `a *= b`
  (multiply), `y = a*x + b*y` (axpy), and a block-reduced `Re(conj(a)·b)` (dot,
  atomicAdd). Launchers take `offloadStream_t`.
- Real full-sum reduction (`fft_gpu_launch_sum_r3d`) for device-resident r3d
  arrays, backing `pw_offload_sum_r3d`.

### 4.3 `pw_offload` (new module)

A `pw_gpu`/`fft_gpu`-independent device-ops layer used from SCF/Poisson:
`pw_offload_copy_c1d`, `pw_offload_axpy_c1d`, `pw_offload_multiply_c1d`,
`pw_offload_dot_c1d`, `pw_offload_g0_c1d`, `pw_offload_sum_r3d`, plus
`pw_offload_available()` and the single toggle
`pw_offload_fused_enabled()` → `CP2K_PW_GPU_FUSED` (default **off**).

### 4.4 Integration points

- **`pw_methods.F`**: device fast paths in `pw_multiply_with` (c1d×c1d),
  `pw_integral_ab` (device dot + HALFSPACE G0 correction), `pw_integrate_function`
  (device G0 read and device r3d full-sum); device-pointer FFT selection
  (`fft_gpu_*_dev` when the input mirror is valid); `invalidate_device`/`sync_host`
  around host-only mutations; read-sync (read-only D2H) of device-current sources
  in `pw_axpy`/`pw_copy`.
- **`pw_poisson_methods.F`**: **M5** — device-only Poisson reciprocal chain
  (copy density→rhog, multiply by influence fn, copy→vhartree) when the density
  is device-resident, `FULLSPACE` (Γ-point and k-point), no aux density, no
  stress, and no one-off `greenfn`. Ensures the *persistent* influence function
  (never a transient value-copy — that leaked).
- **`rs_pw_interface.F`**: grid→device producer — when the realspace collocate
  result lives in the grid's device buffer, D2D it into the pw mirror so the FFT
  can skip its H2D. Single-grid / non-distributed only.
- **`qs_ks_methods.F`**: **M3** — build `rho_tot_gspace` on the device (copy
  `rho_core`/`rho0` + axpy spins) alongside the host copy; consumer keeps the
  Hartree potential device-resident into integrate, syncing to host only for the
  real-space integrator.

### 4.5 Infrastructure / perf (#1, #4, cuSOLVERMp)

- **#1 mempool**: device-only allocs use the mempool.
- **#4 sync coarsening**: removed per-kernel `offloadStreamSynchronize` from the
  pure device ops; ordering is preserved because ops run on the same *blocking*
  stream and every copy boundary is synchronous (`offload_device_d2d`
  device-syncs, `offload_device_d2h` is a synchronous memcpy) under the **legacy
  default stream**.
- **cuSOLVERMp**: eigensolver workspaces (work/matrix/eigen/info) cached in the
  cuSOLVERMp context, grow-on-demand, freed at context destroy (compiled only with
  `CP2K_USE_CUSOLVER_MP=ON`).

### 4.6 Realspace / grid-residency foundation (and why it stops there)

- **`offload_buffer` authority flags** (`host_current`/`device_current`, default
  `host_current=1`; `h2d` sets device-current, `d2h` host-current) plus
  `mark_*`, `is_*`, `sync_to_host/device` (lazy). Additive; consumers opt in
  progressively.
- **Device r3d reduction** (`pw_offload_sum_r3d`) so `pw_integrate_function`
  doesn't force a host round-trip just for a scalar.
- **Conclusion from tracing the grid glue**: `rs%r => rs%buffer%host_buffer` and
  the grid GPU backend copies host↔device in `grid_gpu_context.cu`
  (`copy_to_host` collocate `:571`, `copy_to_gpu` integrate `:625`). The realspace
  `rho`/`rs_v` are read on **host** by many consumers (XC, integrate CPU backend,
  CDFT/Hirshfeld/CNeo `%r`, STM/ELF/cube output). Skipping a single copy only
  swaps one D2H for another — realspace residency is **not** a net win without
  making all those consumers device-aware. The flags + device sum are the safe
  foundation; broad realspace residency was therefore **not** pursued.

### 4.7 Phase A: device-resident grid round-trip (local single grid)

A first, conservative step on top of the authority flags keeps the grid's
`offload_buffer` device copy authoritative across **one grid↔pw round-trip** for
the local single-grid case. Toggle: **`CP2K_GRID_DEVICE_RESIDENT`** (default
**off**, accelerator builds only).

- **Buffer flag**: `offload_buffer` gains `device_resident` (plus
  `offload_buffer_{set,is}_device_resident`). Setting it is an explicit caller
  promise that the grid is consumed on the device before any host read.
- **GPU grid backend** (`grid_gpu_context.cu`): after collocate, a resident
  buffer skips `copy_to_host` and is marked device-current; before integrate, a
  resident *and* device-current buffer skips `copy_to_gpu`. Grid *validation*
  always downloads the host copy (the reference backend compares it).
- **Transfers** (`realspace_grid_types.F`): `transfer_rs2pw` / `transfer_pw2rs`
  take an optional `device` argument. For `border==0`, non-distributed grids with
  matching sizes, and a device-current source, they `offload_device_d2d` instead
  of touching the host. Any host fallback first syncs host from device (and the
  host write marks host authoritative), so the default path is unchanged.
- **Wiring**: `calculate_rho_elec` sets residency for `NGRIDS 1` only and passes
  `device=` to `density_rs2pw`; `integrate_v_rspace` passes `device=` to
  `potential_pw2rs`. The multi-grid spline path and `GRID%VALIDATE` are excluded.

Status / measured: single-grid PBE/OT energies match the host path to ~1e-12
(4- and 16-H₂O), and multi-grid is unchanged (byte-identical path). The
integrate side is now active too: for a single grid `potential_pw2rs` uploads
the summed potential straight into the grid device buffer (D2D if it is already
device-current, otherwise one H2D directly from the host array — no intermediate
`mgrid` staging), so the integrate H2D is skipped. The `v_xc + v_hartree` sum is
formed on the device via `pw_offload_axpy_r3d` when both potentials are
device-current (future GPU-XC backends); otherwise it uses the host `pw_axpy`.
On 4 H₂O nsys
shows host→device transfers drop from 4801 to 4699 calls (~+102 D2D instead) and
total mem-copy time down ~30 %; the collocate D2H and FFT-input H2D also
disappear. The D2D path does a device-wide synchronize, so larger systems (where
the grid transfers dominate) are the real target. The density host copy is
refreshed (`rho%ensure_host()`) so host XC consumers (SKALA/libxc on CPU) still
see current data — this keeps the saved host `dcopy` and FFT-input H2D while
moving only the D2H. Raw `rs%r` host consumers outside the batched transfer
functions still require an explicit sync, and the multi-grid / distributed cases
are intentionally not covered yet.

______________________________________________________________________

## 5. Where the fused region lives (file-by-file)

- **`src/pw/pw_types.F`** — mirror fields + contract methods (`ensure_device`,
  `ensure_host`, `sync_host`, `invalidate_device`, `mark_device_current`,
  `release`) + defined assignment that detaches `device_data`; capacity tracking.
- **`src/pw/pw_offload.F`** — the glue for the device-only region: `copy_c1d`,
  `axpy_c1d`, `multiply_c1d`, `dot_c1d`, `g0_c1d`, `sum_r3d`, availability and
  the `CP2K_PW_GPU_FUSED` toggle, implemented over `offload_api` +
  `fft_gpu_*_dev` C bindings. Guarded by
  `#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)`.
- **`src/pw/pw_methods.F`** — device fast paths and device FFT selection as in
  §4.4; host semantics preserved for the generic entry points.
- **`src/qs_ks_methods.F`** — `calc_rho_tot_gspace` builds the device mirror;
  the consumer after the Poisson solve takes the `_dev` `c1dr3d` path and syncs
  only at the real-space integrator.
- **`src/pw/pw_poisson_methods.F`** — reciprocal branch on device-only copies,
  `pw_multiply_with` device branch, device dot for `ehartree`, no premature
  `vhartree%sync_host()`, persistent `influence_fn` device mirror.
- **`src/pw_env/rs_pw_interface.F`** — grid→device producer (D2D from the grid
  device buffer).
- **`src/fft/fft_gpu.F`, `src/fft/gpu/fft_gpu.c`, `fft_gpu_kernels.{cu,h}`** —
  device-pointer FFT wrappers and elementwise/reduction kernels.

______________________________________________________________________

## 6. Systematic procedure (how to bring another region to the GPU)

Apply this loop to each hot region identified by profiling:

1. **Profile** (nsys): kernel time, `H2D/D2H` bytes and counts,
   `cudaMalloc/free`, `cudaStreamSynchronize`, and the host glue around the
   kernel. Decide the residency boundary (which arrays should stay on the device
   across the step).
1. **Add a device mirror** for those arrays (reuse the pw-type pattern: pointer +
   `device_valid`/`host_valid` + capacity + contract methods + defined
   assignment). If it is not a pw type, introduce an analogous small type.
1. **Implement device kernels** for the elementwise/scatter/reduction parts; wrap
   them `BIND(C)` and expose via a thin Fortran layer (`fft_gpu` / `pw_offload`
   style). Prefer device-pointer variants of existing transforms so no data
   H2D/D2H is needed.
1. **Wire producers** to populate the mirror from the natural device buffer (e.g.
   grid collocate → pw mirror via D2D) instead of from host.
1. **Wire consumers**: keep them device (`mark_device_current` after device
   writes; use device ops downstream); call `ensure_host`/`sync_host` **only** at
   genuine host boundaries (energy/print, host multiplies, integrate inputs).
1. **Gate** the new path behind `CP2K_PW_GPU_FUSED` (or a new toggle); keep the
   host branch byte-for-byte equivalent.
1. **Validate** fused-vs-default energies and reference regtests; measure with
   nsys; iterate.

### Recipes for the remaining gaps

- **Stress/virial device Poisson**: M5 still excludes `h_stress`; the stress path
  needs the reciprocal-space chain plus `pw_derive`/gradient on the device
  (currently host). HALFSPACE (Γ-point) device Poisson is now enabled and
  validated.
- **Distributed-grid FFT**: extend `fft_gpu_*_dev` to the distributed PPC path
  (currently only `PW_MODE_LOCAL` / `ray_distribution`). Needs device-resident
  `ghatmap`/ray mapping and the PPC transpose on device.
- **Interp/integrate glue**: implement device gather/scatter for the sparse
  `gidx` mapping so `density_rs2pw`/`potential_pw2rs`/multigrid stay device-side;
  move the volume-ratio scaling into a device kernel.
- **Defer `ensure_host`**: remove the unconditional D2H after each device FFT by
  tracking which consumers are device-only and syncing only at true host
  boundaries.
- **DBCSR zero-copy**: keep block operands device-resident across a multiply and
  finalize the product on the device; this attacks the host "cannon loop", the
  dominant OT cost.
- **Device-pointer dense LA**: drive cuSOLVERMp/SLATE on device buffers (remove
  the per-solve H2D/D2H of the matrices in `cp_fm_cusolver.c`) so eigensolves stay
  device-resident.

______________________________________________________________________

## 7. Pitfalls / rules learned

- **Streaming model**: the sync coarsening assumes the **legacy default stream**
  (no `--default-stream per-thread`) and blocking `cudaStreamCreate` streams, so
  a synchronous `cudaMemcpy` on stream 0 orders after kernel work on the module
  stream. Do **not** remove the synchronous copy boundaries without unifying the
  streams. `offload_device_d2d` keeps the device sync as the ordering anchor.
- **HALFSPACE G0**: the G=0 coefficient is assumed at `array(1)`; gate the
  correction on `have_g0`, exactly as the host path.
- **Value-copy aliasing**: adding a `C_PTR` component makes intrinsic assignment
  alias device buffers. Provide a **defined assignment** that detaches
  `device_data`; never `ensure_device()` a transient copy (that leaks) — ensure
  the persistent object and use it directly. (This was a real ~20 MB/solve leak
  in the Poisson `influence_fn` value-copy.)
- **Capacity**: track allocated bytes; grow-or-realloc rather than
  allocate-once-from-`SIZE(array)`. Device-mirror capacity is now tracked, but a
  pooled pw object reused with a larger array must still grow (verify).
- **Guard naming**: use the registered `__NO_OFFLOAD_FFT` everywhere (the stale
  `__NO_OFFLOAD_PW` was removed; `make_pretty` rejects unregistered flags).
- **32-bit `n`**: device kernels take `int n`; guard sizes > 2³¹.
- **Leaks**: free cached device buffers at object/context teardown (`dot_dev`
  static, cuSOLVERMp caches).
- **Wavelet Poisson**: the fused `rho_tot_gspace` device mirror is skipped when
  the Poisson solver is `PSOLVER WAVELET` (rs-grid) — the device FFT chain is not
  valid there. This fixed the former GAPW+WAVELET wrong energy
  (`QS/regtest-gpw-1/h2test`), now 56/56 with fused.
- **Gating/coverage**: path is default-**off**; device Poisson covers Γ-point and
  k-point (stress still host); grid→device producer is single-grid/non-distributed;
  multi-rank distributed FFT is still host. Validate before widening.

______________________________________________________________________

## 8. Validation playbook

```bash
# Build
cd /workspace/tools/toolchain && source ./install/setup
./build_cp2k.sh -j 16            # or: cd build && make -j16 cp2k

# Fused vs default (energies must agree, no aborts)
source /workspace/install/cp2k_env
export CP2K_DATA_DIR=/workspace/data OMP_NUM_THREADS=1
BIN=/workspace/build/bin/cp2k.psmp
$BIN -i in.inp -o def.out
CP2K_PW_GPU_FUSED=1 $BIN -i in.inp -o fused.out
grep "ENERGY| Total FORCE_EVAL" def.out fused.out

# Multi-rank (production-like)
OMP_NUM_THREADS=16 mpirun --allow-run-as-root -np 2 $BIN -i in.inp -o out

# Reference regtests (run FULLSPACE/k-point dirs to exercise the M5 device chain)
python3 tests/do_regtest.py /workspace/build/bin psmp \
  --restrictdir "QS/regtest-kp-2" --mpiranks 2 --ompthreads 1 \
  --cp2kdatadir /workspace/data --mpiexec "mpirun --allow-run-as-root -np {N}"
# with CP2K_PW_GPU_FUSED=1 for the fused run

# Force/stress CI gate (default + fused)
tools/run_fused_regtests.sh
```

Also validate with `NGRIDS 1` PBE/OT 4–8 H₂O to confirm the `_dev` FFT path
actually fires, and A/B the fused toggle on/off (energies equal to ~1e-11).

**Measured validation of this branch** (2 ranks, OMP 1): `QS/regtest-admm-dm`
7/7 (default) and 7/7 (fused), `QS/regtest-kp-2` 19/19 (fused, FULLSPACE →
exercises M5), `QS/regtest-gapw-1` 16/16 (fused, GAPW M3). Multi-rank (2×16)
OT/ADMM fused vs default agree to ~1e-12. A/B of #1+#4 on fused OT showed ~2 %
mean wall-time improvement (small sample; GPU-bound/fullspace cases should show
more). nsys counters not yet captured (nsys not present in the dev container);
cuSOLVERMp path not yet built here (`CP2K_USE_CUSOLVER_MP=OFF`).

For running on this repo, the fixed-config A/B harness lives in
`perf-baseline/` (2 ranks × 16 OMP, `REPS=3`); use it to compare `cp2k_min` of
fused vs default interleaved on the same node.

______________________________________________________________________

## 9. Status & open items

**Done (this branch / `jvp-gpu-cache-v2`):** device mirror + contract (+ defined
assignment, capacity); pooled device allocator + copies + **host/device authority
flags**; device FFT variants + c1d kernels + **r3d sum**; `pw_offload`; **M3**
density mirror; **M4/M5** device Poisson chain for **Γ-point and k-point** (stress
still host); grid→device producer; read-sync of stale sources in
`pw_axpy`/`pw_copy`; sync coarsening; cuSOLVERMp workspace pooling;
**residency/leak unit test** (`pw_unittest`); **force/stress CI gate**
(`tools/run_fused_regtests.sh`); Hartree potential kept device-resident into
integrate. Validated: γ-OT fused≈default; force gate `kp-4`/`gpw-8` default+fused;
fused regtests `admm-dm`/`gapw-1`/`kp-2`; `make_pretty` clean.

**Open:** cuSOLVERMp build/run validation (needs cuSOLVERMp); nsys before/after
counters (needs nsys + baseline binary); **stress/virial device Poisson** (device
derive/gradient); **distributed-grid device FFT** (multi-rank PPC); device-pointer
dense LA (cuSOLVERMp/SLATE direct); **DBCSR multiply zero-copy** (host cannon,
external submodule); grid/realspace residency judged **not worth it** (host
consumers) — see §4.6.

Relationship to the other work streams (see `status.md`): the async DBCSR and
pinned-staging streams showed no measurable gain on their own; this fused
G-space residency work is the current focus and the base for the remaining
copy/overlap/DBCSR items.

______________________________________________________________________

### Appendix: toggles & guards

- Runtime toggle: `CP2K_PW_GPU_FUSED=1` (default off) — single switch for the
  fused device G-space pipeline (`pw_offload_fused_enabled()`), including the
  grid→device producer.
- Runtime toggle: `CP2K_GRID_DEVICE_RESIDENT=1` (default off) — Phase A
  device-resident grid round-trip (`offload_grid_resident_enabled()`); local
  single-grid only, disabled under `GRID%VALIDATE` and for multi-grid.
- Compile guards: device code under `#if defined(__OFFLOAD)` (CUDA/HIP); FFT/PW
  device transforms under `#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)`.
- Fallbacks: `offload_api` device primitives are `malloc/free`+"copy" without an
  accelerator; `pw_offload` device ops `CPABORT` if compiled without PW offload
  and are only reached when `device_valid` (never true then).
