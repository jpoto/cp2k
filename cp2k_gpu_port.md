# Bringing CP2K to the GPU — a systematic guide

This document describes a repeatable method for moving CP2K's hot numerical paths onto the GPU, and
documents the concrete implementation done on the `jvp-gpu-cache-v2` branch relative to `master`. It
focuses on **keeping data device‑resident** rather than only offloading kernels, because on modern
GPUs the kernels are usually not the bottleneck — the host↔device round‑trips, host‑side
orchestration and per‑call allocation/synchronisation are.

The reference implementation covers the **plane‑wave (pw) FFT / density / Poisson path** (the "pw
device‑mirror / GPU cache" work) plus the infrastructure that all device work shares (device
allocator, mempool, sync policy, dense‑LA workspace pooling).

______________________________________________________________________

## 1. Guiding principles

1. **Offload hot kernels first, but plan for residency.** Profiling decides what to touch; priority
   = (GPU time) + (copy time) + (host glue around the kernel).
1. **Keep arrays device‑resident across a whole SCF step.** A kernel that reads from and writes to
   device memory is cheap; a D2H followed by a H2D is not.
1. **Never change the numerical algorithm.** The device path must produce the same math as the host
   path; it is gated by a runtime toggle and always has a host fallback.
1. **One explicit residency contract per object.** Flags + helper methods, so every
   producer/consumer knows whose copy is authoritative.
1. **Preserve Fortran value semantics safely.** Value copies (`=`) must not alias device buffers
   (see the defined assignment below).
1. **Make it measurable.** nsys counters (`cudaMalloc/cudaFree`, `cudaMemcpy*`,
   `cudaStreamSynchronize`) before/after, plus fused‑vs‑default regtests.

______________________________________________________________________

## 2. Architecture (layers)

```
Fortran call sites        pw_methods / pw_poisson_methods / qs_ks_methods / rs_pw_interface
        │  (host fallback unchanged; device fast path behind CP2K_PW_GPU_FUSED)
        ▼
pw_offload (new)          device copy/axpy/multiply/dot/G0 + availability + toggle
        │
        ▼
fft_gpu (Fortran)         device-pointer FFT wrappers, prepare_device, multiply_device
        │  BIND(C)
        ▼
fft_gpu.c / kernels.cu    device-pointer FFT + elementwise complex kernels (CUDA/HIP)
        │
        ▼
offload_api / offload_buffer (.F/.c/.h)   device alloc/free/H2D/D2H/D2D; mempool
        │
        ▼
offload_runtime.h         thin CUDA/HIP wrappers
```

### Core abstraction: the pw type device mirror (`pw_types.F`)

Every generated pw type (`pw_<kind>_<space>_type`) carries an optional device mirror of `array`:

```fortran
${type}$, CONTIGUOUS, POINTER :: array => NULL()
TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
TYPE(C_PTR) :: device_data = C_NULL_PTR     ! device mirror of `array`
INTEGER(KIND=C_SIZE_T) :: device_capacity = 0_C_SIZE_T  ! bytes actually allocated
LOGICAL :: device_valid = .FALSE.           ! device copy is authoritative
LOGICAL :: host_valid   = .TRUE.            ! host copy is authoritative
```

Type‑bound methods (the residency contract):

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

______________________________________________________________________

## 3. What was changed relative to `master`

`git diff --stat master..HEAD` (15 files, +1349/−70; commits `ed115b7042`, `684a3c99b0`,
`9b1585887a`, `ad8e161c4d`, `43b940170a`):

| file                             | change | purpose                                                                                                                                      |
| -------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/offload/offload_api.F`      | +102   | Fortran wrappers: `offload_device_alloc/free/h2d/d2h/d2d`                                                                                    |
| `src/offload/offload_buffer.c`   | +78    | C impl of the above; **mempool‑backed** device malloc; D2D async + sync                                                                      |
| `src/offload/offload_buffer.h`   | +27    | declarations                                                                                                                                 |
| `src/pw/pw_types.F`              | +121   | device mirror fields, contract methods, **defined assignment**, capacity                                                                     |
| `src/pw/pw_offload.F` *(new)*    | +238   | device copy/axpy/multiply/dot/G0, `CP2K_PW_GPU_FUSED` toggle, availability                                                                   |
| `src/fft/fft_gpu.F`              | +170   | device‑pointer FFT wrappers (`fft_gpu_*_dev`), `prepare_device_*`, `multiply_device_c1d`                                                     |
| `src/fft/gpu/fft_gpu.c`          | +152   | `fft_gpu_cfffg_dev`/`sfffc_dev` (no data H2D/D2H), c1d multiply/axpy/dot, coarse sync                                                        |
| `src/fft/gpu/fft_gpu_kernels.cu` | +83    | CUDA kernels + launchers (complex multiply/axpy, block‑reduced dot)                                                                          |
| `src/fft/gpu/fft_gpu_kernels.h`  | +19    | launcher declarations                                                                                                                        |
| `src/pw/pw_methods.F`            | +71    | device fast paths (multiply_with/integral_ab/integrate_function), device FFT selection, invalidation on host writes, source‑validity asserts |
| `src/pw/pw_poisson_methods.F`    | +90/−  | M5 device Poisson reciprocal chain (non‑HALFSPACE); leak fix                                                                                 |
| `src/pw_env/rs_pw_interface.F`   | +64    | grid→device producer, device FFT, capacity                                                                                                   |
| `src/qs_ks_methods.F`            | +37    | M3 device `rho_tot_gspace` mirror                                                                                                            |
| `src/fm/cp_fm_cusolver.c`        | +166/− | persistent cuSOLVERMp workspaces (infra; not residency)                                                                                      |
| `src/CMakeSources.cmake`         | +1     | add `pw/pw_offload.F`                                                                                                                        |

### 3.1 Device allocator + mempool (`offload_api`/`offload_buffer`)

New synchronous, device‑only primitives (no host mirror):

```fortran
offload_device_alloc(bytes) -> C_PTR          ! offload_mempool_device_malloc (pooled!)
offload_device_free(ptr)
offload_device_h2d(dev, host, bytes)          ! cudaMemcpy HtoD
offload_device_d2h(host, dev, bytes)          ! cudaMemcpy DtoH
offload_device_d2d(dst, src, bytes)           ! async DtoD (stream 0) + device sync
```

Device‑only allocation goes through the **device mempool**, so repeated allocate/free (e.g. the pw
mirrors) reuses cached chunks instead of hitting `cudaMalloc/cudaFree` every call.

### 3.2 Device kernels (`fft_gpu.c`, `fft_gpu_kernels.cu`)

- Device‑pointer FFT variants `fft_gpu_cfffg_dev` / `fft_gpu_sfffc_dev`: the real or complex
  transform arrays are already on the device, so there is **no data H2D/D2H** — only the `ghatmap`
  index map is uploaded and the result is D2D‑copied to the caller's device buffer.
- Elementwise complex kernels on device‑resident `c1d` arrays: `a *= b` (multiply), `y = a*x + b*y`
  (axpy), and a block‑reduced `Re(conj(a)·b)` (dot, atomicAdd). Launchers take `offloadStream_t`.

### 3.3 `pw_offload` (new module)

A `pw_gpu`/`fft_gpu`‑independent device‑ops layer used from SCF/Poisson: `pw_offload_copy_c1d`,
`pw_offload_axpy_c1d`, `pw_offload_multiply_c1d`, `pw_offload_dot_c1d`, `pw_offload_g0_c1d`, plus
`pw_offload_available()` and the single toggle `pw_offload_fused_enabled()` → `CP2K_PW_GPU_FUSED`
(default **off**).

### 3.4 Integration points

- **`pw_methods.F`**: device fast paths in `pw_multiply_with` (c1d×c1d), `pw_integral_ab` (device
  dot + HALFSPACE G0 correction), `pw_integrate_function` (device G0 read); device‑pointer FFT
  selection (`fft_gpu_*_dev` when the input mirror is valid); `invalidate_device`/`sync_host` around
  host‑only mutations; source‑validity asserts in `pw_axpy`/`pw_copy`.
- **`pw_poisson_methods.F`**: **M5** — device‑only Poisson reciprocal chain (copy density→rhog,
  multiply by influence fn, copy→vhartree) when the density is device‑resident, `FULLSPACE`
  (non‑HALFSPACE), no aux density, no stress, and no one‑off `greenfn`. Ensures the *persistent*
  influence function.
- **`rs_pw_interface.F`**: grid→device producer — when the realspace collocate result lives in the
  grid's device buffer, D2D it into the pw mirror so the FFT can skip its H2D.
- **`qs_ks_methods.F`**: **M3** — build `rho_tot_gspace` on the device (copy rho_core/rho0 + axpy
  spins) alongside the host copy.

### 3.5 Infrastructure / perf (#1, #4, cuSOLVERMp)

- **#1 mempool**: device‑only allocs use the mempool.
- **#4 sync coarsening**: removed per‑kernel `offloadStreamSynchronize` from the pure device ops;
  ordering is preserved because ops run on the same *blocking* stream and every copy boundary is
  synchronous (`offload_device_d2d` device‑syncs, `offload_device_d2h` is a synchronous memcpy)
  under the **legacy default stream**.
- **cuSOLVERMp**: eigensolver workspaces (work/matrix/eigen/info) cached in the cuSOLVERMp context,
  grow‑on‑demand, freed at context destroy (compiled only with `CP2K_USE_CUSOLVER_MP=ON`).

______________________________________________________________________

## 4. Systematic procedure (how to bring another region to the GPU)

Apply this loop to each hot region identified by profiling:

1. **Profile** (nsys): kernel time, `H2D/D2H` bytes and counts, `cudaMalloc/free`,
   `cudaStreamSynchronize`, and the host glue around the kernel. Decide the residency boundary
   (which arrays should stay on the device across the step).
1. **Add a device mirror** for those arrays (reuse the pw‑type pattern: pointer +
   `device_valid`/`host_valid` + capacity + contract methods + defined assignment). If it is not a
   pw type, introduce an analogous small type.
1. **Implement device kernels** for the elementwise/scatter/reduction parts; wrap them `BIND(C)` and
   expose via a thin Fortran layer (`fft_gpu` / `pw_offload` style). Prefer device‑pointer variants
   of existing transforms so no data H2D/D2H is needed.
1. **Wire producers** to populate the mirror from the natural device buffer (e.g. grid collocate →
   pw mirror via D2D) instead of from host.
1. **Wire consumers**: keep them device (`mark_device_current` after device writes; use device ops
   downstream); call `ensure_host`/`sync_host` **only** at genuine host boundaries (energy/print,
   host multiplies, integrate inputs).
1. **Gate** the new path behind `CP2K_PW_GPU_FUSED` (or a new toggle); keep the host branch
   byte‑for‑byte equivalent.
1. **Validate** fused‑vs‑default energies and reference regtests; measure with nsys; iterate.

### Recipes for the remaining gaps

- **HALFSPACE (Γ‑point) Poisson**: relax the `grid_span /= HALFSPACE` gate in the M5 blocks;
  implement the HALFSPACE G0/G² conventions on the device (the device dot already handles the
  `2·dot − g0` identity; guard on `have_g0`). Most OT runs are Γ‑point, so this is the highest‑value
  next step.
- **Distributed‑grid FFT**: extend `fft_gpu_*_dev` to the distributed PPC path (currently only
  `PW_MODE_LOCAL` / `ray_distribution`). Needs device‑resident `ghatmap`/ray mapping and the PPC
  transpose on device.
- **Interp/integrate glue**: implement device gather/scatter for the sparse `gidx` mapping so
  `density_rs2pw`/`potential_pw2rs`/multigrid stay device‑side; move the volume‑ratio scaling into a
  device kernel.
- **Defer `ensure_host`**: remove the unconditional D2H after each device FFT by tracking which
  consumers are device‑only and syncing only at true host boundaries.
- **DBCSR zero‑copy**: keep block operands device‑resident across a multiply and finalize the
  product on the device; this attacks the host "cannon loop", the dominant OT cost.
- **Device‑pointer dense LA**: drive cuSOLVERMp/SLATE on device buffers (remove the per-solve
  H2D/D2H of the matrices in `cp_fm_cusolver.c`) so eigensolves stay device‑resident.

______________________________________________________________________

## 5. Pitfalls / rules learned

- **Streaming model**: the sync coarsening assumes the **legacy default stream** (no
  `--default-stream per-thread`) and blocking `cudaStreamCreate` streams, so a synchronous
  `cudaMemcpy` on stream 0 orders after kernel work on the module stream. Do **not** remove the
  synchronous copy boundaries without unifying the streams. `offload_device_d2d` keeps the device
  sync as the ordering anchor.
- **HALFSPACE G0**: the G=0 coefficient is assumed at `array(1)`; gate the correction on `have_g0`,
  exactly as the host path.
- **Value‑copy aliasing**: adding a `C_PTR` component makes intrinsic assignment alias device
  buffers. Provide a **defined assignment** that detaches `device_data`; never `ensure_device()` a
  transient copy (that leaks) — ensure the persistent object and use it directly.
- **Capacity**: track allocated bytes; grow‑or‑realloc rather than allocate‑once‑from‑`SIZE(array)`.
- **Guard naming**: FFT code uses `__NO_OFFLOAD_FFT`; the ported `pw_offload.F` still tests
  `__NO_OFFLOAD_PW` — unify before relying on `__NO_OFFLOAD_FFT` builds.
- **32‑bit `n`**: device kernels take `int n`; guard sizes > 2³¹.
- **Leaks**: free cached device buffers at object/context teardown (`dot_dev` static, cuSOLVERMp
  caches).
- **Gating/coverage**: path is default‑**off**, `FULLSPACE`‑only, single‑grid producer,
  non‑distributed. Validate before widening.

______________________________________________________________________

## 6. Validation playbook

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
```

**Measured validation of this branch** (2 ranks, OMP 1): `QS/regtest-admm-dm` 7/7 (default) and 7/7
(fused), `QS/regtest-kp-2` 19/19 (fused, FULLSPACE → exercises M5), `QS/regtest-gapw-1` 16/16
(fused, GAPW M3). Multi‑rank (2×16) OT/ADMM fused vs default agree to ~1e‑12. A/B of #1+#4 on fused
OT showed ~2 % mean wall‑time improvement (small sample; GPU‑bound/fullspace cases should show
more). nsys counters not yet captured (nsys not present in the dev container); cuSOLVERMp path not
yet built here (`CP2K_USE_CUSOLVER_MP=OFF`).

______________________________________________________________________

## 7. Status & open items

**Done (this branch):** device mirror + contract; device allocator (pooled) + copies; device FFT
variants + c1d kernels; `pw_offload`; M3 density mirror; M4/M5 Poisson device chain (non‑HALFSPACE);
grid→device producer; defined assignment + capacity hardening; sync coarsening; cuSOLVERMp workspace
pooling; multi‑rank fused regtests green.

**Open:** cuSOLVERMp build/run validation (needs cuSOLVERMp); nsys before/after counters (needs nsys
\+ baseline binary); residency/leak regression test; HALFSPACE device Poisson; distributed‑grid
device FFT; interp/integrate device glue; deferred `ensure_host`; DBCSR zero‑copy; device‑pointer
dense LA; unify `__NO_OFFLOAD_*` guards; broaden to forces/stress.

______________________________________________________________________

### Appendix: toggles & guards

- Runtime toggle: `CP2K_PW_GPU_FUSED=1` (default off) — single switch for the fused device G‑space
  pipeline (`pw_offload_fused_enabled()`).
- Compile guards: device code under `#if defined(__OFFLOAD)` (CUDA/HIP); FFT/PW device transforms
  under `#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)`.
- Fallbacks: `offload_api` device primitives are `malloc/free`+"copy" without an accelerator;
  `pw_offload` device ops `CPABORT` if compiled without PW offload and are only reached when
  `device_valid` (never true then).
