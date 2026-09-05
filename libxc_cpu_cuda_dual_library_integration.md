# Libxc CPU + CUDA Dual-Library Integration

## Purpose

This document describes an approach for integrating CUDA-enabled Libxc into CP2K while keeping the CPU and GPU implementations as two independently linkable libraries:

```text
libxc.so
libxc_cuda.so
```

The motivation is to avoid forcing CP2K's final CUDA device-link step to inspect or extract CUDA device objects hidden inside an externally built Libxc archive. That situation can lead to errors involving symbols such as `__cudaRegisterLinkedBinary_*`.

The proposed architecture is:

```text
                         CP2K
                           |
                  XC backend selection
                    /             \
                   /               \
              CPU path           GPU path
                 |                  |
             libxc.so          libxc_cuda.so
                 |                  |
          xc_func_init()     xc_cuda_func_init()
          xc_lda_exc()       xc_cuda_lda_exc()
          ...                ...
```

The key requirement is that the CPU and CUDA implementations must have different externally visible symbol names if both libraries are linked into the same executable.

## 1. Problem Being Solved

A CUDA-enabled Libxc build contains both host-side and CUDA device code.

A conventional external static library can look conceptually like:

```text
libxc.a
 ├── CPU objects
 ├── CUDA host objects
 └── CUDA device objects
```

If CP2K links such an archive into a CUDA executable, CMake's CUDA device-link stage may not automatically treat all CUDA objects hidden inside that archive as part of the device-link input.

This can produce errors around generated CUDA registration/device-link symbols, for example:

```text
__cudaRegisterLinkedBinary_...
__fatbinwrap_...
```

A more robust architecture is to make Libxc own the CUDA compilation/device-link boundary:

```text
Libxc CUDA build
       |
       +-- CUDA source
       |
       +-- nvcc
       |
       +-- CUDA device link
       |
       +-- libxc_cuda.so
```

CP2K then consumes `libxc_cuda.so` as an already-linked CUDA library.

## 2. Why Two Libraries Need Distinct Symbols

Simply compiling Libxc twice is insufficient if both libraries are linked into one CP2K executable.

For example, these two libraries cannot safely coexist if both export:

```text
xc_func_init
xc_func_end
xc_lda_exc
xc_lda_vxc
...
```

The CUDA library therefore needs a distinct namespace:

```text
libxc.so:
    xc_func_init
    xc_func_end
    xc_lda_exc
    xc_lda_vxc
    ...

libxc_cuda.so:
    xc_cuda_func_init
    xc_cuda_func_end
    xc_cuda_lda_exc
    xc_cuda_lda_vxc
    ...
```

This follows the direction discussed by the Libxc developers for allowing CPU and CUDA implementations to coexist.

## 3. Public API Design

### 3.1 CPU API

The normal CPU API remains unchanged:

```c
xc_func_init(...)
xc_func_end(...)
xc_lda_exc(...)
xc_lda_vxc(...)
```

Existing CPU applications should therefore require no source changes.

### 3.2 CUDA API

The CUDA API should expose explicitly prefixed functions:

```c
xc_cuda_func_init(...)
xc_cuda_func_end(...)
xc_cuda_lda_exc(...)
xc_cuda_lda_vxc(...)
```

A separate public header is preferable:

```text
xc.h
xc_cuda.h
```

so that an application can explicitly request both APIs:

```c
#include <xc.h>
#include <xc_cuda.h>
```

This is preferable to requiring CP2K to define a global macro such as:

```c
#define XC_CUDA
#include <xc.h>
```

because CP2K may need to access both CPU and GPU APIs in the same compilation unit.

## 4. Prefixing Mechanism

The implementation should avoid manually renaming every CUDA function.

A compile-time prefix mechanism should be introduced.

Conceptually:

```c
#ifdef XC_CUDA
#define XC_API(name) xc_cuda_##name
#else
#define XC_API(name) xc_##name
#endif
```

An API definition such as:

```c
int XC_API(func_init)(...);
```

would therefore become:

```text
CPU build:
    xc_func_init

CUDA build:
    xc_cuda_func_init
```

The exact implementation should ideally operate at the Libxc code-generation/build layer, because Libxc generates substantial portions of its functional implementation and dispatch tables.

## 5. Do Not Prefix Only the Public API

The public API is the obvious source of symbol collisions, but it may not be the only one.

Libxc contains internal/global implementation functions and generated functional dispatch structures.

For example, conceptually a CPU functional table might contain:

```c
{
    .init = xc_func_init,
    .lda  = xc_lda,
    .vxc  = xc_lda_vxc
}
```

while the CUDA version needs:

```c
{
    .init = xc_cuda_func_init,
    .lda  = xc_cuda_lda,
    .vxc  = xc_cuda_lda_vxc
}
```

The CUDA build should therefore use the CUDA namespace consistently for all externally visible implementation symbols that could collide when the two libraries are linked together.

A useful target namespace is:

```text
xc_*
```

for CPU and:

```text
xc_cuda_*
```

for CUDA.

Static/local symbols do not need renaming.

## 6. Functional Dispatch Tables

The dispatch layer is particularly important.

Libxc contains metadata and function-pointer tables describing each functional. The CUDA build must construct CUDA-specific tables whose function pointers refer to CUDA-prefixed implementations.

Conceptually:

```text
CPU:

xc_func_info
     |
     +-- init -> xc_func_init
     +-- lda  -> xc_lda_...
     +-- gga  -> xc_gga_...
     +-- mgga -> xc_mgga_...


CUDA:

xc_cuda_func_info
     |
     +-- init -> xc_cuda_func_init
     +-- lda  -> xc_cuda_lda_...
     +-- gga  -> xc_cuda_gga_...
     +-- mgga -> xc_cuda_mgga_...
```

It is important that the CUDA dispatch table does not accidentally reference CPU implementations.

Because Libxc uses generated sources, the preferred implementation is to make the code generator aware of the selected symbol prefix rather than maintaining a separate hand-edited CUDA version of every generated table.

## 7. CUDA Function Type

The CPU and CUDA APIs should ideally have distinct opaque/public types.

Instead of exposing only:

```c
xc_func_type
```

for both implementations, introduce:

```c
xc_cuda_func_type
```

even if the initial implementation is structurally identical.

Conceptually:

```c
typedef struct xc_func_type xc_func_type;
typedef struct xc_cuda_func_type xc_cuda_func_type;
```

Then:

```c
int xc_func_init(xc_func_type *, ...);
int xc_cuda_func_init(xc_cuda_func_type *, ...);
```

This provides a clean ABI boundary and prevents accidental use of a CPU object with the CUDA API.

Whether the two structures can share internal definitions should be decided based on the actual Libxc ABI and CUDA implementation. The public API should not depend on accidental binary compatibility.

## 8. CUDA Memory Semantics

The CUDA Libxc API should retain its existing CUDA semantics.

Libxc should not implicitly become responsible for CP2K's general device-memory management.

The intended model is:

```text
CP2K
 |
 +-- owns device allocations
 |
 +-- passes device-resident data
 |
 +-- calls xc_cuda_*
 |
 +-- receives device-resident results
```

rather than implicitly copying CPU arrays to and from the GPU.

This fits CP2K's existing GPU memory-management model.

## 9. Build Two Independent Libxc Configurations

The source tree should be built twice:

```text
libxc/
├── build-cpu/
└── build-cuda/
```

The CPU build produces:

```text
libxc.so
```

and the CUDA build produces:

```text
libxc_cuda.so
```

Conceptually:

```bash
cmake -S libxc -B build-cpu     -DENABLE_CUDA=OFF     ...
```

and:

```bash
cmake -S libxc -B build-cuda     -DENABLE_CUDA=ON     -DXC_SYMBOL_PREFIX=xc_cuda_     ...
```

The exact Libxc CMake option names should be adapted to the version being used.

The important properties are:

```text
CPU:
    CUDA disabled
    normal xc_* namespace

CUDA:
    CUDA enabled
    xc_cuda_* namespace
```

## 10. Prefer a Shared CUDA Library Initially

The first implementation should preferably produce:

```text
libxc.so
libxc_cuda.so
```

rather than static archives.

The reason is the CUDA device-link boundary.

With a shared CUDA library, Libxc can perform the CUDA-specific compilation/device-linking internally:

```text
CUDA Libxc source
       |
      nvcc
       |
 CUDA object files
       |
 CUDA device link
       |
 libxc_cuda.so
```

CP2K then sees a normal shared-library boundary.

This avoids making CP2K responsible for reaching inside a third-party static archive to find CUDA device objects.

A static-library implementation can be added later once the symbol/device-link behavior is understood.

## 11. CMake Targets

The installed Libxc CMake package should ideally export two targets:

```text
Libxc::xc
Libxc::xc_cuda
```

For example:

```cmake
find_package(Libxc REQUIRED)

target_link_libraries(cp2k_cpu_target
    PRIVATE
    Libxc::xc
)

target_link_libraries(cp2k_gpu_target
    PRIVATE
    Libxc::xc_cuda
)
```

The CUDA target should carry all necessary usage requirements:

```text
include directories
compile definitions
CUDA runtime dependency
library path
```

and should not require CP2K to know how Libxc internally performed its CUDA device linking.

## 12. CP2K Integration

CP2K should not expose the implementation details of the two libraries throughout the codebase.

Instead, introduce a small Libxc backend/dispatch layer.

Conceptually:

```text
                 CP2K XC interface
                         |
                 backend selection
                   /           \
                  /             \
              CPU backend    CUDA backend
                  |               |
              xc_* API        xc_cuda_* API
                  |               |
              libxc.so       libxc_cuda.so
```

The higher-level CP2K code should ask the backend to evaluate an XC functional instead of directly choosing between `xc_*` and `xc_cuda_*` throughout the code.

This keeps the CUDA-specific API localized.

## 13. Example CP2K-Level API

A conceptual wrapper could expose:

```text
cp2k_libxc_init_cpu(...)
cp2k_libxc_init_gpu(...)

cp2k_libxc_lda_exc_cpu(...)
cp2k_libxc_lda_exc_gpu(...)

cp2k_libxc_gga_exc_cpu(...)
cp2k_libxc_gga_exc_gpu(...)
```

The exact names and Fortran/C interoperability layer should follow CP2K conventions.

The important property is that the rest of CP2K does not need to know that the CUDA functions are named `xc_cuda_*`.

## 14. CUDA Device Linking

The desired link topology is:

```text
                     Libxc CUDA build
                           |
                     nvcc / nvlink
                           |
                    libxc_cuda.so
                           |
                           v
CP2K CUDA objects ---> CP2K final link ---> cp2k.psmp
                           |
                           +---- libxc_cuda.so
                           +---- libxc.so
```

The undesirable topology is:

```text
CP2K final CUDA device link
              |
       CP2K CUDA objects
              |
              +---- libxc.a
                       |
                       +---- CUDA object files
                       |
                       +---- device code
```

The latter requires CP2K/CMake to understand the CUDA objects hidden inside the archive.

The proposed design deliberately avoids that dependency.

## 15. Changes to CP2K's Current CUDA CMake Configuration

The CP2K CUDA build should avoid using global settings such as:

```cmake
set(CMAKE_CUDA_FLAGS "-dc")
```

as a workaround for the Libxc problem.

CMake's CUDA support should manage separable compilation and device linking through target properties.

Likewise, `CUDA_RESOLVE_DEVICE_SYMBOLS` should be applied only to targets that actually require it, rather than globally.

The goal of the two-library design is that CP2K does not need to resolve Libxc's CUDA device symbols itself.

## 16. Installation Layout

A possible installation layout is:

```text
prefix/
├── include/
│   ├── xc.h
│   └── xc_cuda.h
│
├── lib/
│   ├── libxc.so
│   ├── libxc_cuda.so
│   └── cmake/
│       └── Libxc/
│           ├── LibxcConfig.cmake
│           └── LibxcTargets.cmake
```

The CMake package should define:

```text
Libxc::xc
Libxc::xc_cuda
```

with the appropriate include paths and link dependencies.

## 17. Backward Compatibility

The CPU Libxc API should remain unchanged.

Existing code:

```c
#include <xc.h>

xc_func_type func;

xc_func_init(&func, ...);
```

should continue to compile and link against:

```text
libxc.so
```

The CUDA API is additive:

```c
#include <xc_cuda.h>

xc_cuda_func_type func;

xc_cuda_func_init(&func, ...);
```

and links against:

```text
libxc_cuda.so
```

This means existing Libxc users do not have to change.

## 18. Recommended Implementation Order

### Step 1 — Build CUDA Libxc independently

Establish that the current Libxc CUDA build works by itself.

Verify:

```text
CUDA compilation succeeds
CUDA device linking succeeds
libxc_cuda.so is produced
```

Do not involve CP2K yet.

### Step 2 — Inspect exported symbols

Compare:

```bash
nm -D libxc.so
nm -D libxc_cuda.so
```

Identify all externally visible symbols that collide.

The goal is approximately:

```text
libxc.so:
    xc_*

libxc_cuda.so:
    xc_cuda_*
```

with no unintended common implementation symbols.

### Step 3 — Introduce the symbol-prefix mechanism

Add a configurable prefix to Libxc's generated and handwritten implementation code.

For the CUDA build:

```text
XC_SYMBOL_PREFIX=xc_cuda_
```

### Step 4 — Add `xc_cuda.h`

Expose the CUDA API explicitly.

### Step 5 — Make the CUDA dispatch tables use the CUDA symbols

Verify that the CUDA functional tables point exclusively to `xc_cuda_*` implementations.

### Step 6 — Install/export `Libxc::xc_cuda`

Make the CUDA library discoverable by CMake.

### Step 7 — Integrate with CP2K

Add the CUDA Libxc target and a thin CP2K wrapper.

### Step 8 — Build CP2K

The final CP2K link should consume:

```text
libxc.so
libxc_cuda.so
```

without needing the individual CUDA Libxc object files.

## 19. Validation

### 19.1 Check symbol separation

```bash
nm -D libxc.so > cpu.symbols
nm -D libxc_cuda.so > cuda.symbols
```

Look for accidental collisions.

A useful check is:

```bash
comm -12     <(nm -D --defined-only libxc.so | awk '{print $3}' | sort)     <(nm -D --defined-only libxc_cuda.so | awk '{print $3}' | sort)
```

There should be no unexpected implementation/API collisions.

### 19.2 Check CUDA code is present

Use CUDA tooling appropriate for the CUDA version, for example:

```bash
cuobjdump --list libxc_cuda.so
```

The CUDA library should contain the expected device code.

### 19.3 Link a minimal test program

Before CP2K, create a small CUDA test that links both:

```text
libxc.so
libxc_cuda.so
```

and invokes one CPU and one CUDA functional.

This isolates Libxc's ABI/linking design from CP2K.

### 19.4 Test the CP2K GPU path

Verify:

```text
CPU CP2K + CPU Libxc
GPU CP2K + CPU Libxc
GPU CP2K + CUDA Libxc
```

The second case is particularly important because it establishes that GPU CP2K can still use the ordinary CPU Libxc backend.

## 20. Important ABI/API Questions to Resolve

Before finalizing the implementation, verify:

1. Which Libxc symbols are exported globally?
2. Which internal symbols are generated?
3. Which structures are shared between CPU and CUDA implementations?
4. Does the CUDA implementation require a distinct `xc_cuda_func_type`?
5. Which functional families have CUDA implementations?
6. Are mixed functionals supported?
7. Are hybrid functionals supported?
8. How are CUDA functionals initialized?
9. What memory space does each CUDA API expect?
10. Does the CUDA library require `cudart` dynamically or statically?
11. Does the CUDA library require a particular CUDA architecture?
12. Does the installed CUDA library need to expose any device-link information to consumers?

These questions should be answered from the exact Libxc version being used.

## 21. Why This Design Is Preferable

The main benefits are:

### Clear ABI boundary

```text
CPU:  xc_*
GPU:  xc_cuda_*
```

### No symbol ambiguity

Both libraries can be linked into the same executable.

### No CP2K access to Libxc CUDA object files

CP2K sees a library, not an archive containing hidden CUDA objects.

### Libxc owns its CUDA device linking

This keeps CUDA-specific build complexity in the component that owns the CUDA code.

### Minimal changes to existing CPU applications

The existing `xc_*` API remains intact.

### Cleaner CP2K architecture

CP2K explicitly chooses:

```text
CPU Libxc backend
```

or:

```text
CUDA Libxc backend
```

instead of modifying every existing Libxc call site.

### Future extensibility

The same model could potentially be extended to additional accelerator backends:

```text
libxc.so
libxc_cuda.so
libxc_hip.so
...
```

with corresponding namespaces:

```text
xc_*
xc_cuda_*
xc_hip_*
```

## 22. Alternative: Keep a Single Libxc CUDA Library

An alternative is to keep the current single-library model and make CP2K's CMake CUDA device-link target aware of Libxc's CUDA objects.

This would require exposing the relevant CUDA object files or device-link information from Libxc to CP2K.

Conceptually:

```text
CP2K CUDA device link
        |
        +-- CP2K CUDA objects
        |
        +-- Libxc CUDA objects
```

This is possible, but it tightly couples CP2K's build system to the internal structure of the Libxc CUDA build.

It also makes static-library handling more complicated.

The two-library design avoids that coupling and gives Libxc ownership of its own CUDA device-link boundary.

## 23. Recommended End State

The recommended final architecture is:

```text
                         CP2K
                           |
                   XC abstraction
                           |
              +------------+------------+
              |                         |
         CPU backend               CUDA backend
              |                         |
        Libxc::xc                Libxc::xc_cuda
              |                         |
         libxc.so                  libxc_cuda.so
              |                         |
        xc_* symbols             xc_cuda_* symbols
                                      |
                                CUDA device code
                                      |
                                device link owned
                                   by Libxc
```

The central design principle is:

> **Libxc's CUDA implementation should be a separately namespaced API and a separately built CUDA library whose device-link boundary is owned by Libxc, while CP2K consumes it through an explicit `Libxc::xc_cuda` target.**

This is preferable to hiding CUDA device objects inside the ordinary `libxc.a` and asking CP2K's final CUDA device-link step to discover them.

## 24. Concrete Files/Areas Likely to Change

The exact filenames depend on the Libxc version, but the work should fall into these categories:

```text
Libxc
├── public headers
│   ├── xc.h
│   └── new xc_cuda.h
│
├── API declarations
│   └── configurable CUDA symbol prefix
│
├── functional implementations
│   └── CUDA-prefixed generated symbols
│
├── functional dispatch tables
│   └── CUDA-specific function pointers
│
├── type definitions
│   └── potentially xc_cuda_func_type
│
├── code generators
│   └── prefix-aware symbol generation
│
└── CMake
    ├── CPU target
    ├── CUDA target
    ├── CUDA device linking
    └── installation/export of Libxc::xc_cuda


CP2K
├── CMake
│   ├── detect CPU Libxc
│   ├── detect CUDA Libxc
│   └── import Libxc::xc_cuda
│
├── Libxc interface
│   ├── CPU wrappers
│   └── CUDA wrappers
│
└── XC backend
    └── choose CPU vs CUDA implementation
```

The exact Libxc source files should be identified against the specific Libxc release/commit being used before implementing the patch.

## 25. Final Recommendation

The implementation should proceed in this order:

1. **Build CPU and CUDA Libxc separately.**
2. **Introduce a systematic `xc_cuda_` namespace for the CUDA implementation.**
3. **Give the CUDA implementation an explicit public header/API.**
4. **Ensure CUDA functional dispatch tables reference only CUDA-prefixed functions.**
5. **Build the CUDA implementation as `libxc_cuda.so`.**
6. **Let Libxc perform its own CUDA device linking.**
7. **Export `Libxc::xc` and `Libxc::xc_cuda` from CMake.**
8. **Add a thin CP2K CUDA-Libxc wrapper.**
9. **Remove the need for CP2K to device-link Libxc's individual CUDA objects.**
10. **Validate CPU Libxc, GPU CP2K + CPU Libxc, and GPU CP2K + CUDA Libxc independently.**

The most important implementation detail is the **symbol namespace separation**. Building two libraries without changing the CUDA symbols does not solve the problem; both libraries must be linkable simultaneously without global symbol collisions.
