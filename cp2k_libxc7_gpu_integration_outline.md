# CP2K + LibXC 7 GPU integration — summary and implementation outline

## 1. Key insights

- **Use upstream LibXC 7**, not `MALBECC/libxc-gpu`, for the CP2K integration.
  `MALBECC/libxc-gpu` is a historical/specialized GPU fork associated with LIO. Modern upstream LibXC has integrated GPU support.

- **LibXC 7 uses the same XC evaluation API for CPU and GPU.**
  The backend is selected when the `xc_func_type` is initialized.

- The relevant LibXC 7 API is:
  ```c
  int xc_func_init(
      xc_func_type *p,
      int functional,
      int nspin);

  int xc_func_init_flags(
      xc_func_type *p,
      int functional,
      int nspin,
      int flags);
  ```

- The relevant device flag is:
  ```c
  XC_FLAGS_ON_DEVICE
  ```
  with the corresponding host flag:
  ```c
  XC_FLAGS_ON_HOST
  ```

- GPU initialization is therefore conceptually:
  ```c
  xc_func_type func;

  xc_func_init_flags(
      &func,
      XC_GGA_X_PBE,
      XC_UNPOLARIZED,
      XC_FLAGS_ON_DEVICE);
  ```

- Evaluation remains the ordinary LibXC call:
  ```c
  xc_gga_exc_vxc(
      &func,
      n,
      rho,
      sigma,
      &out);
  ```

  If the functional was initialized with `XC_FLAGS_ON_DEVICE`, the supplied arrays are expected to be device-resident.

- **There is no need for a separate `xc_gga_gpu()` API** in the CP2K wrapper.

- CP2K's existing offload infrastructure should be reused for device-memory management and, if supported by LibXC, execution-stream integration. The wrapper should not introduce its own CUDA/HIP allocation or host/device copies.

- The clean architecture is:
  ```text
  CP2K Fortran
       |
       v
  thin C wrapper
       |
       +---- CPU: LibXC 7
       |
       +---- GPU: LibXC 7
  ```

- The main CP2K architectural question is therefore **not how to write a GPU XC kernel**, but whether the arrays extracted by the existing `xc_rho_set` / XC machinery are already available as device pointers in the GPU Quickstep path.

---

## 2. Recommended CP2K integration

### Existing conceptual path

```text
xc_libxc.F
    |
    v
xc_libxc_wrap.F
    |
    v
LibXC Fortran/C interface
    |
    v
LibXC
```

### Proposed path

```text
xc_libxc.F
    |
    v
xc_libxc_wrap.F
    |
    | ISO_C_BINDING
    v
small CP2K C wrapper
    |
    +-------------------+
    |                   |
 CPU backend        GPU backend
    |                   |
    v                   v
 LibXC 7 CPU       LibXC 7 GPU
                        |
                  device pointers
```

The C wrapper should contain LibXC-7-specific details and keep them out of the Fortran XC logic.

---

## 3. Proposed C interface

Start with a very small interface.

### Header

```c
#ifndef CP2K_XC_LIBXC_H
#define CP2K_XC_LIBXC_H

#include <stddef.h>
#include <xc.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP2K_LIBXC_CPU = 0,
    CP2K_LIBXC_GPU = 1
} cp2k_libxc_backend;

/* Initialize a LibXC functional for CPU or GPU execution. */
int cp2k_libxc_init(
    xc_func_type *func,
    int functional,
    int nspin,
    cp2k_libxc_backend backend);

/* GGA evaluation. Add LDA/MGGA wrappers following the same pattern. */
int cp2k_libxc_gga(
    xc_func_type *func,
    size_t n,
    const double *rho,
    const double *sigma,
    xc_gga_out_params *out);

#ifdef __cplusplus
}
#endif

#endif
```

---

## 4. C implementation

The important part is initialization.

```c
#include "cp2k_xc_libxc.h"

int cp2k_libxc_init(
    xc_func_type *func,
    int functional,
    int nspin,
    cp2k_libxc_backend backend)
{
    const int flags =
        (backend == CP2K_LIBXC_GPU)
        ? XC_FLAGS_ON_DEVICE
        : XC_FLAGS_ON_HOST;

    return xc_func_init_flags(
        func,
        functional,
        nspin,
        flags);
}
```

The GGA wrapper is deliberately thin:

```c
int cp2k_libxc_gga(
    xc_func_type *func,
    size_t n,
    const double *rho,
    const double *sigma,
    xc_gga_out_params *out)
{
    xc_gga_exc_vxc(
        func,
        n,
        rho,
        sigma,
        out);

    return 0;
}
```

The same pattern can be used for LDA and MGGA.

The wrapper should **not** contain:

```c
cudaMalloc(...)
hipMalloc(...)
cudaMemcpy(...)
hipMemcpy(...)
```

CP2K owns the memory.

---

## 5. CPU and GPU calls

### CPU

```c
xc_func_type func;

xc_func_init_flags(
    &func,
    XC_GGA_X_PBE,
    XC_UNPOLARIZED,
    XC_FLAGS_ON_HOST);

xc_gga_exc_vxc(
    &func,
    n,
    rho_host,
    sigma_host,
    &out);
```

### GPU

```c
xc_func_type func;

xc_func_init_flags(
    &func,
    XC_GGA_X_PBE,
    XC_UNPOLARIZED,
    XC_FLAGS_ON_DEVICE);

xc_gga_exc_vxc(
    &func,
    n,
    rho_device,
    sigma_device,
    &out);
```

The **evaluation call is the same**. The initialization flag selects the implementation.

---

## 6. Fortran side

Keep the existing CP2K XC logic largely unchanged.

Add a C binding in the wrapper module:

```fortran
interface
   function cp2k_libxc_init(func, functional, nspin, backend) &
        bind(C, name="cp2k_libxc_init") result(ierr)

      use iso_c_binding

      type(c_ptr), value :: func
      integer(c_int), value :: functional
      integer(c_int), value :: nspin
      integer(c_int), value :: backend
      integer(c_int) :: ierr

   end function cp2k_libxc_init
end interface
```

The exact declaration should follow how CP2K currently represents `xc_func_t` in `xc_libxc_wrap.F`.

The high-level logic can then remain:

```fortran
if (use_gpu) then
   backend = CP2K_LIBXC_GPU
else
   backend = CP2K_LIBXC_CPU
endif
```

and initialization selects the appropriate LibXC implementation.

---

## 7. Do not initially modify the offload layer

The first prototype should assume that CP2K can already supply device-resident arrays.

Desired data flow:

```text
CP2K GPU density
      |
      | device pointer
      v
C wrapper
      |
      | unchanged pointer
      v
LibXC 7 GPU
      |
      | device pointer
      v
CP2K GPU XC potential
```

Avoid:

```text
GPU
 |
 v
CPU copy
 |
 v
LibXC CPU
 |
 v
GPU copy
```

The existing CP2K offload infrastructure should be used if it is necessary to:

- obtain a device pointer from a CP2K buffer,
- identify memory location,
- synchronize dependencies,
- provide a CUDA/HIP execution stream.

But it should not become a staging layer that copies every XC array.

---

## 8. Most important CP2K source path to investigate

The next source-level investigation should trace:

```text
xc_libxc.F
    |
    +-- xc_rho_set_get(...)
    |
    +-- xc_derivative_set / derivative arrays
    |
    v
rho, sigma, tau, lapl, ...
    |
    v
LibXC
```

The key question is:

> In the GPU Quickstep path, are the arrays handed to `xc_libxc.F` already device-resident?

If yes, the LibXC integration should be relatively small.

If no, the larger task is making the relevant XC data path device-resident.

---

## 9. Recommended implementation order

### Step 1 — CPU wrapper

Route one existing PBE GGA call through:

```text
CP2K → C wrapper → LibXC 7 CPU
```

and verify no numerical/functional change.

### Step 2 — standalone LibXC GPU test

Before involving CP2K:

```text
CUDA/HIP device arrays
        |
        v
xc_func_init_flags(..., XC_FLAGS_ON_DEVICE)
        |
        v
xc_gga_exc_vxc(...)
        |
        v
device outputs
```

This establishes the exact LibXC 7 GPU behavior.

### Step 3 — CP2K GPU pointer test

Pass CP2K's existing device-resident arrays directly:

```text
CP2K device arrays
        |
        v
C wrapper
        |
        v
LibXC GPU
```

### Step 4 — integrate with the GPU Quickstep path

Only after the direct GPU evaluation works should the backend selection become automatic.

### Step 5 — investigate asynchronous execution

Determine whether LibXC 7 exposes CUDA/HIP stream control. If it does, connect it to CP2K's existing offload stream.

The target is:

```text
CP2K stream
   |
   +-- density calculation
   |
   +-- LibXC GPU
   |
   +-- XC potential
   |
   +-- next GPU operation
```

rather than repeated global synchronization.

---

## 10. Suggested initial file changes

A minimal first implementation could be:

```text
src/xc/
    xc_libxc.F              existing
    xc_libxc_wrap.F         existing
    cp2k_xc_libxc.h         new
    cp2k_xc_libxc.c         new
```

Potentially only small build-system changes are needed.

I would **not initially create**:

```text
xc_libxc_gpu.F
```

and I would **not initially modify**:

```text
src/offload/
```

unless tracing the existing device buffers shows that an offload API extension is actually required.

---

## 11. Target architecture

```text
                         CP2K
                           |
                     xc_libxc.F
                           |
                    xc_libxc_wrap.F
                           |
                       C ABI
                           |
                  cp2k_xc_libxc.c
                           |
              +------------+------------+
              |                         |
       XC_FLAGS_ON_HOST         XC_FLAGS_ON_DEVICE
              |                         |
              v                         v
         LibXC 7 CPU               LibXC 7 GPU
              |                         |
        host arrays               device arrays
              |                         |
              +------------+------------+
                           |
                         CP2K
```

### Bottom line

The LibXC 7 interface makes the CP2K integration substantially simpler than initially expected:

```c
xc_func_init_flags(..., XC_FLAGS_ON_DEVICE);
```

followed by the **normal**:

```c
xc_gga_exc_vxc(...);
```

The main CP2K work is therefore the C/Fortran boundary and ensuring that the XC input/output arrays are genuinely device-resident. The existing CP2K offload infrastructure is the natural mechanism to reuse for that part.
