# Kernel Backends (Metal + Triton)

This repo has two GPU kernel backends:

- Metal/MPS: optimized kernels in a single `.metal` file, dispatched via a
  JIT-built PyTorch extension.
- Triton/CUDA: kernels in Python (`@triton.jit`) with optional import-time
  fallback for non-CUDA machines.

This document captures the thermodynamics "fast path" and the Triton port that
targets feature parity with the optimized Metal implementation.

## Metal Backend (MPS)

Source + dispatch:

- `sensorium/kernels/metal/manifold_physics.metal`: all compute kernels.
- `sensorium/kernels/metal/ops.mm`: Objective-C++ dispatch + pybind exports.
- `sensorium/kernels/metal/jit.py`: builds `ops.mm` and `manifold_ops.metallib`
  via `torch.utils.cpp_extension.load` + `xcrun metal/metallib`.
- `sensorium/kernels/metal/manifold_physics.py`: Python domain wrapper using
  the bound ops.

Thermodynamics "optimized" pipeline:

1) Sort-based particle scatter (counting-sort / binning)
   - `scatter_compute_cell_idx`: particle -> periodic cell id
   - `scatter_count_cells`: per-cell counts (atomics)
   - prefix sum for cell starts (exclusive)
   - `scatter_reorder_particles`: reorder into cell-contiguous layout
   - `scatter_sorted`: CIC deposit (8 corners) into `(rho, mom, e_int)`

2) Gas grid RK2 (dual-energy)
   - `gas_rk2_stage1`, `gas_rk2_stage2`
   - Rusanov/LLF flux on conserved variables + pressure work + thermal diffusion
   - Periodic neighbor stencils
   - "NaN poisoning": inadmissible states write qNaNs; host halves `dt` and retries

3) PIC gather + particle update
   - `pic_gather_update_particles`: CIC gather of `(rho,mom,e_int)` at particle pos
   - Vacuum rule: if `rho <= 0` -> `u=0, T=0, heat=0`
   - Gravity: gradient of `phi` via face-interpolated trilinear gradient
   - Periodic wrap of positions into the domain

Important semantics / edge cases (Metal):

- Periodic topology everywhere (grid stencils + particle positions).
- Dual-energy semantics:
  - Grid `E_field` is internal energy density (thermal only).
  - Scatter deposits `particle_heat` only (oscillator energy does not go to grid).
- Low-density envelope and admissibility checks:
  - Near-vacuum cells allow small negative energy envelopes to avoid spurious
    blow-ups; outside the envelope, any invalid state poisons with NaNs.
- Atomics:
  - Uses CAS-based float atomics for `rho/mom/e_int` deposition.

Performance tricks (Metal):

- Scatter is split into a bin+reorder stage to improve memory locality during
  CIC deposit.
- Uses a uniform 1D dispatch with a fixed threadgroup size (256 threads).
- In-kernel logging buffers support debugging NaN sources.

## Triton Backend (CUDA)

Existing patterns and integration:

- `sensorium/kernels/triton/*.py` uses the repo's standard optional-Triton
  import pattern (dummy `triton.jit` decorator + `_require_triton()` guard).
- Triton backend is pure Python (no C++ extension).
- Runtime selection lives in `sensorium/kernels/runtime.py`.

Thermodynamics Metal-parity kernels:

- `sensorium/kernels/triton/thermo_metal_kernels.py`
  - Sort-based scatter: compute cell ids -> count -> reorder -> `scatter_sorted`
  - Gas RK2: stage1/stage2 with admissibility + NaN poisoning
  - PIC gather/update: fused gather + gravity + periodic wrap
  - Autotune configs choose `BLOCK` and `num_warps` per kernel keyed by `N` or
    `num_cells`.

CUDA domain wiring:

- `sensorium/kernels/triton/manifold_physics.py`
  - `ThermodynamicsDomain.pic_scatter_conserved(...)` uses the sort-based Triton
    pipeline and deposits `heat` only into `e_int_field` (Metal semantics).
  - `ThermodynamicsDomain.step_particles(...)` mirrors Metal control flow:
    scatter -> solve gravity -> derive `dt` constraints -> RK2 with dt-halving
    -> fused PIC gather/update -> conservative Planck exchange.

Known deltas (Metal vs Triton):

- Floating-point atomics are non-deterministic across backends; expect small
  numerical drift even with identical initial conditions.
- Triton kernels currently use a 1D flattened launch (like Metal). Further
  optimization opportunities likely exist for RK2 (3D tiling / cache reuse).
