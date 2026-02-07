---
session: ses_3c98
updated: 2026-02-07T04:57:53.870Z
---



# Session Summary

## Goal
Exhaustively identify all stencil/kernel logic performing neighbor reads, periodic boundary handling, and FFT Poisson/gravity handling across the entire thermo_manifold codebase, producing an exact map of read patterns that must be replaced by interior+halo logic for a distributed tile-aware implementation.

## Constraints & Preferences
- Every assistant message must start with exact phrase 'DIGGING IN...'
- Exhaustive search/research mode: maximize search effort, use multiple parallel searches, do not stop at first result
- Must include both Python-side orchestration AND .metal/.triton kernels
- Must identify ABI structures used for buffers
- NO modifications allowed; NO assumptions without evidence
- Context: building a new tile-aware implementation with halo exchange and distributed gravity

## Progress
### Done
- [x] Exhaustive parallel grep for: `neighbor|stencil|halo|adjacent`, `periodic|boundary|wrap|modulo`, `fft|FFT|poisson|gravity|potential`, `kernel|thread_position_in_grid`, `triton|tl.|@triton`, `struct|buffer|MTLBuffer|device *|constant *`, stencil index offset patterns
- [x] Globbed all `*.metal` files (1 file: `sensorium/kernels/metal/manifold_physics.metal`) and all `*.py` files (100+ files)
- [x] Deep-read all critical files: `gas_dynamics.py`, `pic.py`, `ops.mm`, `manifold_physics.metal` (multiple sections), `metal/manifold_physics.py`, `triton/manifold_grid_kernels.py`, `triton/manifold_physics.py`
- [x] Catalogued all ABI param structs from both `.metal` and `.mm` files with exact field layouts
- [x] Produced comprehensive categorized audit with exact file/function/line references and halo width requirements

### In Progress
- (none — audit is complete)

### Blocked
- (none)

## Key Decisions
- **Halo width = 1 cell suffices for ALL non-FFT operations**: Evidence from every stencil pattern found — 7-point Laplacian, 8-corner trilinear CIC, and 27-cell collision neighborhoods all require exactly 1-cell-deep neighbor reads
- **FFT gravity is the hardest to distribute**: Both backends use `torch.fft.fftn` on the entire `(gx,gy,gz)` domain — this is inherently global and cannot be tiled naively. A Jacobi alternative already exists in Triton (`poisson_jacobi_step_kernel`) which IS tileable with halo=1
- **GPE ω-field uses clamped (not periodic) boundaries**: The 1D left/right neighbor read at `coherence_gpe_step` clamps at edges, unlike spatial grid stencils which wrap periodically

## Next Steps
1. Design the tile descriptor struct (local dims, halo width, global offset) to replace `grid_x/y/z` in existing ABI structs (`GasGridParams`, `SortScatterParams`, `PicGatherParams`, `SpatialHashParams`, `SpatialCollisionParams`)
2. Implement halo exchange protocol for grid fields (rho, mom, E, gravity_potential, temperature)
3. Implement ghost particle exchange for particles within 1 cell of tile boundaries (needed by PIC scatter/gather and collision detection)
4. Decide gravity strategy: distributed FFT (pencil decomposition) vs. replacing with tileable Jacobi iteration (already implemented) vs. hybrid
5. Modify `torch.roll`-based periodic wraps in `gas_dynamics.py` to use halo reads instead
6. Modify `wrap_minus_one`/`wrap_plus_one` in Metal and `tl.where` wraps in Triton to read from halo region at tile boundaries
7. Handle GPE ω-field partitioning with 1-bin halo if carrier array is split across devices

## Critical Context
- **Files with stencil patterns needing halo replacement**:
  - `sensorium/kernels/gas_dynamics.py` — `central_diff_periodic()`, `laplacian_periodic()`, `navier_stokes_rhs()` — all use `torch.roll` for periodic BC
  - `sensorium/kernels/metal/manifold_physics.metal` — `gas_rhs_cell()` (L1740-1869), `gas_rk2_stage1` (L1871), `gas_rk2_stage2` (L1945), `trilinear_coords()` (L275-293), `sample_field_trilinear()` (L296-344), `sample_gradient_trilinear()` (L347-404), `scatter_sorted` (L1380), `pic_gather_update_particles` (L2014), `spatial_hash_collisions` (L995-1160, 27-cell loop at L1051-1145), `coherence_gpe_step` (L2693-2830, 1D L/R at L2780-2781)
  - `sensorium/kernels/metal/ops.mm` — C++ dispatch wrappers mirroring all param structs
  - `sensorium/kernels/metal/manifold_physics.py` — `solve_gravity()` (L1117-1140, FFT-based), `_ensure_fft_poisson_cache()` (L915-939)
  - `sensorium/kernels/triton/manifold_grid_kernels.py` — `scatter_particle_kernel` (L134-242), `poisson_jacobi_step_kernel` (L246-301), `diffuse_heat_field_kernel` (L304-358), `pic_gather_update_kernel` (L399-933)
  - `sensorium/kernels/triton/manifold_physics.py` — `solve_gravity()` (L707-724), `_ensure_fft_cache()` (L235-258)
  - `sensorium/kernels/triton/thermo_metal_kernels.py` — `gas_rk2_stage1_kernel` (L506-837), `gas_rk2_stage2_kernel` (L840-1155)
  - `sensorium/kernels/triton/spatial_hash_kernels.py` — `collision_resolve_kernel` (L100-279, 27-cell periodic traversal)
  - `sensorium/kernels/triton/manifold_physics_kernels.py` — `coherence_gpe_step_kernel` (L1055-1224, 1D clamped L/R)
  - `sensorium/kernels/pic.py` — `cic_stencil_periodic()` (L27+), `scatter_conserved_cic()`, `gather_trilinear()`, `gather_trilinear_vec3()`

- **ABI structs that encode global grid dims (must be modified for tiling)**:
  - `GasGridParams` — `{num_cells, grid_x/y/z, dx, dt, gamma, c_v, rho_min, p_min, mu, k_thermal}`
  - `SortScatterParams` — `{num_particles, num_cells, grid_x/y/z, grid_spacing, inv_grid_spacing}`
  - `PicGatherParams` — `{num_particles, grid_x/y/z, grid_spacing, inv_grid_spacing, dt, domain_x/y/z, gamma, R_specific, c_v, rho_min, p_min, gravity_enabled}`
  - `SpatialHashParams` — `{num_particles, grid_x/y/z, cell_size, inv_cell_size, domain_min_x/y/z}`
  - `SpatialCollisionParams` — extends SpatialHashParams + collision fields
  - `GPEParams` — `{dt, hbar_eff, mass_eff, g_interaction, energy_decay, chemical_potential, inv_domega2, anchors, rng_seed, anchor_eps}`

- **Halo requirements summary**: All grid stencils need halo=1. Collision needs ghost particles in 1-cell border. FFT gravity is global (needs distributed FFT or Jacobi replacement). GPE ω-field needs 1-bin halo if partitioned.

## File Operations
### Read
- `sensorium/kernels/gas_dynamics.py` (full)
- `sensorium/kernels/pic.py` (full)
- `sensorium/kernels/metal/ops.mm` (full, truncated at ~50KB)
- `sensorium/kernels/metal/manifold_physics.metal` (lines 270-420, 1040-1170, 1489-1909)
- `sensorium/kernels/metal/manifold_physics.py` (lines 905-1155)
- `sensorium/kernels/triton/manifold_grid_kernels.py` (lines 130-370)
- `sensorium/kernels/triton/manifold_physics.py` (lines 100-280)

### Modified
- (none — read-only audit per constraints)
